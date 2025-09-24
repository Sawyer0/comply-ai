"""
Service Lifecycle Manager for analysis engines.

This manager handles initialization, health checking, and shutdown
of analysis services with proper error handling and monitoring.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from ..domain import IAnalysisEngine

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service lifecycle states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ServiceInfo:
    """Information about a managed service."""
    
    def __init__(self, name: str, service: IAnalysisEngine):
        self.name = name
        self.service = service
        self.state = ServiceState.UNINITIALIZED
        self.health_status = HealthStatus.UNKNOWN
        self.last_health_check = None
        self.start_time = None
        self.error_count = 0
        self.last_error = None
        self.metadata: Dict[str, Any] = {}


class ServiceLifecycleManager:
    """
    Manages the lifecycle of analysis services.
    
    Provides initialization, health monitoring, graceful shutdown,
    and error recovery for analysis engines.
    """
    
    def __init__(self, health_check_interval: int = 30):
        """
        Initialize the service lifecycle manager.
        
        Args:
            health_check_interval: Interval between health checks in seconds
        """
        self.services: Dict[str, ServiceInfo] = {}
        self.health_check_interval = health_check_interval
        self.health_check_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.startup_timeout = 30  # seconds
        self.shutdown_timeout = 15  # seconds
    
    def register_service(self, name: str, service: IAnalysisEngine) -> None:
        """
        Register a service for lifecycle management.
        
        Args:
            name: Service name
            service: Service instance
        """
        if name in self.services:
            logger.warning(f"Service {name} already registered, replacing")
        
        self.services[name] = ServiceInfo(name, service)
        logger.info(f"Registered service: {name}")
    
    async def start_all_services(self) -> Dict[str, bool]:
        """
        Start all registered services.
        
        Returns:
            Dictionary mapping service names to success status
        """
        results = {}
        
        logger.info(f"Starting {len(self.services)} services...")
        
        # Start services in parallel
        tasks = []
        for name, service_info in self.services.items():
            task = asyncio.create_task(self._start_service(service_info))
            tasks.append((name, task))
        
        # Wait for all services to start
        for name, task in tasks:
            try:
                success = await asyncio.wait_for(task, timeout=self.startup_timeout)
                results[name] = success
            except asyncio.TimeoutError:
                logger.error(f"Service {name} startup timed out")
                results[name] = False
                self.services[name].state = ServiceState.FAILED
            except Exception as e:
                logger.error(f"Service {name} startup failed: {e}")
                results[name] = False
                self.services[name].state = ServiceState.FAILED
        
        # Start health monitoring
        if any(results.values()):
            await self._start_health_monitoring()
        
        successful = sum(results.values())
        logger.info(f"Started {successful}/{len(results)} services successfully")
        
        return results
    
    async def stop_all_services(self) -> Dict[str, bool]:
        """
        Stop all services gracefully.
        
        Returns:
            Dictionary mapping service names to success status
        """
        results = {}
        
        logger.info(f"Stopping {len(self.services)} services...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Stop health monitoring
        await self._stop_health_monitoring()
        
        # Stop services in parallel
        tasks = []
        for name, service_info in self.services.items():
            if service_info.state in [ServiceState.RUNNING, ServiceState.DEGRADED]:
                task = asyncio.create_task(self._stop_service(service_info))
                tasks.append((name, task))
        
        # Wait for all services to stop
        for name, task in tasks:
            try:
                success = await asyncio.wait_for(task, timeout=self.shutdown_timeout)
                results[name] = success
            except asyncio.TimeoutError:
                logger.error(f"Service {name} shutdown timed out")
                results[name] = False
                # Force stop
                self.services[name].state = ServiceState.STOPPED
            except Exception as e:
                logger.error(f"Service {name} shutdown failed: {e}")
                results[name] = False
        
        successful = sum(results.values())
        logger.info(f"Stopped {successful}/{len(results)} services successfully")
        
        return results
    
    async def restart_service(self, name: str) -> bool:
        """
        Restart a specific service.
        
        Args:
            name: Service name
            
        Returns:
            True if restart successful, False otherwise
        """
        if name not in self.services:
            logger.error(f"Service {name} not found")
            return False
        
        service_info = self.services[name]
        
        logger.info(f"Restarting service: {name}")
        
        # Stop service
        stop_success = await self._stop_service(service_info)
        if not stop_success:
            logger.warning(f"Failed to stop service {name} cleanly")
        
        # Start service
        start_success = await self._start_service(service_info)
        
        if start_success:
            logger.info(f"Service {name} restarted successfully")
        else:
            logger.error(f"Failed to restart service {name}")
        
        return start_success
    
    def get_service_status(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get status information for a service.
        
        Args:
            name: Service name
            
        Returns:
            Service status dictionary or None if not found
        """
        if name not in self.services:
            return None
        
        service_info = self.services[name]
        
        uptime = None
        if service_info.start_time:
            uptime = (datetime.now(timezone.utc) - service_info.start_time).total_seconds()
        
        return {
            'name': service_info.name,
            'state': service_info.state.value,
            'health_status': service_info.health_status.value,
            'uptime_seconds': uptime,
            'error_count': service_info.error_count,
            'last_error': service_info.last_error,
            'last_health_check': service_info.last_health_check.isoformat() if service_info.last_health_check else None,
            'metadata': service_info.metadata
        }
    
    def get_all_service_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status for all services.
        
        Returns:
            Dictionary mapping service names to status information
        """
        return {name: self.get_service_status(name) for name in self.services.keys()}
    
    def get_healthy_services(self) -> List[str]:
        """
        Get list of healthy service names.
        
        Returns:
            List of healthy service names
        """
        return [
            name for name, service_info in self.services.items()
            if service_info.health_status == HealthStatus.HEALTHY
        ]
    
    def get_unhealthy_services(self) -> List[str]:
        """
        Get list of unhealthy service names.
        
        Returns:
            List of unhealthy service names
        """
        return [
            name for name, service_info in self.services.items()
            if service_info.health_status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        ]
    
    async def _start_service(self, service_info: ServiceInfo) -> bool:
        """Start a single service."""
        try:
            service_info.state = ServiceState.INITIALIZING
            
            # Initialize service if it has an initialize method
            if hasattr(service_info.service, 'initialize'):
                await service_info.service.initialize()
            
            service_info.state = ServiceState.RUNNING
            service_info.start_time = datetime.now(timezone.utc)
            service_info.health_status = HealthStatus.HEALTHY
            
            logger.info(f"Service {service_info.name} started successfully")
            return True
            
        except Exception as e:
            service_info.state = ServiceState.FAILED
            service_info.health_status = HealthStatus.UNHEALTHY
            service_info.error_count += 1
            service_info.last_error = str(e)
            
            logger.error(f"Failed to start service {service_info.name}: {e}")
            return False
    
    async def _stop_service(self, service_info: ServiceInfo) -> bool:
        """Stop a single service."""
        try:
            service_info.state = ServiceState.STOPPING
            
            # Shutdown service if it has a shutdown method
            if hasattr(service_info.service, 'shutdown'):
                await service_info.service.shutdown()
            
            service_info.state = ServiceState.STOPPED
            service_info.health_status = HealthStatus.UNKNOWN
            
            logger.info(f"Service {service_info.name} stopped successfully")
            return True
            
        except Exception as e:
            service_info.error_count += 1
            service_info.last_error = str(e)
            
            logger.error(f"Failed to stop service {service_info.name}: {e}")
            return False
    
    async def _start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        if self.health_check_task and not self.health_check_task.done():
            return
        
        self.health_check_task = asyncio.create_task(self._health_monitoring_loop())
        logger.info("Started health monitoring")
    
    async def _stop_health_monitoring(self) -> None:
        """Stop health monitoring task."""
        if self.health_check_task and not self.health_check_task.done():
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _health_monitoring_loop(self) -> None:
        """Health monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all services."""
        for service_info in self.services.values():
            if service_info.state == ServiceState.RUNNING:
                await self._check_service_health(service_info)
    
    async def _check_service_health(self, service_info: ServiceInfo) -> None:
        """Check health of a single service."""
        try:
            # Use service's health check method if available
            if hasattr(service_info.service, 'get_health_status'):
                health_status = service_info.service.get_health_status()
                
                if health_status == "healthy":
                    service_info.health_status = HealthStatus.HEALTHY
                elif health_status == "degraded":
                    service_info.health_status = HealthStatus.DEGRADED
                    service_info.state = ServiceState.DEGRADED
                else:
                    service_info.health_status = HealthStatus.UNHEALTHY
            else:
                # Basic health check - service is running
                if service_info.state == ServiceState.RUNNING:
                    service_info.health_status = HealthStatus.HEALTHY
                else:
                    service_info.health_status = HealthStatus.UNHEALTHY
            
            service_info.last_health_check = datetime.now(timezone.utc)
            
        except Exception as e:
            service_info.health_status = HealthStatus.UNHEALTHY
            service_info.error_count += 1
            service_info.last_error = str(e)
            
            logger.warning(f"Health check failed for {service_info.name}: {e}")
            
            # Consider restarting if too many errors
            if service_info.error_count >= 3:
                logger.warning(f"Service {service_info.name} has {service_info.error_count} errors, considering restart")


def create_lifecycle_manager(health_check_interval: int = 30) -> ServiceLifecycleManager:
    """
    Create a service lifecycle manager with default settings.
    
    Args:
        health_check_interval: Health check interval in seconds
        
    Returns:
        Configured service lifecycle manager
    """
    return ServiceLifecycleManager(health_check_interval)