"""
Service Health Monitor for Analysis Services.

Provides comprehensive health checking, performance monitoring,
and alerting for analysis services with configurable thresholds.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..domain import IAnalysisEngine

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold: float
    status: HealthStatus
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    unit: str = ""
    description: str = ""


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service_name: str
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    check_duration_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass
class HealthAlert:
    """Health alert notification."""
    service_name: str
    metric_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class IHealthChecker:
    """Interface for service health checkers."""
    
    async def check_health(self, service: IAnalysisEngine) -> HealthCheckResult:
        """Perform health check on a service."""
        pass


class DefaultHealthChecker(IHealthChecker):
    """Default health checker implementation."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize default health checker.
        
        Args:
            thresholds: Custom thresholds for health metrics
        """
        self.thresholds = thresholds or {
            "response_time_ms": 1000.0,
            "error_rate_percent": 5.0,
            "memory_usage_mb": 1024.0,
            "cpu_usage_percent": 80.0
        }
    
    async def check_health(self, service: IAnalysisEngine) -> HealthCheckResult:
        """Perform comprehensive health check."""
        start_time = time.time()
        metrics = []
        overall_status = HealthStatus.HEALTHY
        error_message = None
        
        try:
            # Check response time
            response_time = await self._check_response_time(service)
            metrics.append(HealthMetric(
                name="response_time_ms",
                value=response_time,
                threshold=self.thresholds["response_time_ms"],
                status=self._get_status_for_threshold(response_time, self.thresholds["response_time_ms"]),
                unit="ms",
                description="Service response time"
            ))
            
            # Check error rate
            error_rate = await self._check_error_rate(service)
            metrics.append(HealthMetric(
                name="error_rate_percent",
                value=error_rate,
                threshold=self.thresholds["error_rate_percent"],
                status=self._get_status_for_threshold(error_rate, self.thresholds["error_rate_percent"]),
                unit="%",
                description="Service error rate"
            ))
            
            # Check memory usage
            memory_usage = await self._check_memory_usage(service)
            metrics.append(HealthMetric(
                name="memory_usage_mb",
                value=memory_usage,
                threshold=self.thresholds["memory_usage_mb"],
                status=self._get_status_for_threshold(memory_usage, self.thresholds["memory_usage_mb"]),
                unit="MB",
                description="Service memory usage"
            ))
            
            # Check CPU usage
            cpu_usage = await self._check_cpu_usage(service)
            metrics.append(HealthMetric(
                name="cpu_usage_percent",
                value=cpu_usage,
                threshold=self.thresholds["cpu_usage_percent"],
                status=self._get_status_for_threshold(cpu_usage, self.thresholds["cpu_usage_percent"]),
                unit="%",
                description="Service CPU usage"
            ))
            
            # Determine overall status
            overall_status = self._calculate_overall_status(metrics)
            
        except Exception as e:
            error_message = str(e)
            overall_status = HealthStatus.CRITICAL
            logger.error(f"Health check failed for {service.__class__.__name__}: {e}")
        
        check_duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return HealthCheckResult(
            service_name=service.__class__.__name__,
            overall_status=overall_status,
            metrics=metrics,
            check_duration_ms=check_duration,
            error_message=error_message
        )
    
    async def _check_response_time(self, service: IAnalysisEngine) -> float:
        """Check service response time."""
        if hasattr(service, 'get_response_time'):
            return await service.get_response_time()
        
        # Fallback: measure a simple health ping
        start_time = time.time()
        if hasattr(service, 'health_ping'):
            await service.health_ping()
        end_time = time.time()
        
        return (end_time - start_time) * 1000  # Convert to milliseconds
    
    async def _check_error_rate(self, service: IAnalysisEngine) -> float:
        """Check service error rate."""
        if hasattr(service, 'get_error_rate'):
            return await service.get_error_rate()
        
        # Fallback: return 0 if no error tracking
        return 0.0
    
    async def _check_memory_usage(self, service: IAnalysisEngine) -> float:
        """Check service memory usage."""
        if hasattr(service, 'get_memory_usage'):
            return await service.get_memory_usage()
        
        # Fallback: basic memory check
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    async def _check_cpu_usage(self, service: IAnalysisEngine) -> float:
        """Check service CPU usage."""
        if hasattr(service, 'get_cpu_usage'):
            return await service.get_cpu_usage()
        
        # Fallback: basic CPU check
        import psutil
        return psutil.cpu_percent(interval=0.1)
    
    def _get_status_for_threshold(self, value: float, threshold: float) -> HealthStatus:
        """Get health status based on threshold comparison."""
        if value <= threshold * 0.7:
            return HealthStatus.HEALTHY
        elif value <= threshold:
            return HealthStatus.DEGRADED
        elif value <= threshold * 1.5:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL
    
    def _calculate_overall_status(self, metrics: List[HealthMetric]) -> HealthStatus:
        """Calculate overall health status from individual metrics."""
        if not metrics:
            return HealthStatus.UNKNOWN
        
        # Find the worst status
        status_priority = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
            HealthStatus.CRITICAL: 3,
            HealthStatus.UNKNOWN: 4
        }
        
        worst_status = HealthStatus.HEALTHY
        for metric in metrics:
            if status_priority[metric.status] > status_priority[worst_status]:
                worst_status = metric.status
        
        return worst_status


class ServiceHealthMonitor:
    """
    Comprehensive service health monitor.
    
    Provides continuous health monitoring, alerting, and performance tracking
    for analysis services with configurable check intervals and thresholds.
    """
    
    def __init__(self, 
                 health_checker: Optional[IHealthChecker] = None,
                 check_interval: int = 30,
                 alert_cooldown: int = 300):
        """
        Initialize service health monitor.
        
        Args:
            health_checker: Custom health checker implementation
            check_interval: Health check interval in seconds
            alert_cooldown: Cooldown period for alerts in seconds
        """
        self.health_checker = health_checker or DefaultHealthChecker()
        self.check_interval = check_interval
        self.alert_cooldown = alert_cooldown
        
        # Service registry and state
        self.monitored_services: Dict[str, IAnalysisEngine] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.alert_handlers: List[Callable[[HealthAlert], None]] = []
        self.last_alerts: Dict[str, datetime] = {}
        
        # Monitoring control
        self.monitoring_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.is_monitoring = False
        
        # Configuration
        self.max_history_size = 100
        self.alert_thresholds = {
            HealthStatus.DEGRADED: AlertSeverity.WARNING,
            HealthStatus.UNHEALTHY: AlertSeverity.ERROR,
            HealthStatus.CRITICAL: AlertSeverity.CRITICAL
        }
    
    def register_service(self, name: str, service: IAnalysisEngine) -> None:
        """
        Register a service for health monitoring.
        
        Args:
            name: Service name
            service: Service instance to monitor
        """
        self.monitored_services[name] = service
        self.health_history[name] = []
        logger.info(f"Registered service for health monitoring: {name}")
    
    def unregister_service(self, name: str) -> None:
        """
        Unregister a service from health monitoring.
        
        Args:
            name: Service name to unregister
        """
        if name in self.monitored_services:
            del self.monitored_services[name]
            del self.health_history[name]
            logger.info(f"Unregistered service from health monitoring: {name}")
    
    def add_alert_handler(self, handler: Callable[[HealthAlert], None]) -> None:
        """
        Add an alert handler.
        
        Args:
            handler: Function to handle health alerts
        """
        self.alert_handlers.append(handler)
    
    def remove_alert_handler(self, handler: Callable[[HealthAlert], None]) -> None:
        """
        Remove an alert handler.
        
        Args:
            handler: Handler function to remove
        """
        try:
            self.alert_handlers.remove(handler)
        except ValueError:
            pass
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self.is_monitoring = True
        self.shutdown_event.clear()
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started service health monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.is_monitoring:
            return
        
        self.shutdown_event.set()
        
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.is_monitoring = False
        logger.info("Stopped service health monitoring")
    
    async def check_service_health(self, service_name: str) -> Optional[HealthCheckResult]:
        """
        Perform health check on a specific service.
        
        Args:
            service_name: Name of the service to check
            
        Returns:
            Health check result or None if service not found
        """
        if service_name not in self.monitored_services:
            logger.warning(f"Service not found for health check: {service_name}")
            return None
        
        service = self.monitored_services[service_name]
        result = await self.health_checker.check_health(service)
        
        # Store in history
        self._store_health_result(service_name, result)
        
        # Check for alerts
        await self._check_for_alerts(result)
        
        return result
    
    async def check_all_services(self) -> Dict[str, HealthCheckResult]:
        """
        Perform health check on all registered services.
        
        Returns:
            Dictionary mapping service names to health check results
        """
        results = {}
        
        # Run health checks in parallel
        tasks = []
        for service_name in self.monitored_services.keys():
            task = asyncio.create_task(self.check_service_health(service_name))
            tasks.append((service_name, task))
        
        # Collect results
        for service_name, task in tasks:
            try:
                result = await task
                if result:
                    results[service_name] = result
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
        
        return results
    
    def get_service_health_status(self, service_name: str) -> Optional[HealthStatus]:
        """
        Get current health status of a service.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Current health status or None if not available
        """
        history = self.health_history.get(service_name, [])
        if history:
            return history[-1].overall_status
        return None
    
    def get_service_health_history(self, service_name: str, limit: int = 10) -> List[HealthCheckResult]:
        """
        Get health check history for a service.
        
        Args:
            service_name: Name of the service
            limit: Maximum number of results to return
            
        Returns:
            List of recent health check results
        """
        history = self.health_history.get(service_name, [])
        return history[-limit:] if history else []
    
    def get_all_service_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status summary for all services.
        
        Returns:
            Dictionary mapping service names to status information
        """
        status = {}
        
        for service_name in self.monitored_services.keys():
            current_status = self.get_service_health_status(service_name)
            recent_history = self.get_service_health_history(service_name, 5)
            
            status[service_name] = {
                "current_status": current_status.value if current_status else "unknown",
                "last_check": recent_history[-1].timestamp.isoformat() if recent_history else None,
                "check_count": len(self.health_history.get(service_name, [])),
                "recent_checks": len([r for r in recent_history if r.overall_status == HealthStatus.HEALTHY])
            }
        
        return status
    
    def get_unhealthy_services(self) -> List[str]:
        """
        Get list of services that are not healthy.
        
        Returns:
            List of unhealthy service names
        """
        unhealthy = []
        
        for service_name in self.monitored_services.keys():
            status = self.get_service_health_status(service_name)
            if status and status != HealthStatus.HEALTHY:
                unhealthy.append(service_name)
        
        return unhealthy
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                # Perform health checks on all services
                await self.check_all_services()
                
                # Wait for next check interval
                await asyncio.wait_for(
                    self.shutdown_event.wait(), 
                    timeout=self.check_interval
                )
                
            except asyncio.TimeoutError:
                # Normal timeout, continue monitoring
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    def _store_health_result(self, service_name: str, result: HealthCheckResult) -> None:
        """Store health check result in history."""
        if service_name not in self.health_history:
            self.health_history[service_name] = []
        
        history = self.health_history[service_name]
        history.append(result)
        
        # Limit history size
        if len(history) > self.max_history_size:
            history.pop(0)
    
    async def _check_for_alerts(self, result: HealthCheckResult) -> None:
        """Check if alerts should be generated for health check result."""
        service_name = result.service_name
        
        # Check if we should alert based on overall status
        if result.overall_status in self.alert_thresholds:
            # Check alert cooldown
            alert_key = f"{service_name}:overall"
            if self._should_send_alert(alert_key):
                severity = self.alert_thresholds[result.overall_status]
                alert = HealthAlert(
                    service_name=service_name,
                    metric_name="overall_status",
                    severity=severity,
                    message=f"Service {service_name} is {result.overall_status.value}",
                    current_value=0.0,  # Status doesn't have numeric value
                    threshold=0.0
                )
                
                await self._send_alert(alert)
                self.last_alerts[alert_key] = datetime.now(timezone.utc)
        
        # Check individual metrics for alerts
        for metric in result.metrics:
            if metric.status in self.alert_thresholds:
                alert_key = f"{service_name}:{metric.name}"
                if self._should_send_alert(alert_key):
                    severity = self.alert_thresholds[metric.status]
                    alert = HealthAlert(
                        service_name=service_name,
                        metric_name=metric.name,
                        severity=severity,
                        message=f"Metric {metric.name} is {metric.status.value}: {metric.value}{metric.unit} (threshold: {metric.threshold}{metric.unit})",
                        current_value=metric.value,
                        threshold=metric.threshold
                    )
                    
                    await self._send_alert(alert)
                    self.last_alerts[alert_key] = datetime.now(timezone.utc)
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if an alert should be sent based on cooldown period."""
        if alert_key not in self.last_alerts:
            return True
        
        last_alert_time = self.last_alerts[alert_key]
        time_since_last = (datetime.now(timezone.utc) - last_alert_time).total_seconds()
        
        return time_since_last >= self.alert_cooldown
    
    async def _send_alert(self, alert: HealthAlert) -> None:
        """Send alert to all registered handlers."""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")


def create_default_health_monitor(check_interval: int = 30) -> ServiceHealthMonitor:
    """
    Create a health monitor with default settings.
    
    Args:
        check_interval: Health check interval in seconds
        
    Returns:
        Configured service health monitor
    """
    return ServiceHealthMonitor(
        health_checker=DefaultHealthChecker(),
        check_interval=check_interval,
        alert_cooldown=300  # 5 minutes
    )