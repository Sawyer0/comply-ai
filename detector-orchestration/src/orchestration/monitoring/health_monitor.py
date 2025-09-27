"""Health monitoring functionality following SRP.

This module provides ONLY health monitoring - tracking service health status.
Single Responsibility: Monitor and track health status of detector services.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from shared.utils.correlation import get_correlation_id
from shared.interfaces.common import HealthStatus
from ..utils.registry import run_registry_operation

logger = logging.getLogger(__name__)

class HealthCheck:
    """Health check result - data structure only."""

    def __init__(
        self,
        service_id: str,
        status: HealthStatus,
        response_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.service_id = service_id
        self.status = status
        self.response_time_ms = response_time_ms
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class HealthMonitor:
    """Monitors health status of detector services.

    Single Responsibility: Track and report health status of services.
    Does NOT handle: service discovery, circuit breakers, load balancing.
    """

    def __init__(
        self,
        health_check_interval: int = 30,
        unhealthy_threshold: int = 3,
        degraded_threshold: int = 2,
    ):
        """Initialize health monitor.

        Args:
            health_check_interval: Interval between health checks in seconds
            unhealthy_threshold: Number of consecutive failures before marking unhealthy
            degraded_threshold: Number of consecutive failures before marking degraded
        """
        self.health_check_interval = health_check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.degraded_threshold = degraded_threshold

        # Health status tracking
        self._health_status: Dict[str, HealthStatus] = {}
        self._health_history: Dict[str, List[HealthCheck]] = {}
        self._failure_counts: Dict[str, int] = {}
        self._last_check_time: Dict[str, datetime] = {}

        # Health check clients (injected)
        self._health_check_clients: Dict[str, Any] = {}

    def register_service(self, service_id: str, health_check_client: Any) -> bool:
        """Register a service for health monitoring."""

        context = {"service_id": service_id}

        def _operation() -> bool:
            self._health_check_clients[service_id] = health_check_client
            self._health_status[service_id] = HealthStatus.UNKNOWN
            self._health_history[service_id] = []
            self._failure_counts[service_id] = 0
            self._last_check_time[service_id] = datetime.utcnow()
            return True

        return run_registry_operation(
            _operation,
            logger=logger,
            success_message="Registered service %s for health monitoring",
            success_args=(service_id,),
            error_message="Failed to register service %s for health monitoring",
            error_args=(service_id,),
            log_context=context,
        )

    def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from health monitoring."""

        context = {"service_id": service_id}

        def _operation() -> bool:
            self._health_check_clients.pop(service_id, None)
            self._health_status.pop(service_id, None)
            self._health_history.pop(service_id, None)
            self._failure_counts.pop(service_id, None)
            self._last_check_time.pop(service_id, None)
            return True

        return run_registry_operation(
            _operation,
            logger=logger,
            success_message="Unregistered service %s from health monitoring",
            success_args=(service_id,),
            error_message="Failed to unregister service %s from health monitoring",
            error_args=(service_id,),
            log_context=context,
        )

    async def check_service_health(self, service_id: str) -> HealthCheck:
        """Perform health check on a specific service.

        Args:
            service_id: Unique identifier for the service

        Returns:
            Health check result
        """
        if service_id not in self._health_check_clients:
            return HealthCheck(
                service_id=service_id,
                status=HealthStatus.UNKNOWN,
                error_message="Service not registered for health monitoring",
            )

        client = self._health_check_clients[service_id]
        start_time = datetime.utcnow()

        try:
            # Perform health check (assuming client has health_check method)
            is_healthy = await client.health_check()
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            if is_healthy:
                status = HealthStatus.HEALTHY
                self._failure_counts[service_id] = 0  # Reset failure count
                error_message = None
            else:
                status = HealthStatus.UNHEALTHY
                self._failure_counts[service_id] += 1
                error_message = "Health check returned false"

            health_check = HealthCheck(
                service_id=service_id,
                status=status,
                response_time_ms=response_time,
                error_message=error_message,
            )

        except Exception as e:
            response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._failure_counts[service_id] += 1

            health_check = HealthCheck(
                service_id=service_id,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                error_message=str(e),
            )

        # Update status based on failure count
        failure_count = self._failure_counts[service_id]
        if failure_count >= self.unhealthy_threshold:
            health_check.status = HealthStatus.UNHEALTHY
        elif failure_count >= self.degraded_threshold:
            health_check.status = HealthStatus.DEGRADED

        # Update tracking
        self._health_status[service_id] = health_check.status
        self._last_check_time[service_id] = health_check.timestamp

        # Store in history (keep last 100 checks)
        if service_id not in self._health_history:
            self._health_history[service_id] = []

        self._health_history[service_id].append(health_check)
        if len(self._health_history[service_id]) > 100:
            self._health_history[service_id] = self._health_history[service_id][-100:]

        return health_check

    async def check_all_services(self) -> Dict[str, HealthCheck]:
        """Perform health checks on all registered services.

        Returns:
            Dictionary of service_id -> health check result
        """
        correlation_id = get_correlation_id()

        logger.info(
            "Performing health checks on %d services",
            len(self._health_check_clients),
            extra={
                "correlation_id": correlation_id,
                "service_count": len(self._health_check_clients),
            },
        )

        # Perform health checks in parallel
        tasks = []
        service_ids = list(self._health_check_clients.keys())

        for service_id in service_ids:
            task = self.check_service_health(service_id)
            tasks.append(task)

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_checks = {}
        for i, result in enumerate(results):
            service_id = service_ids[i]
            if isinstance(result, Exception):
                health_checks[service_id] = HealthCheck(
                    service_id=service_id,
                    status=HealthStatus.UNHEALTHY,
                    error_message=str(result),
                )
            else:
                health_checks[service_id] = result

        return health_checks

    def get_service_health(self, service_id: str) -> HealthStatus:
        """Get current health status of a service.

        Args:
            service_id: Unique identifier for the service

        Returns:
            Current health status
        """
        return self._health_status.get(service_id, HealthStatus.UNKNOWN)

    def is_service_healthy(self, service_id: str) -> bool:
        """Check if a service is healthy.

        Args:
            service_id: Unique identifier for the service

        Returns:
            True if service is healthy, False otherwise
        """
        status = self.get_service_health(service_id)
        return status == HealthStatus.HEALTHY

    def get_healthy_services(self) -> List[str]:
        """Get list of healthy services.

        Returns:
            List of service IDs that are currently healthy
        """
        return [
            service_id
            for service_id, status in self._health_status.items()
            if status == HealthStatus.HEALTHY
        ]

    def get_unhealthy_services(self) -> List[str]:
        """Get list of unhealthy services.

        Returns:
            List of service IDs that are currently unhealthy
        """
        return [
            service_id
            for service_id, status in self._health_status.items()
            if status == HealthStatus.UNHEALTHY
        ]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of health status across all services.

        Returns:
            Dictionary with health summary information
        """
        total_services = len(self._health_status)
        if total_services == 0:
            return {
                "total_services": 0,
                "healthy": 0,
                "degraded": 0,
                "unhealthy": 0,
                "unknown": 0,
                "health_percentage": 0.0,
            }

        status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}

        for status in self._health_status.values():
            status_counts[status.value] += 1

        health_percentage = (status_counts["healthy"] / total_services) * 100

        return {
            "total_services": total_services,
            "healthy": status_counts["healthy"],
            "degraded": status_counts["degraded"],
            "unhealthy": status_counts["unhealthy"],
            "unknown": status_counts["unknown"],
            "health_percentage": round(health_percentage, 2),
            "last_check_times": {
                service_id: timestamp.isoformat()
                for service_id, timestamp in self._last_check_time.items()
            },
        }

    def get_service_health_history(
        self, service_id: str, limit: int = 10
    ) -> List[HealthCheck]:
        """Get health check history for a service.

        Args:
            service_id: Unique identifier for the service
            limit: Maximum number of history entries to return

        Returns:
            List of recent health checks (most recent first)
        """
        history = self._health_history.get(service_id, [])
        return list(reversed(history[-limit:]))


# Export only the health monitoring functionality
__all__ = [
    "HealthMonitor",
    "HealthCheck",
    "HealthStatus",
]
