"""
Health Monitor for Mapper Service

Single Responsibility: Monitor service health and provide health check endpoints.
Coordinates health checks across all service components and provides unified status.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    error: Optional[str] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class ServiceHealth:
    """Overall service health status."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    uptime_seconds: float
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "version": self.version,
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "response_time_ms": comp.response_time_ms,
                    "last_check": (
                        comp.last_check.isoformat() if comp.last_check else None
                    ),
                    "error": comp.error,
                    "details": comp.details,
                }
                for name, comp in self.components.items()
            },
        }


class HealthMonitor:
    """
    Health Monitor for service health management.

    Single Responsibility: Coordinate health checks and provide unified health status.

    This class handles:
    - Component health check registration
    - Periodic health check execution
    - Health status aggregation
    - Health degradation detection
    - Health metrics collection
    - Health alerting coordination
    """

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.start_time = time.time()

        # Health check registry
        self._health_checks: Dict[str, Callable] = {}
        self._component_health: Dict[str, ComponentHealth] = {}
        self._overall_status = HealthStatus.UNKNOWN

        # Monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False

        self.logger = logger.bind(component="health_monitor")

    def register_health_check(self, name: str, check_func: Callable) -> None:
        """
        Register a health check function.

        Single Responsibility: Register component health checks.

        Args:
            name: Component name
            check_func: Async function that returns health status dict
        """
        self._health_checks[name] = check_func
        self._component_health[name] = ComponentHealth(
            name=name, status=HealthStatus.UNKNOWN, message="Not checked yet"
        )

        self.logger.info("Health check registered", component=name)

    def unregister_health_check(self, name: str) -> None:
        """
        Unregister a health check function.

        Single Responsibility: Remove component health checks.
        """
        self._health_checks.pop(name, None)
        self._component_health.pop(name, None)

        self.logger.info("Health check unregistered", component=name)

    async def start_monitoring(self) -> None:
        """
        Start periodic health monitoring.

        Single Responsibility: Begin continuous health monitoring.
        """
        if self._is_monitoring:
            self.logger.warning("Health monitoring already started")
            return

        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Health monitoring started", interval=self.check_interval)

    async def stop_monitoring(self) -> None:
        """
        Stop periodic health monitoring.

        Single Responsibility: Stop continuous health monitoring.
        """
        self._is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        self.logger.info("Health monitoring stopped")

    async def check_health(self) -> ServiceHealth:
        """
        Perform immediate health check of all components.

        Single Responsibility: Execute all health checks and aggregate results.
        """
        check_tasks = []

        # Execute all health checks concurrently
        for name, check_func in self._health_checks.items():
            task = asyncio.create_task(self._execute_health_check(name, check_func))
            check_tasks.append(task)

        # Wait for all checks to complete
        if check_tasks:
            await asyncio.gather(*check_tasks, return_exceptions=True)

        # Determine overall status
        self._update_overall_status()

        # Create service health response
        return ServiceHealth(
            status=self._overall_status,
            components=self._component_health.copy(),
            timestamp=datetime.utcnow(),
            uptime_seconds=time.time() - self.start_time,
        )

    async def _execute_health_check(self, name: str, check_func: Callable) -> None:
        """Execute a single health check."""
        start_time = time.time()

        try:
            # Execute health check
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()

            response_time = (time.time() - start_time) * 1000  # Convert to ms

            # Parse result
            if isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                message = result.get("message", "")
                error = result.get("error")
                details = result.get("details", {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "OK" if result else "Check failed"
                error = None
                details = {}
            else:
                status = HealthStatus.UNKNOWN
                message = f"Invalid health check result: {result}"
                error = None
                details = {}

            # Update component health
            self._component_health[name] = ComponentHealth(
                name=name,
                status=status,
                message=message,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                error=error,
                details=details,
            )

        except (RuntimeError, ValueError, TimeoutError, ConnectionError) as e:
            response_time = (time.time() - start_time) * 1000

            self._component_health[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check failed",
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                error=str(e),
            )

            self.logger.error("Health check failed", component=name, error=str(e))

    def _update_overall_status(self) -> None:
        """Update overall service health status based on component health."""
        if not self._component_health:
            self._overall_status = HealthStatus.UNKNOWN
            return

        component_statuses = [comp.status for comp in self._component_health.values()]

        # Determine overall status
        if all(status == HealthStatus.HEALTHY for status in component_statuses):
            self._overall_status = HealthStatus.HEALTHY
        elif any(status == HealthStatus.UNHEALTHY for status in component_statuses):
            self._overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in component_statuses):
            self._overall_status = HealthStatus.DEGRADED
        else:
            self._overall_status = HealthStatus.UNKNOWN

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_monitoring:
            try:
                await self.check_health()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except (RuntimeError, ValueError) as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(self.check_interval)

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        return self._component_health.get(name)

    def get_overall_status(self) -> HealthStatus:
        """Get overall service health status."""
        return self._overall_status

    def get_uptime(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self.start_time

    async def wait_for_healthy(self, timeout: float = 60.0) -> bool:
        """
        Wait for service to become healthy.

        Single Responsibility: Block until service is healthy or timeout.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            health = await self.check_health()

            if health.status == HealthStatus.HEALTHY:
                return True

            await asyncio.sleep(1.0)

        return False

    def create_readiness_check(self) -> Callable:
        """
        Create a readiness check function for Kubernetes.

        Single Responsibility: Provide readiness probe endpoint.
        """

        async def readiness_check():
            health = await self.check_health()
            return health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

        return readiness_check

    def create_liveness_check(self) -> Callable:
        """
        Create a liveness check function for Kubernetes.

        Single Responsibility: Provide liveness probe endpoint.
        """

        async def liveness_check():
            # Service is alive if monitoring is running or can perform basic checks
            if self._is_monitoring:
                return True

            # Perform minimal health check
            try:
                await self.check_health()
                return self._overall_status != HealthStatus.UNHEALTHY
            except Exception:
                return False

        return liveness_check
