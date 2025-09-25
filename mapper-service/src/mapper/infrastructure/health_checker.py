"""
Database Health Checker for Mapper Service

Monitors database health and provides health check endpoints
following Single Responsibility Principle.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class HealthCheckResult(BaseModel):
    """Health check result model."""

    service: str
    status: str  # healthy, unhealthy, degraded
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = {}
    timestamp: str


class DatabaseHealthChecker:
    """
    Database health checker responsible for monitoring database health.

    Follows SRP by focusing solely on health monitoring and reporting.
    """

    def __init__(self, connection_pool_manager):
        self.connection_pool = connection_pool_manager
        self.logger = logger.bind(component="database_health_checker")

        # Health check configuration
        self.check_interval = 30  # seconds
        self.timeout = 10  # seconds
        self.max_response_time = 1000  # milliseconds

        # Health history for trend analysis
        self._health_history: List[HealthCheckResult] = []
        self._max_history_size = 100

        # Background task
        self._health_check_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._is_running:
            return

        self.logger.info("Starting database health monitoring")
        self._is_running = True
        self._health_check_task = asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self._is_running:
            return

        self.logger.info("Stopping database health monitoring")
        self._is_running = False

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def check_database_health(self) -> HealthCheckResult:
        """Perform comprehensive database health check."""
        start_time = datetime.utcnow()

        try:
            # Get connection status from pool manager
            connection_status = await self.connection_pool.get_connection_status()

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000

            # Determine overall health status
            overall_health = connection_status["overall_health"]

            # Create health check result
            result = HealthCheckResult(
                service="database",
                status=overall_health,
                response_time_ms=response_time,
                details=connection_status,
                timestamp=datetime.utcnow().isoformat(),
            )

            # Add to history
            self._add_to_history(result)

            return result

        except Exception as e:
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error("Database health check failed", error=str(e))

            result = HealthCheckResult(
                service="database",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e),
                timestamp=datetime.utcnow().isoformat(),
            )

            self._add_to_history(result)
            return result

    async def check_primary_database(self) -> HealthCheckResult:
        """Check primary database health specifically."""
        start_time = datetime.utcnow()

        try:
            primary_db = self.connection_pool.get_write_db()
            health_status = await primary_db.health_check()

            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000

            return HealthCheckResult(
                service="primary_database",
                status=health_status["status"],
                response_time_ms=response_time,
                details=health_status,
                timestamp=datetime.utcnow().isoformat(),
            )

        except Exception as e:
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds() * 1000

            return HealthCheckResult(
                service="primary_database",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e),
                timestamp=datetime.utcnow().isoformat(),
            )

    async def check_read_replicas(self) -> List[HealthCheckResult]:
        """Check all read replica health."""
        results = []

        for i in range(len(self.connection_pool.read_replicas)):
            start_time = datetime.utcnow()

            try:
                replica_db = self.connection_pool.get_read_db(preferred_replica=i)
                health_status = await replica_db.health_check()

                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds() * 1000

                result = HealthCheckResult(
                    service=f"read_replica_{i}",
                    status=health_status["status"],
                    response_time_ms=response_time,
                    details=health_status,
                    timestamp=datetime.utcnow().isoformat(),
                )

            except Exception as e:
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds() * 1000

                result = HealthCheckResult(
                    service=f"read_replica_{i}",
                    status="unhealthy",
                    response_time_ms=response_time,
                    error=str(e),
                    timestamp=datetime.utcnow().isoformat(),
                )

            results.append(result)

        return results

    async def get_health_summary(self) -> Dict[str, any]:
        """Get comprehensive health summary."""
        # Perform health checks
        database_health = await self.check_database_health()
        primary_health = await self.check_primary_database()
        replica_health = await self.check_read_replicas()

        # Calculate health metrics
        healthy_replicas = len([r for r in replica_health if r.status == "healthy"])
        total_replicas = len(replica_health)

        # Get recent health trends
        recent_checks = self._get_recent_health_checks(minutes=10)
        uptime_percentage = self._calculate_uptime(recent_checks)

        return {
            "overall_status": database_health.status,
            "primary_database": {
                "status": primary_health.status,
                "response_time_ms": primary_health.response_time_ms,
                "error": primary_health.error,
            },
            "read_replicas": {
                "healthy_count": healthy_replicas,
                "total_count": total_replicas,
                "health_percentage": (
                    (healthy_replicas / total_replicas * 100)
                    if total_replicas > 0
                    else 0
                ),
                "details": [r.dict() for r in replica_health],
            },
            "metrics": {
                "uptime_percentage": uptime_percentage,
                "avg_response_time_ms": self._calculate_avg_response_time(
                    recent_checks
                ),
                "total_checks": len(self._health_history),
                "last_check": database_health.timestamp,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_health_history(self, limit: int = 50) -> List[HealthCheckResult]:
        """Get recent health check history."""
        return self._health_history[-limit:] if self._health_history else []

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                await self.check_database_health()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health monitoring loop error", error=str(e))
                await asyncio.sleep(self.check_interval)

    def _add_to_history(self, result: HealthCheckResult) -> None:
        """Add health check result to history."""
        self._health_history.append(result)

        # Trim history if it exceeds max size
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size :]

    def _get_recent_health_checks(self, minutes: int = 10) -> List[HealthCheckResult]:
        """Get health checks from the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)

        return [
            check
            for check in self._health_history
            if datetime.fromisoformat(check.timestamp) > cutoff_time
        ]

    def _calculate_uptime(self, health_checks: List[HealthCheckResult]) -> float:
        """Calculate uptime percentage from health checks."""
        if not health_checks:
            return 0.0

        healthy_checks = len(
            [check for check in health_checks if check.status == "healthy"]
        )
        return (healthy_checks / len(health_checks)) * 100

    def _calculate_avg_response_time(
        self, health_checks: List[HealthCheckResult]
    ) -> float:
        """Calculate average response time from health checks."""
        if not health_checks:
            return 0.0

        response_times = [
            check.response_time_ms
            for check in health_checks
            if check.response_time_ms is not None
        ]

        if not response_times:
            return 0.0

        return sum(response_times) / len(response_times)
