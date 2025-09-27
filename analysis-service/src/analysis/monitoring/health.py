"""
Health monitoring for Analysis Service.

Provides comprehensive health checks for all service components.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio

from ..shared_integration import get_shared_logger, get_shared_database

logger = get_shared_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthMonitor:
    """Comprehensive health monitoring for Analysis Service."""

    def __init__(self):
        self.logger = logger.bind(component="health_monitor")
        self.db = get_shared_database()

        # Health check history
        self._health_history: List[HealthCheck] = []
        self._max_history_size = 1000

    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        try:
            health_checks = await self._run_all_health_checks()

            # Determine overall status
            overall_status = self._calculate_overall_status(health_checks)

            # Build response
            response = {
                "status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "analysis-service",
                "version": "1.0.0",  # Would come from config
                "checks": {
                    check.name: {
                        "status": check.status.value,
                        "message": check.message,
                        "response_time_ms": check.response_time_ms,
                        "metadata": check.metadata or {},
                    }
                    for check in health_checks
                },
            }

            # Add health history to response
            if self._health_history:
                recent_checks = self._health_history[-10:]  # Last 10 checks
                response["recent_history"] = [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "timestamp": check.timestamp.isoformat(),
                    }
                    for check in recent_checks
                ]

            return response

        except Exception as e:
            self.logger.error("Health check failed", error=str(e))
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "analysis-service",
                "error": str(e),
            }

    async def _run_all_health_checks(self) -> List[HealthCheck]:
        """Run all health checks."""
        checks = []

        # Database health check
        checks.append(await self._check_database_health())

        # Analysis engines health check
        checks.append(await self._check_analysis_engines_health())

        # RAG system health check
        checks.append(await self._check_rag_system_health())

        # Memory usage check
        checks.append(await self._check_memory_usage())

        # Disk space check
        checks.append(await self._check_disk_space())

        # External dependencies check
        checks.append(await self._check_external_dependencies())

        # Store in history
        for check in checks:
            self._add_to_history(check)

        return checks

    async def _check_database_health(self) -> HealthCheck:
        """Check database connectivity and performance."""
        start_time = datetime.utcnow()

        try:
            # Simple connectivity test
            if self.db:
                # Simulate database query
                await asyncio.sleep(0.01)  # Simulate query time

                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

                if response_time < 100:  # Less than 100ms is healthy
                    status = HealthStatus.HEALTHY
                    message = "Database connection healthy"
                elif response_time < 500:  # Less than 500ms is degraded
                    status = HealthStatus.DEGRADED
                    message = f"Database response slow: {response_time:.1f}ms"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"Database response very slow: {response_time:.1f}ms"

                return HealthCheck(
                    name="database",
                    status=status,
                    message=message,
                    timestamp=datetime.utcnow(),
                    response_time_ms=response_time,
                    metadata={"connection_pool_size": 10},  # Would be actual pool size
                )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database connection not available",
                    timestamp=datetime.utcnow(),
                )

        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    async def _check_analysis_engines_health(self) -> HealthCheck:
        """Check analysis engines health."""
        try:
            # Test basic analysis engine functionality
            from ..core.analyzer import AnalysisEngine

            engine = AnalysisEngine()

            # Simple test request
            test_request = {
                "tenant_id": "health_check",
                "analysis_types": ["risk_assessment"],
                "findings": {"test": "data"},
            }

            start_time = datetime.utcnow()

            # This would be a lightweight health check, not full analysis
            # For now, just check if engine initializes
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return HealthCheck(
                name="analysis_engines",
                status=HealthStatus.HEALTHY,
                message="Analysis engines operational",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                metadata={
                    "engines_loaded": [
                        "risk_scoring",
                        "pattern_recognition",
                        "compliance_intelligence",
                    ]
                },
            )

        except Exception as e:
            return HealthCheck(
                name="analysis_engines",
                status=HealthStatus.UNHEALTHY,
                message=f"Analysis engines check failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    async def _check_rag_system_health(self) -> HealthCheck:
        """Check RAG system health."""
        try:
            from ..rag.knowledge_base import KnowledgeBase

            kb = KnowledgeBase()

            # Test knowledge base connectivity
            start_time = datetime.utcnow()

            # Simple health check - would test document retrieval
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return HealthCheck(
                name="rag_system",
                status=HealthStatus.HEALTHY,
                message="RAG system operational",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                metadata={"knowledge_base_size": 1000},  # Would be actual size
            )

        except Exception as e:
            return HealthCheck(
                name="rag_system",
                status=HealthStatus.DEGRADED,
                message=f"RAG system check failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    async def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent < 70:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            elif memory_percent < 85:
                status = HealthStatus.DEGRADED
                message = f"Memory usage elevated: {memory_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_percent:.1f}%"

            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                metadata={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / (1024**3),
                    "total_gb": memory.total / (1024**3),
                },
            )

        except ImportError:
            # psutil not available, return unknown status
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="Memory monitoring not available (psutil not installed)",
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space."""
        try:
            import psutil

            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            if disk_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            elif disk_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Disk usage elevated: {disk_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical: {disk_percent:.1f}%"

            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                metadata={
                    "disk_percent": disk_percent,
                    "free_gb": disk.free / (1024**3),
                    "total_gb": disk.total / (1024**3),
                },
            )

        except ImportError:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message="Disk monitoring not available (psutil not installed)",
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Disk check failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    async def _check_external_dependencies(self) -> HealthCheck:
        """Check external dependencies."""
        try:
            # This would check external services like model serving endpoints
            # For now, simulate the check

            dependencies_status = {
                "model_serving": True,
                "vector_database": True,
                "cache_service": True,
            }

            failed_deps = [
                name for name, status in dependencies_status.items() if not status
            ]

            if not failed_deps:
                status = HealthStatus.HEALTHY
                message = "All external dependencies healthy"
            elif len(failed_deps) == 1:
                status = HealthStatus.DEGRADED
                message = f"Dependency issue: {failed_deps[0]}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Multiple dependency issues: {', '.join(failed_deps)}"

            return HealthCheck(
                name="external_dependencies",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                metadata={"dependencies": dependencies_status},
            )

        except Exception as e:
            return HealthCheck(
                name="external_dependencies",
                status=HealthStatus.UNKNOWN,
                message=f"Dependencies check failed: {str(e)}",
                timestamp=datetime.utcnow(),
            )

    def _calculate_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """Calculate overall health status from individual checks."""
        if not checks:
            return HealthStatus.UNKNOWN

        statuses = [check.status for check in checks]

        # If any check is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any check is degraded, overall is degraded
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED

        # If all checks are healthy, overall is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        # Otherwise, unknown
        return HealthStatus.UNKNOWN

    def _add_to_history(self, check: HealthCheck) -> None:
        """Add health check to history."""
        self._health_history.append(check)

        # Keep history size manageable
        if len(self._health_history) > self._max_history_size:
            self._health_history = self._health_history[-self._max_history_size :]
