"""Helpers for summarising service health and metrics for observability endpoints."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from detector_orchestration.config import Settings
from detector_orchestration.health_monitor import HealthMonitor
from detector_orchestration.metrics import OrchestrationMetricsCollector


class ServiceHealthReporter:
    """Builds cached service health snapshots for API handlers."""

    def __init__(
        self,
        *,
        settings: Settings,
        health_monitor: HealthMonitor,
        metrics_collector: OrchestrationMetricsCollector,
        cache_ttl_seconds: int = 10,
    ) -> None:
        self._settings = settings
        self._health_monitor = health_monitor
        self._metrics = metrics_collector
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._cached_status: Optional[Tuple[datetime, Dict[str, Any]]] = None

    def get_service_status(self) -> Dict[str, Any]:
        """Return a cached snapshot of the orchestrator status."""

        now = datetime.now(timezone.utc)
        if self._cached_status:
            cached_at, payload = self._cached_status
            if now - cached_at < self._cache_ttl:
                return payload

        status = {
            "timestamp": now.isoformat(),
            "environment": self._settings.environment,
            "version": getattr(self._settings, "version", "unknown"),
            "detectors": self._detector_status(),
            "infrastructure": self._infrastructure_status(),
            "policies": self._policy_status(),
        }
        self._cached_status = (now, status)
        return status

    async def get_detailed_health_report(self) -> Dict[str, Any]:
        """Return a full health report including derived metrics."""

        status = self.get_service_status().copy()
        status["metrics"] = {
            "total_requests": self._metrics.get_total_requests(),
            "error_rate": self._metrics.get_error_rate(),
            "average_latency_ms": self._metrics.get_average_latency(),
        }
        return status

    def _detector_status(self) -> Dict[str, Any]:
        healthy: List[str] = []
        unhealthy: List[str] = []
        for name in self._settings.detectors:
            if self._health_monitor.is_healthy(name):
                healthy.append(name)
            else:
                unhealthy.append(name)
        return {
            "total": len(self._settings.detectors),
            "healthy": len(healthy),
            "unhealthy": len(unhealthy),
            "healthy_list": healthy,
            "unhealthy_list": unhealthy,
        }

    def _infrastructure_status(self) -> Dict[str, Any]:
        config = self._settings.config
        return {
            "cache_backend": config.cache_backend,
            "redis_available": config.cache_backend == "redis"
            and config.redis_url is not None,
            "opa_enabled": config.opa_enabled,
        }

    def _policy_status(self) -> Dict[str, Any]:
        config = self._settings.config
        return {
            "opa_enabled": config.opa_enabled,
            "opa_url": config.opa_url,
            "policy_dir": config.policy_dir,
        }

