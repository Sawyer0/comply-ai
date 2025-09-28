"""Health monitoring CLI commands following SRP.

This module provides ONLY health monitoring CLI commands.
Single Responsibility: Handle CLI commands for health check operations.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from shared.exceptions.base import BaseServiceException

try:
    import yaml
except ImportError:  # pragma: no cover - yaml optional for CLI formatting
    yaml = None

logger = logging.getLogger(__name__)


class HealthCLI:
    """CLI commands for health monitoring."""

    def __init__(self, orchestration_service) -> None:
        self.service = orchestration_service

    async def service_status(self, output_format: str = "table") -> str:
        """Show overall service status."""

        try:
            status = await self.service.get_service_status()
        except BaseServiceException as exc:
            return self._format_error("get service status", exc)

        if output_format == "json":
            return json.dumps(status, indent=2)

        if output_format == "yaml":
            if yaml is None:
                return "PyYAML not installed; use --format json instead"
            return yaml.dump(status, default_flow_style=False)  # type: ignore[arg-type]

        lines = ["Orchestration Service Status:", "=" * 50]
        lines.append(f"Service: {status.get('service', 'Unknown')}")
        lines.append(f"Status: {status.get('status', 'Unknown')}")
        lines.append(f"Version: {status.get('version', 'Unknown')}")
        lines.append(f"Uptime: {status.get('uptime', 'Unknown')}")

        components = status.get("components", {})
        if components:
            lines.append("\nComponents:")
            for component, comp_status in components.items():
                lines.append(f"  - {component}: {comp_status}")

        metrics = status.get("metrics", {})
        if metrics:
            lines.append("\nMetrics:")
            for metric, value in metrics.items():
                lines.append(f"  {metric}: {value}")

        return "\n".join(lines)

    async def health_check(self, component: Optional[str] = None) -> str:
        """Perform health check."""

        monitor = getattr(self.service, "health_monitor", None)
        if not monitor:
            return "Health monitoring is disabled"

        try:
            if component:
                health_check = await monitor.check_service_health(component)
                if not health_check:
                    return f"Component '{component}' not found"

                lines = [f"Health Check - {component}:"]
                lines.append(f"Status: {health_check.status.value}")
                if health_check.response_time_ms:
                    lines.append(f"Response Time: {health_check.response_time_ms}ms")
                if health_check.error_message:
                    lines.append(f"Error: {health_check.error_message}")
                return "\n".join(lines)

            await monitor.check_all_services()
            health_summary = monitor.get_health_summary()
        except BaseServiceException as exc:
            return self._format_error("perform health check", exc)

        lines = ["System Health Check:", "=" * 50]
        lines.append(f"Total Services: {health_summary.get('total_services', 0)}")
        lines.append(f"Healthy Services: {health_summary.get('healthy_services', 0)}")
        lines.append(f"Unhealthy Services: {health_summary.get('unhealthy_services', 0)}")
        return "\n".join(lines)

    async def metrics_summary(self) -> str:
        """Show metrics summary."""

        collector = getattr(self.service, "metrics_collector", None)
        if not collector:
            return "Metrics collection is disabled"

        try:
            metrics = collector.get_metrics_summary()
        except BaseServiceException as exc:
            return self._format_error("get metrics summary", exc)

        lines = ["Metrics Summary:", "=" * 50]
        for metric_name, metric_value in metrics.items():
            lines.append(f"{metric_name}: {metric_value}")
        return "\n".join(lines)

    async def cache_status(self) -> str:
        """Show cache status."""

        cache_component = getattr(self.service, "cache", None)
        if not cache_component:
            return "Caching is disabled"

        try:
            cache_stats = cache_component.get_statistics()
            is_healthy = await cache_component.health_check()
        except BaseServiceException as exc:
            return self._format_error("get cache status", exc)

        lines = ["Cache Status:", "=" * 50]
        lines.append(f"Cache Hits: {cache_stats.get('hits', 0)}")
        lines.append(f"Cache Misses: {cache_stats.get('misses', 0)}")
        lines.append(f"Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
        lines.append(f"Total Keys: {cache_stats.get('total_keys', 0)}")
        lines.append(f"Health: {'Healthy' if is_healthy else 'Unhealthy'}")
        return "\n".join(lines)

    async def job_status(self) -> str:
        """Show async job processor status."""

        job_processor = getattr(self.service, "job_processor", None)
        if not job_processor:
            return "Async job processing is disabled"

        try:
            job_stats = job_processor.get_statistics()
        except BaseServiceException as exc:
            return self._format_error("get job status", exc)

        lines = ["Job Processor Status:", "=" * 50]
        lines.append(f"Total Jobs: {job_stats.get('total_jobs', 0)}")
        lines.append(f"Completed Jobs: {job_stats.get('completed_jobs', 0)}")
        lines.append(f"Failed Jobs: {job_stats.get('failed_jobs', 0)}")
        lines.append(f"Active Jobs: {job_stats.get('active_jobs', 0)}")
        return "\n".join(lines)

    async def tenant_stats(self, tenant_id: Optional[str] = None) -> str:
        """Show tenant statistics."""

        tenant_manager = getattr(self.service, "tenant_manager", None)
        if not tenant_manager:
            return "Tenant management is disabled"

        try:
            if tenant_id:
                tenant = tenant_manager.get_tenant(tenant_id)
                if not tenant:
                    return f"Tenant '{tenant_id}' not found"

                lines = [f"Tenant Statistics - {tenant_id}:", "=" * 50]
                lines.append(f"Name: {tenant.name}")
                lines.append(f"Status: {tenant.status.value}")
                lines.append(f"Tier: {tenant.tier.value}")
                lines.append(f"Contact: {tenant.contact_email}")
                return "\n".join(lines)

            tenant_stats = tenant_manager.get_tenant_stats()
        except BaseServiceException as exc:
            return self._format_error("get tenant stats", exc)

        lines = ["Tenant System Statistics:", "=" * 50]
        lines.append(f"Total Tenants: {tenant_stats.get('total_tenants', 0)}")
        lines.append(f"Active Tenants: {tenant_stats.get('active_tenants', 0)}")
        lines.append(f"Suspended Tenants: {tenant_stats.get('suspended_tenants', 0)}")
        return "\n".join(lines)

    @staticmethod
    def _format_error(action: str, exc: BaseServiceException) -> str:
        logger.error("Failed to %s: %s", action, exc)
        return f"Error attempting to {action}: {exc}"


__all__ = ["HealthCLI"]
