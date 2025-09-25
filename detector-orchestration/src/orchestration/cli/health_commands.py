"""Health monitoring CLI commands following SRP.

This module provides ONLY health monitoring CLI commands.
Single Responsibility: Handle CLI commands for health check operations.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HealthCLI:
    """CLI commands for health monitoring.

    Single Responsibility: Provide CLI interface for health operations.
    Does NOT handle: business logic, validation, orchestration.
    """

    def __init__(self, orchestration_service):
        """Initialize health CLI.

        Args:
            orchestration_service: OrchestrationService instance
        """
        self.service = orchestration_service

    async def service_status(self, output_format: str = "table") -> str:
        """Show overall service status.

        Args:
            output_format: Output format (table, json, yaml)

        Returns:
            Service status report
        """
        try:
            status = await self.service.get_service_status()

            if output_format == "json":
                return json.dumps(status, indent=2)

            if output_format == "yaml":
                import yaml

                return yaml.dump(status, default_flow_style=False)

            # Table format
            output = "Orchestration Service Status:\\n"
            output += "=" * 50 + "\\n"
            output += f"Service: {status.get('service', 'Unknown')}\\n"
            output += f"Status: {status.get('status', 'Unknown')}\\n"
            output += f"Version: {status.get('version', 'Unknown')}\\n"
            output += f"Uptime: {status.get('uptime', 'Unknown')}\\n\\n"

            # Component status
            components = status.get("components", {})
            if components:
                output += "Components:\\n"
                for component, comp_status in components.items():
                    status_icon = "✓" if comp_status == "enabled" else "✗"
                    output += f"  {status_icon} {component}: {comp_status}\\n"

            # Metrics
            metrics = status.get("metrics", {})
            if metrics:
                output += "\\nMetrics:\\n"
                for metric, value in metrics.items():
                    output += f"  {metric}: {value}\\n"

            return output

        except Exception as e:
            logger.error("Failed to get service status: %s", str(e))
            return f"Error getting service status: {str(e)}"

    async def health_check(self, component: Optional[str] = None) -> str:
        """Perform health check.

        Args:
            component: Optional specific component to check

        Returns:
            Health check result
        """
        try:
            if not self.service.health_monitor:
                return "Health monitoring is disabled"

            if component:
                # Check specific component
                health_check = await self.service.health_monitor.check_service_health(
                    component
                )

                output = f"Health Check - {component}:\\n"
                output += f"Status: {health_check.status.value}\\n"

                if health_check.response_time_ms:
                    output += f"Response Time: {health_check.response_time_ms}ms\\n"

                if health_check.error_message:
                    output += f"Error: {health_check.error_message}\\n"

                return output

            # Check all components
            await self.service.health_monitor.check_all_services()
            health_summary = self.service.health_monitor.get_health_summary()

            output = "System Health Check:\\n"
            output += "=" * 50 + "\\n"
            output += f"Total Services: {health_summary.get('total_services', 0)}\\n"
            output += (
                f"Healthy Services: {health_summary.get('healthy_services', 0)}\\n"
            )
            output += (
                f"Unhealthy Services: {health_summary.get('unhealthy_services', 0)}\\n"
            )

            return output

        except Exception as e:
            logger.error("Failed to perform health check: %s", str(e))
            return f"Error performing health check: {str(e)}"

    async def metrics_summary(self) -> str:
        """Show metrics summary.

        Returns:
            Metrics summary report
        """
        try:
            if not self.service.metrics_collector:
                return "Metrics collection is disabled"

            metrics = self.service.metrics_collector.get_metrics_summary()

            output = "Metrics Summary:\\n"
            output += "=" * 50 + "\\n"

            for metric_name, metric_value in metrics.items():
                output += f"{metric_name}: {metric_value}\\n"

            return output

        except Exception as e:
            logger.error("Failed to get metrics summary: %s", str(e))
            return f"Error getting metrics summary: {str(e)}"

    async def cache_status(self) -> str:
        """Show cache status.

        Returns:
            Cache status report
        """
        try:
            if not self.service.cache:
                return "Caching is disabled"

            cache_stats = self.service.cache.get_statistics()

            output = "Cache Status:\\n"
            output += "=" * 50 + "\\n"
            output += f"Cache Hits: {cache_stats.get('hits', 0)}\\n"
            output += f"Cache Misses: {cache_stats.get('misses', 0)}\\n"
            output += f"Hit Rate: {cache_stats.get('hit_rate', 0):.2%}\\n"
            output += f"Total Keys: {cache_stats.get('total_keys', 0)}\\n"

            # Health check
            is_healthy = await self.service.cache.health_check()
            health_status = "Healthy" if is_healthy else "Unhealthy"
            output += f"Health: {health_status}\\n"

            return output

        except Exception as e:
            logger.error("Failed to get cache status: %s", str(e))
            return f"Error getting cache status: {str(e)}"

    async def job_status(self) -> str:
        """Show async job processor status.

        Returns:
            Job processor status report
        """
        try:
            if not self.service.job_processor:
                return "Async job processing is disabled"

            job_stats = self.service.job_processor.get_statistics()

            output = "Job Processor Status:\\n"
            output += "=" * 50 + "\\n"
            output += f"Total Jobs: {job_stats.get('total_jobs', 0)}\\n"
            output += f"Completed Jobs: {job_stats.get('completed_jobs', 0)}\\n"
            output += f"Failed Jobs: {job_stats.get('failed_jobs', 0)}\\n"
            output += f"Active Jobs: {job_stats.get('active_jobs', 0)}\\n"

            return output

        except Exception as e:
            logger.error("Failed to get job status: %s", str(e))
            return f"Error getting job status: {str(e)}"

    async def tenant_stats(self, tenant_id: Optional[str] = None) -> str:
        """Show tenant statistics.

        Args:
            tenant_id: Optional specific tenant to check

        Returns:
            Tenant statistics report
        """
        try:
            if tenant_id:
                # Specific tenant stats
                tenant = self.service.tenant_manager.get_tenant(tenant_id)
                if not tenant:
                    return f"Tenant '{tenant_id}' not found"

                output = f"Tenant Statistics - {tenant_id}:\\n"
                output += "=" * 50 + "\\n"
                output += f"Name: {tenant.name}\\n"
                output += f"Status: {tenant.status.value}\\n"
                output += f"Tier: {tenant.tier.value}\\n"
                output += f"Contact: {tenant.contact_email}\\n"

                return output

            # All tenant stats
            tenant_stats = self.service.tenant_manager.get_tenant_stats()

            output = "Tenant System Statistics:\\n"
            output += "=" * 50 + "\\n"
            output += f"Total Tenants: {tenant_stats.get('total_tenants', 0)}\\n"
            output += f"Active Tenants: {tenant_stats.get('active_tenants', 0)}\\n"
            output += (
                f"Suspended Tenants: {tenant_stats.get('suspended_tenants', 0)}\\n"
            )

            return output

        except Exception as e:
            logger.error("Failed to get tenant stats: %s", str(e))
            return f"Error getting tenant stats: {str(e)}"


# Export only the health CLI functionality
__all__ = [
    "HealthCLI",
]
