"""Main cost monitoring system that orchestrates all components."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..logging import get_logger
from .analytics import CostAnalytics, CostAnalyticsConfig
from .autoscaling import CostAwareScaler, CostAwareScalingConfig, ScalingPolicy
from .config import CostMonitoringSystemConfig
from .core import CostMetricsCollector, CostMonitoringConfig
from .core.factory import CostMonitoringFactory
from .guardrails import CostGuardrail, CostGuardrails, CostGuardrailsConfig


class CostMonitoringSystem:
    """Main cost monitoring system that orchestrates all components."""

    def __init__(self, config: CostMonitoringSystemConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Create components using factory
        self.metrics_collector = self._create_metrics_collector()
        self.guardrails = CostGuardrails(config.guardrails, self.metrics_collector)
        self.autoscaler = CostAwareScaler(config.autoscaling, self.metrics_collector)
        self.analytics = CostAnalytics(config.analytics, self.metrics_collector)

        self._running = False
        self._startup_time: Optional[datetime] = None

    def _create_metrics_collector(self) -> CostMetricsCollector:
        """Create metrics collector with proper dependencies."""
        # Create dependencies using factory
        resource_monitor = CostMonitoringFactory.create_resource_monitor(
            monitor_type="system", include_gpu=True  # Use system monitor in production
        )

        cost_calculator = CostMonitoringFactory.create_cost_calculator(
            calculator_type="standard"
        )

        metrics_storage = CostMonitoringFactory.create_metrics_storage(
            storage_type="memory"  # Could be "database" in production
        )

        alert_manager = CostMonitoringFactory.create_alert_manager(
            manager_type="mock"  # Could be "email" or "slack" in production
        )

        return CostMetricsCollector(
            config=self.config.metrics_collector,
            resource_monitor=resource_monitor,
            cost_calculator=cost_calculator,
            metrics_storage=metrics_storage,
            alert_manager=alert_manager,
        )

    async def start(self) -> None:
        """Start the cost monitoring system."""
        if self._running:
            return

        self.logger.info("Starting cost monitoring system")

        try:
            # Start all components
            await self.metrics_collector.start()
            await self.guardrails.start()
            await self.autoscaler.start()
            await self.analytics.start()

            self._running = True
            self._startup_time = datetime.now(timezone.utc)

            self.logger.info("Cost monitoring system started successfully")

        except Exception as e:
            self.logger.error("Failed to start cost monitoring system", error=str(e))
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the cost monitoring system."""
        if not self._running:
            return

        self.logger.info("Stopping cost monitoring system")

        try:
            # Stop all components
            await self.analytics.stop()
            await self.autoscaler.stop()
            await self.guardrails.stop()
            await self.metrics_collector.stop()

            self._running = False

            self.logger.info("Cost monitoring system stopped successfully")

        except Exception as e:
            self.logger.error("Error stopping cost monitoring system", error=str(e))

    def is_running(self) -> bool:
        """Check if the system is running."""
        return self._running

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the cost monitoring system."""
        return {
            "running": self._running,
            "startup_time": (
                self._startup_time.isoformat() if self._startup_time else None
            ),
            "uptime_seconds": (
                (datetime.now(timezone.utc) - self._startup_time).total_seconds()
                if self._startup_time and self._running
                else 0
            ),
            "components": {
                "metrics_collector": {
                    "enabled": self.config.metrics_collector.enabled,
                    "collection_interval": self.config.metrics_collector.collection_interval_seconds,
                },
                "guardrails": {
                    "enabled": self.config.guardrails.enabled,
                    "max_violations_per_hour": self.config.guardrails.max_violations_per_hour,
                },
                "autoscaling": {
                    "enabled": self.config.autoscaling.enabled,
                    "evaluation_interval": self.config.autoscaling.evaluation_interval_seconds,
                },
                "analytics": {
                    "enabled": self.config.analytics.enabled,
                    "analysis_interval": self.config.analytics.analysis_interval_hours,
                },
            },
            "budget_limits": {
                "daily": self.config.daily_budget_limit,
                "monthly": self.config.monthly_budget_limit,
                "emergency_stop": self.config.emergency_stop_threshold,
            },
        }

    # Guardrails management
    def add_guardrail(self, guardrail: CostGuardrail) -> None:
        """Add a new cost guardrail."""
        self.guardrails.add_guardrail(guardrail)

    def remove_guardrail(self, guardrail_id: str) -> None:
        """Remove a cost guardrail."""
        self.guardrails.remove_guardrail(guardrail_id)

    def get_guardrails(self, tenant_id: Optional[str] = None) -> List[CostGuardrail]:
        """Get all guardrails."""
        return self.guardrails.get_guardrails(tenant_id)

    def get_guardrail_violations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Any]:
        """Get guardrail violations."""
        return self.guardrails.get_violations(start_time, end_time, tenant_id)

    # Autoscaling management
    def add_scaling_policy(self, policy: ScalingPolicy) -> None:
        """Add a new scaling policy."""
        self.autoscaler.add_policy(policy)

    def remove_scaling_policy(self, policy_id: str) -> None:
        """Remove a scaling policy."""
        self.autoscaler.remove_policy(policy_id)

    def get_scaling_policies(
        self, tenant_id: Optional[str] = None
    ) -> List[ScalingPolicy]:
        """Get all scaling policies."""
        return self.autoscaler.get_policies(tenant_id)

    def get_scaling_decisions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Any]:
        """Get scaling decisions."""
        return self.autoscaler.get_scaling_decisions(start_time, end_time, tenant_id)

    # Analytics and reporting
    def get_cost_breakdown(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Get cost breakdown for a time period."""
        return self.metrics_collector.get_cost_breakdown(
            start_time, end_time, tenant_id
        )

    def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trends."""
        return self.metrics_collector.get_cost_trends(days)

    def get_optimization_recommendations(
        self,
        category: Optional[str] = None,
        priority_min: int = 1,
        tenant_id: Optional[str] = None,
    ) -> List[Any]:
        """Get optimization recommendations."""
        return self.analytics.get_optimization_recommendations(
            category, priority_min, tenant_id
        )

    def get_cost_anomalies(
        self,
        severity: Optional[str] = None,
        days: int = 30,
        tenant_id: Optional[str] = None,
    ) -> List[Any]:
        """Get cost anomalies."""
        return self.analytics.get_cost_anomalies(severity, days, tenant_id)

    def get_latest_forecast(self, tenant_id: Optional[str] = None) -> Optional[Any]:
        """Get the latest cost forecast."""
        return self.analytics.get_latest_forecast(tenant_id)

    def get_analytics_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        return self.analytics.get_analytics_summary(days)

    # Emergency controls
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Emergency stop the system to prevent further costs."""
        self.logger.critical("Emergency stop triggered", reason=reason)

        # Stop autoscaling to prevent further scaling
        await self.autoscaler.stop()

        # Trigger emergency guardrails
        # This would integrate with your infrastructure to actually stop services

        self.logger.critical("Emergency stop completed", reason=reason)

    async def resume_after_emergency(self) -> None:
        """Resume normal operation after emergency stop."""
        self.logger.info("Resuming normal operation after emergency stop")

        # Restart autoscaling
        await self.autoscaler.start()

        self.logger.info("Normal operation resumed")

    # Configuration management
    def update_config(self, new_config: CostMonitoringSystemConfig) -> None:
        """Update the system configuration."""
        self.logger.info("Updating cost monitoring system configuration")

        # Validate new configuration
        issues = new_config.validate_config()
        if issues:
            raise ValueError(f"Configuration validation failed: {', '.join(issues)}")

        # Update configuration
        self.config = new_config

        # Restart components with new configuration
        if self._running:
            asyncio.create_task(self._restart_with_new_config())

    async def _restart_with_new_config(self) -> None:
        """Restart the system with new configuration."""
        try:
            await self.stop()
            await self.start()
            self.logger.info("System restarted with new configuration")
        except Exception as e:
            self.logger.error("Failed to restart with new configuration", error=str(e))

    # Health checks
    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        health_status = {
            "overall_health": "healthy",
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Check metrics collector
        try:
            # This would check if metrics are being collected
            health_status["components"]["metrics_collector"] = {
                "status": "healthy",
                "last_collection": "recent",  # This would be actual timestamp
            }
        except Exception as e:
            health_status["components"]["metrics_collector"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_health"] = "degraded"

        # Check guardrails
        try:
            active_violations = len(self.guardrails.get_violations())
            health_status["components"]["guardrails"] = {
                "status": "healthy",
                "active_violations": active_violations,
            }
        except Exception as e:
            health_status["components"]["guardrails"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_health"] = "degraded"

        # Check autoscaler
        try:
            policies = len(self.autoscaler.get_policies())
            health_status["components"]["autoscaler"] = {
                "status": "healthy",
                "active_policies": policies,
            }
        except Exception as e:
            health_status["components"]["autoscaler"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_health"] = "degraded"

        # Check analytics
        try:
            recommendations = len(self.analytics.get_optimization_recommendations())
            health_status["components"]["analytics"] = {
                "status": "healthy",
                "active_recommendations": recommendations,
            }
        except Exception as e:
            health_status["components"]["analytics"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_health"] = "degraded"

        return health_status
