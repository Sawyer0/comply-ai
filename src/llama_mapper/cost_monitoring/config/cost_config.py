"""Configuration management for cost monitoring and autoscaling."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from ..analytics.cost_analytics import CostAnalyticsConfig
from ..autoscaling.cost_aware_scaler import CostAwareScalingConfig
from ..core.metrics_collector import CostMonitoringConfig
from ..guardrails.cost_guardrails import CostGuardrailsConfig


class CostMonitoringSystemConfig(BaseModel):
    """Complete configuration for the cost monitoring system."""

    # Core components
    metrics_collector: CostMonitoringConfig = Field(
        default_factory=CostMonitoringConfig,
        description="Configuration for cost metrics collection",
    )
    guardrails: CostGuardrailsConfig = Field(
        default_factory=CostGuardrailsConfig,
        description="Configuration for cost guardrails",
    )
    autoscaling: CostAwareScalingConfig = Field(
        default_factory=CostAwareScalingConfig,
        description="Configuration for cost-aware autoscaling",
    )
    analytics: CostAnalyticsConfig = Field(
        default_factory=CostAnalyticsConfig,
        description="Configuration for cost analytics",
    )

    # System-wide settings
    enabled: bool = Field(
        default=True, description="Enable the entire cost monitoring system"
    )
    environment: str = Field(
        default="production",
        description="Environment (development, staging, production)",
    )
    tenant_id: Optional[str] = Field(default=None, description="Default tenant ID")

    # Integration settings
    prometheus_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics export"
    )
    grafana_enabled: bool = Field(default=True, description="Enable Grafana dashboard")
    alertmanager_enabled: bool = Field(
        default=True, description="Enable Alertmanager integration"
    )

    # Cost thresholds (system-wide)
    daily_budget_limit: float = Field(default=1000.0, description="Daily budget limit")
    monthly_budget_limit: float = Field(
        default=30000.0, description="Monthly budget limit"
    )
    emergency_stop_threshold: float = Field(
        default=5000.0, description="Emergency stop threshold"
    )

    # Notification settings
    notification_channels: Dict[str, Any] = Field(
        default_factory=lambda: {
            "email": {"enabled": True, "recipients": []},
            "slack": {"enabled": False, "webhook_url": None},
            "webhook": {"enabled": False, "url": None},
        },
        description="Notification channel configurations",
    )

    # Data retention
    metrics_retention_days: int = Field(
        default=90, description="Metrics retention period"
    )
    alerts_retention_days: int = Field(
        default=30, description="Alerts retention period"
    )
    reports_retention_days: int = Field(
        default=365, description="Reports retention period"
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CostMonitoringSystemConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def validate_config(self) -> list[str]:
        """Validate the configuration and return any issues."""
        issues = []

        # Validate budget limits
        if self.daily_budget_limit <= 0:
            issues.append("Daily budget limit must be positive")

        if self.monthly_budget_limit <= 0:
            issues.append("Monthly budget limit must be positive")

        if self.daily_budget_limit * 30 > self.monthly_budget_limit:
            issues.append(
                "Daily budget limit * 30 should not exceed monthly budget limit"
            )

        # Validate retention periods
        if self.metrics_retention_days < 1:
            issues.append("Metrics retention period must be at least 1 day")

        if self.alerts_retention_days < 1:
            issues.append("Alerts retention period must be at least 1 day")

        # Validate notification channels
        for (
            channel,
            config,
        ) in self.notification_channels.items():  # pylint: disable=no-member
            if config.get("enabled", False):
                if channel == "email" and not config.get("recipients"):
                    issues.append(
                        "Email notifications enabled but no recipients configured"
                    )
                elif channel == "slack" and not config.get("webhook_url"):
                    issues.append(
                        "Slack notifications enabled but no webhook URL configured"
                    )
                elif channel == "webhook" and not config.get("url"):
                    issues.append("Webhook notifications enabled but no URL configured")

        return issues


class CostMonitoringFactory:
    """Factory for creating cost monitoring system components."""

    @staticmethod
    def create_default_config() -> CostMonitoringSystemConfig:
        """Create a default configuration for cost monitoring."""
        return CostMonitoringSystemConfig()

    @staticmethod
    def create_development_config() -> CostMonitoringSystemConfig:
        """Create a configuration suitable for development."""
        config = CostMonitoringSystemConfig()
        config.environment = "development"
        config.daily_budget_limit = 100.0
        config.monthly_budget_limit = 3000.0
        config.emergency_stop_threshold = 500.0

        # More lenient settings for development
        config.guardrails.max_violations_per_hour = 20
        config.autoscaling.max_cost_increase_percent = 100.0

        return config

    @staticmethod
    def create_production_config() -> CostMonitoringSystemConfig:
        """Create a configuration suitable for production."""
        config = CostMonitoringSystemConfig()
        config.environment = "production"
        config.daily_budget_limit = 1000.0
        config.monthly_budget_limit = 30000.0
        config.emergency_stop_threshold = 5000.0

        # Stricter settings for production
        config.guardrails.max_violations_per_hour = 5
        config.autoscaling.max_cost_increase_percent = 25.0
        config.analytics.anomaly_threshold = 1.5  # More sensitive anomaly detection

        return config

    @staticmethod
    def create_high_performance_config() -> CostMonitoringSystemConfig:
        """Create a configuration for high-performance scenarios."""
        config = CostMonitoringSystemConfig()
        config.environment = "production"
        config.daily_budget_limit = 5000.0
        config.monthly_budget_limit = 150000.0
        config.emergency_stop_threshold = 25000.0

        # Optimized for performance over cost
        config.autoscaling.cost_threshold_percent = 90.0
        config.autoscaling.performance_threshold_percent = 50.0
        config.autoscaling.max_cost_increase_percent = 100.0

        return config

    @staticmethod
    def create_cost_optimized_config() -> CostMonitoringSystemConfig:
        """Create a configuration optimized for cost savings."""
        config = CostMonitoringSystemConfig()
        config.environment = "production"
        config.daily_budget_limit = 500.0
        config.monthly_budget_limit = 15000.0
        config.emergency_stop_threshold = 2500.0

        # Optimized for cost over performance
        config.autoscaling.cost_threshold_percent = 60.0
        config.autoscaling.performance_threshold_percent = 80.0
        config.autoscaling.max_cost_increase_percent = 10.0

        # More aggressive cost controls
        config.guardrails.max_violations_per_hour = 3
        config.analytics.anomaly_threshold = 1.0  # Very sensitive anomaly detection

        return config
