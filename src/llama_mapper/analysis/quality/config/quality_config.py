"""
Quality alerting system configuration.

This module provides configuration classes for the quality alerting system
including thresholds, alert handlers, and system settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from ..interfaces import (
    QualityMetricType, QualityThreshold, AlertSeverity
)


class EmailConfig(BaseModel):
    """Email alert handler configuration."""
    smtp_server: str = Field(..., description="SMTP server hostname")
    smtp_port: int = Field(587, description="SMTP server port")
    username: str = Field(..., description="SMTP username")
    password: str = Field(..., description="SMTP password")
    from_email: str = Field(..., description="From email address")
    to_emails: List[str] = Field(..., description="List of recipient email addresses")
    use_tls: bool = Field(True, description="Whether to use TLS")
    enabled: bool = Field(True, description="Whether email alerts are enabled")


class SlackConfig(BaseModel):
    """Slack alert handler configuration."""
    webhook_url: str = Field(..., description="Slack webhook URL")
    channel: Optional[str] = Field(None, description="Slack channel")
    username: str = Field("Quality Monitor", description="Bot username")
    icon_emoji: str = Field(":warning:", description="Bot icon emoji")
    enabled: bool = Field(True, description="Whether Slack alerts are enabled")


class WebhookConfig(BaseModel):
    """Webhook alert handler configuration."""
    webhook_url: str = Field(..., description="Webhook URL")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    timeout: int = Field(10, description="Request timeout in seconds")
    enabled: bool = Field(True, description="Whether webhook alerts are enabled")


class QualityThresholdConfig(BaseModel):
    """Quality threshold configuration."""
    metric_type: QualityMetricType = Field(..., description="Metric type")
    warning_threshold: float = Field(..., description="Warning threshold value")
    critical_threshold: float = Field(..., description="Critical threshold value")
    min_samples: int = Field(10, description="Minimum samples for detection")
    time_window_minutes: int = Field(60, description="Time window in minutes")
    enabled: bool = Field(True, description="Whether threshold is enabled")


class QualityAlertingConfig(BaseModel):
    """Complete quality alerting system configuration."""
    
    # System settings
    monitoring_interval_seconds: int = Field(60, description="Monitoring interval in seconds")
    max_metrics_per_type: int = Field(10000, description="Maximum metrics per type")
    alert_retention_days: int = Field(30, description="Alert retention period in days")
    deduplication_window_minutes: int = Field(15, description="Alert deduplication window in minutes")
    
    # Alert handlers
    email_config: Optional[EmailConfig] = Field(None, description="Email alert configuration")
    slack_config: Optional[SlackConfig] = Field(None, description="Slack alert configuration")
    webhook_config: Optional[WebhookConfig] = Field(None, description="Webhook alert configuration")
    
    # Quality thresholds
    thresholds: List[QualityThresholdConfig] = Field(
        default_factory=list,
        description="Quality thresholds"
    )
    
    # Detection settings
    anomaly_sensitivity: float = Field(2.0, description="Anomaly detection sensitivity")
    trend_window_minutes: int = Field(30, description="Trend analysis window in minutes")
    min_samples_for_detection: int = Field(5, description="Minimum samples for detection")
    
    # Logging settings
    log_level: str = Field("WARNING", description="Log level for alerts")
    
    def get_default_thresholds(self) -> List[QualityThresholdConfig]:
        """Get default quality thresholds."""
        return [
            QualityThresholdConfig(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                warning_threshold=0.95,
                critical_threshold=0.90,
                min_samples=10,
                time_window_minutes=60
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.TEMPLATE_FALLBACK_RATE,
                warning_threshold=0.20,
                critical_threshold=0.30,
                min_samples=10,
                time_window_minutes=60
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
                warning_threshold=0.95,
                critical_threshold=0.90,
                min_samples=10,
                time_window_minutes=60
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.CONFIDENCE_SCORE,
                warning_threshold=0.70,
                critical_threshold=0.60,
                min_samples=10,
                time_window_minutes=60
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.RESPONSE_TIME,
                warning_threshold=2.0,
                critical_threshold=5.0,
                min_samples=10,
                time_window_minutes=60
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.ERROR_RATE,
                warning_threshold=0.05,
                critical_threshold=0.10,
                min_samples=10,
                time_window_minutes=60
            )
        ]
    
    def to_quality_thresholds(self) -> List[QualityThreshold]:
        """Convert to QualityThreshold objects."""
        from ..interfaces import QualityThreshold
        
        thresholds = []
        for config in self.thresholds:
            threshold = QualityThreshold(
                metric_type=config.metric_type,
                warning_threshold=config.warning_threshold,
                critical_threshold=config.critical_threshold,
                min_samples=config.min_samples,
                time_window_minutes=config.time_window_minutes,
                enabled=config.enabled
            )
            thresholds.append(threshold)
        
        return thresholds


@dataclass
class QualityAlertingSettings:
    """Quality alerting system settings."""
    
    # System settings
    monitoring_interval_seconds: int = 60
    max_metrics_per_type: int = 10000
    alert_retention_days: int = 30
    deduplication_window_minutes: int = 15
    
    # Detection settings
    anomaly_sensitivity: float = 2.0
    trend_window_minutes: int = 30
    min_samples_for_detection: int = 5
    
    # Logging settings
    log_level: str = "WARNING"
    
    # Email settings
    email_enabled: bool = False
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    email_use_tls: bool = True
    
    # Slack settings
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: Optional[str] = None
    slack_username: str = "Quality Monitor"
    slack_icon_emoji: str = ":warning:"
    
    # Webhook settings
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    webhook_timeout: int = 10
    
    # Threshold settings
    schema_validation_warning: float = 0.95
    schema_validation_critical: float = 0.90
    template_fallback_warning: float = 0.20
    template_fallback_critical: float = 0.30
    opa_compilation_warning: float = 0.95
    opa_compilation_critical: float = 0.90
    confidence_score_warning: float = 0.70
    confidence_score_critical: float = 0.60
    response_time_warning: float = 2.0
    response_time_critical: float = 5.0
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.10
    
    def to_config(self) -> QualityAlertingConfig:
        """Convert to QualityAlertingConfig."""
        config = QualityAlertingConfig(
            monitoring_interval_seconds=self.monitoring_interval_seconds,
            max_metrics_per_type=self.max_metrics_per_type,
            alert_retention_days=self.alert_retention_days,
            deduplication_window_minutes=self.deduplication_window_minutes,
            anomaly_sensitivity=self.anomaly_sensitivity,
            trend_window_minutes=self.trend_window_minutes,
            min_samples_for_detection=self.min_samples_for_detection,
            log_level=self.log_level
        )
        
        # Add email config if enabled
        if self.email_enabled:
            config.email_config = EmailConfig(
                smtp_server=self.email_smtp_server,
                smtp_port=self.email_smtp_port,
                username=self.email_username,
                password=self.email_password,
                from_email=self.email_from,
                to_emails=self.email_to,
                use_tls=self.email_use_tls
            )
        
        # Add Slack config if enabled
        if self.slack_enabled:
            config.slack_config = SlackConfig(
                webhook_url=self.slack_webhook_url,
                channel=self.slack_channel,
                username=self.slack_username,
                icon_emoji=self.slack_icon_emoji
            )
        
        # Add webhook config if enabled
        if self.webhook_enabled:
            config.webhook_config = WebhookConfig(
                webhook_url=self.webhook_url,
                headers=self.webhook_headers,
                timeout=self.webhook_timeout
            )
        
        # Add thresholds
        config.thresholds = [
            QualityThresholdConfig(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                warning_threshold=self.schema_validation_warning,
                critical_threshold=self.schema_validation_critical
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.TEMPLATE_FALLBACK_RATE,
                warning_threshold=self.template_fallback_warning,
                critical_threshold=self.template_fallback_critical
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
                warning_threshold=self.opa_compilation_warning,
                critical_threshold=self.opa_compilation_critical
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.CONFIDENCE_SCORE,
                warning_threshold=self.confidence_score_warning,
                critical_threshold=self.confidence_score_critical
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.RESPONSE_TIME,
                warning_threshold=self.response_time_warning,
                critical_threshold=self.response_time_critical
            ),
            QualityThresholdConfig(
                metric_type=QualityMetricType.ERROR_RATE,
                warning_threshold=self.error_rate_warning,
                critical_threshold=self.error_rate_critical
            )
        ]
        
        return config
