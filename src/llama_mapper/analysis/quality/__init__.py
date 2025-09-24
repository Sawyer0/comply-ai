"""
Quality monitoring and alerting system.

This package provides comprehensive quality monitoring, degradation detection,
and alerting capabilities for the analysis module.

Key Components:
- Quality monitoring with metrics tracking and statistical analysis
- Degradation detection with multiple algorithms (thresholds, anomalies, trends)
- Alert management with deduplication and lifecycle management
- Multiple alert handlers (email, Slack, webhooks, logging)
- Configurable thresholds and alert routing
"""

from .alerting.alert_handlers import (
    CompositeAlertHandler,
    EmailAlertHandler,
    LoggingAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
)
from .alerting.alert_manager import AlertManager
from .config.quality_config import (
    EmailConfig,
    QualityAlertingConfig,
    QualityAlertingSettings,
    QualityThresholdConfig,
    SlackConfig,
    WebhookConfig,
)
from .detection.degradation_detector import QualityDegradationDetector
from .interfaces import (
    Alert,
    AlertSeverity,
    AlertStatus,
    DegradationDetection,
    DegradationType,
    IAlertHandler,
    IAlertManager,
    IQualityAlertingSystem,
    IQualityDetector,
    IQualityMonitor,
    QualityMetric,
    QualityMetricType,
    QualityThreshold,
)
from .monitoring.quality_monitor import QualityMonitor
from .quality_alerting_system import QualityAlertingSystem

__all__ = [
    # Interfaces
    "QualityMetricType",
    "AlertSeverity",
    "AlertStatus",
    "DegradationType",
    "QualityMetric",
    "QualityThreshold",
    "DegradationDetection",
    "Alert",
    "IQualityMonitor",
    "IQualityDetector",
    "IAlertHandler",
    "IAlertManager",
    "IQualityAlertingSystem",
    # Implementations
    "QualityMonitor",
    "QualityDegradationDetector",
    "AlertManager",
    "LoggingAlertHandler",
    "EmailAlertHandler",
    "SlackAlertHandler",
    "WebhookAlertHandler",
    "CompositeAlertHandler",
    "QualityAlertingSystem",
    # Configuration
    "QualityAlertingConfig",
    "QualityAlertingSettings",
    "EmailConfig",
    "SlackConfig",
    "WebhookConfig",
    "QualityThresholdConfig",
]
