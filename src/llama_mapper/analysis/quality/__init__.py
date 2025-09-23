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

from .interfaces import (
    QualityMetricType, AlertSeverity, AlertStatus, DegradationType,
    QualityMetric, QualityThreshold, DegradationDetection, Alert,
    IQualityMonitor, IQualityDetector, IAlertHandler, IAlertManager,
    IQualityAlertingSystem
)

from .monitoring.quality_monitor import QualityMonitor
from .detection.degradation_detector import QualityDegradationDetector
from .alerting.alert_manager import AlertManager
from .alerting.alert_handlers import (
    LoggingAlertHandler, EmailAlertHandler, SlackAlertHandler,
    WebhookAlertHandler, CompositeAlertHandler
)
from .quality_alerting_system import QualityAlertingSystem
from .config.quality_config import (
    QualityAlertingConfig, QualityAlertingSettings,
    EmailConfig, SlackConfig, WebhookConfig, QualityThresholdConfig
)

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
    "QualityThresholdConfig"
]
