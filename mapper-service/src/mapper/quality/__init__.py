"""
Quality management for mapper service.

This package provides comprehensive quality monitoring, alerting, and
degradation detection specifically designed for mapper service operations.
"""

from .alert_generator import AlertGenerator
from .alert_handlers import (
    BaseAlertHandler,
    CompositeAlertHandler,
    EmailAlertHandler,
    LoggingAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
)
from .alert_manager import AlertManager
from .alerting_system import QualityAlertingSystem
from .degradation import QualityDegradationDetector
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
from .models import QualityAlert, ThresholdConfig, AlertRule
from .monitoring import QualityMonitor
from .quality_manager import QualityManager
from .notification_system import (
    EmailNotificationChannel,
    LogNotificationChannel,
    NotificationChannel,
    NotificationSystem,
    SlackNotificationChannel,
    WebhookNotificationChannel,
)

__all__ = [
    # Interfaces
    "IQualityMonitor",
    "IQualityDetector", 
    "IAlertHandler",
    "IAlertManager",
    "IQualityAlertingSystem",
    # Models
    "QualityMetric",
    "QualityMetricType",
    "QualityThreshold",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "DegradationDetection",
    "DegradationType",
    "QualityAlert",
    "ThresholdConfig",
    "AlertRule",
    # Implementations
    "QualityMonitor",
    "QualityManager",
    "QualityDegradationDetector",
    "AlertManager",
    "QualityAlertingSystem",
    "AlertGenerator",
    # Alert Handlers
    "BaseAlertHandler",
    "LoggingAlertHandler",
    "EmailAlertHandler",
    "SlackAlertHandler",
    "WebhookAlertHandler",
    "CompositeAlertHandler",
    # Notification System
    "NotificationChannel",
    "LogNotificationChannel",
    "EmailNotificationChannel",
    "SlackNotificationChannel",
    "WebhookNotificationChannel",
    "NotificationSystem",
]
