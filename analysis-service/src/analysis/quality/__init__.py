"""
Quality monitoring and alerting system for the analysis service.

This module provides comprehensive quality monitoring with SRP-compliant components
consolidated from the original analysis module:
- Interfaces and data structures
- Quality monitoring with metrics storage
- Degradation detection algorithms
- Alert management and lifecycle
- Multi-channel alert handlers
- Complete alerting system orchestration
"""

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
from .monitoring import QualityMonitor
from .notification_system import NotificationSystem

# Legacy models for backward compatibility
from .models import (
    AlertRule,
    QualityAlert,
    ThresholdConfig,
)

__all__ = [
    # Interfaces
    "IQualityMonitor",
    "IQualityDetector",
    "IAlertHandler",
    "IAlertManager",
    "IQualityAlertingSystem",
    # Data structures
    "QualityMetric",
    "QualityMetricType",
    "QualityThreshold",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "DegradationDetection",
    "DegradationType",
    # Core components
    "QualityMonitor",
    "QualityDegradationDetector",
    "AlertManager",
    "QualityAlertingSystem",
    # Alert handlers
    "BaseAlertHandler",
    "LoggingAlertHandler",
    "EmailAlertHandler",
    "SlackAlertHandler",
    "WebhookAlertHandler",
    "CompositeAlertHandler",
    # Legacy components
    "NotificationSystem",
    "QualityAlert",
    "AlertRule",
    "ThresholdConfig",
]
