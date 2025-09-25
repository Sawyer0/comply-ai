"""
Quality monitoring interfaces for the analysis service.

This module defines the core interfaces and data structures for quality monitoring,
degradation detection, and alerting systems following the Interface Segregation Principle.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class QualityMetricType(Enum):
    """Types of quality metrics."""

    SCHEMA_VALIDATION_RATE = "schema_validation_rate"
    TEMPLATE_FALLBACK_RATE = "template_fallback_rate"
    OPA_COMPILATION_SUCCESS_RATE = "opa_compilation_success_rate"
    CONFIDENCE_SCORE = "confidence_score"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM_METRIC = "custom_metric"


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class DegradationType(Enum):
    """Types of quality degradation."""

    SUDDEN_DROP = "sudden_drop"
    GRADUAL_DECLINE = "gradual_decline"
    THRESHOLD_BREACH = "threshold_breach"
    ANOMALY = "anomaly"
    TREND_REVERSAL = "trend_reversal"


@dataclass
class QualityMetric:
    """Quality metric data point."""

    metric_type: QualityMetricType
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate metric data after initialization."""
        if self.value < 0:
            raise ValueError(f"Metric value cannot be negative: {self.value}")

        # Validate rate metrics (0.0-1.0)
        rate_metrics = {
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            QualityMetricType.TEMPLATE_FALLBACK_RATE,
            QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
            QualityMetricType.ERROR_RATE,
        }

        if self.metric_type in rate_metrics and self.value > 1.0:
            raise ValueError(
                f"Rate metric {self.metric_type.value} cannot exceed 1.0: {self.value}"
            )


@dataclass
class QualityThreshold:
    """Quality threshold configuration."""

    metric_type: QualityMetricType
    warning_threshold: float
    critical_threshold: float
    min_samples: int = 10
    time_window_minutes: int = 60
    enabled: bool = True

    def __post_init__(self):
        """Validate threshold configuration."""
        if self.warning_threshold <= 0:
            raise ValueError(
                f"Warning threshold must be positive: {self.warning_threshold}"
            )
        if self.critical_threshold <= 0:
            raise ValueError(
                f"Critical threshold must be positive: {self.critical_threshold}"
            )
        if self.critical_threshold == self.warning_threshold:
            raise ValueError(
                "Critical threshold must be different from warning threshold"
            )


@dataclass
class DegradationDetection:
    """Quality degradation detection result."""

    metric_type: QualityMetricType
    degradation_type: DegradationType
    severity: AlertSeverity
    current_value: float
    expected_value: float
    deviation_percentage: float
    confidence: float
    timestamp: datetime
    description: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Alert:
    """Alert data structure."""

    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    metric_type: QualityMetricType
    degradation_detection: Optional[DegradationDetection]
    created_at: datetime
    updated_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    labels: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        title: str,
        description: str,
        severity: AlertSeverity,
        metric_type: QualityMetricType,
        degradation_detection: Optional[DegradationDetection] = None,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Alert":
        """Create a new alert with auto-generated ID and timestamps."""
        now = datetime.now()
        return cls(
            alert_id=str(uuid.uuid4()),
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            metric_type=metric_type,
            degradation_detection=degradation_detection,
            created_at=now,
            updated_at=now,
            labels=labels,
            metadata=metadata,
        )


class IQualityMonitor(ABC):
    """Interface for quality monitoring."""

    @abstractmethod
    def record_metric(self, metric: QualityMetric) -> None:
        """Record a quality metric."""

    @abstractmethod
    def get_metrics(
        self,
        metric_type: QualityMetricType,
        start_time: datetime,
        end_time: datetime,
        labels: Optional[Dict[str, str]] = None,
    ) -> List[QualityMetric]:
        """Get metrics for a time range."""

    @abstractmethod
    def get_current_metrics(self) -> Dict[QualityMetricType, float]:
        """Get current metric values."""

    @abstractmethod
    def get_metric_statistics(
        self, metric_type: QualityMetricType, time_window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get metric statistics for a time window."""


class IQualityDetector(ABC):
    """Interface for quality degradation detection."""

    @abstractmethod
    def detect_degradation(
        self,
        metric_type: QualityMetricType,
        metrics: List[QualityMetric],
        threshold: QualityThreshold,
    ) -> Optional[DegradationDetection]:
        """Detect quality degradation for a metric."""

    @abstractmethod
    def detect_anomalies(
        self, metric_type: QualityMetricType, metrics: List[QualityMetric]
    ) -> List[DegradationDetection]:
        """Detect anomalies in metric data."""

    @abstractmethod
    def detect_trends(
        self, metric_type: QualityMetricType, metrics: List[QualityMetric]
    ) -> List[DegradationDetection]:
        """Detect trends in metric data."""


class IAlertHandler(ABC):
    """Interface for alert handlers."""

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert."""

    @abstractmethod
    def can_handle_alert(self, alert: Alert) -> bool:
        """Check if this handler can handle the alert."""

    @abstractmethod
    def get_handler_name(self) -> str:
        """Get handler name."""

    @property
    @abstractmethod
    def sent_count(self) -> int:
        """Get count of successfully sent alerts."""

    @property
    @abstractmethod
    def failed_count(self) -> int:
        """Get count of failed alert sends."""


class IAlertManager(ABC):
    """Interface for alert management."""

    @abstractmethod
    def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        metric_type: QualityMetricType,
        degradation_detection: Optional[DegradationDetection] = None,
    ) -> Alert:
        """Create a new alert."""

    @abstractmethod
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert through appropriate handlers."""

    @abstractmethod
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""

    @abstractmethod
    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""

    @abstractmethod
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""


class IQualityAlertingSystem(ABC):
    """Interface for the complete quality alerting system."""

    @abstractmethod
    def start_monitoring(self) -> None:
        """Start quality monitoring."""

    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop quality monitoring."""

    @abstractmethod
    def add_threshold(self, threshold: QualityThreshold) -> None:
        """Add a quality threshold."""

    @abstractmethod
    def process_metric(self, metric: QualityMetric) -> None:
        """Process a new metric and check for degradation."""

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
