"""
Interfaces for quality monitoring and alerting.

This module defines the core interfaces and abstract base classes
for quality monitoring, degradation detection, and alerting systems.

The interfaces follow the Interface Segregation Principle and provide
clear contracts for all quality monitoring components.
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
    """
    Quality metric data point.

    Represents a single measurement of a quality metric with associated
    metadata and labels for categorization and filtering.

    Args:
        metric_type: Type of quality metric being measured
        value: Numeric value of the metric (0.0-1.0 for rates, seconds for time, etc.)
        timestamp: When the metric was recorded
        labels: Key-value pairs for categorizing the metric
        metadata: Additional context information

    Raises:
        ValueError: If value is negative for rate metrics or invalid
    """

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

        # Validate timestamp (allow small future timestamps for testing)
        now = datetime.now()
        if self.timestamp > now + timedelta(seconds=1):
            raise ValueError(
                "Metric timestamp cannot be more than 1 second in the future"
            )


@dataclass
class QualityThreshold:
    """
    Quality threshold configuration.

    Defines thresholds for quality metrics with configurable warning and
    critical levels, sample requirements, and time windows.

    Args:
        metric_type: Type of metric this threshold applies to
        warning_threshold: Value below which a warning alert is triggered
        critical_threshold: Value below which a critical alert is triggered
        min_samples: Minimum number of samples required for detection
        time_window_minutes: Time window for threshold evaluation
        enabled: Whether this threshold is active

    Raises:
        ValueError: If thresholds are invalid or inconsistent
    """

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

        # For most metrics, critical should be lower than warning (lower values are worse)
        # But for some metrics like error rates, critical should be higher than warning
        # We'll allow both configurations and let the detection logic handle it
        if self.critical_threshold == self.warning_threshold:
            raise ValueError(
                f"Critical threshold ({self.critical_threshold}) must be different from "
                f"warning threshold ({self.warning_threshold})"
            )

        if self.min_samples < 1:
            raise ValueError(f"Minimum samples must be at least 1: {self.min_samples}")

        if self.time_window_minutes < 1:
            raise ValueError(
                f"Time window must be at least 1 minute: {self.time_window_minutes}"
            )

        # Validate rate metrics thresholds
        rate_metrics = {
            QualityMetricType.SCHEMA_VALIDATION_RATE,
            QualityMetricType.TEMPLATE_FALLBACK_RATE,
            QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
            QualityMetricType.ERROR_RATE,
        }

        if self.metric_type in rate_metrics:
            if self.warning_threshold > 1.0 or self.critical_threshold > 1.0:
                raise ValueError(
                    f"Rate metric thresholds cannot exceed 1.0: "
                    f"warning={self.warning_threshold}, critical={self.critical_threshold}"
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
    """
    Alert data structure.

    Represents a quality alert with lifecycle management and metadata.

    Args:
        alert_id: Unique identifier for the alert
        title: Brief title describing the alert
        description: Detailed description of the issue
        severity: Alert severity level
        status: Current alert status
        metric_type: Type of metric that triggered the alert
        degradation_detection: Associated degradation detection details
        created_at: When the alert was created
        updated_at: When the alert was last updated
        acknowledged_at: When the alert was acknowledged (if applicable)
        resolved_at: When the alert was resolved (if applicable)
        labels: Key-value pairs for categorizing the alert
        metadata: Additional context information

    Raises:
        ValueError: If alert data is invalid or inconsistent
    """

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

    def __post_init__(self):
        """Validate alert data after initialization."""
        if not self.alert_id or not self.alert_id.strip():
            raise ValueError("Alert ID cannot be empty")

        if not self.title or not self.title.strip():
            raise ValueError("Alert title cannot be empty")

        if not self.description or not self.description.strip():
            raise ValueError("Alert description cannot be empty")

        # Validate timestamps
        if self.updated_at < self.created_at:
            raise ValueError("Updated timestamp cannot be before created timestamp")

        if self.acknowledged_at and self.acknowledged_at < self.created_at:
            raise ValueError(
                "Acknowledged timestamp cannot be before created timestamp"
            )

        if self.resolved_at and self.resolved_at < self.created_at:
            raise ValueError("Resolved timestamp cannot be before created timestamp")

        # Validate status consistency
        if self.status == AlertStatus.ACKNOWLEDGED and not self.acknowledged_at:
            raise ValueError("Acknowledged alerts must have acknowledged_at timestamp")

        if self.status == AlertStatus.RESOLVED and not self.resolved_at:
            raise ValueError("Resolved alerts must have resolved_at timestamp")

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
        """
        Create a new alert with auto-generated ID and timestamps.

        Args:
            title: Alert title
            description: Alert description
            severity: Alert severity
            metric_type: Associated metric type
            degradation_detection: Optional degradation details
            labels: Optional labels
            metadata: Optional metadata

        Returns:
            New Alert instance
        """
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
    def suppress_alert(self, alert_id: str, reason: str) -> bool:
        """Suppress an alert."""

    @abstractmethod
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""

    @abstractmethod
    def get_alert_history(
        self,
        start_time: datetime,
        end_time: datetime,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get alert history for a time range."""


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
    def remove_threshold(self, metric_type: QualityMetricType) -> None:
        """Remove a quality threshold."""

    @abstractmethod
    def get_thresholds(self) -> List[QualityThreshold]:
        """Get all quality thresholds."""

    @abstractmethod
    def process_metric(self, metric: QualityMetric) -> None:
        """Process a new metric and check for degradation."""

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
