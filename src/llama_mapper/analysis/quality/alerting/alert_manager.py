"""
Alert manager implementation.

This module provides the core alert management functionality
including alert creation, routing, and lifecycle management.
"""

import logging
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Dict, List, Optional

from ..interfaces import (
    Alert,
    AlertSeverity,
    AlertStatus,
    DegradationDetection,
    IAlertManager,
    QualityMetricType,
)

logger = logging.getLogger(__name__)


class AlertManager(IAlertManager):
    """
    Alert manager implementation.

    Provides comprehensive alert management including creation,
    routing, lifecycle management, and deduplication.
    """

    def __init__(
        self,
        max_alerts: int = 10000,
        alert_retention_days: int = 30,
        deduplication_window_minutes: int = 15,
    ):
        """
        Initialize alert manager.

        Args:
            max_alerts: Maximum number of alerts to store
            alert_retention_days: Alert retention period in days
            deduplication_window_minutes: Window for alert deduplication
        """
        self.max_alerts = max_alerts
        self.alert_retention_days = alert_retention_days
        self.deduplication_window_minutes = deduplication_window_minutes

        # Alert storage
        self._alerts: Dict[str, Alert] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=max_alerts)

        # Deduplication tracking
        self._recent_alerts: Dict[str, List[datetime]] = defaultdict(list)

        # Thread safety
        self._lock = Lock()

        # Statistics
        self._stats = {
            "total_alerts_created": 0,
            "total_alerts_sent": 0,
            "alerts_by_severity": defaultdict(int),
            "alerts_by_metric_type": defaultdict(int),
            "last_cleanup": datetime.now(),
        }

        logger.info("Alert manager initialized with %s max alerts", max_alerts)

    def create_alert(
        self,
        title: str,
        description: str,
        severity: AlertSeverity,
        metric_type: QualityMetricType,
        degradation_detection: Optional[DegradationDetection] = None,
    ) -> Alert:
        """Create a new alert."""
        alert_id = str(uuid.uuid4())
        current_time = datetime.now()

        alert = Alert(
            alert_id=alert_id,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            metric_type=metric_type,
            degradation_detection=degradation_detection,
            created_at=current_time,
            updated_at=current_time,
            labels=degradation_detection.metadata if degradation_detection else None,
        )

        with self._lock:
            # Check for deduplication
            if self._should_deduplicate_alert(alert):
                logger.debug("Alert deduplicated: %s", title)
                return self._get_duplicate_alert(alert)

            # Store alert
            self._alerts[alert_id] = alert
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)

            # Update deduplication tracking
            self._update_deduplication_tracking(alert)

            # Update statistics
            self._stats["total_alerts_created"] += 1
            self._stats["alerts_by_severity"][severity.value] += 1
            self._stats["alerts_by_metric_type"][metric_type.value] += 1

            logger.info("Created alert: %s - %s (%s)", alert_id, title, severity.value)

        return alert

    def send_alert(self, alert: Alert) -> bool:
        """Send an alert through appropriate handlers."""
        with self._lock:
            try:
                # Update alert status
                alert.updated_at = datetime.now()

                # Here we would integrate with actual alert handlers
                # For now, we'll just log the alert
                logger.warning(
                    f"ALERT SENT: {alert.severity.value.upper()} - {alert.title}\n"
                    f"Description: {alert.description}\n"
                    f"Metric: {alert.metric_type.value}\n"
                    f"Alert ID: {alert.alert_id}\n"
                    f"Created: {alert.created_at}"
                )

                # Update statistics
                self._stats["total_alerts_sent"] += 1

                return True

            except Exception as e:
                logger.error("Failed to send alert %s: %s", alert.alert_id, e)
                return False

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                logger.warning("Alert %s not found for acknowledgment", alert_id)
                return False

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.updated_at = datetime.now()

            logger.info("Alert %s acknowledged by %s", alert_id, user)
            return True

    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                logger.warning("Alert %s not found for resolution", alert_id)
                return False

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            alert.updated_at = datetime.now()

            # Remove from active alerts
            if alert_id in self._active_alerts:
                del self._active_alerts[alert_id]

            logger.info("Alert %s resolved by %s", alert_id, user)
            return True

    def suppress_alert(self, alert_id: str, reason: str) -> bool:
        """Suppress an alert."""
        with self._lock:
            if alert_id not in self._alerts:
                logger.warning("Alert %s not found for suppression", alert_id)
                return False

            alert = self._alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.updated_at = datetime.now()

            # Add suppression reason to metadata
            if alert.metadata is None:
                alert.metadata = {}
            alert.metadata["suppression_reason"] = reason
            alert.metadata["suppressed_at"] = datetime.now().isoformat()

            # Remove from active alerts
            if alert_id in self._active_alerts:
                del self._active_alerts[alert_id]

            logger.info("Alert %s suppressed: %s", alert_id, reason)
            return True

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(
        self,
        start_time: datetime,
        end_time: datetime,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get alert history for a time range."""
        with self._lock:
            alerts = []

            for alert in self._alert_history:
                if start_time <= alert.created_at <= end_time:
                    if severity is None or alert.severity == severity:
                        alerts.append(alert)

            return alerts

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_alerts_by_metric_type(
        self, metric_type: QualityMetricType, status: Optional[AlertStatus] = None
    ) -> List[Alert]:
        """Get alerts by metric type."""
        with self._lock:
            alerts = []

            for alert in self._alerts.values():
                if alert.metric_type == metric_type:
                    if status is None or alert.status == status:
                        alerts.append(alert)

            return alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            # Clean up old deduplication data
            self._cleanup_deduplication_data()

            # Calculate additional statistics
            active_count = len(self._active_alerts)
            total_count = len(self._alerts)

            # Alerts by status
            status_counts = defaultdict(int)
            for alert in self._alerts.values():
                status_counts[alert.status.value] += 1

            # Recent alert rate (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_alerts = [
                a for a in self._alerts.values() if a.created_at >= recent_cutoff
            ]
            recent_rate = len(recent_alerts) / 24  # alerts per hour

            return {
                "total_alerts": total_count,
                "active_alerts": active_count,
                "alerts_by_severity": dict(self._stats["alerts_by_severity"]),
                "alerts_by_metric_type": dict(self._stats["alerts_by_metric_type"]),
                "alerts_by_status": dict(status_counts),
                "total_created": self._stats["total_alerts_created"],
                "total_sent": self._stats["total_alerts_sent"],
                "recent_rate_per_hour": recent_rate,
                "last_cleanup": self._stats["last_cleanup"],
            }

    def cleanup_old_alerts(self) -> int:
        """Clean up old alerts beyond retention period."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(days=self.alert_retention_days)
            alerts_to_remove = []

            for alert_id, alert in self._alerts.items():
                if alert.created_at < cutoff_time:
                    alerts_to_remove.append(alert_id)

            # Remove old alerts
            for alert_id in alerts_to_remove:
                del self._alerts[alert_id]
                if alert_id in self._active_alerts:
                    del self._active_alerts[alert_id]

            self._stats["last_cleanup"] = datetime.now()

            logger.info("Cleaned up %s old alerts", len(alerts_to_remove))
            return len(alerts_to_remove)

    def _should_deduplicate_alert(self, alert: Alert) -> bool:
        """Check if alert should be deduplicated."""
        # Create deduplication key
        dedup_key = self._create_dedup_key(alert)

        # Check recent alerts
        recent_alerts = self._recent_alerts[dedup_key]
        cutoff_time = datetime.now() - timedelta(
            minutes=self.deduplication_window_minutes
        )

        # Remove old entries
        recent_alerts[:] = [t for t in recent_alerts if t > cutoff_time]

        # Check if we have recent similar alerts
        return len(recent_alerts) > 0

    def _get_duplicate_alert(self, alert: Alert) -> Alert:
        """Get the most recent duplicate alert."""
        dedup_key = self._create_dedup_key(alert)

        # Find the most recent alert with the same dedup key
        for existing_alert in reversed(self._alert_history):
            if self._create_dedup_key(existing_alert) == dedup_key:
                return existing_alert

        # Fallback to the new alert if no duplicate found
        return alert

    def _create_dedup_key(self, alert: Alert) -> str:
        """Create deduplication key for alert."""
        # Use metric type, severity, and degradation type for deduplication
        degradation_type = ""
        if alert.degradation_detection:
            degradation_type = alert.degradation_detection.degradation_type.value

        return f"{alert.metric_type.value}:{alert.severity.value}:{degradation_type}"

    def _update_deduplication_tracking(self, alert: Alert) -> None:
        """Update deduplication tracking."""
        dedup_key = self._create_dedup_key(alert)
        self._recent_alerts[dedup_key].append(alert.created_at)

    def _cleanup_deduplication_data(self) -> None:
        """Clean up old deduplication data."""
        cutoff_time = datetime.now() - timedelta(
            minutes=self.deduplication_window_minutes
        )

        for dedup_key in list(self._recent_alerts.keys()):
            self._recent_alerts[dedup_key] = [
                t for t in self._recent_alerts[dedup_key] if t > cutoff_time
            ]

            # Remove empty entries
            if not self._recent_alerts[dedup_key]:
                del self._recent_alerts[dedup_key]
