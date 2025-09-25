"""
Alert Manager

This module manages the lifecycle of quality alerts.
Follows SRP by focusing solely on alert lifecycle management.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .models import QualityAlert, AlertStatus

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages quality alert lifecycle.

    Single responsibility: Store, update, and track alert states.
    """

    def __init__(self):
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_history: List[QualityAlert] = []
        self.suppressed_alert_types: set = set()

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "alerts_by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "alerts_by_status": {"active": 0, "acknowledged": 0, "resolved": 0},
            "escalations": 0,
        }

        logger.info("Alert Manager initialized")

    def add_alert(self, alert: QualityAlert) -> bool:
        """
        Add new alert to management system.

        Args:
            alert: Alert to add

        Returns:
            True if alert was added, False if suppressed or duplicate
        """
        try:
            # Check if alert type is suppressed
            alert_type_key = (
                f"{alert.source_component}:{alert.alert_type}:{alert.severity.value}"
            )
            if alert_type_key in self.suppressed_alert_types:
                logger.debug("Alert suppressed", alert_id=alert.id, type=alert_type_key)
                return False

            # Check for existing similar alert (deduplication)
            existing_alert = self._find_similar_alert(alert)
            if existing_alert:
                self._update_existing_alert(existing_alert, alert)
                return False

            # Add new alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)

            # Update statistics
            self._update_stats_for_new_alert(alert)

            logger.info("Alert added", alert_id=alert.id, severity=alert.severity.value)
            return True

        except Exception as e:
            logger.error("Failed to add alert", alert_id=alert.id, error=str(e))
            return False

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: User who acknowledged the alert

        Returns:
            True if successful, False otherwise
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning("Alert not found for acknowledgment", alert_id=alert_id)
                return False

            alert = self.active_alerts[alert_id]

            if alert.status != AlertStatus.ACTIVE:
                logger.warning(
                    "Alert not in active state",
                    alert_id=alert_id,
                    status=alert.status.value,
                )
                return False

            # Update alert
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)

            # Update statistics
            self.stats["alerts_by_status"]["active"] -= 1
            self.stats["alerts_by_status"]["acknowledged"] += 1

            logger.info(
                "Alert acknowledged", alert_id=alert_id, acknowledged_by=acknowledged_by
            )
            return True

        except Exception as e:
            logger.error("Failed to acknowledge alert", alert_id=alert_id, error=str(e))
            return False

    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: ID of alert to resolve
            resolved_by: User who resolved the alert

        Returns:
            True if successful, False otherwise
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning("Alert not found for resolution", alert_id=alert_id)
                return False

            alert = self.active_alerts[alert_id]
            old_status = alert.status

            # Update alert
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)
            alert.updated_at = datetime.now(timezone.utc)

            # Remove from active alerts
            del self.active_alerts[alert_id]

            # Update statistics
            if old_status == AlertStatus.ACTIVE:
                self.stats["alerts_by_status"]["active"] -= 1
            elif old_status == AlertStatus.ACKNOWLEDGED:
                self.stats["alerts_by_status"]["acknowledged"] -= 1

            self.stats["alerts_by_status"]["resolved"] += 1

            logger.info("Alert resolved", alert_id=alert_id, resolved_by=resolved_by)
            return True

        except Exception as e:
            logger.error("Failed to resolve alert", alert_id=alert_id, error=str(e))
            return False

    def escalate_alert(self, alert_id: str) -> bool:
        """
        Escalate an alert.

        Args:
            alert_id: ID of alert to escalate

        Returns:
            True if successful, False otherwise
        """
        try:
            if alert_id not in self.active_alerts:
                logger.warning("Alert not found for escalation", alert_id=alert_id)
                return False

            alert = self.active_alerts[alert_id]
            alert.escalation_level += 1
            alert.updated_at = datetime.now(timezone.utc)

            self.stats["escalations"] += 1

            logger.warning(
                "Alert escalated",
                alert_id=alert_id,
                escalation_level=alert.escalation_level,
            )
            return True

        except Exception as e:
            logger.error("Failed to escalate alert", alert_id=alert_id, error=str(e))
            return False

    def suppress_alert_type(self, component: str, alert_type: str, severity: str):
        """
        Suppress alerts of a specific type.

        Args:
            component: Component name
            alert_type: Type of alert to suppress
            severity: Severity level to suppress
        """
        alert_type_key = f"{component}:{alert_type}:{severity}"
        self.suppressed_alert_types.add(alert_type_key)

        logger.info(
            "Alert type suppressed",
            component=component,
            alert_type=alert_type,
            severity=severity,
        )

    def unsuppress_alert_type(self, component: str, alert_type: str, severity: str):
        """
        Remove suppression for alert type.

        Args:
            component: Component name
            alert_type: Type of alert to unsuppress
            severity: Severity level to unsuppress
        """
        alert_type_key = f"{component}:{alert_type}:{severity}"
        self.suppressed_alert_types.discard(alert_type_key)

        logger.info(
            "Alert type suppression removed",
            component=component,
            alert_type=alert_type,
            severity=severity,
        )

    def get_active_alerts(self) -> List[QualityAlert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_by_id(self, alert_id: str) -> Optional[QualityAlert]:
        """Get alert by ID."""
        return self.active_alerts.get(alert_id)

    def get_alerts_by_component(self, component: str) -> List[QualityAlert]:
        """Get all active alerts for a component."""
        return [
            alert
            for alert in self.active_alerts.values()
            if alert.source_component == component
        ]

    def get_alerts_by_severity(self, severity: str) -> List[QualityAlert]:
        """Get all active alerts by severity."""
        return [
            alert
            for alert in self.active_alerts.values()
            if alert.severity.value == severity
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert management statistics."""
        return {
            "active_alerts_count": len(self.active_alerts),
            "total_alerts_processed": self.stats["total_alerts"],
            "alerts_by_severity": self.stats["alerts_by_severity"].copy(),
            "alerts_by_status": self.stats["alerts_by_status"].copy(),
            "escalations": self.stats["escalations"],
            "suppressed_types_count": len(self.suppressed_alert_types),
            "alert_resolution_rate": self._calculate_resolution_rate(),
        }

    def _find_similar_alert(self, new_alert: QualityAlert) -> Optional[QualityAlert]:
        """Find existing similar alert for deduplication."""
        for alert in self.active_alerts.values():
            if (
                alert.source_component == new_alert.source_component
                and alert.alert_type == new_alert.alert_type
                and alert.severity == new_alert.severity
                and alert.status == AlertStatus.ACTIVE
            ):

                # Check if it's the same metric/rule
                if (
                    "metric" in alert.threshold_violated
                    and "metric" in new_alert.threshold_violated
                ):
                    if (
                        alert.threshold_violated["metric"]
                        == new_alert.threshold_violated["metric"]
                    ):
                        return alert
                elif (
                    "rule_name" in alert.threshold_violated
                    and "rule_name" in new_alert.threshold_violated
                ):
                    if (
                        alert.threshold_violated["rule_name"]
                        == new_alert.threshold_violated["rule_name"]
                    ):
                        return alert

        return None

    def _update_existing_alert(
        self, existing_alert: QualityAlert, new_alert: QualityAlert
    ):
        """Update existing alert with new information."""
        existing_alert.updated_at = datetime.now(timezone.utc)
        existing_alert.metrics.update(new_alert.metrics)

        # Update threshold violation info if it's a metric alert
        if "value" in new_alert.threshold_violated:
            existing_alert.threshold_violated.update(new_alert.threshold_violated)

        logger.debug("Updated existing alert", alert_id=existing_alert.id)

    def _update_stats_for_new_alert(self, alert: QualityAlert):
        """Update statistics for new alert."""
        self.stats["total_alerts"] += 1
        self.stats["alerts_by_severity"][alert.severity.value] += 1
        self.stats["alerts_by_status"]["active"] += 1

    def _calculate_resolution_rate(self) -> float:
        """Calculate alert resolution rate."""
        total_alerts = self.stats["total_alerts"]
        if total_alerts == 0:
            return 0.0

        resolved_alerts = self.stats["alerts_by_status"]["resolved"]
        return resolved_alerts / total_alerts

    def clear_resolved_alerts_history(self, older_than_days: int = 30):
        """Clear old resolved alerts from history."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)

            original_count = len(self.alert_history)
            self.alert_history = [
                alert
                for alert in self.alert_history
                if alert.status != AlertStatus.RESOLVED
                or alert.resolved_at is None
                or alert.resolved_at > cutoff_date
            ]

            cleared_count = original_count - len(self.alert_history)

            logger.info(
                "Cleared old resolved alerts",
                cleared_count=cleared_count,
                older_than_days=older_than_days,
            )

        except Exception as e:
            logger.error("Failed to clear old alerts", error=str(e))
