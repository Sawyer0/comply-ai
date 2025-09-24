"""
Quality alerting system implementation.

This module provides the complete quality alerting system that integrates
monitoring, detection, and alerting components.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .alerting.alert_handlers import (
    CompositeAlertHandler,
    EmailAlertHandler,
    LoggingAlertHandler,
    SlackAlertHandler,
    WebhookAlertHandler,
)
from .alerting.alert_manager import AlertManager
from .detection.degradation_detector import QualityDegradationDetector
from .interfaces import (
    AlertSeverity,
    DegradationDetection,
    IQualityAlertingSystem,
    QualityMetric,
    QualityMetricType,
    QualityThreshold,
)
from .monitoring.quality_monitor import QualityMonitor

logger = logging.getLogger(__name__)


class QualityAlertingSystem(IQualityAlertingSystem):
    """
    Complete quality alerting system implementation.

    Integrates quality monitoring, degradation detection, and alerting
    to provide comprehensive quality monitoring and alerting capabilities.
    """

    def __init__(
        self,
        monitoring_interval_seconds: int = 60,
        max_metrics_per_type: int = 10000,
        alert_retention_days: int = 30,
        deduplication_window_minutes: int = 15,
    ):
        """
        Initialize quality alerting system.

        Args:
            monitoring_interval_seconds: Interval for monitoring checks
            max_metrics_per_type: Maximum metrics to store per type
            alert_retention_days: Alert retention period
            deduplication_window_minutes: Alert deduplication window

        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate parameters
        if monitoring_interval_seconds <= 0:
            raise ValueError(
                f"monitoring_interval_seconds must be positive: {monitoring_interval_seconds}"
            )
        if max_metrics_per_type <= 0:
            raise ValueError(
                f"max_metrics_per_type must be positive: {max_metrics_per_type}"
            )
        if alert_retention_days <= 0:
            raise ValueError(
                f"alert_retention_days must be positive: {alert_retention_days}"
            )
        if deduplication_window_minutes <= 0:
            raise ValueError(
                f"deduplication_window_minutes must be positive: {deduplication_window_minutes}"
            )

        self.monitoring_interval_seconds = monitoring_interval_seconds

        # Initialize components
        self.quality_monitor = QualityMonitor(max_metrics_per_type=max_metrics_per_type)

        self.degradation_detector = QualityDegradationDetector()

        self.alert_manager = AlertManager(
            max_alerts=max_metrics_per_type,
            alert_retention_days=alert_retention_days,
            deduplication_window_minutes=deduplication_window_minutes,
        )

        # Alert handlers
        self.alert_handlers: List[Any] = []
        self._setup_default_handlers()

        # Quality thresholds
        self.thresholds: Dict[QualityMetricType, QualityThreshold] = {}
        self._setup_default_thresholds()

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Statistics
        self._stats = {
            "metrics_processed": 0,
            "degradations_detected": 0,
            "alerts_created": 0,
            "last_monitoring_check": None,
            "system_start_time": datetime.now(),
        }

        logger.info("Quality alerting system initialized")

    def start_monitoring(self) -> None:
        """Start quality monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return

        self._monitoring_active = True
        self._stop_event.clear()

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

        logger.info("Quality monitoring started")

    def stop_monitoring(self) -> None:
        """Stop quality monitoring."""
        if not self._monitoring_active:
            logger.warning("Monitoring is not active")
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

        logger.info("Quality monitoring stopped")

    def add_threshold(self, threshold: QualityThreshold) -> None:
        """Add a quality threshold."""
        self.thresholds[threshold.metric_type] = threshold
        logger.info("Added threshold for %s", threshold.metric_type.value)

    def remove_threshold(self, metric_type: QualityMetricType) -> None:
        """Remove a quality threshold."""
        if metric_type in self.thresholds:
            del self.thresholds[metric_type]
            logger.info("Removed threshold for %s", metric_type.value)

    def get_thresholds(self) -> List[QualityThreshold]:
        """Get all quality thresholds."""
        return list(self.thresholds.values())

    def process_metric(self, metric: QualityMetric) -> None:
        """Process a new metric and check for degradation."""
        try:
            # Record metric
            self.quality_monitor.record_metric(metric)
            self._stats["metrics_processed"] += 1

            # Check for degradation if threshold exists
            if metric.metric_type in self.thresholds:
                threshold = self.thresholds[metric.metric_type]

                # Get recent metrics for analysis
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=threshold.time_window_minutes)
                recent_metrics = self.quality_monitor.get_metrics(
                    metric.metric_type, start_time, end_time
                )

                # Detect degradation
                degradation = self.degradation_detector.detect_degradation(
                    metric.metric_type, recent_metrics, threshold
                )

                if degradation:
                    self._handle_degradation_detected(degradation)

            # Check for anomalies
            anomalies = self.degradation_detector.detect_anomalies(
                metric.metric_type, [metric]
            )

            for anomaly in anomalies:
                self._handle_degradation_detected(anomaly)

            logger.debug(
                "Processed metric: %s=%s", metric.metric_type.value, metric.value
            )

        except Exception as e:
            logger.error("Error processing metric: %s", e)

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        return {
            "monitoring_active": self._monitoring_active,
            "thresholds_configured": len(self.thresholds),
            "alert_handlers": len(self.alert_handlers),
            "statistics": self._stats.copy(),
            "quality_monitor_stats": self.quality_monitor.get_storage_statistics(),
            "alert_manager_stats": self.alert_manager.get_alert_statistics(),
            "uptime_seconds": (
                datetime.now() - self._stats["system_start_time"]
            ).total_seconds(),
        }

    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality dashboard."""
        current_metrics = self.quality_monitor.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()

        # Get trends for each metric type
        trends = {}
        for metric_type in current_metrics.keys():
            trends[metric_type.value] = self.quality_monitor.get_metric_trends(
                metric_type, time_window_minutes=60
            )

        # Get recent alert history
        recent_alerts = self.alert_manager.get_alert_history(
            start_time=datetime.now() - timedelta(hours=24), end_time=datetime.now()
        )

        return {
            "current_metrics": {k.value: v for k, v in current_metrics.items()},
            "trends": trends,
            "active_alerts": [
                {
                    "id": alert.alert_id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "metric_type": alert.metric_type.value,
                    "created_at": alert.created_at.isoformat(),
                }
                for alert in active_alerts
            ],
            "recent_alerts_count": len(recent_alerts),
            "thresholds": [
                {
                    "metric_type": t.metric_type.value,
                    "warning_threshold": t.warning_threshold,
                    "critical_threshold": t.critical_threshold,
                    "enabled": t.enabled,
                }
                for t in self.thresholds.values()
            ],
        }

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Quality monitoring loop started")

        while not self._stop_event.is_set():
            try:
                self._perform_monitoring_check()
                self._stats["last_monitoring_check"] = datetime.now()

                # Wait for next check
                self._stop_event.wait(self.monitoring_interval_seconds)

            except Exception as e:
                logger.error("Error in monitoring loop: %s", e)
                time.sleep(5)  # Brief pause before retrying

        logger.info("Quality monitoring loop stopped")

    async def send_evaluation_notification(
        self,
        recipient: str,
        report_path: str,
        quality_metrics: Any,  # QualityMetrics from domain entities
        alerts: List[str],
    ) -> bool:
        """
        Send evaluation notification to a recipient.

        Args:
            recipient: Email address or notification target
            report_path: Path to the evaluation report
            quality_metrics: Quality metrics from the evaluation
            alerts: List of alert messages

        Returns:
            True if notification sent successfully
        """
        try:
            from datetime import datetime

            from .interfaces import Alert, AlertSeverity, QualityMetricType

            # Create evaluation alert
            alert_title = "Weekly Quality Evaluation Report"
            alert_message = self._format_evaluation_message(
                quality_metrics, alerts, report_path
            )

            # Determine severity based on alerts
            severity = AlertSeverity.LOW
            if alerts:
                severity = AlertSeverity.MEDIUM
                if any("critical" in alert.lower() for alert in alerts):
                    severity = AlertSeverity.HIGH

            from .interfaces import AlertStatus

            current_time = datetime.now()
            evaluation_alert = Alert(
                alert_id=f"eval_{current_time.strftime('%Y%m%d_%H%M%S')}",
                title=alert_title,
                description=alert_message,
                severity=severity,
                status=AlertStatus.ACTIVE,
                metric_type=QualityMetricType.CUSTOM_METRIC,
                degradation_detection=None,
                created_at=current_time,
                updated_at=current_time,
                labels={
                    "report_path": report_path,
                    "evaluation_type": "weekly",
                    "recipient": recipient,
                },
            )

            # Send through appropriate handlers
            success_count = 0
            total_handlers = 0

            for handler in self.alert_handlers:
                if handler.can_handle_alert(evaluation_alert):
                    total_handlers += 1
                    try:
                        # For email handler, set recipient
                        if hasattr(handler, "send_to_recipient"):
                            success = await handler.send_to_recipient(
                                recipient, evaluation_alert
                            )
                        else:
                            success = await handler.send_alert(evaluation_alert)

                        if success:
                            success_count += 1
                            logger.info(
                                f"Sent evaluation notification to {recipient} via {handler.get_handler_name()}"
                            )
                        else:
                            logger.warning(
                                f"Failed to send evaluation notification to {recipient} via {handler.get_handler_name()}"
                            )

                    except Exception as e:
                        logger.error(
                            f"Error sending evaluation notification via {handler.get_handler_name()}: {e}"
                        )

            # Store alert in manager
            self.alert_manager.create_alert(
                title=evaluation_alert.title,
                description=evaluation_alert.description,
                severity=evaluation_alert.severity,
                metric_type=evaluation_alert.metric_type,
            )

            success_rate = success_count / total_handlers if total_handlers > 0 else 0
            logger.info(
                f"Evaluation notification sent to {recipient}: {success_count}/{total_handlers} handlers succeeded"
            )

            return success_rate > 0

        except Exception as e:
            logger.error(
                "Failed to send evaluation notification to %s: %s", recipient, e
            )
            return False

    def _format_evaluation_message(
        self,
        quality_metrics: Any,
        alerts: List[str],
        report_path: str,
    ) -> str:
        """Format evaluation notification message."""
        message_parts = [
            "📊 Weekly Quality Evaluation Report",
            "",
            "Quality Metrics:",
        ]

        # Add quality metrics
        if hasattr(quality_metrics, "schema_valid_rate"):
            message_parts.append(
                f"• Schema Validation Rate: {quality_metrics.schema_valid_rate:.2%}"
            )
        if hasattr(quality_metrics, "rubric_score"):
            message_parts.append(f"• Rubric Score: {quality_metrics.rubric_score:.2f}")
        if hasattr(quality_metrics, "opa_compile_success_rate"):
            message_parts.append(
                f"• OPA Compilation Success Rate: {quality_metrics.opa_compile_success_rate:.2%}"
            )
        if hasattr(quality_metrics, "evidence_accuracy"):
            message_parts.append(
                f"• Evidence Accuracy: {quality_metrics.evidence_accuracy:.2f}"
            )

        # Add alerts if any
        if alerts:
            message_parts.extend(["", "⚠️ Alerts:", *[f"• {alert}" for alert in alerts]])
        else:
            message_parts.extend(["", "✅ No quality issues detected"])

        message_parts.extend(
            [
                "",
                f"📄 Full report available at: {report_path}",
                "",
                f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ]
        )

        return "\n".join(message_parts)

    def _perform_monitoring_check(self) -> None:
        """Perform periodic monitoring checks."""
        # Check for trends in all monitored metrics
        for metric_type in self.quality_monitor._current_metrics.keys():
            try:
                # Get recent metrics for trend analysis
                end_time = datetime.now()
                start_time = end_time - timedelta(minutes=30)
                recent_metrics = self.quality_monitor.get_metrics(
                    metric_type, start_time, end_time
                )

                if len(recent_metrics) >= 5:  # Minimum samples for trend analysis
                    # Detect trends
                    trends = self.degradation_detector.detect_trends(
                        metric_type, recent_metrics
                    )

                    for trend in trends:
                        self._handle_degradation_detected(trend)

            except Exception as e:
                logger.error("Error checking trends for %s: %s", metric_type.value, e)

        # Cleanup old alerts periodically
        if (
            self._stats["last_monitoring_check"] is None
            or (datetime.now() - self._stats["last_monitoring_check"]).total_seconds()
            > 3600
        ):  # 1 hour
            self.alert_manager.cleanup_old_alerts()

    def _handle_degradation_detected(self, degradation: DegradationDetection) -> None:
        """Handle detected quality degradation."""
        try:
            self._stats["degradations_detected"] += 1

            # Create alert
            title = f"Quality Degradation: {degradation.metric_type.value}"
            description = degradation.description

            alert = self.alert_manager.create_alert(
                title=title,
                description=description,
                severity=degradation.severity,
                metric_type=degradation.metric_type,
                degradation_detection=degradation,
            )

            # Send alert through handlers
            if self.alert_handlers:
                composite_handler = CompositeAlertHandler(self.alert_handlers)
                success = composite_handler.send_alert(alert)

                if success:
                    self.alert_manager.send_alert(alert)
                    self._stats["alerts_created"] += 1
                    logger.warning("Quality degradation alert sent: %s", title)
                else:
                    logger.error("Failed to send quality degradation alert: %s", title)
            else:
                # No handlers configured, just log
                logger.warning("Quality degradation detected (no handlers): %s", title)

        except Exception as e:
            logger.error("Error handling degradation detection: %s", e)

    def _setup_default_handlers(self) -> None:
        """Setup default alert handlers."""
        # Add logging handler by default
        logging_handler = LoggingAlertHandler()
        self.alert_handlers.append(logging_handler)

        logger.info("Default alert handlers configured")

    def _setup_default_thresholds(self) -> None:
        """Setup default quality thresholds."""
        from .interfaces import QualityThreshold

        # Schema validation rate threshold
        self.add_threshold(
            QualityThreshold(
                metric_type=QualityMetricType.SCHEMA_VALIDATION_RATE,
                warning_threshold=0.95,
                critical_threshold=0.90,
                min_samples=10,
                time_window_minutes=60,
            )
        )

        # Template fallback rate threshold (higher values are worse)
        self.add_threshold(
            QualityThreshold(
                metric_type=QualityMetricType.TEMPLATE_FALLBACK_RATE,
                warning_threshold=0.20,
                critical_threshold=0.30,
                min_samples=10,
                time_window_minutes=60,
            )
        )

        # OPA compilation success rate threshold
        self.add_threshold(
            QualityThreshold(
                metric_type=QualityMetricType.OPA_COMPILATION_SUCCESS_RATE,
                warning_threshold=0.95,
                critical_threshold=0.90,
                min_samples=10,
                time_window_minutes=60,
            )
        )

        # Confidence score threshold
        self.add_threshold(
            QualityThreshold(
                metric_type=QualityMetricType.CONFIDENCE_SCORE,
                warning_threshold=0.70,
                critical_threshold=0.60,
                min_samples=10,
                time_window_minutes=60,
            )
        )

        logger.info("Default quality thresholds configured")

    def add_email_handler(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: List[str],
    ) -> None:
        """Add email alert handler."""
        email_handler = EmailAlertHandler(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            from_email=from_email,
            to_emails=to_emails,
        )
        self.alert_handlers.append(email_handler)
        logger.info("Email alert handler added")

    def add_slack_handler(
        self, webhook_url: str, channel: Optional[str] = None
    ) -> None:
        """Add Slack alert handler."""
        slack_handler = SlackAlertHandler(webhook_url=webhook_url, channel=channel)
        self.alert_handlers.append(slack_handler)
        logger.info("Slack alert handler added")

    def add_webhook_handler(
        self, webhook_url: str, headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Add webhook alert handler."""
        webhook_handler = WebhookAlertHandler(webhook_url=webhook_url, headers=headers)
        self.alert_handlers.append(webhook_handler)
        logger.info("Webhook alert handler added")
