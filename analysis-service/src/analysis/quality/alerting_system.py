"""
Complete quality alerting system implementation.

This module provides the main quality alerting system that orchestrates
monitoring, detection, and alerting components following SRP.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .alert_handlers import CompositeAlertHandler
from .alert_manager import AlertManager
from .degradation import QualityDegradationDetector
from .interfaces import (
    IAlertHandler,
    IQualityAlertingSystem,
    QualityMetric,
    QualityMetricType,
    QualityThreshold,
)
from .monitoring import QualityMonitor

logger = logging.getLogger(__name__)


class QualityAlertingSystem(IQualityAlertingSystem):
    """
    Complete quality alerting system.

    Single Responsibility: Orchestrate quality monitoring, detection, and alerting
    by coordinating the specialized components.
    """

    def __init__(
        self,
        monitor: Optional[QualityMonitor] = None,
        detector: Optional[QualityDegradationDetector] = None,
        alert_manager: Optional[AlertManager] = None,
        alert_handlers: Optional[List[IAlertHandler]] = None,
        monitoring_interval_seconds: int = 60,
    ):
        """Initialize quality alerting system."""
        self.monitor = monitor or QualityMonitor()
        self.detector = detector or QualityDegradationDetector()
        self.alert_manager = alert_manager or AlertManager()

        # Set up composite alert handler
        if alert_handlers:
            composite_handler = CompositeAlertHandler(alert_handlers)
            self.alert_manager.add_handler(composite_handler)

        self.monitoring_interval_seconds = monitoring_interval_seconds

        # Thresholds configuration
        self._thresholds: Dict[QualityMetricType, QualityThreshold] = {}

        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # System statistics
        self._system_stats = {
            "start_time": None,
            "metrics_processed": 0,
            "alerts_generated": 0,
            "degradations_detected": 0,
            "last_monitoring_cycle": None,
            "monitoring_errors": 0,
        }

        logger.info("Quality alerting system initialized")

    def start_monitoring(self) -> None:
        """Start quality monitoring."""
        if self._monitoring_active:
            logger.warning("Quality monitoring is already active")
            return

        self._monitoring_active = True
        self._stop_event.clear()
        self._system_stats["start_time"] = datetime.now()

        # Start monitoring thread
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, name="QualityMonitoring", daemon=True
        )
        self._monitoring_thread.start()

        logger.info("Quality monitoring started")

    def stop_monitoring(self) -> None:
        """Stop quality monitoring."""
        if not self._monitoring_active:
            logger.warning("Quality monitoring is not active")
            return

        self._monitoring_active = False
        self._stop_event.set()

        # Wait for monitoring thread to finish
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
            if self._monitoring_thread.is_alive():
                logger.warning("Monitoring thread did not stop gracefully")

        logger.info("Quality monitoring stopped")

    def add_threshold(self, threshold: QualityThreshold) -> None:
        """Add a quality threshold."""
        self._thresholds[threshold.metric_type] = threshold
        logger.info(
            "Added threshold for %s: warning=%.3f, critical=%.3f",
            threshold.metric_type.value,
            threshold.warning_threshold,
            threshold.critical_threshold,
        )

    def remove_threshold(self, metric_type: QualityMetricType) -> None:
        """Remove a quality threshold."""
        if metric_type in self._thresholds:
            del self._thresholds[metric_type]
            logger.info("Removed threshold for %s", metric_type.value)

    def get_thresholds(self) -> List[QualityThreshold]:
        """Get all quality thresholds."""
        return list(self._thresholds.values())

    def process_metric(self, metric: QualityMetric) -> None:
        """Process a new metric and check for degradation."""
        try:
            # Record metric
            self.monitor.record_metric(metric)
            self._system_stats["metrics_processed"] += 1

            # Check for degradation if threshold exists
            threshold = self._thresholds.get(metric.metric_type)
            if threshold:
                self._check_metric_degradation(metric.metric_type, threshold)

        except Exception as e:
            logger.error("Failed to process metric %s: %s", metric.metric_type.value, e)
            self._system_stats["monitoring_errors"] += 1

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        monitor_stats = self.monitor.get_storage_statistics()
        alert_stats = self.alert_manager.get_statistics()

        return {
            "monitoring_active": self._monitoring_active,
            "system_stats": self._system_stats.copy(),
            "monitor_stats": monitor_stats,
            "alert_stats": alert_stats,
            "thresholds_count": len(self._thresholds),
            "current_metrics": self.monitor.get_current_metrics(),
        }

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Quality monitoring loop started")

        while self._monitoring_active and not self._stop_event.is_set():
            try:
                cycle_start = time.time()

                # Check all configured thresholds
                for metric_type, threshold in self._thresholds.items():
                    if not self._monitoring_active:
                        break

                    self._check_metric_degradation(metric_type, threshold)

                self._system_stats["last_monitoring_cycle"] = datetime.now()

                # Calculate sleep time to maintain interval
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.monitoring_interval_seconds - cycle_duration)

                if sleep_time > 0:
                    self._stop_event.wait(sleep_time)

            except Exception as e:
                logger.error("Error in monitoring loop: %s", e)
                self._system_stats["monitoring_errors"] += 1
                # Sleep before retrying
                self._stop_event.wait(min(30, self.monitoring_interval_seconds))

        logger.info("Quality monitoring loop stopped")

    def _check_metric_degradation(
        self, metric_type: QualityMetricType, threshold: QualityThreshold
    ) -> None:
        """Check for degradation in a specific metric type."""
        try:
            # Get recent metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=threshold.time_window_minutes)

            metrics = self.monitor.get_metrics(metric_type, start_time, end_time)

            if len(metrics) < threshold.min_samples:
                return

            # Detect degradation
            degradation = self.detector.detect_degradation(
                metric_type, metrics, threshold
            )

            if degradation:
                self._handle_degradation_detection(degradation)

            # Also check for anomalies and trends
            anomalies = self.detector.detect_anomalies(metric_type, metrics)
            for anomaly in anomalies:
                self._handle_degradation_detection(anomaly)

            trends = self.detector.detect_trends(metric_type, metrics)
            for trend in trends:
                self._handle_degradation_detection(trend)

        except Exception as e:
            logger.error("Failed to check degradation for %s: %s", metric_type.value, e)
            self._system_stats["monitoring_errors"] += 1

    def _handle_degradation_detection(self, degradation) -> None:
        """Handle a degradation detection by creating and sending an alert."""
        try:
            # Create alert
            alert = self.alert_manager.create_alert(
                title=f"Quality Degradation: {degradation.metric_type.value}",
                description=degradation.description,
                severity=degradation.severity,
                metric_type=degradation.metric_type,
                degradation_detection=degradation,
            )

            # Send alert
            success = self.alert_manager.send_alert(alert)

            if success:
                self._system_stats["alerts_generated"] += 1
                logger.info(
                    "Alert generated for %s degradation: %s",
                    degradation.degradation_type.value,
                    alert.alert_id,
                )
            else:
                logger.error("Failed to send alert: %s", alert.alert_id)

            self._system_stats["degradations_detected"] += 1

        except Exception as e:
            logger.error("Failed to handle degradation detection: %s", e)
            self._system_stats["monitoring_errors"] += 1

    def get_active_alerts(self) -> List:
        """Get all active alerts."""
        return self.alert_manager.get_active_alerts()

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        return self.alert_manager.acknowledge_alert(alert_id, user)

    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""
        return self.alert_manager.resolve_alert(alert_id, user)

    def get_alert_history(self, hours: int = 24) -> List:
        """Get alert history."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        return self.alert_manager.get_alert_history(start_time, end_time)

    def get_metric_statistics(
        self, metric_type: QualityMetricType, time_window_minutes: int = 60
    ) -> Dict[str, float]:
        """Get statistics for a specific metric type."""
        return self.monitor.get_metric_statistics(metric_type, time_window_minutes)

    def get_metric_trends(
        self, metric_type: QualityMetricType, time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get trend analysis for a specific metric type."""
        return self.monitor.get_metric_trends(metric_type, time_window_minutes)

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
