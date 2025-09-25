"""
Quality Alerting System

This module orchestrates the complete quality alerting system.
Follows SRP by focusing solely on system coordination.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .interfaces import (
    IQualityAlertingSystem,
    QualityMetric,
    QualityThreshold,
    QualityMetricType,
    DegradationDetection,
)
from .monitoring import QualityMonitor
from .degradation import QualityDegradationDetector
from .alert_manager import AlertManager
from .alert_generator import AlertGenerator
from .alert_handlers import LoggingAlertHandler, CompositeAlertHandler

logger = logging.getLogger(__name__)


class QualityAlertingSystem(IQualityAlertingSystem):
    """
    Complete quality alerting system.

    Coordinates monitoring, detection, alert generation, and delivery.
    """

    def __init__(
        self,
        monitor: Optional[QualityMonitor] = None,
        detector: Optional[QualityDegradationDetector] = None,
        alert_manager: Optional[AlertManager] = None,
        alert_generator: Optional[AlertGenerator] = None,
    ):
        """Initialize quality alerting system."""
        self.monitor = monitor or QualityMonitor()
        self.detector = detector or QualityDegradationDetector()
        self.alert_manager = alert_manager or AlertManager()
        self.alert_generator = alert_generator or AlertGenerator()

        # Alert handlers
        self.alert_handlers = [
            LoggingAlertHandler(),
        ]

        # Thresholds
        self.thresholds: Dict[QualityMetricType, QualityThreshold] = {}

        # System state
        self.monitoring_active = False
        self.start_time = None

        logger.info("Quality Alerting System initialized")

    def start_monitoring(self) -> None:
        """Start quality monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.start_time = datetime.now()

        logger.info("Quality monitoring started")

    def stop_monitoring(self) -> None:
        """Stop quality monitoring."""
        if not self.monitoring_active:
            logger.warning("Monitoring not active")
            return

        self.monitoring_active = False

        logger.info("Quality monitoring stopped")

    def add_threshold(self, threshold: QualityThreshold) -> None:
        """Add a quality threshold."""
        self.thresholds[threshold.metric_type] = threshold
        logger.info(
            "Threshold added",
            metric_type=threshold.metric_type.value,
            warning_threshold=threshold.warning_threshold,
            critical_threshold=threshold.critical_threshold,
        )

    def process_metric(self, metric: QualityMetric) -> None:
        """Process a new metric and check for degradation."""
        if not self.monitoring_active:
            return

        try:
            # Record metric
            self.monitor.record_metric(metric)

            # Check if we have a threshold for this metric type
            if metric.metric_type not in self.thresholds:
                return

            threshold = self.thresholds[metric.metric_type]

            # Get recent metrics for analysis
            end_time = datetime.now()
            start_time = end_time - threshold.time_window_minutes
            recent_metrics = self.monitor.get_metrics(
                metric.metric_type, start_time, end_time
            )

            # Detect degradation
            degradation = self.detector.detect_degradation(
                metric.metric_type, recent_metrics, threshold
            )

            if degradation:
                # Generate alert
                alert = self.alert_generator.generate_alert(
                    degradation, "mapper_service", {"metric_value": metric.value}
                )

                # Add to alert manager
                if self.alert_manager.add_alert(alert):
                    # Send alert through handlers
                    self._send_alert(alert)

        except Exception as e:
            logger.error("Error processing metric", metric_type=metric.metric_type.value, error=str(e))

    def _send_alert(self, alert) -> None:
        """Send alert through all handlers."""
        for handler in self.alert_handlers:
            try:
                handler.send_alert(alert)
            except Exception as e:
                logger.error("Handler failed", handler=handler.name, error=str(e))

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics."""
        return {
            "monitoring_active": self.monitoring_active,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "thresholds_count": len(self.thresholds),
            "monitor_stats": self.monitor.get_storage_statistics(),
            "alert_manager_stats": self.alert_manager.get_statistics(),
            "handler_stats": {
                handler.name: {
                    "sent_count": handler.sent_count,
                    "failed_count": handler.failed_count,
                }
                for handler in self.alert_handlers
            },
        }
