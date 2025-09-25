"""
Alert Generator

This module generates quality alerts based on degradation detection.
Follows SRP by focusing solely on alert generation logic.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .interfaces import Alert, AlertSeverity, QualityMetricType, DegradationDetection
from .models import QualityAlert, AlertStatus

logger = logging.getLogger(__name__)


class AlertGenerator:
    """
    Generates quality alerts from degradation detection results.
    
    Single responsibility: Convert degradation detection into alerts.
    """

    def __init__(self):
        self.alert_count = 0
        logger.info("Alert Generator initialized")

    def generate_alert(
        self,
        degradation: DegradationDetection,
        source_component: str = "mapper_service",
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> QualityAlert:
        """
        Generate alert from degradation detection.

        Args:
            degradation: Degradation detection result
            source_component: Component that detected the degradation
            additional_metrics: Additional metrics to include

        Returns:
            Generated quality alert
        """
        try:
            # Create alert ID
            self.alert_count += 1
            alert_id = f"mapper_quality_{self.alert_count}_{int(datetime.now().timestamp())}"

            # Determine alert type based on degradation type
            alert_type = self._get_alert_type(degradation.degradation_type)

            # Create title and description
            title = self._create_alert_title(degradation)
            description = self._create_alert_description(degradation)

            # Prepare metrics
            metrics = {
                "current_value": degradation.current_value,
                "expected_value": degradation.expected_value,
                "deviation_percentage": degradation.deviation_percentage,
                "confidence": degradation.confidence,
            }
            if additional_metrics:
                metrics.update(additional_metrics)

            # Prepare threshold violation info
            threshold_violated = {
                "metric": degradation.metric_type.value,
                "degradation_type": degradation.degradation_type.value,
                "severity": degradation.severity.value,
                "confidence": degradation.confidence,
            }

            # Create alert
            alert = QualityAlert(
                id=alert_id,
                alert_type=alert_type,
                severity=degradation.severity,
                title=title,
                description=description,
                source_component=source_component,
                metrics=metrics,
                threshold_violated=threshold_violated,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                status=AlertStatus.ACTIVE,
            )

            logger.info(
                "Alert generated",
                alert_id=alert_id,
                severity=degradation.severity.value,
                degradation_type=degradation.degradation_type.value,
            )

            return alert

        except Exception as e:
            logger.error("Failed to generate alert", error=str(e))
            raise RuntimeError(f"Failed to generate alert: {e}") from e

    def _get_alert_type(self, degradation_type) -> str:
        """Get alert type based on degradation type."""
        type_mapping = {
            "sudden_drop": "quality_degradation",
            "gradual_decline": "quality_trend",
            "threshold_breach": "threshold_violation",
            "anomaly": "statistical_anomaly",
            "trend_reversal": "quality_trend",
        }
        return type_mapping.get(degradation_type.value, "quality_issue")

    def _create_alert_title(self, degradation: DegradationDetection) -> str:
        """Create alert title."""
        metric_name = degradation.metric_type.value.replace("_", " ").title()
        degradation_name = degradation.degradation_type.value.replace("_", " ").title()
        
        return f"{metric_name} {degradation_name} Detected"

    def _create_alert_description(self, degradation: DegradationDetection) -> str:
        """Create alert description."""
        return (
            f"{degradation.description}. "
            f"Current value: {degradation.current_value:.3f}, "
            f"Expected: {degradation.expected_value:.3f} "
            f"({degradation.deviation_percentage:+.1f}% deviation). "
            f"Confidence: {degradation.confidence:.2f}"
        )
