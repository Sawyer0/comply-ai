"""
Quality degradation detection implementation.

This module provides algorithms for detecting various types of
quality degradation including threshold breaches, anomalies, and trends.
"""

import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..interfaces import (
    AlertSeverity,
    DegradationDetection,
    DegradationType,
    IQualityDetector,
    QualityMetric,
    QualityMetricType,
    QualityThreshold,
)

logger = logging.getLogger(__name__)


class QualityDegradationDetector(IQualityDetector):
    """
    Quality degradation detector implementation.

    Provides multiple algorithms for detecting quality degradation
    including threshold-based, statistical, and trend-based detection.
    """

    def __init__(
        self,
        anomaly_sensitivity: float = 2.0,
        trend_window_minutes: int = 30,
        min_samples_for_detection: int = 5,
    ):
        """
        Initialize degradation detector.

        Args:
            anomaly_sensitivity: Sensitivity for anomaly detection (standard deviations)
            trend_window_minutes: Window for trend analysis in minutes
            min_samples_for_detection: Minimum samples required for detection
        """
        self.anomaly_sensitivity = anomaly_sensitivity
        self.trend_window_minutes = trend_window_minutes
        self.min_samples_for_detection = min_samples_for_detection

        logger.info(
            f"Quality degradation detector initialized with sensitivity {anomaly_sensitivity}"
        )

    def detect_degradation(
        self,
        metric_type: QualityMetricType,
        metrics: List[QualityMetric],
        threshold: QualityThreshold,
    ) -> Optional[DegradationDetection]:
        """Detect quality degradation for a metric."""
        if not metrics or len(metrics) < threshold.min_samples:
            return None

        # Check for sudden drops first (higher priority)
        sudden_drop = self._detect_sudden_drop(metric_type, metrics)
        if sudden_drop:
            return sudden_drop

        # Check threshold breaches
        threshold_detection = self._detect_threshold_breach(
            metric_type, metrics, threshold
        )
        if threshold_detection:
            return threshold_detection

        # Check for gradual decline
        gradual_decline = self._detect_gradual_decline(metric_type, metrics)
        if gradual_decline:
            return gradual_decline

        return None

    def detect_anomalies(
        self, metric_type: QualityMetricType, metrics: List[QualityMetric]
    ) -> List[DegradationDetection]:
        """Detect anomalies in metric data."""
        if len(metrics) < self.min_samples_for_detection:
            return []

        anomalies = []
        values = [m.value for m in metrics]

        # Calculate statistical measures
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        if std == 0:
            return []

        # Detect statistical anomalies
        for i, metric in enumerate(metrics):
            z_score = abs(metric.value - mean) / std

            if z_score > self.anomaly_sensitivity:
                severity = self._determine_anomaly_severity(z_score)

                detection = DegradationDetection(
                    metric_type=metric_type,
                    degradation_type=DegradationType.ANOMALY,
                    severity=severity,
                    current_value=metric.value,
                    expected_value=mean,
                    deviation_percentage=(
                        ((metric.value - mean) / mean) * 100 if mean != 0 else 0
                    ),
                    confidence=min(z_score / self.anomaly_sensitivity, 1.0),
                    timestamp=metric.timestamp,
                    description=f"Statistical anomaly detected: z-score={z_score:.2f}",
                    metadata={
                        "z_score": z_score,
                        "mean": mean,
                        "std": std,
                        "anomaly_sensitivity": self.anomaly_sensitivity,
                    },
                )
                anomalies.append(detection)

        logger.debug("Detected %s anomalies in %s", len(anomalies), metric_type.value)
        return anomalies

    def detect_trends(
        self, metric_type: QualityMetricType, metrics: List[QualityMetric]
    ) -> List[DegradationDetection]:
        """Detect trends in metric data."""
        if len(metrics) < self.min_samples_for_detection:
            return []

        trends = []

        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Calculate trend using linear regression
        timestamps = [
            (m.timestamp - sorted_metrics[0].timestamp).total_seconds()
            for m in sorted_metrics
        ]
        values = [m.value for m in sorted_metrics]

        slope, r_squared = self._linear_regression(timestamps, values)

        # Determine if trend is significant
        if r_squared > 0.7:  # Strong correlation
            trend_type = (
                DegradationType.GRADUAL_DECLINE
                if slope < 0
                else DegradationType.TREND_REVERSAL
            )

            # Calculate trend strength
            trend_strength = abs(slope) * r_squared

            # Determine severity based on trend strength (very sensitive thresholds)
            if trend_strength > 0.001:  # Very sensitive for high severity
                severity = AlertSeverity.HIGH
            elif trend_strength > 0.0001:  # Very sensitive for medium severity
                severity = AlertSeverity.MEDIUM
            else:
                severity = AlertSeverity.LOW

            # Calculate expected vs actual values
            current_value = values[-1]
            expected_value = values[0] + slope * (timestamps[-1] - timestamps[0])

            detection = DegradationDetection(
                metric_type=metric_type,
                degradation_type=trend_type,
                severity=severity,
                current_value=current_value,
                expected_value=expected_value,
                deviation_percentage=(
                    ((current_value - expected_value) / expected_value) * 100
                    if expected_value != 0
                    else 0
                ),
                confidence=r_squared,
                timestamp=sorted_metrics[-1].timestamp,
                description=f"Trend detected: slope={slope:.4f}, R²={r_squared:.3f}",
                metadata={
                    "slope": slope,
                    "r_squared": r_squared,
                    "trend_strength": trend_strength,
                    "data_points": len(metrics),
                },
            )
            trends.append(detection)

        logger.debug("Detected %s trends in %s", len(trends), metric_type.value)
        return trends

    def _detect_threshold_breach(
        self,
        metric_type: QualityMetricType,
        metrics: List[QualityMetric],
        threshold: QualityThreshold,
    ) -> Optional[DegradationDetection]:
        """Detect threshold breaches."""
        if not threshold.enabled:
            return None

        # Get recent metrics within time window
        cutoff_time = datetime.now() - timedelta(minutes=threshold.time_window_minutes)
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]

        if len(recent_metrics) < threshold.min_samples:
            return None

        # Check for critical threshold breach
        critical_breaches = [
            m for m in recent_metrics if m.value <= threshold.critical_threshold
        ]
        if critical_breaches:
            latest_breach = max(critical_breaches, key=lambda m: m.timestamp)
            return DegradationDetection(
                metric_type=metric_type,
                degradation_type=DegradationType.THRESHOLD_BREACH,
                severity=AlertSeverity.CRITICAL,
                current_value=latest_breach.value,
                expected_value=threshold.critical_threshold,
                deviation_percentage=(
                    (latest_breach.value - threshold.critical_threshold)
                    / threshold.critical_threshold
                )
                * 100,
                confidence=1.0,
                timestamp=latest_breach.timestamp,
                description=f"Critical threshold breached: {latest_breach.value:.3f} <= {threshold.critical_threshold:.3f}",
                metadata={
                    "threshold_type": "critical",
                    "threshold_value": threshold.critical_threshold,
                    "breach_count": len(critical_breaches),
                    "time_window_minutes": threshold.time_window_minutes,
                },
            )

        # Check for warning threshold breach
        warning_breaches = [
            m for m in recent_metrics if m.value <= threshold.warning_threshold
        ]
        if warning_breaches:
            latest_breach = max(warning_breaches, key=lambda m: m.timestamp)
            return DegradationDetection(
                metric_type=metric_type,
                degradation_type=DegradationType.THRESHOLD_BREACH,
                severity=AlertSeverity.MEDIUM,
                current_value=latest_breach.value,
                expected_value=threshold.warning_threshold,
                deviation_percentage=(
                    (latest_breach.value - threshold.warning_threshold)
                    / threshold.warning_threshold
                )
                * 100,
                confidence=0.8,
                timestamp=latest_breach.timestamp,
                description=f"Warning threshold breached: {latest_breach.value:.3f} <= {threshold.warning_threshold:.3f}",
                metadata={
                    "threshold_type": "warning",
                    "threshold_value": threshold.warning_threshold,
                    "breach_count": len(warning_breaches),
                    "time_window_minutes": threshold.time_window_minutes,
                },
            )

        return None

    def _detect_sudden_drop(
        self, metric_type: QualityMetricType, metrics: List[QualityMetric]
    ) -> Optional[DegradationDetection]:
        """Detect sudden drops in metric values."""
        if len(metrics) < 3:
            return None

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Look for significant drops in recent metrics
        for i in range(2, len(sorted_metrics)):
            current = sorted_metrics[i]
            previous = sorted_metrics[i - 1]
            baseline = sorted_metrics[i - 2]

            # Calculate drop percentage
            drop_percentage = (
                ((previous.value - current.value) / previous.value) * 100
                if previous.value != 0
                else 0
            )

            # Check if this is a significant sudden drop (>20% decrease)
            if drop_percentage > 20:
                severity = self._determine_drop_severity(drop_percentage)

                return DegradationDetection(
                    metric_type=metric_type,
                    degradation_type=DegradationType.SUDDEN_DROP,
                    severity=severity,
                    current_value=current.value,
                    expected_value=previous.value,
                    deviation_percentage=drop_percentage,
                    confidence=min(drop_percentage / 20, 1.0),
                    timestamp=current.timestamp,
                    description=f"Sudden drop detected: {drop_percentage:.1f}% decrease",
                    metadata={
                        "drop_percentage": drop_percentage,
                        "previous_value": previous.value,
                        "baseline_value": baseline.value,
                        "time_window": (
                            current.timestamp - previous.timestamp
                        ).total_seconds(),
                    },
                )

        return None

    def _detect_gradual_decline(
        self, metric_type: QualityMetricType, metrics: List[QualityMetric]
    ) -> Optional[DegradationDetection]:
        """Detect gradual decline in metric values."""
        if len(metrics) < self.min_samples_for_detection:
            return None

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Calculate trend over the entire period
        timestamps = [
            (m.timestamp - sorted_metrics[0].timestamp).total_seconds()
            for m in sorted_metrics
        ]
        values = [m.value for m in sorted_metrics]

        slope, r_squared = self._linear_regression(timestamps, values)

        # Check for significant negative trend
        if slope < -0.01 and r_squared > 0.5:  # Negative slope with good correlation
            # Calculate total decline
            total_decline = values[-1] - values[0]
            decline_percentage = (
                (total_decline / values[0]) * 100 if values[0] != 0 else 0
            )

            if decline_percentage < -10:  # At least 10% decline
                severity = self._determine_decline_severity(abs(decline_percentage))

                return DegradationDetection(
                    metric_type=metric_type,
                    degradation_type=DegradationType.GRADUAL_DECLINE,
                    severity=severity,
                    current_value=values[-1],
                    expected_value=values[0],
                    deviation_percentage=decline_percentage,
                    confidence=r_squared,
                    timestamp=sorted_metrics[-1].timestamp,
                    description=f"Gradual decline detected: {decline_percentage:.1f}% over time period",
                    metadata={
                        "slope": slope,
                        "r_squared": r_squared,
                        "total_decline": total_decline,
                        "decline_percentage": decline_percentage,
                        "data_points": len(metrics),
                    },
                )

        return None

    def _determine_anomaly_severity(self, z_score: float) -> AlertSeverity:
        """Determine alert severity based on z-score."""
        if z_score > 4:
            return AlertSeverity.CRITICAL
        elif z_score > 3:
            return AlertSeverity.HIGH
        elif z_score > 2:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _determine_drop_severity(self, drop_percentage: float) -> AlertSeverity:
        """Determine alert severity based on drop percentage."""
        if drop_percentage > 50:
            return AlertSeverity.CRITICAL
        elif drop_percentage > 30:
            return AlertSeverity.HIGH
        elif drop_percentage > 20:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _determine_decline_severity(self, decline_percentage: float) -> AlertSeverity:
        """Determine alert severity based on decline percentage."""
        if decline_percentage > 30:
            return AlertSeverity.CRITICAL
        elif decline_percentage > 20:
            return AlertSeverity.HIGH
        elif decline_percentage > 10:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def _linear_regression(
        self, x_values: List[float], y_values: List[float]
    ) -> tuple[float, float]:
        """Perform linear regression and return slope and R-squared."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0, 0.0

        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)

        # Calculate slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum(
            (y - (slope * x + (sum_y - slope * sum_x) / n)) ** 2
            for x, y in zip(x_values, y_values)
        )

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, r_squared
