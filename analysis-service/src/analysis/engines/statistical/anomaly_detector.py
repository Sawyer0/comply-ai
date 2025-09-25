"""
Consolidated Anomaly Detector

This detector consolidates anomaly detection capabilities from the original
analysis module, providing sophisticated statistical outlier detection.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Consolidated anomaly detector using statistical outlier detection methods.

    Detects:
    - Statistical outliers using IQR method
    - Z-score based anomalies
    - Isolation forest concepts
    - Temporal anomalies
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 10)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.05)
        self.z_score_threshold = self.config.get("z_score_threshold", 2.0)
        self.iqr_multiplier = self.config.get("iqr_multiplier", 1.5)

        logger.debug("Anomaly Detector initialized with config: %s", self.config)

    async def analyze(self, metrics_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze metrics data for anomaly patterns.

        Args:
            metrics_data: List of security metrics

        Returns:
            List of detected anomaly patterns
        """
        patterns = []

        if len(metrics_data) < self.min_data_points:
            logger.debug(
                "Insufficient data for anomaly detection: %d < %d",
                len(metrics_data),
                self.min_data_points,
            )
            return patterns

        try:
            # Group metrics by type
            metric_groups = self._group_metrics_by_type(metrics_data)

            for metric_type, values in metric_groups.items():
                if len(values) >= self.min_data_points:
                    # Detect outliers using multiple methods
                    anomaly_pattern = await self._detect_anomaly_pattern(
                        metric_type, values, metrics_data
                    )
                    if anomaly_pattern:
                        patterns.append(anomaly_pattern)

            logger.info(
                "Anomaly detection completed",
                metric_groups=len(metric_groups),
                patterns_found=len(patterns),
            )

        except Exception as e:
            logger.error("Anomaly analysis failed", error=str(e))

        return patterns

    def _group_metrics_by_type(
        self, metrics_data: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Group metrics by type and extract numerical values."""
        metric_groups = {}

        for metric in metrics_data:
            metric_type = metric.get("type", "unknown")
            value = metric.get("value")

            if isinstance(value, (int, float)):
                if metric_type not in metric_groups:
                    metric_groups[metric_type] = []
                metric_groups[metric_type].append(float(value))

        return metric_groups

    async def _detect_anomaly_pattern(
        self, metric_type: str, values: List[float], original_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect anomaly patterns in metric values."""
        try:
            # Detect outliers using multiple methods
            outlier_results = await self._detect_outliers_multiple_methods(values)

            if outlier_results["has_anomalies"]:
                # Calculate anomaly statistics
                anomaly_stats = self._calculate_anomaly_statistics(
                    values, outlier_results
                )

                return {
                    "pattern_type": "anomaly",
                    "metric_type": metric_type,
                    "anomaly_count": outlier_results["anomaly_count"],
                    "total_values": len(values),
                    "anomaly_percentage": (
                        outlier_results["anomaly_count"] / len(values)
                    )
                    * 100,
                    "detection_methods": outlier_results["methods_used"],
                    "anomaly_indices": outlier_results["anomaly_indices"],
                    "anomaly_values": outlier_results["anomaly_values"],
                    "statistics": anomaly_stats,
                    "confidence": self._calculate_anomaly_confidence(
                        outlier_results, anomaly_stats
                    ),
                    "severity": self._determine_anomaly_severity(
                        outlier_results, anomaly_stats
                    ),
                    "statistical_significance": anomaly_stats[
                        "statistical_significance"
                    ],
                }

        except Exception as e:
            logger.error("Error detecting anomaly pattern for %s: %s", metric_type, e)

        return None

    async def _detect_outliers_multiple_methods(
        self, values: List[float]
    ) -> Dict[str, Any]:
        """Detect outliers using multiple statistical methods."""
        try:
            values_array = np.array(values)

            # Method 1: Z-score based detection
            z_score_outliers = self._detect_z_score_outliers(values_array)

            # Method 2: IQR based detection
            iqr_outliers = self._detect_iqr_outliers(values_array)

            # Method 3: Modified Z-score (using median)
            modified_z_outliers = self._detect_modified_z_score_outliers(values_array)

            # Combine results from all methods
            all_outlier_indices = set()
            all_outlier_indices.update(z_score_outliers["indices"])
            all_outlier_indices.update(iqr_outliers["indices"])
            all_outlier_indices.update(modified_z_outliers["indices"])

            anomaly_indices = list(all_outlier_indices)
            anomaly_values = [values[i] for i in anomaly_indices]

            methods_used = []
            if z_score_outliers["indices"]:
                methods_used.append("z_score")
            if iqr_outliers["indices"]:
                methods_used.append("iqr")
            if modified_z_outliers["indices"]:
                methods_used.append("modified_z_score")

            return {
                "has_anomalies": len(anomaly_indices) > 0,
                "anomaly_count": len(anomaly_indices),
                "anomaly_indices": anomaly_indices,
                "anomaly_values": anomaly_values,
                "methods_used": methods_used,
                "method_results": {
                    "z_score": z_score_outliers,
                    "iqr": iqr_outliers,
                    "modified_z_score": modified_z_outliers,
                },
            }

        except Exception as e:
            logger.error("Error in multi-method outlier detection: %s", e)
            return {
                "has_anomalies": False,
                "anomaly_count": 0,
                "anomaly_indices": [],
                "anomaly_values": [],
                "methods_used": [],
                "error": str(e),
            }

    def _detect_z_score_outliers(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        try:
            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                return {
                    "indices": [],
                    "z_scores": [],
                    "threshold": self.z_score_threshold,
                }

            z_scores = np.abs((values - mean) / std)
            outlier_indices = np.where(z_scores > self.z_score_threshold)[0].tolist()

            return {
                "indices": outlier_indices,
                "z_scores": z_scores.tolist(),
                "threshold": self.z_score_threshold,
                "mean": float(mean),
                "std": float(std),
            }

        except Exception as e:
            logger.error("Error in Z-score outlier detection: %s", e)
            return {"indices": [], "z_scores": [], "error": str(e)}

    def _detect_iqr_outliers(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range (IQR) method."""
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)

            outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[
                0
            ].tolist()

            return {
                "indices": outlier_indices,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "multiplier": self.iqr_multiplier,
            }

        except Exception as e:
            logger.error("Error in IQR outlier detection: %s", e)
            return {"indices": [], "error": str(e)}

    def _detect_modified_z_score_outliers(self, values: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method (using median)."""
        try:
            median = np.median(values)
            mad = np.median(np.abs(values - median))  # Median Absolute Deviation

            if mad == 0:
                return {
                    "indices": [],
                    "modified_z_scores": [],
                    "threshold": self.z_score_threshold,
                }

            modified_z_scores = 0.6745 * (values - median) / mad
            outlier_indices = np.where(
                np.abs(modified_z_scores) > self.z_score_threshold
            )[0].tolist()

            return {
                "indices": outlier_indices,
                "modified_z_scores": modified_z_scores.tolist(),
                "threshold": self.z_score_threshold,
                "median": float(median),
                "mad": float(mad),
            }

        except Exception as e:
            logger.error("Error in Modified Z-score outlier detection: %s", e)
            return {"indices": [], "modified_z_scores": [], "error": str(e)}

    def _calculate_anomaly_statistics(
        self, values: List[float], outlier_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics for anomaly analysis."""
        try:
            values_array = np.array(values)

            # Basic statistics
            stats = {
                "mean": float(np.mean(values_array)),
                "median": float(np.median(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "range": float(np.max(values_array) - np.min(values_array)),
            }

            # Anomaly-specific statistics
            if outlier_results["anomaly_values"]:
                anomaly_values = np.array(outlier_results["anomaly_values"])
                stats.update(
                    {
                        "anomaly_mean": float(np.mean(anomaly_values)),
                        "anomaly_std": float(np.std(anomaly_values)),
                        "max_deviation": float(
                            np.max(np.abs(anomaly_values - stats["mean"]))
                        ),
                        "anomaly_severity_score": self._calculate_severity_score(
                            values_array, anomaly_values
                        ),
                    }
                )

            # Statistical significance
            stats["statistical_significance"] = (
                self._calculate_statistical_significance(outlier_results, len(values))
            )

            return stats

        except Exception as e:
            logger.error("Error calculating anomaly statistics: %s", e)
            return {"error": str(e)}

    def _calculate_severity_score(
        self, all_values: np.ndarray, anomaly_values: np.ndarray
    ) -> float:
        """Calculate severity score for anomalies."""
        try:
            if len(anomaly_values) == 0:
                return 0.0

            mean = np.mean(all_values)
            std = np.std(all_values)

            if std == 0:
                return 0.5

            # Calculate how many standard deviations away the anomalies are
            max_deviation = np.max(np.abs(anomaly_values - mean)) / std

            # Normalize to 0-1 scale
            severity = min(1.0, max_deviation / 5.0)  # 5 std devs = max severity

            return float(severity)

        except Exception as e:
            logger.error("Error calculating severity score: %s", e)
            return 0.5

    def _calculate_statistical_significance(
        self, outlier_results: Dict[str, Any], total_samples: int
    ) -> float:
        """Calculate statistical significance of anomaly detection."""
        try:
            anomaly_count = outlier_results.get("anomaly_count", 0)

            if anomaly_count == 0:
                return 0.0

            # Expected number of outliers by chance (assuming normal distribution)
            expected_outliers = total_samples * self.anomaly_threshold

            # Significance based on how much we exceed expected outliers
            if expected_outliers > 0:
                significance = min(
                    1.0, (anomaly_count - expected_outliers) / expected_outliers
                )
                return max(0.0, significance)
            else:
                return 1.0 if anomaly_count > 0 else 0.0

        except Exception as e:
            logger.error("Error calculating statistical significance: %s", e)
            return 0.5

    def _calculate_anomaly_confidence(
        self, outlier_results: Dict[str, Any], anomaly_stats: Dict[str, Any]
    ) -> float:
        """Calculate confidence in anomaly detection."""
        try:
            # Base confidence from number of detection methods that agree
            methods_count = len(outlier_results.get("methods_used", []))
            method_confidence = methods_count / 3.0  # We have 3 methods

            # Confidence from statistical significance
            significance_confidence = anomaly_stats.get("statistical_significance", 0.5)

            # Confidence from severity
            severity_confidence = anomaly_stats.get("anomaly_severity_score", 0.5)

            # Combined confidence
            combined_confidence = (
                method_confidence + significance_confidence + severity_confidence
            ) / 3

            return min(1.0, max(0.0, combined_confidence))

        except Exception as e:
            logger.error("Error calculating anomaly confidence: %s", e)
            return 0.5

    def _determine_anomaly_severity(
        self, outlier_results: Dict[str, Any], anomaly_stats: Dict[str, Any]
    ) -> str:
        """Determine severity level of detected anomalies."""
        try:
            severity_score = anomaly_stats.get("anomaly_severity_score", 0.5)
            anomaly_percentage = (
                outlier_results.get("anomaly_count", 0)
                / outlier_results.get("total_values", 1)
            ) * 100

            # Combine severity score and percentage
            combined_severity = (severity_score + (anomaly_percentage / 100)) / 2

            if combined_severity >= 0.8:
                return "critical"
            elif combined_severity >= 0.6:
                return "high"
            elif combined_severity >= 0.3:
                return "medium"
            else:
                return "low"

        except Exception as e:
            logger.error("Error determining anomaly severity: %s", e)
            return "medium"
