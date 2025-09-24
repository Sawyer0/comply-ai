"""
Anomaly Detector using statistical outlier detection methods.

This module implements sophisticated anomaly detection algorithms
to identify statistical outliers and anomalous patterns in security data.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

# Using numpy-based statistical functions instead of scipy

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    TimeRange,
)

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detects anomalies in security metrics using statistical methods.

    Uses various statistical techniques including IQR method,
    Z-score analysis, and isolation forest concepts.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 10)
        self.anomaly_threshold = self.config.get("anomaly_threshold", 0.05)
        self.z_score_threshold = self.config.get("z_score_threshold", 2.0)

    async def analyze(self, metrics_data: List[Dict[str, any]]) -> List[Pattern]:
        """
        Analyze metrics data for anomaly patterns.

        Args:
            metrics_data: List of security metrics

        Returns:
            List of detected anomaly patterns
        """
        patterns = []

        if len(metrics_data) < self.min_data_points:
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

        except Exception as e:
            logger.error("Anomaly analysis failed", error=str(e))

        return patterns

    def _group_metrics_by_type(
        self, metrics_data: List[Dict[str, any]]
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
        self, metric_type: str, values: List[float], original_data: List[Dict[str, any]]
    ) -> Optional[Pattern]:
        """Detect anomaly patterns in metric values."""
        try:
            # Detect outliers using multiple methods
            outlier_results = await self._detect_outliers_multiple_methods(values)

            if outlier_results["has_anomalies"]:
                pattern = Pattern(
                    pattern_type=PatternType.ANOMALY,
                    strength=self._determine_anomaly_strength(outlier_results),
                    confidence=outlier_results["confidence"],
                    description=f"Anomaly pattern detected in {metric_type} metrics",
                    affected_detectors=[metric_type],
                    time_range=TimeRange(
                        start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
                    ),
                    statistical_significance=outlier_results[
                        "statistical_significance"
                    ],
                    business_relevance=self._assess_anomaly_business_relevance(
                        metric_type, outlier_results
                    ),
                    supporting_evidence=[
                        {
                            "outlier_count": outlier_results["outlier_count"],
                            "total_samples": len(values),
                            "outlier_ratio": outlier_results["outlier_ratio"],
                            "detection_methods": outlier_results["methods_used"],
                            "outlier_indices": outlier_results["outlier_indices"][
                                :10
                            ],  # Limit for brevity
                            "statistical_summary": outlier_results[
                                "statistical_summary"
                            ],
                        }
                    ],
                )

                return pattern

        except Exception as e:
            logger.error("Anomaly pattern detection failed", error=str(e))

        return None

    async def _detect_outliers_multiple_methods(
        self, values: List[float]
    ) -> Dict[str, any]:
        """Detect outliers using multiple statistical methods."""
        try:
            values_array = np.array(values)

            # Method 1: IQR method
            iqr_outliers = self._detect_outliers_iqr(values_array)

            # Method 2: Z-score method
            zscore_outliers = self._detect_outliers_zscore(values_array)

            # Method 3: Modified Z-score method
            modified_zscore_outliers = self._detect_outliers_modified_zscore(
                values_array
            )

            # Combine results
            all_outlier_indices = (
                set(iqr_outliers) | set(zscore_outliers) | set(modified_zscore_outliers)
            )
            outlier_count = len(all_outlier_indices)
            outlier_ratio = outlier_count / len(values)

            # Calculate confidence and significance
            confidence = min(1.0, outlier_ratio * 5)  # Scale to 0-1
            statistical_significance = min(1.0, outlier_ratio * 3)

            # Determine if anomalies are significant
            has_anomalies = outlier_ratio > self.anomaly_threshold

            return {
                "has_anomalies": has_anomalies,
                "outlier_count": outlier_count,
                "outlier_ratio": outlier_ratio,
                "outlier_indices": list(all_outlier_indices),
                "confidence": confidence,
                "statistical_significance": statistical_significance,
                "methods_used": ["iqr", "zscore", "modified_zscore"],
                "statistical_summary": {
                    "mean": float(np.mean(values_array)),
                    "std": float(np.std(values_array)),
                    "median": float(np.median(values_array)),
                    "q1": float(np.percentile(values_array, 25)),
                    "q3": float(np.percentile(values_array, 75)),
                },
            }

        except Exception as e:
            logger.error("Multiple outlier detection failed", error=str(e))
            return {
                "has_anomalies": False,
                "outlier_count": 0,
                "outlier_ratio": 0.0,
                "confidence": 0.0,
                "statistical_significance": 0.0,
            }

    def _detect_outliers_iqr(self, values: np.ndarray) -> List[int]:
        """Detect outliers using Interquartile Range (IQR) method."""
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_indices = []
            for i, value in enumerate(values):
                if value < lower_bound or value > upper_bound:
                    outlier_indices.append(i)

            return outlier_indices

        except Exception:
            return []

    def _detect_outliers_zscore(self, values: np.ndarray) -> List[int]:
        """Detect outliers using Z-score method."""
        try:
            # Calculate z-scores manually
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = (
                np.abs((values - mean_val) / std_val)
                if std_val > 0
                else np.zeros_like(values)
            )
            outlier_indices = []

            for i, z_score in enumerate(z_scores):
                if z_score > self.z_score_threshold:
                    outlier_indices.append(i)

            return outlier_indices

        except Exception:
            return []

    def _detect_outliers_modified_zscore(self, values: np.ndarray) -> List[int]:
        """Detect outliers using Modified Z-score method."""
        try:
            median = np.median(values)
            mad = np.median(np.abs(values - median))

            if mad == 0:
                return []

            modified_z_scores = 0.6745 * (values - median) / mad
            outlier_indices = []

            for i, score in enumerate(np.abs(modified_z_scores)):
                if score > 3.5:  # Common threshold for modified Z-score
                    outlier_indices.append(i)

            return outlier_indices

        except Exception:
            return []

    def _determine_anomaly_strength(
        self, outlier_results: Dict[str, any]
    ) -> PatternStrength:
        """Determine the strength of anomaly pattern."""
        outlier_ratio = outlier_results.get("outlier_ratio", 0)

        if outlier_ratio > 0.2:
            return PatternStrength.STRONG
        elif outlier_ratio > 0.1:
            return PatternStrength.MODERATE
        else:
            return PatternStrength.WEAK

    def _assess_anomaly_business_relevance(
        self, metric_type: str, outlier_results: Dict[str, any]
    ) -> BusinessRelevance:
        """Assess business relevance of anomaly patterns."""
        outlier_ratio = outlier_results.get("outlier_ratio", 0)

        if outlier_ratio > 0.2:
            return BusinessRelevance.CRITICAL
        elif outlier_ratio > 0.1:
            return BusinessRelevance.HIGH
        elif outlier_ratio > 0.05:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
