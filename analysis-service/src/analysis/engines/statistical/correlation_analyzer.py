"""
Consolidated Correlation Analyzer

This analyzer consolidates correlation analysis capabilities from the original
analysis module, providing sophisticated correlation detection for security data.
"""

import logging
import math
import importlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Consolidated correlation analyzer for cross-detector relationship detection.

    Detects:
    - Statistical correlations between security metrics
    - Cross-detector relationships
    - Multi-dimensional correlations
    - Temporal correlation patterns
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 5)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)
        self.significance_threshold = self.config.get("significance_threshold", 0.05)

        logger.debug("Correlation Analyzer initialized with config: %s", self.config)

    async def analyze(
        self, multi_dimensional_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multi-dimensional data for correlation patterns.

        Args:
            multi_dimensional_data: List of multi-dimensional data points

        Returns:
            List of detected correlation patterns
        """
        patterns = []

        if len(multi_dimensional_data) < self.min_data_points:
            logger.debug(
                "Insufficient data for correlation analysis: %d < %d",
                len(multi_dimensional_data),
                self.min_data_points,
            )
            return patterns

        try:
            # Extract numerical features
            feature_data = self._extract_numerical_features(multi_dimensional_data)

            if len(feature_data) < 2:  # Need at least 2 features for correlation
                logger.debug(
                    "Insufficient features for correlation analysis: %d",
                    len(feature_data),
                )
                return patterns

            # Find correlated feature pairs
            correlation_patterns = await self._detect_correlation_patterns(
                feature_data, multi_dimensional_data
            )
            patterns.extend(correlation_patterns)

            logger.info(
                "Correlation analysis completed",
                extra={
                    "features": len(feature_data),
                    "patterns_found": len(patterns),
                },
            )

        except Exception as e:
            logger.error(
                "Correlation analysis failed", extra={"error": str(e)}
            )

        return patterns

    def _extract_numerical_features(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Extract numerical features for correlation analysis."""
        feature_data = {}

        for item in data:
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != "timestamp":
                    if key not in feature_data:
                        feature_data[key] = []
                    feature_data[key].append(float(value))

        # Filter features with sufficient data points
        filtered_features = {
            key: values
            for key, values in feature_data.items()
            if len(values) >= self.min_data_points
        }

        return filtered_features

    async def _detect_correlation_patterns(
        self, feature_data: Dict[str, List[float]], original_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect correlation patterns between features."""
        patterns = []

        feature_names = list(feature_data.keys())

        # Analyze all pairs of features
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                feature1 = feature_names[i]
                feature2 = feature_names[j]

                values1 = feature_data[feature1]
                values2 = feature_data[feature2]

                # Ensure equal length arrays
                min_length = min(len(values1), len(values2))
                values1 = values1[:min_length]
                values2 = values2[:min_length]

                if len(values1) >= self.min_data_points:
                    correlation_result = self._calculate_correlation(values1, values2)

                    if (
                        abs(correlation_result["correlation"])
                        >= self.correlation_threshold
                    ):
                        pattern = {
                            "pattern_type": "correlation",
                            "feature1": feature1,
                            "feature2": feature2,
                            "correlation_coefficient": correlation_result[
                                "correlation"
                            ],
                            "correlation_strength": self._classify_correlation_strength(
                                correlation_result["correlation"]
                            ),
                            "p_value": correlation_result["p_value"],
                            "significance": correlation_result["significance"],
                            "sample_size": len(values1),
                            "confidence": self._calculate_correlation_confidence(
                                correlation_result, len(values1)
                            ),
                            "relationship_type": self._determine_relationship_type(
                                correlation_result["correlation"]
                            ),
                            "statistical_significance": correlation_result[
                                "significance"
                            ],
                        }
                        patterns.append(pattern)

        return patterns

    def _calculate_correlation(
        self, values1: List[float], values2: List[float]
    ) -> Dict[str, float]:
        """Calculate Pearson correlation coefficient and significance."""
        try:
            # Convert to numpy arrays
            x = np.array(values1)
            y = np.array(values2)

            # Calculate Pearson correlation coefficient
            correlation_matrix = np.corrcoef(x, y)
            correlation = correlation_matrix[0, 1]

            # Handle NaN case
            if np.isnan(correlation):
                correlation = 0.0

            # Calculate p-value using a t-distribution based approach.
            n = len(values1)
            if n > 2:
                # t-statistic for correlation
                t_stat = (
                    correlation * np.sqrt((n - 2) / (1 - correlation**2))
                    if correlation != 1.0
                    else float("inf")
                )

                # Prefer a high-quality implementation from scipy.stats if
                # available, otherwise fall back to an analytical
                # approximation based on the normal distribution.
                p_value = 1.0
                try:
                    stats_module = importlib.import_module("scipy.stats")
                except Exception:
                    stats_module = None

                if stats_module is not None:
                    df = n - 2
                    p_value = float(2.0 * stats_module.t.sf(abs(t_stat), df=df))
                else:
                    p_value = self._approximate_p_value(abs(t_stat), n - 2)
            else:
                t_stat = 0.0
                p_value = 1.0

            # Determine significance
            significance = p_value < self.significance_threshold

            return {
                "correlation": float(correlation),
                "p_value": float(p_value),
                "significance": significance,
                "t_statistic": float(t_stat) if "t_stat" in locals() else 0.0,
            }

        except Exception as e:
            logger.error("Error calculating correlation: %s", e)
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "significance": False,
                "t_statistic": 0.0,
            }

    def _approximate_p_value(self, t_stat: float, degrees_of_freedom: int) -> float:
        """Approximate two-sided p-value for a t-statistic.

        Uses a smooth approximation based on the normal distribution, with a
        small-sample correction for low degrees of freedom. This avoids the
        need for heavy statistical dependencies while remaining numerically
        stable for common analysis ranges.
        """
        try:
            if degrees_of_freedom <= 0:
                return 1.0

            df = float(degrees_of_freedom)
            abs_t = float(abs(t_stat))

            # Adjust the statistic for low degrees of freedom to better match
            # the heavier tails of the t-distribution.
            if df < 30.0 and df > 2.0:
                adjusted_t = abs_t * math.sqrt((df - 2.0) / df)
            else:
                adjusted_t = abs_t

            # Two-sided p-value using the complementary error function for the
            # standard normal distribution.
            z = adjusted_t
            p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

            # Clamp to [0, 1] to protect against numerical noise.
            return max(0.0, min(1.0, p))

        except Exception:
            return 1.0

    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength based on coefficient."""
        abs_corr = abs(correlation)

        if abs_corr >= 0.9:
            return "very_strong"
        elif abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.5:
            return "moderate"
        elif abs_corr >= 0.3:
            return "weak"
        else:
            return "very_weak"

    def _determine_relationship_type(self, correlation: float) -> str:
        """Determine the type of relationship based on correlation sign."""
        if correlation > 0:
            return "positive"
        elif correlation < 0:
            return "negative"
        else:
            return "none"

    def _calculate_correlation_confidence(
        self, correlation_result: Dict[str, float], sample_size: int
    ) -> float:
        """Calculate confidence in the correlation result."""
        try:
            # Base confidence from correlation strength
            abs_correlation = abs(correlation_result["correlation"])
            strength_confidence = abs_correlation

            # Adjust for sample size
            size_confidence = min(
                1.0, sample_size / 20.0
            )  # Full confidence at 20+ samples

            # Adjust for statistical significance
            significance_confidence = 0.9 if correlation_result["significance"] else 0.5

            # Combined confidence
            combined_confidence = (
                strength_confidence + size_confidence + significance_confidence
            ) / 3

            return min(1.0, max(0.0, combined_confidence))

        except Exception as e:
            logger.error("Error calculating correlation confidence: %s", e)
            return 0.5
