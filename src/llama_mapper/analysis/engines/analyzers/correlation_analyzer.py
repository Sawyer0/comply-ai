"""
Correlation Analyzer for cross-detector relationship detection.

This module implements statistical correlation analysis to detect
relationships between different security detectors and metrics.
"""

import logging
from typing import Dict, List, Optional
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


class CorrelationAnalyzer:
    """
    Analyzes correlation patterns in multi-dimensional security data.

    Detects statistical correlations between different security
    metrics and detector outputs.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 5)
        self.correlation_threshold = self.config.get("correlation_threshold", 0.7)

    async def analyze(
        self, multi_dimensional_data: List[Dict[str, any]]
    ) -> List[Pattern]:
        """
        Analyze multi-dimensional data for correlation patterns.

        Args:
            multi_dimensional_data: List of multi-dimensional data points

        Returns:
            List of detected correlation patterns
        """
        patterns = []

        if len(multi_dimensional_data) < self.min_data_points:
            return patterns

        try:
            # Extract numerical features
            feature_data = self._extract_numerical_features(multi_dimensional_data)

            if len(feature_data) < 2:  # Need at least 2 features for correlation
                return patterns

            # Find correlated feature pairs
            correlation_patterns = await self._detect_correlation_patterns(
                feature_data, multi_dimensional_data
            )
            patterns.extend(correlation_patterns)

        except Exception as e:
            logger.error("Correlation analysis failed", error=str(e))

        return patterns

    def _extract_numerical_features(
        self, data: List[Dict[str, any]]
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
        self, feature_data: Dict[str, List[float]], original_data: List[Dict[str, any]]
    ) -> List[Pattern]:
        """Detect correlation patterns between features."""
        patterns = []

        try:
            feature_names = list(feature_data.keys())

            # Compare each feature with every other feature
            for i, feature1 in enumerate(feature_names):
                for feature2 in feature_names[i + 1 :]:
                    values1 = feature_data[feature1]
                    values2 = feature_data[feature2]

                    # Ensure equal length
                    min_length = min(len(values1), len(values2))
                    if min_length >= self.min_data_points:
                        correlation_pattern = await self._analyze_feature_correlation(
                            feature1,
                            values1[:min_length],
                            feature2,
                            values2[:min_length],
                            original_data,
                        )

                        if correlation_pattern:
                            patterns.append(correlation_pattern)

        except Exception as e:
            logger.error("Correlation pattern detection failed", error=str(e))

        return patterns

    async def _analyze_feature_correlation(
        self,
        feature1: str,
        values1: List[float],
        feature2: str,
        values2: List[float],
        original_data: List[Dict[str, any]],
    ) -> Optional[Pattern]:
        """Analyze correlation between two features."""
        try:
            # Calculate Pearson correlation using numpy
            correlation_matrix = np.corrcoef(values1, values2)
            correlation = (
                correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0
            )

            # Simple p-value approximation
            n = len(values1)
            t_stat = (
                correlation * np.sqrt((n - 2) / (1 - correlation**2))
                if abs(correlation) < 1
                else 0
            )
            p_value = 0.05 if abs(t_stat) > 2 else 0.5  # Simplified p-value

            if abs(correlation) >= self.correlation_threshold and p_value < 0.05:
                # Create correlation pattern
                pattern = Pattern(
                    pattern_type=PatternType.CORRELATION,
                    strength=(
                        PatternStrength.STRONG
                        if abs(correlation) > 0.85
                        else PatternStrength.MODERATE
                    ),
                    confidence=abs(correlation),
                    description=f"Strong correlation between {feature1} and {feature2} (r={correlation:.3f})",
                    affected_detectors=[feature1, feature2],
                    time_range=TimeRange(
                        start=datetime.now(timezone.utc), end=datetime.now(timezone.utc)
                    ),
                    statistical_significance=1 - p_value,
                    business_relevance=self._assess_correlation_business_relevance(
                        feature1, feature2
                    ),
                    supporting_evidence=[
                        {
                            "correlation_coefficient": correlation,
                            "p_value": p_value,
                            "sample_size": len(values1),
                            "feature_1": feature1,
                            "feature_2": feature2,
                            "correlation_type": (
                                "positive" if correlation > 0 else "negative"
                            ),
                        }
                    ],
                )

                return pattern

        except Exception as e:
            logger.error("Feature correlation analysis failed", error=str(e))

        return None

    def _assess_correlation_business_relevance(
        self, feature1: str, feature2: str
    ) -> BusinessRelevance:
        """Assess business relevance of correlation patterns."""
        critical_features = ["pii", "gdpr", "hipaa", "financial"]

        if any(
            cf in feature1.lower() or cf in feature2.lower() for cf in critical_features
        ):
            return BusinessRelevance.HIGH
        else:
            return BusinessRelevance.MEDIUM
