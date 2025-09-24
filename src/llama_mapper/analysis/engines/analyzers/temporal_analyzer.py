"""
Temporal Analyzer for time-series pattern detection using statistical methods.

This module implements sophisticated temporal analysis to detect
time-based patterns and trends in security data.
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


class TemporalAnalyzer:
    """
    Analyzes temporal patterns in time-series security data.

    Uses statistical methods to detect trends, seasonality,
    and other time-based patterns in security metrics.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 5)
        self.trend_threshold = self.config.get("trend_threshold", 0.5)

    async def analyze(self, time_series_data: List[Dict[str, any]]) -> List[Pattern]:
        """
        Analyze time series data for temporal patterns.

        Args:
            time_series_data: List of time-series data points

        Returns:
            List of detected temporal patterns
        """
        patterns = []

        if len(time_series_data) < self.min_data_points:
            return patterns

        try:
            # Convert to time-value pairs
            time_values = self._extract_time_values(time_series_data)

            if len(time_values) < self.min_data_points:
                return patterns

            # Detect trend patterns
            trend_pattern = await self._detect_trend_pattern(
                time_values, time_series_data
            )
            if trend_pattern:
                patterns.append(trend_pattern)

            # Detect seasonal patterns
            seasonal_pattern = await self._detect_seasonal_pattern(
                time_values, time_series_data
            )
            if seasonal_pattern:
                patterns.append(seasonal_pattern)

            # Detect cyclical patterns
            cyclical_pattern = await self._detect_cyclical_pattern(
                time_values, time_series_data
            )
            if cyclical_pattern:
                patterns.append(cyclical_pattern)

        except Exception as e:
            logger.error("Temporal analysis failed", error=str(e))

        return patterns

    def _extract_time_values(
        self, time_series_data: List[Dict[str, any]]
    ) -> List[Tuple[datetime, float]]:
        """Extract time-value pairs from time series data."""
        time_values = []

        for item in time_series_data:
            timestamp = item.get("timestamp")
            value = item.get("value", 0)

            if timestamp and isinstance(timestamp, (str, datetime)):
                try:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                    time_values.append((timestamp, float(value)))
                except (ValueError, TypeError):
                    continue

        # Sort by timestamp
        time_values.sort(key=lambda x: x[0])
        return time_values

    async def _detect_trend_pattern(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, any]],
    ) -> Optional[Pattern]:
        """Detect trend patterns in time series."""
        if len(time_values) < 3:
            return None

        try:
            # Calculate trend using linear regression
            trend_info = await self._calculate_trend(time_values)

            if trend_info["confidence"] >= self.trend_threshold:
                # Extract affected detectors
                detectors = list(
                    set(item.get("detector", "unknown") for item in original_data)
                )

                # Create time range
                start_time = min(tv[0] for tv in time_values)
                end_time = max(tv[0] for tv in time_values)

                pattern = Pattern(
                    pattern_type=PatternType.TEMPORAL,
                    strength=(
                        PatternStrength.MODERATE
                        if trend_info["confidence"] > 0.7
                        else PatternStrength.WEAK
                    ),
                    confidence=trend_info["confidence"],
                    description=f"Temporal {trend_info['trend']} trend detected with {trend_info['confidence']:.2f} confidence",
                    affected_detectors=detectors,
                    time_range=TimeRange(start=start_time, end=end_time),
                    statistical_significance=trend_info["confidence"],
                    business_relevance=self._assess_temporal_business_relevance(
                        trend_info
                    ),
                    supporting_evidence=[
                        {
                            "trend_direction": trend_info["trend"],
                            "slope": trend_info["slope"],
                            "r_squared": trend_info["r_squared"],
                            "p_value": trend_info["p_value"],
                            "sample_size": trend_info["sample_size"],
                        }
                    ],
                )

                return pattern

        except Exception as e:
            logger.error("Trend pattern detection failed", error=str(e))

        return None

    async def _detect_seasonal_pattern(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, any]],
    ) -> Optional[Pattern]:
        """Detect seasonal patterns in time series."""
        if len(time_values) < 10:  # Need more data for seasonal analysis
            return None

        try:
            # Simple seasonal detection using hour-of-day analysis
            hourly_values = {}
            for timestamp, value in time_values:
                hour = timestamp.hour
                if hour not in hourly_values:
                    hourly_values[hour] = []
                hourly_values[hour].append(value)

            # Check if there's significant variation by hour
            if len(hourly_values) >= 3:
                hour_means = {
                    hour: np.mean(values) for hour, values in hourly_values.items()
                }
                mean_values = list(hour_means.values())

                if len(mean_values) >= 3:
                    # Calculate coefficient of variation
                    cv = (
                        np.std(mean_values) / np.mean(mean_values)
                        if np.mean(mean_values) > 0
                        else 0
                    )

                    if cv > 0.2:  # Significant hourly variation
                        detectors = list(
                            set(
                                item.get("detector", "unknown")
                                for item in original_data
                            )
                        )
                        start_time = min(tv[0] for tv in time_values)
                        end_time = max(tv[0] for tv in time_values)

                        pattern = Pattern(
                            pattern_type=PatternType.TEMPORAL,
                            strength=PatternStrength.MODERATE,
                            confidence=min(1.0, cv),
                            description=f"Seasonal hourly pattern detected with {cv:.2f} variation coefficient",
                            affected_detectors=detectors,
                            time_range=TimeRange(start=start_time, end=end_time),
                            statistical_significance=min(1.0, cv),
                            business_relevance=BusinessRelevance.MEDIUM,
                            supporting_evidence=[
                                {
                                    "pattern_type": "hourly_seasonal",
                                    "coefficient_of_variation": cv,
                                    "hourly_means": hour_means,
                                    "peak_hours": [
                                        h
                                        for h, v in hour_means.items()
                                        if v == max(mean_values)
                                    ],
                                }
                            ],
                        )

                        return pattern

        except Exception as e:
            logger.error("Seasonal pattern detection failed", error=str(e))

        return None

    async def _detect_cyclical_pattern(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, any]],
    ) -> Optional[Pattern]:
        """Detect cyclical patterns in time series."""
        if len(time_values) < 8:  # Need sufficient data for cycle detection
            return None

        try:
            # Simple cycle detection using autocorrelation
            values = [tv[1] for tv in time_values]

            # Calculate autocorrelation for different lags
            max_lag = min(len(values) // 2, 10)
            autocorrelations = []

            for lag in range(1, max_lag + 1):
                if len(values) > lag:
                    corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrelations.append((lag, abs(corr)))

            if autocorrelations:
                # Find the lag with highest autocorrelation
                best_lag, best_corr = max(autocorrelations, key=lambda x: x[1])

                if best_corr > 0.5:  # Significant autocorrelation
                    detectors = list(
                        set(item.get("detector", "unknown") for item in original_data)
                    )
                    start_time = min(tv[0] for tv in time_values)
                    end_time = max(tv[0] for tv in time_values)

                    pattern = Pattern(
                        pattern_type=PatternType.TEMPORAL,
                        strength=(
                            PatternStrength.MODERATE
                            if best_corr > 0.7
                            else PatternStrength.WEAK
                        ),
                        confidence=best_corr,
                        description=f"Cyclical pattern detected with {best_lag} period lag and {best_corr:.2f} correlation",
                        affected_detectors=detectors,
                        time_range=TimeRange(start=start_time, end=end_time),
                        statistical_significance=best_corr,
                        business_relevance=BusinessRelevance.MEDIUM,
                        supporting_evidence=[
                            {
                                "pattern_type": "cyclical",
                                "cycle_lag": best_lag,
                                "autocorrelation": best_corr,
                                "all_autocorrelations": autocorrelations,
                            }
                        ],
                    )

                    return pattern

        except Exception as e:
            logger.error("Cyclical pattern detection failed", error=str(e))

        return None

    async def _calculate_trend(
        self, time_values: List[Tuple[datetime, float]]
    ) -> Dict[str, any]:
        """Calculate trend information using linear regression."""
        try:
            # Convert timestamps to numeric values (seconds since first timestamp)
            first_timestamp = time_values[0][0]
            x_values = [(tv[0] - first_timestamp).total_seconds() for tv in time_values]
            y_values = [tv[1] for tv in time_values]

            # Perform linear regression using numpy
            slope, intercept = np.polyfit(x_values, y_values, 1)

            # Calculate correlation coefficient
            correlation_matrix = np.corrcoef(x_values, y_values)
            r_value = (
                correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0
            )

            # Simple p-value approximation
            n = len(x_values)
            t_stat = (
                r_value * np.sqrt((n - 2) / (1 - r_value**2)) if abs(r_value) < 1 else 0
            )
            p_value = 0.05 if abs(t_stat) > 2 else 0.5  # Simplified p-value

            # Determine trend direction
            if abs(slope) < 1e-10:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"

            # Calculate confidence based on R-squared and p-value
            r_squared = r_value**2
            confidence = r_squared * (1 - p_value) if p_value < 1.0 else 0.0

            return {
                "trend": trend_direction,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "p_value": p_value,
                "confidence": min(1.0, confidence),
                "sample_size": len(time_values),
            }

        except Exception as e:
            logger.error("Trend calculation failed", error=str(e))
            return {
                "trend": "unknown",
                "slope": 0,
                "confidence": 0,
                "sample_size": len(time_values),
            }

    def _assess_temporal_business_relevance(
        self, trend_info: Dict[str, any]
    ) -> BusinessRelevance:
        """Assess business relevance of temporal patterns."""
        if trend_info["trend"] == "increasing" and trend_info["confidence"] > 0.8:
            return BusinessRelevance.HIGH
        elif (
            trend_info["trend"] in ["increasing", "decreasing"]
            and trend_info["confidence"] > 0.6
        ):
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
