"""
Consolidated Temporal Analyzer

This analyzer consolidates temporal pattern detection capabilities from the original
analysis module, providing sophisticated time-series analysis for security data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Consolidated temporal analyzer for time-series pattern detection.

    Detects:
    - Trend patterns (increasing, decreasing, stable)
    - Seasonal patterns (hourly, daily variations)
    - Cyclical patterns (recurring cycles)
    - Anomalous temporal behavior
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_data_points = self.config.get("min_data_points", 5)
        self.trend_threshold = self.config.get("trend_threshold", 0.5)
        self.seasonal_threshold = self.config.get("seasonal_threshold", 0.2)
        self.cycle_threshold = self.config.get("cycle_threshold", 0.5)

        logger.debug("Temporal Analyzer initialized with config: %s", self.config)

    async def analyze(
        self, time_series_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze time series data for temporal patterns.

        Args:
            time_series_data: List of time-series data points with timestamp and value

        Returns:
            List of detected temporal patterns
        """
        patterns = []

        if len(time_series_data) < self.min_data_points:
            logger.debug(
                "Insufficient data points for temporal analysis: %d",
                len(time_series_data),
            )
            return patterns

        try:
            # Extract and validate time-value pairs
            time_values = self._extract_time_values(time_series_data)

            if len(time_values) < self.min_data_points:
                logger.debug(
                    "Insufficient valid time-value pairs: %d", len(time_values)
                )
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

            # Detect anomalous temporal behavior
            anomaly_patterns = await self._detect_temporal_anomalies(
                time_values, time_series_data
            )
            patterns.extend(anomaly_patterns)

            logger.info("Temporal analysis completed", patterns_detected=len(patterns))

        except Exception as e:
            logger.error("Temporal analysis failed", error=str(e))

        return patterns

    def _extract_time_values(
        self, time_series_data: List[Dict[str, Any]]
    ) -> List[Tuple[datetime, float]]:
        """Extract and validate time-value pairs from time series data."""
        time_values = []

        for item in time_series_data:
            timestamp = item.get("timestamp")
            value = item.get("value", 0)

            if timestamp and isinstance(timestamp, (str, datetime)):
                try:
                    if isinstance(timestamp, str):
                        # Handle various timestamp formats
                        if timestamp.endswith("Z"):
                            timestamp = timestamp.replace("Z", "+00:00")
                        timestamp = datetime.fromisoformat(timestamp)

                    # Ensure timezone awareness
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)

                    time_values.append((timestamp, float(value)))

                except (ValueError, TypeError) as e:
                    logger.debug(
                        "Failed to parse timestamp: %s, error: %s", timestamp, e
                    )
                    continue

        # Sort by timestamp
        time_values.sort(key=lambda x: x[0])
        return time_values

    async def _detect_trend_pattern(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Detect trend patterns using linear regression analysis."""
        if len(time_values) < 3:
            return None

        try:
            # Calculate trend using linear regression
            trend_info = await self._calculate_trend(time_values)

            if trend_info["confidence"] >= self.trend_threshold:
                # Extract metadata
                detectors = list(
                    set(item.get("detector", "unknown") for item in original_data)
                )
                start_time = min(tv[0] for tv in time_values)
                end_time = max(tv[0] for tv in time_values)

                pattern = {
                    "type": "temporal_trend",
                    "subtype": trend_info["trend"],
                    "confidence": trend_info["confidence"],
                    "strength": (
                        "moderate" if trend_info["confidence"] > 0.7 else "weak"
                    ),
                    "description": f"Temporal {trend_info['trend']} trend detected with {trend_info['confidence']:.2f} confidence",
                    "affected_detectors": detectors,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                    "statistical_significance": trend_info["confidence"],
                    "business_relevance": self._assess_temporal_business_relevance(
                        trend_info
                    ),
                    "evidence": {
                        "trend_direction": trend_info["trend"],
                        "slope": trend_info["slope"],
                        "r_squared": trend_info["r_squared"],
                        "p_value": trend_info["p_value"],
                        "sample_size": trend_info["sample_size"],
                    },
                }

                return pattern

        except Exception as e:
            logger.error("Trend pattern detection failed", error=str(e))

        return None

    async def _detect_seasonal_pattern(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Detect seasonal patterns using hour-of-day and day-of-week analysis."""
        if len(time_values) < 10:  # Need more data for seasonal analysis
            return None

        try:
            # Analyze hourly patterns
            hourly_pattern = self._analyze_hourly_patterns(time_values)

            # Analyze daily patterns
            daily_pattern = self._analyze_daily_patterns(time_values)

            # Choose the most significant pattern
            best_pattern = None
            if (
                hourly_pattern
                and hourly_pattern["coefficient_of_variation"] > self.seasonal_threshold
            ):
                best_pattern = hourly_pattern
                best_pattern["pattern_type"] = "hourly_seasonal"
            elif (
                daily_pattern
                and daily_pattern["coefficient_of_variation"] > self.seasonal_threshold
            ):
                best_pattern = daily_pattern
                best_pattern["pattern_type"] = "daily_seasonal"

            if best_pattern:
                detectors = list(
                    set(item.get("detector", "unknown") for item in original_data)
                )
                start_time = min(tv[0] for tv in time_values)
                end_time = max(tv[0] for tv in time_values)

                pattern = {
                    "type": "temporal_seasonal",
                    "subtype": best_pattern["pattern_type"],
                    "confidence": min(1.0, best_pattern["coefficient_of_variation"]),
                    "strength": (
                        "moderate"
                        if best_pattern["coefficient_of_variation"] > 0.4
                        else "weak"
                    ),
                    "description": f"Seasonal {best_pattern['pattern_type']} pattern detected with {best_pattern['coefficient_of_variation']:.2f} variation coefficient",
                    "affected_detectors": detectors,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                    "statistical_significance": min(
                        1.0, best_pattern["coefficient_of_variation"]
                    ),
                    "business_relevance": "medium",
                    "evidence": best_pattern,
                }

                return pattern

        except Exception as e:
            logger.error("Seasonal pattern detection failed", error=str(e))

        return None

    async def _detect_cyclical_pattern(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Detect cyclical patterns using autocorrelation analysis."""
        if len(time_values) < 8:  # Need sufficient data for cycle detection
            return None

        try:
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

                if best_corr > self.cycle_threshold:
                    detectors = list(
                        set(item.get("detector", "unknown") for item in original_data)
                    )
                    start_time = min(tv[0] for tv in time_values)
                    end_time = max(tv[0] for tv in time_values)

                    pattern = {
                        "type": "temporal_cyclical",
                        "subtype": "autocorrelation_cycle",
                        "confidence": best_corr,
                        "strength": "moderate" if best_corr > 0.7 else "weak",
                        "description": f"Cyclical pattern detected with {best_lag} period lag and {best_corr:.2f} correlation",
                        "affected_detectors": detectors,
                        "time_range": {
                            "start": start_time.isoformat(),
                            "end": end_time.isoformat(),
                        },
                        "statistical_significance": best_corr,
                        "business_relevance": "medium",
                        "evidence": {
                            "pattern_type": "cyclical",
                            "cycle_lag": best_lag,
                            "autocorrelation": best_corr,
                            "all_autocorrelations": autocorrelations,
                        },
                    }

                    return pattern

        except Exception as e:
            logger.error("Cyclical pattern detection failed", error=str(e))

        return None

    async def _detect_temporal_anomalies(
        self,
        time_values: List[Tuple[datetime, float]],
        original_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect temporal anomalies using statistical methods."""
        anomalies = []

        if len(time_values) < 5:
            return anomalies

        try:
            values = [tv[1] for tv in time_values]

            # Calculate statistical thresholds
            mean_val = np.mean(values)
            std_val = np.std(values)

            if std_val == 0:
                return anomalies

            # Detect outliers using z-score
            z_threshold = 2.5
            anomalous_points = []

            for i, (timestamp, value) in enumerate(time_values):
                z_score = abs(value - mean_val) / std_val
                if z_score > z_threshold:
                    anomalous_points.append(
                        {
                            "timestamp": timestamp,
                            "value": value,
                            "z_score": z_score,
                            "index": i,
                        }
                    )

            if anomalous_points:
                detectors = list(
                    set(item.get("detector", "unknown") for item in original_data)
                )

                pattern = {
                    "type": "temporal_anomaly",
                    "subtype": "statistical_outliers",
                    "confidence": min(
                        1.0, len(anomalous_points) / len(time_values) * 5
                    ),  # Scale confidence
                    "strength": (
                        "high"
                        if len(anomalous_points) > len(time_values) * 0.1
                        else "moderate"
                    ),
                    "description": f"Temporal anomalies detected: {len(anomalous_points)} outliers in {len(time_values)} data points",
                    "affected_detectors": detectors,
                    "time_range": {
                        "start": min(tv[0] for tv in time_values).isoformat(),
                        "end": max(tv[0] for tv in time_values).isoformat(),
                    },
                    "statistical_significance": min(
                        1.0, len(anomalous_points) / len(time_values) * 3
                    ),
                    "business_relevance": (
                        "high" if len(anomalous_points) > 2 else "medium"
                    ),
                    "evidence": {
                        "anomalous_points": len(anomalous_points),
                        "total_points": len(time_values),
                        "z_threshold": z_threshold,
                        "mean_value": mean_val,
                        "std_deviation": std_val,
                        "outlier_details": [
                            {
                                "timestamp": ap["timestamp"].isoformat(),
                                "value": ap["value"],
                                "z_score": ap["z_score"],
                            }
                            for ap in anomalous_points[
                                :5
                            ]  # Limit to first 5 for brevity
                        ],
                    },
                }

                anomalies.append(pattern)

        except Exception as e:
            logger.error("Temporal anomaly detection failed", error=str(e))

        return anomalies

    def _analyze_hourly_patterns(
        self, time_values: List[Tuple[datetime, float]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze hourly patterns in the data."""
        try:
            hourly_values = {}
            for timestamp, value in time_values:
                hour = timestamp.hour
                if hour not in hourly_values:
                    hourly_values[hour] = []
                hourly_values[hour].append(value)

            if len(hourly_values) >= 3:
                hour_means = {
                    hour: np.mean(values) for hour, values in hourly_values.items()
                }
                mean_values = list(hour_means.values())

                if len(mean_values) >= 3 and np.mean(mean_values) > 0:
                    cv = np.std(mean_values) / np.mean(mean_values)

                    return {
                        "coefficient_of_variation": cv,
                        "hourly_means": hour_means,
                        "peak_hours": [
                            h for h, v in hour_means.items() if v == max(mean_values)
                        ],
                        "low_hours": [
                            h for h, v in hour_means.items() if v == min(mean_values)
                        ],
                    }

        except Exception as e:
            logger.error("Hourly pattern analysis failed", error=str(e))

        return None

    def _analyze_daily_patterns(
        self, time_values: List[Tuple[datetime, float]]
    ) -> Optional[Dict[str, Any]]:
        """Analyze daily patterns in the data."""
        try:
            daily_values = {}
            for timestamp, value in time_values:
                day = timestamp.weekday()  # 0=Monday, 6=Sunday
                if day not in daily_values:
                    daily_values[day] = []
                daily_values[day].append(value)

            if len(daily_values) >= 3:
                day_means = {
                    day: np.mean(values) for day, values in daily_values.items()
                }
                mean_values = list(day_means.values())

                if len(mean_values) >= 3 and np.mean(mean_values) > 0:
                    cv = np.std(mean_values) / np.mean(mean_values)

                    day_names = [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]

                    return {
                        "coefficient_of_variation": cv,
                        "daily_means": {
                            day_names[day]: mean for day, mean in day_means.items()
                        },
                        "peak_days": [
                            day_names[d]
                            for d, v in day_means.items()
                            if v == max(mean_values)
                        ],
                        "low_days": [
                            day_names[d]
                            for d, v in day_means.items()
                            if v == min(mean_values)
                        ],
                    }

        except Exception as e:
            logger.error("Daily pattern analysis failed", error=str(e))

        return None

    async def _calculate_trend(
        self, time_values: List[Tuple[datetime, float]]
    ) -> Dict[str, Any]:
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
            if abs(r_value) < 1:
                t_stat = r_value * np.sqrt((n - 2) / (1 - r_value**2))
                p_value = 0.05 if abs(t_stat) > 2 else 0.5  # Simplified p-value
            else:
                p_value = 0.05

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
                "confidence": min(1.0, max(0.0, confidence)),
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

    def _assess_temporal_business_relevance(self, trend_info: Dict[str, Any]) -> str:
        """Assess business relevance of temporal patterns."""
        if trend_info["trend"] == "increasing" and trend_info["confidence"] > 0.8:
            return "high"
        elif (
            trend_info["trend"] in ["increasing", "decreasing"]
            and trend_info["confidence"] > 0.6
        ):
            return "medium"
        else:
            return "low"
