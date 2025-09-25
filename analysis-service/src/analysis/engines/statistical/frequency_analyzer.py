"""
Consolidated Frequency Analyzer

This analyzer consolidates frequency pattern detection capabilities from the original
analysis module, providing sophisticated frequency analysis for security events.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Consolidated frequency analyzer for recurring event pattern identification.

    Detects:
    - Recurring event patterns
    - Regular interval patterns
    - Frequency-based anomalies
    - Event clustering patterns
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_events = self.config.get("min_events", 5)
        self.regularity_threshold = self.config.get("regularity_threshold", 0.3)
        self.frequency_threshold = self.config.get("frequency_threshold", 0.1)

        logger.debug("Frequency Analyzer initialized with config: %s", self.config)

    async def analyze(self, events_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze events data for frequency patterns.

        Args:
            events_data: List of security events

        Returns:
            List of detected frequency patterns
        """
        patterns = []

        if len(events_data) < self.min_events:
            logger.debug(
                "Insufficient events for frequency analysis: %d < %d",
                len(events_data),
                self.min_events,
            )
            return patterns

        try:
            # Group events by detector/type
            event_groups = self._group_events(events_data)

            for group_key, group_events in event_groups.items():
                if len(group_events) >= self.min_events:
                    # Detect regular frequency patterns
                    frequency_pattern = await self._detect_frequency_pattern(
                        group_key, group_events
                    )
                    if frequency_pattern:
                        patterns.append(frequency_pattern)

            logger.info(
                "Frequency analysis completed",
                event_groups=len(event_groups),
                patterns_found=len(patterns),
            )

        except Exception as e:
            logger.error("Frequency analysis failed", error=str(e))

        return patterns

    def _group_events(
        self, events_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by detector and type."""
        groups = defaultdict(list)

        for event in events_data:
            detector = event.get("detector", "unknown")
            event_type = event.get("type", "unknown")
            key = f"{detector}:{event_type}"
            groups[key].append(event)

        return dict(groups)

    async def _detect_frequency_pattern(
        self, group_key: str, events: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect frequency patterns in a group of events."""
        try:
            # Extract timestamps
            timestamps = []
            for event in events:
                timestamp = event.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                    timestamps.append(timestamp)

            if len(timestamps) < self.min_events:
                return None

            # Sort timestamps
            timestamps.sort()

            # Calculate intervals between events
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
                intervals.append(interval)

            if not intervals:
                return None

            # Analyze frequency characteristics
            frequency_stats = self._calculate_frequency_statistics(intervals)

            # Determine if pattern is regular enough
            if frequency_stats["regularity_score"] >= self.regularity_threshold:
                return {
                    "pattern_type": "frequency",
                    "group_key": group_key,
                    "event_count": len(events),
                    "time_span": (timestamps[-1] - timestamps[0]).total_seconds(),
                    "average_interval": frequency_stats["mean_interval"],
                    "interval_variance": frequency_stats["variance"],
                    "regularity_score": frequency_stats["regularity_score"],
                    "frequency_hz": (
                        1.0 / frequency_stats["mean_interval"]
                        if frequency_stats["mean_interval"] > 0
                        else 0
                    ),
                    "confidence": min(1.0, frequency_stats["regularity_score"] * 1.2),
                    "statistical_significance": frequency_stats["significance"],
                    "timestamps": [
                        ts.isoformat() for ts in timestamps[:5]
                    ],  # Sample timestamps
                }

        except Exception as e:
            logger.error("Error detecting frequency pattern for %s: %s", group_key, e)

        return None

    def _calculate_frequency_statistics(
        self, intervals: List[float]
    ) -> Dict[str, float]:
        """Calculate statistical measures for frequency analysis."""
        try:
            intervals_array = np.array(intervals)

            mean_interval = float(np.mean(intervals_array))
            variance = float(np.var(intervals_array))
            std_dev = float(np.std(intervals_array))

            # Calculate coefficient of variation (lower = more regular)
            cv = std_dev / mean_interval if mean_interval > 0 else float("inf")

            # Regularity score (1 = perfectly regular, 0 = completely irregular)
            regularity_score = max(0.0, 1.0 - cv)

            # Statistical significance based on sample size and regularity
            significance = min(1.0, (len(intervals) / 10.0) * regularity_score)

            return {
                "mean_interval": mean_interval,
                "variance": variance,
                "std_dev": std_dev,
                "coefficient_of_variation": cv,
                "regularity_score": regularity_score,
                "significance": significance,
            }

        except Exception as e:
            logger.error("Error calculating frequency statistics: %s", e)
            return {
                "mean_interval": 0.0,
                "variance": 0.0,
                "std_dev": 0.0,
                "coefficient_of_variation": float("inf"),
                "regularity_score": 0.0,
                "significance": 0.0,
            }
