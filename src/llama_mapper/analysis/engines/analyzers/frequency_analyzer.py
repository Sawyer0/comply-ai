"""
Frequency Analyzer for recurring event pattern identification.

This module implements statistical analysis to detect
frequency-based patterns in security events.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    TimeRange,
)

logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Analyzes frequency patterns in security event data.

    Detects recurring events, regular intervals, and
    frequency-based anomalies in security data.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.min_events = self.config.get("min_events", 5)
        self.regularity_threshold = self.config.get("regularity_threshold", 0.3)

    async def analyze(self, events_data: List[Dict[str, any]]) -> List[Pattern]:
        """
        Analyze events data for frequency patterns.

        Args:
            events_data: List of security events

        Returns:
            List of detected frequency patterns
        """
        patterns = []

        if len(events_data) < self.min_events:
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

        except Exception as e:
            logger.error("Frequency analysis failed", error=str(e))

        return patterns

    def _group_events(
        self, events_data: List[Dict[str, any]]
    ) -> Dict[str, List[Dict[str, any]]]:
        """Group events by detector and type."""
        groups = defaultdict(list)

        for event in events_data:
            detector = event.get("detector", "unknown")
            event_type = event.get("type", "unknown")
            key = f"{detector}:{event_type}"
            groups[key].append(event)

        return dict(groups)

    async def _detect_frequency_pattern(
        self, group_key: str, events: List[Dict[str, any]]
    ) -> Optional[Pattern]:
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

            if len(timestamps) < 3:
                return None

            # Sort timestamps
            timestamps.sort()

            # Calculate intervals between events
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
                intervals.append(interval)

            # Analyze interval regularity
            regularity_info = self._analyze_interval_regularity(intervals)

            if regularity_info["is_regular"]:
                detector, event_type = group_key.split(":", 1)

                pattern = Pattern(
                    pattern_type=PatternType.FREQUENCY,
                    strength=(
                        PatternStrength.STRONG
                        if regularity_info["regularity_score"] > 0.8
                        else PatternStrength.MODERATE
                    ),
                    confidence=regularity_info["regularity_score"],
                    description=f"Regular frequency pattern in {detector} {event_type} events",
                    affected_detectors=[detector],
                    time_range=TimeRange(start=timestamps[0], end=timestamps[-1]),
                    statistical_significance=regularity_info["regularity_score"],
                    business_relevance=self._assess_frequency_business_relevance(
                        detector, len(events)
                    ),
                    supporting_evidence=[
                        {
                            "event_count": len(events),
                            "average_interval": regularity_info["average_interval"],
                            "interval_variance": regularity_info["interval_variance"],
                            "regularity_score": regularity_info["regularity_score"],
                            "coefficient_of_variation": regularity_info[
                                "coefficient_of_variation"
                            ],
                        }
                    ],
                )

                return pattern

        except Exception as e:
            logger.error("Frequency pattern detection failed", error=str(e))

        return None

    def _analyze_interval_regularity(self, intervals: List[float]) -> Dict[str, any]:
        """Analyze regularity of intervals between events."""
        if not intervals:
            return {"is_regular": False, "regularity_score": 0.0}

        try:
            intervals_array = np.array(intervals)

            # Calculate statistics
            mean_interval = np.mean(intervals_array)
            std_interval = np.std(intervals_array)
            cv = std_interval / mean_interval if mean_interval > 0 else float("inf")

            # Regular patterns have low coefficient of variation
            is_regular = cv < self.regularity_threshold
            regularity_score = max(0.0, 1.0 - cv) if cv < 1.0 else 0.0

            return {
                "is_regular": is_regular,
                "regularity_score": regularity_score,
                "average_interval": mean_interval,
                "interval_variance": std_interval**2,
                "coefficient_of_variation": cv,
            }

        except Exception as e:
            logger.error("Interval regularity analysis failed", error=str(e))
            return {"is_regular": False, "regularity_score": 0.0}

    def _assess_frequency_business_relevance(
        self, detector: str, event_count: int
    ) -> BusinessRelevance:
        """Assess business relevance of frequency patterns."""
        high_impact_detectors = ["presidio", "pii-detector", "gdpr-scanner"]

        if detector in high_impact_detectors and event_count > 10:
            return BusinessRelevance.CRITICAL
        elif event_count > 20:
            return BusinessRelevance.HIGH
        elif event_count > 10:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW
