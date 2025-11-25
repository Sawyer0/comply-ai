"""
Pattern analysis engines for statistical pattern recognition.

This module provides various statistical analyzers for pattern recognition
including temporal, frequency, correlation, and anomaly detection.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from .correlation_analyzer import CorrelationAnalyzer as StatisticalCorrelationAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Result of pattern analysis."""

    pattern_type: str
    confidence: float
    strength: float
    metadata: Dict[str, Any]


class TemporalAnalyzer:
    """Analyzes temporal patterns in data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_window = config.get("time_window_hours", 24)

    async def analyze_temporal_patterns(
        self, events: List[Dict[str, Any]]
    ) -> List[PatternResult]:
        """Analyze temporal patterns in events."""
        patterns = []

        # Simple temporal clustering analysis
        if len(events) >= 3:
            time_gaps = self._calculate_time_gaps(events)
            if self._detect_clustering(time_gaps):
                patterns.append(
                    PatternResult(
                        pattern_type="temporal_clustering",
                        confidence=0.7,
                        strength=0.6,
                        metadata={"time_gaps": time_gaps},
                    )
                )

        return patterns

    def _calculate_time_gaps(self, events: List[Dict[str, Any]]) -> List[float]:
        """Calculate time gaps between events."""
        gaps = []
        for i in range(1, len(events)):
            prev_time = events[i - 1].get("timestamp", datetime.now())
            curr_time = events[i].get("timestamp", datetime.now())
            if isinstance(prev_time, str):
                prev_time = datetime.fromisoformat(prev_time)
            if isinstance(curr_time, str):
                curr_time = datetime.fromisoformat(curr_time)
            gap = (curr_time - prev_time).total_seconds()
            gaps.append(gap)
        return gaps

    def _detect_clustering(self, time_gaps: List[float]) -> bool:
        """Detect if events are clustered in time."""
        if not time_gaps:
            return False

        avg_gap = sum(time_gaps) / len(time_gaps)
        short_gaps = [gap for gap in time_gaps if gap < avg_gap * 0.3]
        return len(short_gaps) >= len(time_gaps) * 0.6


class FrequencyAnalyzer:
    """Analyzes frequency patterns in data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.baseline_threshold = config.get("baseline_threshold", 2.0)

    async def analyze_frequency_patterns(
        self, events: List[Dict[str, Any]]
    ) -> List[PatternResult]:
        """Analyze frequency patterns in events."""
        patterns = []

        # Simple frequency spike detection
        if len(events) > 10:
            frequency_score = len(events) / 24  # Events per hour
            if frequency_score > self.baseline_threshold:
                patterns.append(
                    PatternResult(
                        pattern_type="frequency_spike",
                        confidence=min(
                            1.0, frequency_score / self.baseline_threshold / 2
                        ),
                        strength=min(1.0, frequency_score / 10),
                        metadata={"frequency_score": frequency_score},
                    )
                )

        return patterns


class CorrelationAnalyzer:
    """Analyzes correlation patterns between different data points."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.correlation_threshold = config.get("correlation_threshold", 0.7)
        self._statistical_analyzer = StatisticalCorrelationAnalyzer(config)

    async def analyze_correlation_patterns(
        self, data_points: List[Dict[str, Any]]
    ) -> List[PatternResult]:
        """Analyze correlation patterns in data."""
        patterns: List[PatternResult] = []

        if len(data_points) < self._statistical_analyzer.min_data_points:
            logger.debug(
                "Insufficient data points for correlation analysis",
                extra={"data_points": len(data_points)},
            )
            return patterns

        try:
            raw_patterns = await self._statistical_analyzer.analyze(data_points)

            for pattern in raw_patterns:
                coefficient = float(pattern.get("correlation_coefficient", 0.0))
                strength = abs(coefficient)

                if strength < self.correlation_threshold:
                    continue

                confidence = float(
                    pattern.get(
                        "confidence",
                        min(1.0, max(0.0, strength)),
                    )
                )

                patterns.append(
                    PatternResult(
                        pattern_type="data_correlation",
                        confidence=confidence,
                        strength=strength,
                        metadata=pattern,
                    )
                )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Correlation pattern analysis failed",
                extra={"error": str(exc)},
            )

        return patterns

    def _calculate_simple_correlation(self, data_points: List[Dict[str, Any]]) -> float:
        """Lightweight fallback correlation score estimation.

        Used only when the advanced statistical analyzer is unavailable.
        """

        if not data_points:
            return 0.0

        return min(1.0, len(data_points) / 10)


class AnomalyDetector:
    """Detects anomalous patterns in data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0)

    async def detect_anomalies(
        self, data_points: List[Dict[str, Any]]
    ) -> List[PatternResult]:
        """Detect anomalous patterns."""
        patterns = []

        if not data_points:
            return patterns

        unusual_points: List[Dict[str, Any]] = []

        # Primary path: numeric anomaly detection using z-scores on common fields such
        # as "value", "rate", etc. Falls back to severity-based heuristics.
        numeric_keys: List[str] = []
        for point in data_points:
            for key, value in point.items():
                if key == "timestamp":
                    continue
                if isinstance(value, (int, float)) and key not in numeric_keys:
                    numeric_keys.append(key)

        def _detect_numeric_anomalies(key: str) -> List[int]:
            series: List[float] = []
            for point in data_points:
                value = point.get(key)
                if isinstance(value, (int, float)):
                    series.append(float(value))

            if len(series) < 3:
                return []

            mean = sum(series) / len(series)
            variance = sum((v - mean) ** 2 for v in series) / len(series)
            std = variance**0.5
            if std == 0:
                return []

            indices: List[int] = []
            for idx, point in enumerate(data_points):
                value = point.get(key)
                if not isinstance(value, (int, float)):
                    continue
                z_score = (float(value) - mean) / std
                if abs(z_score) >= self.anomaly_threshold:
                    indices.append(idx)

            return indices

        anomalous_indices: List[int] = []
        for key in numeric_keys:
            anomalous_indices.extend(_detect_numeric_anomalies(key))

        if anomalous_indices:
            seen_idx = set()
            for idx in anomalous_indices:
                if idx not in seen_idx:
                    seen_idx.add(idx)
                    unusual_points.append(data_points[idx])

        # Fallback to severity-based heuristic when numeric analysis finds nothing
        if not unusual_points:
            unusual_points = [point for point in data_points if self._is_anomalous(point)]

        if unusual_points:
            anomaly_ratio = len(unusual_points) / len(data_points)
            if anomaly_ratio > 0.1:  # More than 10% anomalous
                patterns.append(
                    PatternResult(
                        pattern_type="behavioral_anomaly",
                        confidence=min(1.0, anomaly_ratio * 2),
                        strength=anomaly_ratio,
                        metadata={
                            "anomaly_count": len(unusual_points),
                            "total_points": len(data_points),
                        },
                    )
                )

        return patterns

    def _is_anomalous(self, data_point: Dict[str, Any]) -> bool:
        """Check if a data point is anomalous."""
        # Heuristic fallback using severity classification when numeric context
        # is unavailable or insufficient for statistical detection.
        severity = str(data_point.get("severity", "low")).lower()
        return severity in ["critical", "high"]


class PatternClassifier:
    """Classifies detected patterns into categories."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def classify_patterns(
        self, patterns: List[PatternResult]
    ) -> Dict[str, List[PatternResult]]:
        """Classify patterns into categories."""
        classified = {
            "security": [],
            "performance": [],
            "behavioral": [],
            "operational": [],
        }

        for pattern in patterns:
            category = self._classify_pattern(pattern)
            classified[category].append(pattern)

        return classified

    def _classify_pattern(self, pattern: PatternResult) -> str:
        """Classify a single pattern."""
        pattern_type = pattern.pattern_type

        if "anomaly" in pattern_type or "spike" in pattern_type:
            return "security"
        elif "temporal" in pattern_type or "frequency" in pattern_type:
            return "performance"
        elif "behavioral" in pattern_type:
            return "behavioral"
        else:
            return "operational"


class PatternStrengthCalculator:
    """Calculates strength scores for patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def calculate_strength(
        self, pattern: PatternResult, context: Dict[str, Any]
    ) -> float:
        """Calculate pattern strength score."""
        base_strength = pattern.strength
        confidence_factor = pattern.confidence

        # Adjust based on context
        context_factor = 1.0
        if context.get("high_risk_environment", False):
            context_factor = 1.2

        return min(1.0, base_strength * confidence_factor * context_factor)


class BusinessRelevanceAssessor:
    """Assesses business relevance of patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def assess_relevance(
        self, pattern: PatternResult, business_context: Dict[str, Any]
    ) -> float:
        """Assess business relevance of a pattern."""
        base_relevance = 0.5

        # Adjust based on business impact
        if business_context.get("affects_revenue", False):
            base_relevance += 0.3
        if business_context.get("affects_compliance", False):
            base_relevance += 0.2

        return min(1.0, base_relevance)


class PatternConfidenceCalculator:
    """Calculates confidence scores for patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def calculate_confidence(
        self, pattern: PatternResult, evidence: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for a pattern."""
        base_confidence = pattern.confidence

        # Adjust based on evidence quality
        evidence_factor = min(
            1.0, len(evidence) / 5
        )  # More evidence = higher confidence

        return min(1.0, base_confidence * (1 + evidence_factor * 0.2))


class MultiPatternAnalyzer:
    """Analyzes interactions between multiple patterns."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def analyze_pattern_interactions(
        self, patterns: List[PatternResult]
    ) -> Dict[str, Any]:
        """Analyze interactions between patterns."""
        if len(patterns) < 2:
            return {"interaction_score": 0.0, "interactions": []}

        interactions = []
        for i, pattern1 in enumerate(patterns):
            for pattern2 in patterns[i + 1 :]:
                interaction = self._analyze_pair_interaction(pattern1, pattern2)
                if interaction["strength"] > 0.3:
                    interactions.append(interaction)

        interaction_score = sum(i["strength"] for i in interactions) / len(patterns)

        return {
            "interaction_score": min(1.0, interaction_score),
            "interactions": interactions,
        }

    def _analyze_pair_interaction(
        self, pattern1: PatternResult, pattern2: PatternResult
    ) -> Dict[str, Any]:
        """Analyze interaction between two patterns."""
        # Simple interaction analysis
        strength = (
            pattern1.strength + pattern2.strength
        ) / 4  # Reduced for interaction

        return {
            "pattern1": pattern1.pattern_type,
            "pattern2": pattern2.pattern_type,
            "strength": strength,
            "type": "amplification" if strength > 0.5 else "correlation",
        }


class PatternEvolutionTracker:
    """Tracks evolution of patterns over time."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def track_evolution(
        self,
        current_patterns: List[PatternResult],
        historical_patterns: List[List[PatternResult]],
    ) -> Dict[str, Any]:
        """Track pattern evolution over time."""
        if not historical_patterns:
            return {"evolution_score": 0.0, "trends": []}

        trends = []
        for pattern in current_patterns:
            trend = self._analyze_pattern_trend(pattern, historical_patterns)
            trends.append(trend)

        evolution_score = (
            sum(t["change_rate"] for t in trends) / len(trends) if trends else 0.0
        )

        return {"evolution_score": abs(evolution_score), "trends": trends}

    def _analyze_pattern_trend(
        self, pattern: PatternResult, historical_patterns: List[List[PatternResult]]
    ) -> Dict[str, Any]:
        """Analyze trend for a specific pattern."""
        # Simple trend analysis
        historical_strengths = []
        for historical_set in historical_patterns:
            for hist_pattern in historical_set:
                if hist_pattern.pattern_type == pattern.pattern_type:
                    historical_strengths.append(hist_pattern.strength)

        if len(historical_strengths) >= 2:
            change_rate = (
                pattern.strength - historical_strengths[-1]
            ) / historical_strengths[-1]
        else:
            change_rate = 0.0

        return {
            "pattern_type": pattern.pattern_type,
            "change_rate": change_rate,
            "trend": (
                "increasing"
                if change_rate > 0.1
                else "decreasing" if change_rate < -0.1 else "stable"
            ),
        }


class PatternInteractionMatrix:
    """Analyzes complex interactions between patterns using matrix analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def build_interaction_matrix(
        self, patterns: List[PatternResult]
    ) -> Dict[str, Any]:
        """Build interaction matrix for patterns."""
        if len(patterns) < 2:
            return {"matrix": [], "complexity_score": 0.0}

        matrix = []
        for i, pattern1 in enumerate(patterns):
            row = []
            for j, pattern2 in enumerate(patterns):
                if i == j:
                    interaction_strength = 1.0  # Self-interaction
                else:
                    interaction_strength = self._calculate_interaction_strength(
                        pattern1, pattern2
                    )
                row.append(interaction_strength)
            matrix.append(row)

        # Calculate complexity score
        total_interactions = sum(sum(row) for row in matrix)
        complexity_score = total_interactions / (len(patterns) ** 2)

        return {
            "matrix": matrix,
            "complexity_score": min(1.0, complexity_score),
            "pattern_types": [p.pattern_type for p in patterns],
        }

    def _calculate_interaction_strength(
        self, pattern1: PatternResult, pattern2: PatternResult
    ) -> float:
        """Calculate interaction strength between two patterns."""
        # Simple interaction calculation
        strength_product = pattern1.strength * pattern2.strength
        confidence_product = pattern1.confidence * pattern2.confidence

        return (strength_product + confidence_product) / 2
