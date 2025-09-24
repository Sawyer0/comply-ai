"""
Pattern Evolution Tracker for monitoring pattern changes over time.

This module tracks how security patterns evolve, change, and develop
over time to identify trends and predict future pattern behavior.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    SecurityData,
    TimeRange,
)

logger = logging.getLogger(__name__)


class PatternEvolutionTracker:
    """
    Tracks evolution and changes in security patterns over time.

    Monitors pattern lifecycle, strength changes, relationship evolution,
    and predicts future pattern behavior based on historical trends.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.tracking_window_days = self.config.get("tracking_window_days", 30)
        self.evolution_threshold = self.config.get("evolution_threshold", 0.1)
        self.prediction_horizon_days = self.config.get("prediction_horizon_days", 7)
        self.pattern_history = {}  # In production, this would be persistent storage

    async def track_pattern_evolution(
        self,
        current_patterns: List[Pattern],
        historical_patterns: List[List[Pattern]],
        context_data: SecurityData,
    ) -> Dict[str, any]:
        """
        Track evolution of patterns over time.

        Args:
            current_patterns: Current set of patterns
            historical_patterns: Historical pattern snapshots
            context_data: Security data for context

        Returns:
            Dictionary containing evolution analysis results
        """
        try:
            # Update pattern history
            self._update_pattern_history(current_patterns)

            # Analyze pattern lifecycle
            lifecycle_analysis = await self._analyze_pattern_lifecycle(
                current_patterns, historical_patterns
            )

            # Track strength evolution
            strength_evolution = await self._track_strength_evolution(
                current_patterns, historical_patterns
            )

            # Monitor relationship evolution
            relationship_evolution = await self._monitor_relationship_evolution(
                current_patterns, historical_patterns
            )

            # Detect emerging patterns
            emerging_patterns = await self._detect_emerging_patterns(
                current_patterns, historical_patterns
            )

            # Identify declining patterns
            declining_patterns = await self._identify_declining_patterns(
                current_patterns, historical_patterns
            )

            # Predict future evolution
            evolution_predictions = await self._predict_pattern_evolution(
                current_patterns, historical_patterns
            )

            # Calculate evolution metrics
            evolution_metrics = self._calculate_evolution_metrics(
                lifecycle_analysis, strength_evolution, relationship_evolution
            )

            evolution_result = {
                "lifecycle_analysis": lifecycle_analysis,
                "strength_evolution": strength_evolution,
                "relationship_evolution": relationship_evolution,
                "emerging_patterns": emerging_patterns,
                "declining_patterns": declining_patterns,
                "evolution_predictions": evolution_predictions,
                "evolution_metrics": evolution_metrics,
                "tracking_metadata": {
                    "current_pattern_count": len(current_patterns),
                    "historical_snapshots": len(historical_patterns),
                    "tracking_window_days": self.tracking_window_days,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                },
            }

            logger.info(
                "Pattern evolution tracking completed",
                current_patterns=len(current_patterns),
                emerging_patterns=len(emerging_patterns),
                declining_patterns=len(declining_patterns),
            )

            return evolution_result

        except Exception as e:
            logger.error("Pattern evolution tracking failed", error=str(e))
            return {}

    def _update_pattern_history(self, current_patterns: List[Pattern]):
        """Update pattern history with current patterns."""
        try:
            current_time = datetime.utcnow()

            for pattern in current_patterns:
                if pattern.pattern_id not in self.pattern_history:
                    self.pattern_history[pattern.pattern_id] = []

                # Add current state to history
                self.pattern_history[pattern.pattern_id].append(
                    {
                        "timestamp": current_time,
                        "confidence": pattern.confidence,
                        "strength": pattern.strength.value,
                        "business_relevance": pattern.business_relevance.value,
                        "affected_detectors": pattern.affected_detectors.copy(),
                        "statistical_significance": pattern.statistical_significance,
                    }
                )

                # Keep only recent history
                cutoff_time = current_time - timedelta(days=self.tracking_window_days)
                self.pattern_history[pattern.pattern_id] = [
                    entry
                    for entry in self.pattern_history[pattern.pattern_id]
                    if entry["timestamp"] > cutoff_time
                ]

        except Exception as e:
            logger.error("Pattern history update failed", error=str(e))

    async def _analyze_pattern_lifecycle(
        self, current_patterns: List[Pattern], historical_patterns: List[List[Pattern]]
    ) -> Dict[str, any]:
        """Analyze pattern lifecycle stages."""
        try:
            lifecycle_stages = {
                "emerging": [],
                "growing": [],
                "mature": [],
                "declining": [],
                "dormant": [],
            }

            # Track pattern IDs across time
            all_pattern_ids = set()
            for patterns in historical_patterns + [current_patterns]:
                all_pattern_ids.update(p.pattern_id for p in patterns)

            for pattern_id in all_pattern_ids:
                lifecycle_stage = self._determine_lifecycle_stage(
                    pattern_id, historical_patterns, current_patterns
                )
                lifecycle_stages[lifecycle_stage].append(pattern_id)

            # Calculate lifecycle statistics
            lifecycle_stats = {
                stage: len(patterns) for stage, patterns in lifecycle_stages.items()
            }

            # Identify lifecycle transitions
            transitions = self._identify_lifecycle_transitions(
                historical_patterns, current_patterns
            )

            return {
                "lifecycle_stages": lifecycle_stages,
                "lifecycle_statistics": lifecycle_stats,
                "lifecycle_transitions": transitions,
                "total_patterns_tracked": len(all_pattern_ids),
            }

        except Exception as e:
            logger.error("Pattern lifecycle analysis failed", error=str(e))
            return {}

    def _determine_lifecycle_stage(
        self,
        pattern_id: str,
        historical_patterns: List[List[Pattern]],
        current_patterns: List[Pattern],
    ) -> str:
        """Determine lifecycle stage for a pattern."""
        try:
            # Check if pattern exists in current patterns
            current_pattern = next(
                (p for p in current_patterns if p.pattern_id == pattern_id), None
            )

            # Count historical appearances
            historical_appearances = 0
            for patterns in historical_patterns:
                if any(p.pattern_id == pattern_id for p in patterns):
                    historical_appearances += 1

            if not current_pattern:
                if historical_appearances > 0:
                    return "dormant"
                else:
                    return "declining"

            # Pattern exists currently
            if historical_appearances == 0:
                return "emerging"
            elif historical_appearances < len(historical_patterns) * 0.3:
                return "growing"
            elif historical_appearances > len(historical_patterns) * 0.8:
                return "mature"
            else:
                # Check trend
                if self._is_pattern_growing(pattern_id, historical_patterns):
                    return "growing"
                else:
                    return "declining"

        except Exception:
            return "unknown"

    def _is_pattern_growing(
        self, pattern_id: str, historical_patterns: List[List[Pattern]]
    ) -> bool:
        """Check if pattern is growing based on confidence trend."""
        try:
            confidences = []

            for patterns in historical_patterns[-5:]:  # Last 5 snapshots
                pattern = next(
                    (p for p in patterns if p.pattern_id == pattern_id), None
                )
                if pattern:
                    confidences.append(pattern.confidence)

            if len(confidences) < 2:
                return False

            # Simple trend check
            return confidences[-1] > confidences[0]

        except Exception:
            return False

    def _identify_lifecycle_transitions(
        self, historical_patterns: List[List[Pattern]], current_patterns: List[Pattern]
    ) -> List[Dict[str, any]]:
        """Identify patterns that have transitioned between lifecycle stages."""
        transitions = []

        try:
            if len(historical_patterns) < 2:
                return transitions

            # Compare last two snapshots
            previous_patterns = historical_patterns[-1]

            # Track patterns that changed significantly
            for current_pattern in current_patterns:
                previous_pattern = next(
                    (
                        p
                        for p in previous_patterns
                        if p.pattern_id == current_pattern.pattern_id
                    ),
                    None,
                )

                if previous_pattern:
                    # Check for significant changes
                    confidence_change = (
                        current_pattern.confidence - previous_pattern.confidence
                    )
                    strength_change = self._compare_strengths(
                        current_pattern.strength, previous_pattern.strength
                    )

                    if (
                        abs(confidence_change) > self.evolution_threshold
                        or strength_change != 0
                    ):
                        transitions.append(
                            {
                                "pattern_id": current_pattern.pattern_id,
                                "transition_type": self._determine_transition_type(
                                    confidence_change, strength_change
                                ),
                                "confidence_change": confidence_change,
                                "strength_change": strength_change,
                                "previous_confidence": previous_pattern.confidence,
                                "current_confidence": current_pattern.confidence,
                            }
                        )

        except Exception as e:
            logger.error("Lifecycle transitions identification failed", error=str(e))

        return transitions

    def _compare_strengths(
        self, current_strength: PatternStrength, previous_strength: PatternStrength
    ) -> int:
        """Compare pattern strengths and return change direction."""
        strength_values = {
            PatternStrength.WEAK: 1,
            PatternStrength.MODERATE: 2,
            PatternStrength.STRONG: 3,
        }

        current_val = strength_values.get(current_strength, 2)
        previous_val = strength_values.get(previous_strength, 2)

        if current_val > previous_val:
            return 1  # Strengthening
        elif current_val < previous_val:
            return -1  # Weakening
        else:
            return 0  # No change

    def _determine_transition_type(
        self, confidence_change: float, strength_change: int
    ) -> str:
        """Determine the type of transition based on changes."""
        if confidence_change > self.evolution_threshold and strength_change >= 0:
            return "strengthening"
        elif confidence_change < -self.evolution_threshold and strength_change <= 0:
            return "weakening"
        elif confidence_change > self.evolution_threshold:
            return "confidence_increase"
        elif confidence_change < -self.evolution_threshold:
            return "confidence_decrease"
        elif strength_change > 0:
            return "strength_increase"
        elif strength_change < 0:
            return "strength_decrease"
        else:
            return "stable"

    async def _track_strength_evolution(
        self, current_patterns: List[Pattern], historical_patterns: List[List[Pattern]]
    ) -> Dict[str, any]:
        """Track how pattern strengths evolve over time."""
        try:
            strength_trends = {}

            for pattern in current_patterns:
                pattern_id = pattern.pattern_id

                # Collect historical strength data
                strength_history = []
                for patterns in historical_patterns:
                    historical_pattern = next(
                        (p for p in patterns if p.pattern_id == pattern_id), None
                    )
                    if historical_pattern:
                        strength_history.append(
                            {
                                "timestamp": datetime.utcnow(),  # Would be actual timestamp in production
                                "strength": historical_pattern.strength.value,
                                "confidence": historical_pattern.confidence,
                            }
                        )

                # Add current strength
                strength_history.append(
                    {
                        "timestamp": datetime.utcnow(),
                        "strength": pattern.strength.value,
                        "confidence": pattern.confidence,
                    }
                )

                if len(strength_history) > 1:
                    trend = self._calculate_strength_trend(strength_history)
                    strength_trends[pattern_id] = {
                        "current_strength": pattern.strength.value,
                        "trend": trend,
                        "history_length": len(strength_history),
                        "volatility": self._calculate_strength_volatility(
                            strength_history
                        ),
                    }

            # Identify patterns with significant strength changes
            significant_changes = [
                {"pattern_id": pid, **data}
                for pid, data in strength_trends.items()
                if abs(data["trend"]) > 0.2
            ]

            return {
                "strength_trends": strength_trends,
                "significant_changes": significant_changes,
                "average_trend": (
                    np.mean([data["trend"] for data in strength_trends.values()])
                    if strength_trends
                    else 0
                ),
                "trend_statistics": self._calculate_trend_statistics(strength_trends),
            }

        except Exception as e:
            logger.error("Strength evolution tracking failed", error=str(e))
            return {}

    def _calculate_strength_trend(
        self, strength_history: List[Dict[str, any]]
    ) -> float:
        """Calculate trend in pattern strength over time."""
        try:
            if len(strength_history) < 2:
                return 0.0

            # Convert strength to numeric values
            strength_values = {"weak": 1, "moderate": 2, "strong": 3}

            values = [
                strength_values.get(entry["strength"], 2) for entry in strength_history
            ]

            # Simple linear trend calculation
            n = len(values)
            x = list(range(n))

            # Calculate slope using least squares
            x_mean = sum(x) / n
            y_mean = sum(values) / n

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return 0.0

            slope = numerator / denominator
            return slope

        except Exception:
            return 0.0

    def _calculate_strength_volatility(
        self, strength_history: List[Dict[str, any]]
    ) -> float:
        """Calculate volatility in pattern strength."""
        try:
            if len(strength_history) < 2:
                return 0.0

            strength_values = {"weak": 1, "moderate": 2, "strong": 3}

            values = [
                strength_values.get(entry["strength"], 2) for entry in strength_history
            ]

            # Calculate standard deviation
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)

            return variance**0.5

        except Exception:
            return 0.0

    def _calculate_trend_statistics(
        self, strength_trends: Dict[str, any]
    ) -> Dict[str, any]:
        """Calculate statistics for strength trends."""
        try:
            if not strength_trends:
                return {}

            trends = [data["trend"] for data in strength_trends.values()]
            volatilities = [data["volatility"] for data in strength_trends.values()]

            return {
                "strengthening_patterns": len([t for t in trends if t > 0.1]),
                "weakening_patterns": len([t for t in trends if t < -0.1]),
                "stable_patterns": len([t for t in trends if abs(t) <= 0.1]),
                "average_volatility": np.mean(volatilities) if volatilities else 0,
                "max_trend": max(trends) if trends else 0,
                "min_trend": min(trends) if trends else 0,
            }

        except Exception:
            return {}

    async def _monitor_relationship_evolution(
        self, current_patterns: List[Pattern], historical_patterns: List[List[Pattern]]
    ) -> Dict[str, any]:
        """Monitor how pattern relationships evolve over time."""
        try:
            # This would track relationship changes between patterns
            # For now, return a simplified implementation

            relationship_changes = []

            if len(historical_patterns) > 0:
                previous_patterns = historical_patterns[-1]

                # Compare detector overlaps between current and previous
                for current_pattern in current_patterns:
                    previous_pattern = next(
                        (
                            p
                            for p in previous_patterns
                            if p.pattern_id == current_pattern.pattern_id
                        ),
                        None,
                    )

                    if previous_pattern:
                        current_detectors = set(current_pattern.affected_detectors)
                        previous_detectors = set(previous_pattern.affected_detectors)

                        added_detectors = current_detectors - previous_detectors
                        removed_detectors = previous_detectors - current_detectors

                        if added_detectors or removed_detectors:
                            relationship_changes.append(
                                {
                                    "pattern_id": current_pattern.pattern_id,
                                    "added_detectors": list(added_detectors),
                                    "removed_detectors": list(removed_detectors),
                                    "change_type": "detector_scope_change",
                                }
                            )

            return {
                "relationship_changes": relationship_changes,
                "total_changes": len(relationship_changes),
                "change_types": list(
                    set(change["change_type"] for change in relationship_changes)
                ),
            }

        except Exception as e:
            logger.error("Relationship evolution monitoring failed", error=str(e))
            return {}

    async def _detect_emerging_patterns(
        self, current_patterns: List[Pattern], historical_patterns: List[List[Pattern]]
    ) -> List[Dict[str, any]]:
        """Detect newly emerging patterns."""
        emerging = []

        try:
            # Get all historical pattern IDs
            historical_ids = set()
            for patterns in historical_patterns:
                historical_ids.update(p.pattern_id for p in patterns)

            # Find patterns that are new
            for pattern in current_patterns:
                if pattern.pattern_id not in historical_ids:
                    emerging.append(
                        {
                            "pattern_id": pattern.pattern_id,
                            "pattern_type": pattern.pattern_type.value,
                            "confidence": pattern.confidence,
                            "strength": pattern.strength.value,
                            "business_relevance": pattern.business_relevance.value,
                            "affected_detectors": pattern.affected_detectors,
                            "emergence_indicators": self._assess_emergence_indicators(
                                pattern
                            ),
                        }
                    )

        except Exception as e:
            logger.error("Emerging patterns detection failed", error=str(e))

        return emerging

    def _assess_emergence_indicators(self, pattern: Pattern) -> Dict[str, any]:
        """Assess indicators that suggest pattern emergence."""
        try:
            indicators = {
                "high_confidence": pattern.confidence > 0.8,
                "strong_pattern": pattern.strength == PatternStrength.STRONG,
                "high_business_relevance": pattern.business_relevance
                in [BusinessRelevance.HIGH, BusinessRelevance.CRITICAL],
                "multiple_detectors": len(pattern.affected_detectors) > 2,
                "recent_timeframe": True,  # Would check if pattern is very recent
            }

            indicators["emergence_score"] = sum(indicators.values()) / len(indicators)

            return indicators

        except Exception:
            return {"emergence_score": 0.0}

    async def _identify_declining_patterns(
        self, current_patterns: List[Pattern], historical_patterns: List[List[Pattern]]
    ) -> List[Dict[str, any]]:
        """Identify patterns that are declining or disappearing."""
        declining = []

        try:
            if not historical_patterns:
                return declining

            # Get recent historical pattern IDs
            recent_historical_ids = set()
            for patterns in historical_patterns[-3:]:  # Last 3 snapshots
                recent_historical_ids.update(p.pattern_id for p in patterns)

            current_ids = set(p.pattern_id for p in current_patterns)

            # Find patterns that were recent but are now missing
            missing_patterns = recent_historical_ids - current_ids

            for pattern_id in missing_patterns:
                # Get last known state
                last_known_pattern = None
                for patterns in reversed(historical_patterns):
                    last_known_pattern = next(
                        (p for p in patterns if p.pattern_id == pattern_id), None
                    )
                    if last_known_pattern:
                        break

                if last_known_pattern:
                    declining.append(
                        {
                            "pattern_id": pattern_id,
                            "last_known_confidence": last_known_pattern.confidence,
                            "last_known_strength": last_known_pattern.strength.value,
                            "decline_indicators": self._assess_decline_indicators(
                                pattern_id, historical_patterns
                            ),
                        }
                    )

        except Exception as e:
            logger.error("Declining patterns identification failed", error=str(e))

        return declining

    def _assess_decline_indicators(
        self, pattern_id: str, historical_patterns: List[List[Pattern]]
    ) -> Dict[str, any]:
        """Assess indicators that suggest pattern decline."""
        try:
            # Track confidence over time
            confidences = []
            for patterns in historical_patterns[-5:]:  # Last 5 snapshots
                pattern = next(
                    (p for p in patterns if p.pattern_id == pattern_id), None
                )
                if pattern:
                    confidences.append(pattern.confidence)

            indicators = {
                "declining_confidence": len(confidences) > 1
                and confidences[-1] < confidences[0],
                "low_final_confidence": confidences[-1] < 0.5 if confidences else False,
                "intermittent_appearance": len(confidences)
                < len(historical_patterns[-5:]) * 0.6,
                "recent_absence": True,  # Pattern is currently absent
            }

            indicators["decline_score"] = sum(indicators.values()) / len(indicators)

            return indicators

        except Exception:
            return {"decline_score": 0.0}

    async def _predict_pattern_evolution(
        self, current_patterns: List[Pattern], historical_patterns: List[List[Pattern]]
    ) -> Dict[str, any]:
        """Predict future pattern evolution based on trends."""
        try:
            predictions = {}

            for pattern in current_patterns:
                if pattern.pattern_id in self.pattern_history:
                    history = self.pattern_history[pattern.pattern_id]

                    if len(history) >= 3:  # Need sufficient history for prediction
                        prediction = self._predict_single_pattern_evolution(
                            pattern, history
                        )
                        predictions[pattern.pattern_id] = prediction

            # Generate overall predictions
            overall_predictions = self._generate_overall_predictions(predictions)

            return {
                "individual_predictions": predictions,
                "overall_predictions": overall_predictions,
                "prediction_horizon_days": self.prediction_horizon_days,
                "prediction_confidence": self._calculate_prediction_confidence(
                    predictions
                ),
            }

        except Exception as e:
            logger.error("Pattern evolution prediction failed", error=str(e))
            return {}

    def _predict_single_pattern_evolution(
        self, pattern: Pattern, history: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Predict evolution for a single pattern."""
        try:
            # Extract confidence trend
            confidences = [entry["confidence"] for entry in history]

            # Simple linear prediction
            if len(confidences) >= 2:
                trend = (confidences[-1] - confidences[0]) / len(confidences)
                predicted_confidence = confidences[-1] + (
                    trend * self.prediction_horizon_days
                )
                predicted_confidence = max(0.0, min(1.0, predicted_confidence))

                return {
                    "current_confidence": pattern.confidence,
                    "predicted_confidence": predicted_confidence,
                    "confidence_trend": trend,
                    "prediction_type": "linear_extrapolation",
                    "prediction_reliability": min(
                        1.0, len(history) / 10
                    ),  # More history = more reliable
                }

            return {
                "current_confidence": pattern.confidence,
                "predicted_confidence": pattern.confidence,
                "confidence_trend": 0.0,
                "prediction_type": "stable",
                "prediction_reliability": 0.5,
            }

        except Exception:
            return {"prediction_reliability": 0.0}

    def _generate_overall_predictions(
        self, individual_predictions: Dict[str, any]
    ) -> Dict[str, any]:
        """Generate overall system-level predictions."""
        try:
            if not individual_predictions:
                return {}

            trends = [
                pred["confidence_trend"] for pred in individual_predictions.values()
            ]

            return {
                "overall_trend": (
                    "improving"
                    if np.mean(trends) > 0.05
                    else "declining" if np.mean(trends) < -0.05 else "stable"
                ),
                "patterns_improving": len([t for t in trends if t > 0.05]),
                "patterns_declining": len([t for t in trends if t < -0.05]),
                "patterns_stable": len([t for t in trends if abs(t) <= 0.05]),
                "average_trend": np.mean(trends),
                "trend_volatility": np.std(trends),
            }

        except Exception:
            return {}

    def _calculate_prediction_confidence(self, predictions: Dict[str, any]) -> float:
        """Calculate overall confidence in predictions."""
        try:
            if not predictions:
                return 0.0

            reliabilities = [
                pred["prediction_reliability"] for pred in predictions.values()
            ]
            return np.mean(reliabilities)

        except Exception:
            return 0.0

    def _calculate_evolution_metrics(
        self,
        lifecycle_analysis: Dict[str, any],
        strength_evolution: Dict[str, any],
        relationship_evolution: Dict[str, any],
    ) -> Dict[str, any]:
        """Calculate comprehensive evolution metrics."""
        try:
            metrics = {
                "pattern_stability": self._calculate_pattern_stability(
                    lifecycle_analysis
                ),
                "evolution_velocity": self._calculate_evolution_velocity(
                    strength_evolution
                ),
                "relationship_dynamism": self._calculate_relationship_dynamism(
                    relationship_evolution
                ),
                "overall_evolution_score": 0.0,
            }

            # Calculate overall evolution score
            metrics["overall_evolution_score"] = (
                metrics["pattern_stability"] * 0.4
                + metrics["evolution_velocity"] * 0.3
                + metrics["relationship_dynamism"] * 0.3
            )

            return metrics

        except Exception as e:
            logger.error("Evolution metrics calculation failed", error=str(e))
            return {}

    def _calculate_pattern_stability(self, lifecycle_analysis: Dict[str, any]) -> float:
        """Calculate pattern stability metric."""
        try:
            stats = lifecycle_analysis.get("lifecycle_statistics", {})
            total_patterns = sum(stats.values())

            if total_patterns == 0:
                return 0.0

            # Stable patterns are mature and growing
            stable_count = stats.get("mature", 0) + stats.get("growing", 0)
            return stable_count / total_patterns

        except Exception:
            return 0.0

    def _calculate_evolution_velocity(
        self, strength_evolution: Dict[str, any]
    ) -> float:
        """Calculate evolution velocity metric."""
        try:
            avg_trend = abs(strength_evolution.get("average_trend", 0))
            return min(1.0, avg_trend * 5)  # Scale to 0-1

        except Exception:
            return 0.0

    def _calculate_relationship_dynamism(
        self, relationship_evolution: Dict[str, any]
    ) -> float:
        """Calculate relationship dynamism metric."""
        try:
            total_changes = relationship_evolution.get("total_changes", 0)
            # Normalize by some expected baseline
            return min(1.0, total_changes / 10)

        except Exception:
            return 0.0
