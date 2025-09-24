"""
Pattern Classifier for categorizing detected patterns.

This module implements sophisticated pattern classification logic
using statistical methods and business rules.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    SecurityData,
)

logger = logging.getLogger(__name__)


class PatternClassifier:
    """
    Classifies detected patterns into categories and assesses their characteristics.

    Uses statistical analysis and business logic to categorize patterns
    and determine their significance for security operations.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.classification_rules = self._load_classification_rules()
        self.business_context = self.config.get("business_context", {})

    def classify_pattern(self, pattern: Pattern, context_data: SecurityData) -> Pattern:
        """
        Classify a pattern and enhance it with additional metadata.

        Args:
            pattern: The pattern to classify
            context_data: Additional security data for context

        Returns:
            Enhanced pattern with classification metadata
        """
        try:
            # Enhance pattern type classification
            refined_type = self._refine_pattern_type(pattern, context_data)

            # Classify pattern characteristics
            characteristics = self._classify_characteristics(pattern, context_data)

            # Assess pattern stability
            stability = self._assess_pattern_stability(pattern, context_data)

            # Update pattern with enhanced classification
            enhanced_pattern = Pattern(
                pattern_id=pattern.pattern_id,
                pattern_type=refined_type,
                strength=pattern.strength,
                confidence=pattern.confidence,
                description=self._generate_enhanced_description(
                    pattern, characteristics
                ),
                affected_detectors=pattern.affected_detectors,
                time_range=pattern.time_range,
                statistical_significance=pattern.statistical_significance,
                business_relevance=pattern.business_relevance,
                supporting_evidence=pattern.supporting_evidence
                + [
                    {
                        "classification_metadata": {
                            "characteristics": characteristics,
                            "stability": stability,
                            "classification_confidence": self._calculate_classification_confidence(
                                pattern, characteristics, stability
                            ),
                            "pattern_category": self._determine_pattern_category(
                                pattern, characteristics
                            ),
                            "risk_indicators": self._identify_risk_indicators(
                                pattern, characteristics
                            ),
                        }
                    }
                ],
            )

            logger.info(
                "Pattern classified successfully",
                pattern_id=pattern.pattern_id,
                pattern_type=refined_type.value,
                characteristics=characteristics,
            )

            return enhanced_pattern

        except Exception as e:
            logger.error(
                "Pattern classification failed",
                error=str(e),
                pattern_id=pattern.pattern_id,
            )
            return pattern  # Return original pattern if classification fails

    def classify_multiple_patterns(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> List[Pattern]:
        """
        Classify multiple patterns with cross-pattern analysis.

        Args:
            patterns: List of patterns to classify
            context_data: Security data for context

        Returns:
            List of classified patterns with enhanced metadata
        """
        classified_patterns = []

        # First pass: individual classification
        for pattern in patterns:
            classified_pattern = self.classify_pattern(pattern, context_data)
            classified_patterns.append(classified_pattern)

        # Second pass: cross-pattern analysis
        self._perform_cross_pattern_analysis(classified_patterns, context_data)

        return classified_patterns

    def _refine_pattern_type(
        self, pattern: Pattern, context_data: SecurityData
    ) -> PatternType:
        """Refine pattern type based on detailed analysis."""
        current_type = pattern.pattern_type

        # Analyze pattern characteristics to potentially refine type
        if current_type == PatternType.TEMPORAL:
            # Check if it's actually a frequency pattern with temporal component
            if self._has_regular_frequency(pattern, context_data):
                return PatternType.FREQUENCY

        elif current_type == PatternType.FREQUENCY:
            # Check if it's actually a correlation pattern
            if self._has_strong_correlations(pattern, context_data):
                return PatternType.CORRELATION

        elif current_type == PatternType.ANOMALY:
            # Check if anomaly is part of a larger temporal trend
            if self._is_part_of_trend(pattern, context_data):
                return PatternType.TEMPORAL

        return current_type

    def _classify_characteristics(
        self, pattern: Pattern, context_data: SecurityData
    ) -> Dict[str, any]:
        """Classify pattern characteristics."""
        characteristics = {
            "persistence": self._assess_persistence(pattern, context_data),
            "scope": self._assess_scope(pattern, context_data),
            "intensity": self._assess_intensity(pattern, context_data),
            "predictability": self._assess_predictability(pattern, context_data),
            "impact_potential": self._assess_impact_potential(pattern, context_data),
            "detection_difficulty": self._assess_detection_difficulty(
                pattern, context_data
            ),
        }

        return characteristics

    def _assess_pattern_stability(
        self, pattern: Pattern, context_data: SecurityData
    ) -> Dict[str, any]:
        """Assess how stable/consistent the pattern is."""
        stability_metrics = {
            "consistency_score": self._calculate_consistency_score(
                pattern, context_data
            ),
            "variance_level": self._calculate_variance_level(pattern, context_data),
            "trend_stability": self._assess_trend_stability(pattern, context_data),
            "seasonal_component": self._detect_seasonal_component(
                pattern, context_data
            ),
        }

        return stability_metrics

    def _assess_persistence(self, pattern: Pattern, context_data: SecurityData) -> str:
        """Assess how persistent the pattern is over time."""
        duration = (pattern.time_range.end - pattern.time_range.start).total_seconds()

        if duration > 7 * 24 * 3600:  # More than a week
            return "high"
        elif duration > 24 * 3600:  # More than a day
            return "medium"
        else:
            return "low"

    def _assess_scope(self, pattern: Pattern, context_data: SecurityData) -> str:
        """Assess the scope of the pattern across detectors/systems."""
        detector_count = len(pattern.affected_detectors)

        if detector_count >= 5:
            return "wide"
        elif detector_count >= 2:
            return "moderate"
        else:
            return "narrow"

    def _assess_intensity(self, pattern: Pattern, context_data: SecurityData) -> str:
        """Assess the intensity/magnitude of the pattern."""
        # Use statistical significance and confidence as proxies for intensity
        combined_score = (pattern.statistical_significance + pattern.confidence) / 2

        if combined_score >= 0.8:
            return "high"
        elif combined_score >= 0.6:
            return "medium"
        else:
            return "low"

    def _assess_predictability(
        self, pattern: Pattern, context_data: SecurityData
    ) -> str:
        """Assess how predictable the pattern is."""
        if pattern.pattern_type == PatternType.FREQUENCY:
            return "high"  # Frequency patterns are typically predictable
        elif pattern.pattern_type == PatternType.TEMPORAL:
            return "medium"  # Temporal patterns have some predictability
        elif pattern.pattern_type == PatternType.CORRELATION:
            return "medium"  # Correlations can be somewhat predictable
        else:  # ANOMALY
            return "low"  # Anomalies are by definition unpredictable

    def _assess_impact_potential(
        self, pattern: Pattern, context_data: SecurityData
    ) -> str:
        """Assess the potential business impact of the pattern."""
        # Consider business relevance and affected detectors
        if pattern.business_relevance == BusinessRelevance.CRITICAL:
            return "critical"
        elif pattern.business_relevance == BusinessRelevance.HIGH:
            return "high"
        elif pattern.business_relevance == BusinessRelevance.MEDIUM:
            return "medium"
        else:
            return "low"

    def _assess_detection_difficulty(
        self, pattern: Pattern, context_data: SecurityData
    ) -> str:
        """Assess how difficult this pattern would be to detect manually."""
        # Complex patterns across multiple detectors are harder to detect
        complexity_score = len(pattern.affected_detectors) * (1 - pattern.confidence)

        if complexity_score > 2.0:
            return "high"
        elif complexity_score > 1.0:
            return "medium"
        else:
            return "low"

    def _calculate_consistency_score(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Calculate how consistent the pattern is."""
        # Use statistical significance as a proxy for consistency
        return pattern.statistical_significance

    def _calculate_variance_level(
        self, pattern: Pattern, context_data: SecurityData
    ) -> str:
        """Calculate the variance level of the pattern."""
        # Higher confidence typically means lower variance
        if pattern.confidence >= 0.8:
            return "low"
        elif pattern.confidence >= 0.6:
            return "medium"
        else:
            return "high"

    def _assess_trend_stability(
        self, pattern: Pattern, context_data: SecurityData
    ) -> str:
        """Assess if the trend within the pattern is stable."""
        if pattern.pattern_type == PatternType.TEMPORAL:
            # For temporal patterns, check if trend is consistent
            return "stable" if pattern.confidence > 0.7 else "unstable"
        else:
            return "not_applicable"

    def _detect_seasonal_component(
        self, pattern: Pattern, context_data: SecurityData
    ) -> bool:
        """Detect if pattern has seasonal/cyclical components."""
        # Simple heuristic: patterns lasting more than a week might have seasonal components
        duration = (pattern.time_range.end - pattern.time_range.start).total_seconds()
        return duration > 7 * 24 * 3600 and pattern.pattern_type in [
            PatternType.TEMPORAL,
            PatternType.FREQUENCY,
        ]

    def _has_regular_frequency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> bool:
        """Check if pattern has regular frequency characteristics."""
        # Look for frequency-related evidence in supporting data
        for evidence in pattern.supporting_evidence:
            if isinstance(evidence, dict) and "average_interval" in evidence:
                return True
        return False

    def _has_strong_correlations(
        self, pattern: Pattern, context_data: SecurityData
    ) -> bool:
        """Check if pattern shows strong correlations."""
        # Look for correlation-related evidence
        for evidence in pattern.supporting_evidence:
            if isinstance(evidence, dict) and "correlation_coefficient" in evidence:
                correlation = abs(evidence["correlation_coefficient"])
                return correlation > 0.7
        return False

    def _is_part_of_trend(self, pattern: Pattern, context_data: SecurityData) -> bool:
        """Check if anomaly is part of a larger trend."""
        # Look for trend-related evidence
        for evidence in pattern.supporting_evidence:
            if isinstance(evidence, dict) and "trend_direction" in evidence:
                return True
        return False

    def _generate_enhanced_description(
        self, pattern: Pattern, characteristics: Dict[str, any]
    ) -> str:
        """Generate enhanced description with classification details."""
        base_description = pattern.description

        # Add classification details
        persistence = characteristics.get("persistence", "unknown")
        scope = characteristics.get("scope", "unknown")
        intensity = characteristics.get("intensity", "unknown")

        enhanced_description = (
            f"{base_description} "
            f"[Persistence: {persistence}, Scope: {scope}, Intensity: {intensity}]"
        )

        return enhanced_description

    def _calculate_classification_confidence(
        self,
        pattern: Pattern,
        characteristics: Dict[str, any],
        stability: Dict[str, any],
    ) -> float:
        """Calculate confidence in the classification."""
        # Combine pattern confidence with classification certainty
        base_confidence = pattern.confidence

        # Adjust based on classification characteristics
        consistency_score = stability.get("consistency_score", 0.5)

        # Higher consistency increases classification confidence
        classification_confidence = (base_confidence + consistency_score) / 2

        return min(1.0, classification_confidence)

    def _determine_pattern_category(
        self, pattern: Pattern, characteristics: Dict[str, any]
    ) -> str:
        """Determine high-level pattern category."""
        persistence = characteristics.get("persistence", "low")
        scope = characteristics.get("scope", "narrow")
        intensity = characteristics.get("intensity", "low")

        if persistence == "high" and scope == "wide" and intensity == "high":
            return "critical_systemic"
        elif persistence == "high" and intensity == "high":
            return "persistent_threat"
        elif scope == "wide" and intensity == "high":
            return "widespread_issue"
        elif intensity == "high":
            return "high_impact"
        elif persistence == "high":
            return "chronic_issue"
        elif scope == "wide":
            return "systemic_pattern"
        else:
            return "localized_pattern"

    def _identify_risk_indicators(
        self, pattern: Pattern, characteristics: Dict[str, any]
    ) -> List[str]:
        """Identify risk indicators based on pattern characteristics."""
        risk_indicators = []

        if characteristics.get("persistence") == "high":
            risk_indicators.append("persistent_threat")

        if characteristics.get("scope") == "wide":
            risk_indicators.append("widespread_impact")

        if characteristics.get("intensity") == "high":
            risk_indicators.append("high_severity")

        if characteristics.get("predictability") == "low":
            risk_indicators.append("unpredictable_behavior")

        if characteristics.get("impact_potential") in ["critical", "high"]:
            risk_indicators.append("high_business_risk")

        if characteristics.get("detection_difficulty") == "high":
            risk_indicators.append("difficult_to_detect")

        return risk_indicators

    def _perform_cross_pattern_analysis(
        self, patterns: List[Pattern], context_data: SecurityData
    ):
        """Perform analysis across multiple patterns to identify relationships."""
        # This could identify pattern clusters, hierarchies, or dependencies
        # For now, we'll add metadata about pattern relationships

        for i, pattern in enumerate(patterns):
            related_patterns = []

            for j, other_pattern in enumerate(patterns):
                if i != j and self._are_patterns_related(pattern, other_pattern):
                    related_patterns.append(other_pattern.pattern_id)

            if related_patterns:
                # Add cross-pattern metadata to supporting evidence
                pattern.supporting_evidence.append(
                    {
                        "cross_pattern_analysis": {
                            "related_patterns": related_patterns,
                            "relationship_strength": len(related_patterns)
                            / len(patterns),
                            "pattern_cluster": self._determine_pattern_cluster(
                                pattern, patterns
                            ),
                        }
                    }
                )

    def _are_patterns_related(self, pattern1: Pattern, pattern2: Pattern) -> bool:
        """Determine if two patterns are related."""
        # Check for overlapping detectors
        detector_overlap = set(pattern1.affected_detectors) & set(
            pattern2.affected_detectors
        )

        # Check for temporal overlap
        temporal_overlap = (
            pattern1.time_range.start <= pattern2.time_range.end
            and pattern2.time_range.start <= pattern1.time_range.end
        )

        return len(detector_overlap) > 0 and temporal_overlap

    def _determine_pattern_cluster(
        self, pattern: Pattern, all_patterns: List[Pattern]
    ) -> str:
        """Determine which cluster this pattern belongs to."""
        # Simple clustering based on pattern type and affected detectors
        same_type_patterns = [
            p for p in all_patterns if p.pattern_type == pattern.pattern_type
        ]

        if len(same_type_patterns) > len(all_patterns) * 0.5:
            return f"dominant_{pattern.pattern_type.value}"
        else:
            return f"minority_{pattern.pattern_type.value}"

    def _load_classification_rules(self) -> Dict[str, any]:
        """Load classification rules from configuration."""
        # Default classification rules
        return {
            "temporal_thresholds": {
                "short_term": 3600,  # 1 hour
                "medium_term": 86400,  # 1 day
                "long_term": 604800,  # 1 week
            },
            "scope_thresholds": {"narrow": 1, "moderate": 2, "wide": 5},
            "intensity_thresholds": {"low": 0.5, "medium": 0.7, "high": 0.8},
        }
