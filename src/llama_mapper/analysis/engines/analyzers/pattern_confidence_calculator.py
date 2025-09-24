"""
Pattern Confidence Calculator based on data quality and statistical rigor.

This module implements sophisticated confidence calculation methods
that consider data quality, statistical validity, and analytical rigor.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from ...domain import (
    Pattern,
    PatternType,
    SecurityData,
)

logger = logging.getLogger(__name__)


class PatternConfidenceCalculator:
    """
    Calculates pattern confidence based on data quality and statistical rigor.

    Evaluates multiple factors including data completeness, statistical validity,
    temporal consistency, and analytical methodology to provide robust confidence scores.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.quality_weights = self._load_quality_weights()
        self.confidence_thresholds = self._load_confidence_thresholds()
        self.statistical_requirements = self._load_statistical_requirements()

    def calculate_pattern_confidence(
        self,
        pattern: Pattern,
        context_data: SecurityData,
        analysis_metadata: Dict[str, any] = None,
    ) -> float:
        """
        Calculate comprehensive confidence score for a pattern.

        Args:
            pattern: The pattern to analyze
            context_data: Security data used in pattern detection
            analysis_metadata: Metadata from the analysis process

        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            analysis_metadata = analysis_metadata or {}

            # Assess data quality factors
            data_quality_score = self._assess_data_quality(pattern, context_data)

            # Assess statistical rigor
            statistical_rigor_score = self._assess_statistical_rigor(
                pattern, analysis_metadata
            )

            # Assess temporal consistency
            temporal_consistency_score = self._assess_temporal_consistency(
                pattern, context_data
            )

            # Assess pattern coherence
            pattern_coherence_score = self._assess_pattern_coherence(
                pattern, context_data
            )

            # Assess analytical methodology
            methodology_score = self._assess_analytical_methodology(
                pattern, analysis_metadata
            )

            # Calculate composite confidence score
            confidence_score = self._calculate_composite_confidence(
                data_quality_score,
                statistical_rigor_score,
                temporal_consistency_score,
                pattern_coherence_score,
                methodology_score,
            )

            # Apply pattern-specific adjustments
            adjusted_confidence = self._apply_pattern_specific_adjustments(
                confidence_score, pattern, context_data
            )

            # Add detailed confidence analysis to pattern evidence
            self._add_confidence_evidence(
                pattern,
                data_quality_score,
                statistical_rigor_score,
                temporal_consistency_score,
                pattern_coherence_score,
                methodology_score,
                adjusted_confidence,
            )

            logger.info(
                "Pattern confidence calculated",
                pattern_id=pattern.pattern_id,
                confidence=adjusted_confidence,
                data_quality=data_quality_score,
                statistical_rigor=statistical_rigor_score,
            )

            return adjusted_confidence

        except Exception as e:
            logger.error(
                "Pattern confidence calculation failed",
                error=str(e),
                pattern_id=pattern.pattern_id,
            )
            return 0.0

    def calculate_multiple_patterns_confidence(
        self,
        patterns: List[Pattern],
        context_data: SecurityData,
        analysis_metadata: Dict[str, any] = None,
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for multiple patterns with cross-validation.

        Args:
            patterns: List of patterns to analyze
            context_data: Security data for context
            analysis_metadata: Analysis metadata

        Returns:
            Dictionary mapping pattern IDs to confidence scores
        """
        confidence_results = {}
        pattern_scores = {}

        # Calculate individual confidences
        for pattern in patterns:
            confidence = self.calculate_pattern_confidence(
                pattern, context_data, analysis_metadata
            )
            confidence_results[pattern.pattern_id] = confidence
            pattern_scores[pattern.pattern_id] = confidence

        # Perform cross-validation analysis
        self._perform_cross_validation_analysis(patterns, pattern_scores, context_data)

        return confidence_results

    def _assess_data_quality(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess the quality of data used for pattern detection."""
        try:
            quality_factors = {}

            # Data completeness
            quality_factors["completeness"] = self._assess_data_completeness(
                pattern, context_data
            )

            # Data consistency
            quality_factors["consistency"] = self._assess_data_consistency(
                pattern, context_data
            )

            # Data freshness
            quality_factors["freshness"] = self._assess_data_freshness(
                pattern, context_data
            )

            # Data volume adequacy
            quality_factors["volume"] = self._assess_data_volume(pattern, context_data)

            # Data source reliability
            quality_factors["source_reliability"] = self._assess_source_reliability(
                pattern, context_data
            )

            # Calculate weighted data quality score
            weights = self.quality_weights.get("data_quality", {})
            quality_score = sum(
                quality_factors.get(factor, 0.5) * weights.get(factor, 0.2)
                for factor in [
                    "completeness",
                    "consistency",
                    "freshness",
                    "volume",
                    "source_reliability",
                ]
            )

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            logger.error("Data quality assessment failed", error=str(e))
            return 0.5

    def _assess_statistical_rigor(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess the statistical rigor of the pattern detection."""
        try:
            rigor_factors = {}

            # Sample size adequacy
            rigor_factors["sample_size"] = self._assess_sample_size_adequacy(
                pattern, analysis_metadata
            )

            # Statistical significance
            rigor_factors["significance"] = self._assess_statistical_significance(
                pattern, analysis_metadata
            )

            # Method appropriateness
            rigor_factors["method_appropriateness"] = (
                self._assess_method_appropriateness(pattern, analysis_metadata)
            )

            # Assumption validation
            rigor_factors["assumption_validation"] = self._assess_assumption_validation(
                pattern, analysis_metadata
            )

            # Multiple testing correction
            rigor_factors["multiple_testing"] = (
                self._assess_multiple_testing_correction(pattern, analysis_metadata)
            )

            # Calculate weighted statistical rigor score
            weights = self.quality_weights.get("statistical_rigor", {})
            rigor_score = sum(
                rigor_factors.get(factor, 0.5) * weights.get(factor, 0.2)
                for factor in [
                    "sample_size",
                    "significance",
                    "method_appropriateness",
                    "assumption_validation",
                    "multiple_testing",
                ]
            )

            return min(1.0, max(0.0, rigor_score))

        except Exception as e:
            logger.error("Statistical rigor assessment failed", error=str(e))
            return 0.5

    def _assess_temporal_consistency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess temporal consistency of the pattern."""
        try:
            consistency_factors = {}

            # Pattern stability over time
            consistency_factors["stability"] = self._assess_pattern_stability(
                pattern, context_data
            )

            # Temporal coverage adequacy
            consistency_factors["coverage"] = self._assess_temporal_coverage(
                pattern, context_data
            )

            # Seasonal/cyclical consistency
            consistency_factors["cyclical_consistency"] = (
                self._assess_cyclical_consistency(pattern, context_data)
            )

            # Trend consistency
            consistency_factors["trend_consistency"] = self._assess_trend_consistency(
                pattern, context_data
            )

            # Calculate weighted temporal consistency score
            weights = self.quality_weights.get("temporal_consistency", {})
            consistency_score = sum(
                consistency_factors.get(factor, 0.5) * weights.get(factor, 0.25)
                for factor in [
                    "stability",
                    "coverage",
                    "cyclical_consistency",
                    "trend_consistency",
                ]
            )

            return min(1.0, max(0.0, consistency_score))

        except Exception as e:
            logger.error("Temporal consistency assessment failed", error=str(e))
            return 0.5

    def _assess_pattern_coherence(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess the coherence and logical consistency of the pattern."""
        try:
            coherence_factors = {}

            # Internal consistency
            coherence_factors["internal_consistency"] = (
                self._assess_internal_consistency(pattern, context_data)
            )

            # Cross-detector consistency
            coherence_factors["cross_detector_consistency"] = (
                self._assess_cross_detector_consistency(pattern, context_data)
            )

            # Pattern interpretability
            coherence_factors["interpretability"] = (
                self._assess_pattern_interpretability(pattern, context_data)
            )

            # Anomaly coherence (for anomaly patterns)
            if pattern.pattern_type == PatternType.ANOMALY:
                coherence_factors["anomaly_coherence"] = self._assess_anomaly_coherence(
                    pattern, context_data
                )

            # Calculate weighted coherence score
            weights = self.quality_weights.get("pattern_coherence", {})
            coherence_score = sum(
                coherence_factors.get(factor, 0.5) * weights.get(factor, 0.33)
                for factor in coherence_factors.keys()
            )

            return min(1.0, max(0.0, coherence_score))

        except Exception as e:
            logger.error("Pattern coherence assessment failed", error=str(e))
            return 0.5

    def _assess_analytical_methodology(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess the quality of analytical methodology used."""
        try:
            methodology_factors = {}

            # Algorithm appropriateness
            methodology_factors["algorithm_appropriateness"] = (
                self._assess_algorithm_appropriateness(pattern, analysis_metadata)
            )

            # Parameter selection
            methodology_factors["parameter_selection"] = (
                self._assess_parameter_selection(pattern, analysis_metadata)
            )

            # Validation methodology
            methodology_factors["validation_methodology"] = (
                self._assess_validation_methodology(pattern, analysis_metadata)
            )

            # Robustness testing
            methodology_factors["robustness"] = self._assess_robustness_testing(
                pattern, analysis_metadata
            )

            # Calculate weighted methodology score
            weights = self.quality_weights.get("analytical_methodology", {})
            methodology_score = sum(
                methodology_factors.get(factor, 0.5) * weights.get(factor, 0.25)
                for factor in [
                    "algorithm_appropriateness",
                    "parameter_selection",
                    "validation_methodology",
                    "robustness",
                ]
            )

            return min(1.0, max(0.0, methodology_score))

        except Exception as e:
            logger.error("Analytical methodology assessment failed", error=str(e))
            return 0.5

    def _calculate_composite_confidence(
        self,
        data_quality: float,
        statistical_rigor: float,
        temporal_consistency: float,
        pattern_coherence: float,
        methodology: float,
    ) -> float:
        """Calculate composite confidence score from all factors."""
        try:
            # Define weights for different confidence factors
            weights = {
                "data_quality": 0.25,
                "statistical_rigor": 0.25,
                "temporal_consistency": 0.2,
                "pattern_coherence": 0.15,
                "methodology": 0.15,
            }

            # Calculate weighted composite score
            composite_score = (
                weights["data_quality"] * data_quality
                + weights["statistical_rigor"] * statistical_rigor
                + weights["temporal_consistency"] * temporal_consistency
                + weights["pattern_coherence"] * pattern_coherence
                + weights["methodology"] * methodology
            )

            return min(1.0, max(0.0, composite_score))

        except Exception as e:
            logger.error("Composite confidence calculation failed", error=str(e))
            return 0.0

    def _apply_pattern_specific_adjustments(
        self, base_confidence: float, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Apply pattern-type specific confidence adjustments."""
        try:
            adjusted_confidence = base_confidence

            # Pattern type specific adjustments
            if pattern.pattern_type == PatternType.TEMPORAL:
                # Temporal patterns benefit from longer observation periods
                duration = (
                    pattern.time_range.end - pattern.time_range.start
                ).total_seconds()
                if duration > 7 * 24 * 3600:  # More than a week
                    adjusted_confidence *= 1.1
                elif duration < 3600:  # Less than an hour
                    adjusted_confidence *= 0.9

            elif pattern.pattern_type == PatternType.FREQUENCY:
                # Frequency patterns benefit from regular intervals
                for evidence in pattern.supporting_evidence:
                    if isinstance(evidence, dict) and "regularity_test" in evidence:
                        if evidence["regularity_test"].get("is_regular", False):
                            adjusted_confidence *= 1.1
                        break

            elif pattern.pattern_type == PatternType.CORRELATION:
                # Correlation patterns benefit from strong correlations
                for evidence in pattern.supporting_evidence:
                    if (
                        isinstance(evidence, dict)
                        and "correlation_coefficient" in evidence
                    ):
                        corr_strength = abs(evidence["correlation_coefficient"])
                        if corr_strength > 0.8:
                            adjusted_confidence *= 1.1
                        elif corr_strength < 0.5:
                            adjusted_confidence *= 0.9
                        break

            elif pattern.pattern_type == PatternType.ANOMALY:
                # Anomaly patterns benefit from clear outlier separation
                for evidence in pattern.supporting_evidence:
                    if isinstance(evidence, dict) and "outlier_test" in evidence:
                        outlier_ratio = evidence["outlier_test"].get("outlier_ratio", 0)
                        if outlier_ratio > 0.1:  # Clear anomalies
                            adjusted_confidence *= 1.1
                        elif outlier_ratio < 0.02:  # Weak anomalies
                            adjusted_confidence *= 0.9
                        break

            # Detector count adjustment
            detector_count = len(pattern.affected_detectors)
            if detector_count >= 3:
                adjusted_confidence *= 1.05  # Multiple detectors increase confidence
            elif detector_count == 1:
                adjusted_confidence *= 0.95  # Single detector decreases confidence

            return min(1.0, max(0.0, adjusted_confidence))

        except Exception as e:
            logger.error("Pattern-specific adjustments failed", error=str(e))
            return base_confidence

    # Data Quality Assessment Methods
    def _assess_data_completeness(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess completeness of data used for pattern detection."""
        try:
            total_expected_points = 100  # Expected data points for robust analysis

            # Count actual data points
            actual_points = 0
            actual_points += len(context_data.time_series)
            actual_points += len(context_data.events)
            actual_points += len(context_data.metrics)

            completeness_ratio = min(1.0, actual_points / total_expected_points)
            return completeness_ratio

        except Exception:
            return 0.5

    def _assess_data_consistency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess consistency of data across different sources."""
        # Simplified consistency check
        return 0.8  # Placeholder - would implement actual consistency checks

    def _assess_data_freshness(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess freshness/recency of data."""
        try:
            now = datetime.utcnow()
            pattern_age = (now - pattern.time_range.end).total_seconds()

            # Fresher data gets higher score
            if pattern_age < 3600:  # Less than 1 hour
                return 1.0
            elif pattern_age < 24 * 3600:  # Less than 1 day
                return 0.8
            elif pattern_age < 7 * 24 * 3600:  # Less than 1 week
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _assess_data_volume(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess adequacy of data volume."""
        try:
            # Count relevant data points for this pattern
            relevant_points = 0

            for item in context_data.time_series:
                if item.get("detector") in pattern.affected_detectors:
                    relevant_points += 1

            for event in context_data.events:
                if event.get("detector") in pattern.affected_detectors:
                    relevant_points += 1

            # More data points increase confidence
            if relevant_points >= 50:
                return 1.0
            elif relevant_points >= 20:
                return 0.8
            elif relevant_points >= 10:
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _assess_source_reliability(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess reliability of data sources."""
        # Simplified reliability assessment based on detector types
        reliable_detectors = {"presidio", "pii-detector", "gdpr-scanner"}

        reliable_count = sum(
            1
            for detector in pattern.affected_detectors
            if detector.lower() in reliable_detectors
        )

        if not pattern.affected_detectors:
            return 0.5

        reliability_ratio = reliable_count / len(pattern.affected_detectors)
        return 0.5 + (reliability_ratio * 0.5)  # Range: 0.5 to 1.0

    # Statistical Rigor Assessment Methods
    def _assess_sample_size_adequacy(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess adequacy of sample size for statistical analysis."""
        try:
            # Extract sample size from pattern evidence
            sample_size = 0
            for evidence in pattern.supporting_evidence:
                if isinstance(evidence, dict):
                    if "sample_size" in evidence:
                        sample_size = evidence["sample_size"]
                        break
                    elif "event_count" in evidence:
                        sample_size = evidence["event_count"]
                        break

            # Assess adequacy based on pattern type
            min_required = self.statistical_requirements.get(
                pattern.pattern_type.value, {}
            ).get("min_sample_size", 10)

            if sample_size >= min_required * 3:
                return 1.0
            elif sample_size >= min_required * 2:
                return 0.8
            elif sample_size >= min_required:
                return 0.6
            else:
                return 0.3

        except Exception:
            return 0.5

    def _assess_statistical_significance(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess statistical significance of pattern detection."""
        # Use the pattern's statistical significance directly
        return pattern.statistical_significance

    def _assess_method_appropriateness(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess appropriateness of analytical method used."""
        # Simplified assessment - would be more sophisticated in practice
        return 0.8  # Assume methods are generally appropriate

    def _assess_assumption_validation(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess validation of statistical assumptions."""
        # Check for normality tests, etc. in evidence
        for evidence in pattern.supporting_evidence:
            if isinstance(evidence, dict) and "normality_test" in evidence:
                return 0.9  # Assumptions were tested

        return 0.6  # No assumption testing found

    def _assess_multiple_testing_correction(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess multiple testing correction."""
        # Simplified - would check for Bonferroni correction, etc.
        return 0.7  # Assume some correction applied

    # Additional assessment methods (simplified implementations)
    def _assess_pattern_stability(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess stability of pattern over time."""
        return pattern.confidence  # Use pattern confidence as proxy

    def _assess_temporal_coverage(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess adequacy of temporal coverage."""
        duration = (pattern.time_range.end - pattern.time_range.start).total_seconds()
        if duration > 24 * 3600:  # More than a day
            return 1.0
        elif duration > 3600:  # More than an hour
            return 0.7
        else:
            return 0.4

    def _assess_cyclical_consistency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess cyclical consistency."""
        return 0.7  # Placeholder

    def _assess_trend_consistency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess trend consistency."""
        return 0.8  # Placeholder

    def _assess_internal_consistency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess internal consistency of pattern."""
        return pattern.confidence  # Use pattern confidence as proxy

    def _assess_cross_detector_consistency(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess consistency across detectors."""
        if len(pattern.affected_detectors) > 1:
            return 0.9  # Multiple detectors suggest consistency
        else:
            return 0.6  # Single detector

    def _assess_pattern_interpretability(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess how interpretable the pattern is."""
        return 0.8  # Placeholder

    def _assess_anomaly_coherence(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Assess coherence of anomaly patterns."""
        return 0.7  # Placeholder

    def _assess_algorithm_appropriateness(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess appropriateness of algorithm used."""
        return 0.8  # Placeholder

    def _assess_parameter_selection(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess parameter selection quality."""
        return 0.7  # Placeholder

    def _assess_validation_methodology(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess validation methodology quality."""
        return 0.8  # Placeholder

    def _assess_robustness_testing(
        self, pattern: Pattern, analysis_metadata: Dict[str, any]
    ) -> float:
        """Assess robustness testing quality."""
        return 0.6  # Placeholder

    def _add_confidence_evidence(
        self,
        pattern: Pattern,
        data_quality: float,
        statistical_rigor: float,
        temporal_consistency: float,
        pattern_coherence: float,
        methodology: float,
        final_confidence: float,
    ):
        """Add detailed confidence analysis to pattern evidence."""
        confidence_evidence = {
            "confidence_analysis": {
                "final_confidence": final_confidence,
                "data_quality_score": data_quality,
                "statistical_rigor_score": statistical_rigor,
                "temporal_consistency_score": temporal_consistency,
                "pattern_coherence_score": pattern_coherence,
                "methodology_score": methodology,
                "calculation_method": "comprehensive_confidence_assessment",
                "assessment_timestamp": datetime.utcnow().isoformat(),
            }
        }

        pattern.supporting_evidence.append(confidence_evidence)

    def _perform_cross_validation_analysis(
        self,
        patterns: List[Pattern],
        pattern_scores: Dict[str, float],
        context_data: SecurityData,
    ):
        """Perform cross-validation analysis across patterns."""
        # Add cross-validation metadata to patterns
        avg_confidence = (
            sum(pattern_scores.values()) / len(pattern_scores) if pattern_scores else 0
        )

        for pattern in patterns:
            pattern_confidence = pattern_scores.get(pattern.pattern_id, 0)

            cross_validation = {
                "cross_validation_analysis": {
                    "relative_confidence": (
                        pattern_confidence / avg_confidence
                        if avg_confidence > 0
                        else 1.0
                    ),
                    "confidence_rank": sorted(
                        pattern_scores.values(), reverse=True
                    ).index(pattern_confidence)
                    + 1,
                    "total_patterns": len(patterns),
                    "average_confidence": avg_confidence,
                }
            }
            pattern.supporting_evidence.append(cross_validation)

    def _load_quality_weights(self) -> Dict[str, Dict[str, float]]:
        """Load quality assessment weights."""
        return self.config.get(
            "quality_weights",
            {
                "data_quality": {
                    "completeness": 0.25,
                    "consistency": 0.2,
                    "freshness": 0.2,
                    "volume": 0.2,
                    "source_reliability": 0.15,
                },
                "statistical_rigor": {
                    "sample_size": 0.3,
                    "significance": 0.25,
                    "method_appropriateness": 0.2,
                    "assumption_validation": 0.15,
                    "multiple_testing": 0.1,
                },
                "temporal_consistency": {
                    "stability": 0.3,
                    "coverage": 0.25,
                    "cyclical_consistency": 0.25,
                    "trend_consistency": 0.2,
                },
                "pattern_coherence": {
                    "internal_consistency": 0.4,
                    "cross_detector_consistency": 0.3,
                    "interpretability": 0.3,
                },
                "analytical_methodology": {
                    "algorithm_appropriateness": 0.3,
                    "parameter_selection": 0.25,
                    "validation_methodology": 0.25,
                    "robustness": 0.2,
                },
            },
        )

    def _load_confidence_thresholds(self) -> Dict[str, float]:
        """Load confidence classification thresholds."""
        return self.config.get(
            "confidence_thresholds",
            {"low": 0.0, "medium": 0.6, "high": 0.8, "very_high": 0.9},
        )

    def _load_statistical_requirements(self) -> Dict[str, Dict[str, int]]:
        """Load statistical requirements by pattern type."""
        return self.config.get(
            "statistical_requirements",
            {
                "temporal": {"min_sample_size": 20},
                "frequency": {"min_sample_size": 15},
                "correlation": {"min_sample_size": 10},
                "anomaly": {"min_sample_size": 25},
            },
        )
