"""
Consolidated Pattern Classifier

This classifier consolidates pattern classification capabilities from the original
analysis module, providing sophisticated pattern categorization and assessment.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PatternClassifier:
    """
    Consolidated pattern classifier for categorizing and assessing detected patterns.

    Classifies:
    - Pattern types and categories
    - Pattern strength and significance
    - Business relevance and impact
    - Risk implications
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.classification_rules = self._load_classification_rules()
        self.strength_thresholds = self.config.get(
            "strength_thresholds", {"weak": 0.3, "moderate": 0.6, "strong": 0.8}
        )

        logger.debug("Pattern Classifier initialized with config: %s", self.config)

    def classify_pattern(
        self, pattern: Dict[str, Any], context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a pattern with enhanced categorization and assessment.

        Args:
            pattern: Pattern to classify
            context_data: Additional context for classification

        Returns:
            Enhanced pattern with classification information
        """
        try:
            # Create enhanced pattern copy
            classified_pattern = pattern.copy()

            # Classify pattern type and category
            type_classification = self._classify_pattern_type(pattern)
            classified_pattern.update(type_classification)

            # Classify pattern strength
            strength_classification = self._classify_pattern_strength(pattern)
            classified_pattern.update(strength_classification)

            # Classify business relevance
            relevance_classification = self._classify_business_relevance(
                pattern, context_data
            )
            classified_pattern.update(relevance_classification)

            # Classify risk implications
            risk_classification = self._classify_risk_implications(
                pattern, context_data
            )
            classified_pattern.update(risk_classification)

            # Add classification metadata
            classified_pattern["classification_metadata"] = {
                "classified_at": datetime.now(timezone.utc).isoformat(),
                "classifier_version": "1.0.0",
                "classification_confidence": self._calculate_classification_confidence(
                    type_classification,
                    strength_classification,
                    relevance_classification,
                ),
            }

            logger.debug(
                "Pattern classified",
                pattern_id=pattern.get("pattern_id", "unknown"),
                pattern_type=type_classification.get("pattern_type"),
                strength=strength_classification.get("strength"),
            )

            return classified_pattern

        except Exception as e:
            logger.error("Error classifying pattern: %s", e)
            # Return original pattern with error information
            error_pattern = pattern.copy()
            error_pattern["classification_error"] = str(e)
            return error_pattern

    def _classify_pattern_type(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the type and category of the pattern."""
        try:
            pattern_type = pattern.get("pattern_type", "unknown")

            # Enhanced type classification
            type_info = self.classification_rules.get("types", {}).get(pattern_type, {})

            classification = {
                "pattern_type": pattern_type,
                "pattern_category": type_info.get("category", "general"),
                "pattern_subcategory": type_info.get("subcategory", "unspecified"),
                "detection_domain": type_info.get("domain", "security"),
                "analysis_complexity": type_info.get("complexity", "medium"),
            }

            # Add pattern characteristics based on type
            if pattern_type == "temporal":
                classification.update(self._classify_temporal_characteristics(pattern))
            elif pattern_type == "frequency":
                classification.update(self._classify_frequency_characteristics(pattern))
            elif pattern_type == "correlation":
                classification.update(
                    self._classify_correlation_characteristics(pattern)
                )
            elif pattern_type == "anomaly":
                classification.update(self._classify_anomaly_characteristics(pattern))

            return classification

        except Exception as e:
            logger.error("Error classifying pattern type: %s", e)
            return {
                "pattern_type": "unknown",
                "pattern_category": "general",
                "classification_error": str(e),
            }

    def _classify_pattern_strength(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the strength and significance of the pattern."""
        try:
            # Get base strength indicators
            confidence = pattern.get("confidence", 0.5)
            statistical_significance = pattern.get("statistical_significance", 0.5)

            # Calculate composite strength score
            strength_score = (confidence + statistical_significance) / 2

            # Classify strength level
            if strength_score >= self.strength_thresholds["strong"]:
                strength_level = "strong"
            elif strength_score >= self.strength_thresholds["moderate"]:
                strength_level = "moderate"
            else:
                strength_level = "weak"

            # Additional strength indicators
            strength_indicators = []
            if confidence > 0.8:
                strength_indicators.append("high_confidence")
            if statistical_significance > 0.8:
                strength_indicators.append("statistically_significant")
            if pattern.get("sample_size", 0) > 100:
                strength_indicators.append("large_sample")

            return {
                "strength": strength_level,
                "strength_score": strength_score,
                "strength_indicators": strength_indicators,
                "confidence_level": self._classify_confidence_level(confidence),
                "significance_level": self._classify_significance_level(
                    statistical_significance
                ),
            }

        except Exception as e:
            logger.error("Error classifying pattern strength: %s", e)
            return {
                "strength": "unknown",
                "strength_score": 0.5,
                "classification_error": str(e),
            }

    def _classify_business_relevance(
        self, pattern: Dict[str, Any], context_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify the business relevance and impact of the pattern."""
        try:
            # Base relevance from pattern type
            pattern_type = pattern.get("pattern_type", "unknown")
            base_relevance = self.classification_rules.get(
                "business_relevance", {}
            ).get(pattern_type, "medium")

            # Context-based adjustments
            relevance_adjustments = []
            adjusted_relevance = base_relevance

            if context_data:
                # Check for business impact indicators
                if context_data.get("affects_critical_systems", False):
                    relevance_adjustments.append("critical_systems_impact")
                    adjusted_relevance = "high"

                if context_data.get("compliance_implications", False):
                    relevance_adjustments.append("compliance_implications")
                    if adjusted_relevance != "high":
                        adjusted_relevance = "medium"

                if context_data.get("customer_impact", False):
                    relevance_adjustments.append("customer_impact")
                    adjusted_relevance = "high"

            # Pattern-specific relevance factors
            pattern_relevance_factors = self._identify_pattern_relevance_factors(
                pattern
            )

            return {
                "business_relevance": adjusted_relevance,
                "base_relevance": base_relevance,
                "relevance_adjustments": relevance_adjustments,
                "relevance_factors": pattern_relevance_factors,
                "impact_assessment": self._assess_business_impact(
                    pattern, context_data
                ),
            }

        except Exception as e:
            logger.error("Error classifying business relevance: %s", e)
            return {"business_relevance": "medium", "classification_error": str(e)}

    def _classify_risk_implications(
        self, pattern: Dict[str, Any], context_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify the risk implications of the pattern."""
        try:
            pattern_type = pattern.get("pattern_type", "unknown")
            strength = pattern.get("strength", "medium")

            # Base risk level from pattern type and strength
            base_risk = self._calculate_base_risk_level(pattern_type, strength)

            # Risk escalation factors
            risk_escalators = []
            escalated_risk = base_risk

            # Check for risk escalation conditions
            if pattern.get("confidence", 0) > 0.9:
                risk_escalators.append("high_confidence")

            if pattern_type == "anomaly" and pattern.get("severity") == "critical":
                risk_escalators.append("critical_anomaly")
                escalated_risk = "high"

            if context_data and context_data.get("security_sensitive", False):
                risk_escalators.append("security_sensitive_context")
                if escalated_risk == "low":
                    escalated_risk = "medium"

            # Risk mitigation factors
            risk_mitigators = []
            if pattern.get("false_positive_likelihood", 0) > 0.7:
                risk_mitigators.append("high_false_positive_likelihood")

            return {
                "risk_level": escalated_risk,
                "base_risk_level": base_risk,
                "risk_escalators": risk_escalators,
                "risk_mitigators": risk_mitigators,
                "risk_score": self._calculate_risk_score(escalated_risk),
                "risk_categories": self._identify_risk_categories(
                    pattern, context_data
                ),
            }

        except Exception as e:
            logger.error("Error classifying risk implications: %s", e)
            return {"risk_level": "medium", "classification_error": str(e)}

    def _classify_temporal_characteristics(
        self, pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify characteristics specific to temporal patterns."""
        characteristics = {}

        if "trend_direction" in pattern:
            characteristics["temporal_trend"] = pattern["trend_direction"]

        if "seasonality" in pattern:
            characteristics["temporal_seasonality"] = pattern["seasonality"]

        if "cycle_length" in pattern:
            characteristics["temporal_cycle"] = pattern["cycle_length"]

        return {"temporal_characteristics": characteristics}

    def _classify_frequency_characteristics(
        self, pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify characteristics specific to frequency patterns."""
        characteristics = {}

        if "frequency_hz" in pattern:
            characteristics["frequency_rate"] = pattern["frequency_hz"]

        if "regularity_score" in pattern:
            characteristics["frequency_regularity"] = pattern["regularity_score"]

        if "interval_variance" in pattern:
            characteristics["frequency_stability"] = 1.0 - pattern["interval_variance"]

        return {"frequency_characteristics": characteristics}

    def _classify_correlation_characteristics(
        self, pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify characteristics specific to correlation patterns."""
        characteristics = {}

        if "correlation_coefficient" in pattern:
            coeff = pattern["correlation_coefficient"]
            characteristics["correlation_strength"] = abs(coeff)
            characteristics["correlation_direction"] = (
                "positive" if coeff > 0 else "negative"
            )

        if "p_value" in pattern:
            characteristics["statistical_significance"] = pattern["p_value"] < 0.05

        return {"correlation_characteristics": characteristics}

    def _classify_anomaly_characteristics(
        self, pattern: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify characteristics specific to anomaly patterns."""
        characteristics = {}

        if "anomaly_percentage" in pattern:
            characteristics["anomaly_prevalence"] = pattern["anomaly_percentage"]

        if "detection_methods" in pattern:
            characteristics["detection_consensus"] = len(pattern["detection_methods"])

        if "severity" in pattern:
            characteristics["anomaly_severity"] = pattern["severity"]

        return {"anomaly_characteristics": characteristics}

    def _classify_confidence_level(self, confidence: float) -> str:
        """Classify confidence level."""
        if confidence >= 0.9:
            return "very_high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        elif confidence >= 0.3:
            return "low"
        else:
            return "very_low"

    def _classify_significance_level(self, significance: float) -> str:
        """Classify statistical significance level."""
        if significance >= 0.95:
            return "highly_significant"
        elif significance >= 0.8:
            return "significant"
        elif significance >= 0.5:
            return "moderately_significant"
        else:
            return "not_significant"

    def _identify_pattern_relevance_factors(self, pattern: Dict[str, Any]) -> List[str]:
        """Identify factors that affect pattern business relevance."""
        factors = []

        if pattern.get("confidence", 0) > 0.8:
            factors.append("high_confidence_detection")

        if pattern.get("pattern_type") in ["security", "compliance", "anomaly"]:
            factors.append("security_related")

        if pattern.get("affects_multiple_detectors", False):
            factors.append("cross_detector_impact")

        return factors

    def _assess_business_impact(
        self, pattern: Dict[str, Any], context_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess the business impact of the pattern."""
        impact = {
            "operational_impact": "low",
            "financial_impact": "low",
            "compliance_impact": "low",
            "reputation_impact": "low",
        }

        # Pattern-based impact assessment
        if (
            pattern.get("pattern_type") == "anomaly"
            and pattern.get("severity") == "critical"
        ):
            impact["operational_impact"] = "high"

        if pattern.get("pattern_type") == "compliance":
            impact["compliance_impact"] = "high"

        # Context-based impact assessment
        if context_data:
            if context_data.get("affects_revenue", False):
                impact["financial_impact"] = "high"

            if context_data.get("public_facing", False):
                impact["reputation_impact"] = "medium"

        return impact

    def _calculate_base_risk_level(self, pattern_type: str, strength: str) -> str:
        """Calculate base risk level from pattern type and strength."""
        risk_matrix = {
            ("anomaly", "strong"): "high",
            ("anomaly", "moderate"): "medium",
            ("anomaly", "weak"): "low",
            ("security", "strong"): "high",
            ("security", "moderate"): "high",
            ("security", "weak"): "medium",
            ("compliance", "strong"): "high",
            ("compliance", "moderate"): "medium",
            ("compliance", "weak"): "low",
        }

        return risk_matrix.get((pattern_type, strength), "medium")

    def _calculate_risk_score(self, risk_level: str) -> float:
        """Convert risk level to numerical score."""
        risk_scores = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        return risk_scores.get(risk_level, 0.5)

    def _identify_risk_categories(
        self, pattern: Dict[str, Any], context_data: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Identify risk categories associated with the pattern."""
        categories = []

        pattern_type = pattern.get("pattern_type", "unknown")

        if pattern_type in ["security", "anomaly"]:
            categories.append("security_risk")

        if pattern_type == "compliance":
            categories.append("compliance_risk")

        if context_data and context_data.get("affects_availability", False):
            categories.append("availability_risk")

        return categories

    def _calculate_classification_confidence(
        self,
        type_classification: Dict[str, Any],
        strength_classification: Dict[str, Any],
        relevance_classification: Dict[str, Any],
    ) -> float:
        """Calculate confidence in the overall classification."""
        try:
            # Base confidence from successful classifications
            successful_classifications = 0
            total_classifications = 3

            if "classification_error" not in type_classification:
                successful_classifications += 1
            if "classification_error" not in strength_classification:
                successful_classifications += 1
            if "classification_error" not in relevance_classification:
                successful_classifications += 1

            base_confidence = successful_classifications / total_classifications

            # Adjust based on pattern strength
            strength_adjustment = (
                strength_classification.get("strength_score", 0.5) * 0.2
            )

            final_confidence = min(1.0, base_confidence + strength_adjustment)
            return final_confidence

        except Exception as e:
            logger.error("Error calculating classification confidence: %s", e)
            return 0.5

    def _load_classification_rules(self) -> Dict[str, Any]:
        """Load classification rules and mappings."""
        return {
            "types": {
                "temporal": {
                    "category": "time_series",
                    "subcategory": "trend_analysis",
                    "domain": "temporal_security",
                    "complexity": "medium",
                },
                "frequency": {
                    "category": "event_analysis",
                    "subcategory": "recurring_patterns",
                    "domain": "behavioral_security",
                    "complexity": "low",
                },
                "correlation": {
                    "category": "relationship_analysis",
                    "subcategory": "cross_correlation",
                    "domain": "statistical_security",
                    "complexity": "high",
                },
                "anomaly": {
                    "category": "outlier_detection",
                    "subcategory": "statistical_anomaly",
                    "domain": "anomaly_security",
                    "complexity": "medium",
                },
            },
            "business_relevance": {
                "temporal": "medium",
                "frequency": "low",
                "correlation": "high",
                "anomaly": "high",
                "security": "high",
                "compliance": "high",
            },
        }
