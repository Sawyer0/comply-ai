"""
Compound Risk Calculator for assessing combined pattern impacts.

This module implements sophisticated risk calculation methods to assess
the compound effects of multiple interacting security patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from enum import Enum

from ...domain import (
    Pattern,
    PatternType,
    PatternStrength,
    BusinessRelevance,
    RiskLevel,
    SecurityData,
)

logger = logging.getLogger(__name__)


class RiskInteractionType(Enum):
    """Types of risk interactions between patterns."""

    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    SYNERGISTIC = "synergistic"
    MITIGATING = "mitigating"
    INDEPENDENT = "independent"


class CompoundRiskCalculator:
    """
    Calculates compound risk scores for multiple interacting patterns.

    Uses sophisticated risk modeling to account for pattern interactions,
    diminishing returns, and synergistic effects in risk assessment.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.base_risk_weights = self._load_base_risk_weights()
        self.interaction_matrix = self._load_interaction_matrix()
        self.diminishing_returns_factor = self.config.get(
            "diminishing_returns_factor", 0.8
        )
        self.synergy_threshold = self.config.get("synergy_threshold", 0.7)

    async def calculate_compound_risk(
        self,
        patterns: List[Pattern],
        pattern_relationships: List[Dict[str, any]],
        context_data: SecurityData,
    ) -> Dict[str, any]:
        """
        Calculate compound risk score for multiple patterns.

        Args:
            patterns: List of patterns to assess
            pattern_relationships: Relationships between patterns
            context_data: Security data for context

        Returns:
            Dictionary containing compound risk assessment
        """
        if not patterns:
            return {"compound_risk_score": 0.0, "risk_level": RiskLevel.LOW}

        try:
            # Calculate individual pattern risks
            individual_risks = await self._calculate_individual_risks(
                patterns, context_data
            )

            # Analyze pattern interactions
            interaction_effects = await self._analyze_interaction_effects(
                patterns, pattern_relationships, individual_risks
            )

            # Calculate base compound risk
            base_compound_risk = self._calculate_base_compound_risk(individual_risks)

            # Apply interaction effects
            adjusted_compound_risk = self._apply_interaction_effects(
                base_compound_risk, interaction_effects
            )

            # Apply diminishing returns
            final_compound_risk = self._apply_diminishing_returns(
                adjusted_compound_risk, len(patterns)
            )

            # Determine risk level
            risk_level = self._determine_risk_level(final_compound_risk)

            # Calculate risk breakdown
            risk_breakdown = self._calculate_risk_breakdown(
                individual_risks, interaction_effects, final_compound_risk
            )

            # Assess risk factors
            risk_factors = await self._assess_risk_factors(
                patterns, pattern_relationships
            )

            # Generate risk scenarios
            risk_scenarios = self._generate_risk_scenarios(
                patterns, individual_risks, interaction_effects
            )

            compound_risk_result = {
                "compound_risk_score": final_compound_risk,
                "risk_level": risk_level,
                "individual_risks": individual_risks,
                "interaction_effects": interaction_effects,
                "risk_breakdown": risk_breakdown,
                "risk_factors": risk_factors,
                "risk_scenarios": risk_scenarios,
                "calculation_metadata": {
                    "total_patterns": len(patterns),
                    "base_compound_risk": base_compound_risk,
                    "interaction_adjustment": adjusted_compound_risk
                    - base_compound_risk,
                    "diminishing_returns_applied": final_compound_risk
                    < adjusted_compound_risk,
                    "calculation_timestamp": datetime.utcnow().isoformat(),
                },
            }

            logger.info(
                "Compound risk calculated",
                total_patterns=len(patterns),
                compound_risk_score=final_compound_risk,
                risk_level=risk_level.value,
            )

            return compound_risk_result

        except Exception as e:
            logger.error("Compound risk calculation failed", error=str(e))
            return {"compound_risk_score": 0.0, "risk_level": RiskLevel.LOW}

    async def _calculate_individual_risks(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> List[Dict[str, any]]:
        """Calculate individual risk scores for each pattern."""
        individual_risks = []

        try:
            for pattern in patterns:
                risk_score = await self._calculate_pattern_risk_score(
                    pattern, context_data
                )

                individual_risk = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type.value,
                    "base_risk_score": risk_score,
                    "confidence_weighted_risk": risk_score * pattern.confidence,
                    "business_relevance_factor": self._get_business_relevance_factor(
                        pattern.business_relevance
                    ),
                    "strength_factor": self._get_strength_factor(pattern.strength),
                    "detector_criticality": self._calculate_detector_criticality(
                        pattern.affected_detectors
                    ),
                    "temporal_urgency": self._calculate_temporal_urgency(pattern),
                    "final_individual_risk": self._calculate_final_individual_risk(
                        pattern, risk_score
                    ),
                }

                individual_risks.append(individual_risk)

        except Exception as e:
            logger.error("Individual risk calculation failed", error=str(e))

        return individual_risks

    async def _calculate_pattern_risk_score(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Calculate base risk score for a single pattern."""
        try:
            # Base risk from pattern type
            type_risk = self.base_risk_weights.get(pattern.pattern_type.value, 0.5)

            # Adjust for pattern strength
            strength_multiplier = {
                PatternStrength.WEAK: 0.7,
                PatternStrength.MODERATE: 1.0,
                PatternStrength.STRONG: 1.3,
            }.get(pattern.strength, 1.0)

            # Adjust for confidence
            confidence_factor = 0.5 + (pattern.confidence * 0.5)  # Range: 0.5 to 1.0

            # Adjust for business relevance
            relevance_multiplier = {
                BusinessRelevance.LOW: 0.6,
                BusinessRelevance.MEDIUM: 0.8,
                BusinessRelevance.HIGH: 1.2,
                BusinessRelevance.CRITICAL: 1.5,
            }.get(pattern.business_relevance, 1.0)

            # Calculate base risk
            base_risk = (
                type_risk
                * strength_multiplier
                * confidence_factor
                * relevance_multiplier
            )

            return min(1.0, max(0.0, base_risk))

        except Exception as e:
            logger.error("Pattern risk score calculation failed", error=str(e))
            return 0.5

    def _get_business_relevance_factor(self, relevance: BusinessRelevance) -> float:
        """Get business relevance factor."""
        factors = {
            BusinessRelevance.LOW: 0.6,
            BusinessRelevance.MEDIUM: 0.8,
            BusinessRelevance.HIGH: 1.2,
            BusinessRelevance.CRITICAL: 1.5,
        }
        return factors.get(relevance, 1.0)

    def _get_strength_factor(self, strength: PatternStrength) -> float:
        """Get pattern strength factor."""
        factors = {
            PatternStrength.WEAK: 0.7,
            PatternStrength.MODERATE: 1.0,
            PatternStrength.STRONG: 1.3,
        }
        return factors.get(strength, 1.0)

    def _calculate_detector_criticality(self, detectors: List[str]) -> float:
        """Calculate criticality factor based on affected detectors."""
        try:
            criticality_scores = {
                "presidio": 0.9,
                "pii-detector": 0.9,
                "gdpr-scanner": 0.85,
                "hipaa-detector": 0.85,
                "financial-detector": 0.8,
                "malware-detector": 0.95,
                "vulnerability-scanner": 0.9,
                "threat-detector": 0.85,
                "deberta-toxicity": 0.6,
            }

            if not detectors:
                return 0.5

            detector_scores = [
                criticality_scores.get(detector.lower(), 0.5) for detector in detectors
            ]

            # Use maximum criticality (most critical detector drives the score)
            return max(detector_scores)

        except Exception:
            return 0.5

    def _calculate_temporal_urgency(self, pattern: Pattern) -> float:
        """Calculate temporal urgency factor."""
        try:
            now = datetime.utcnow().replace(tzinfo=timezone.utc)
            pattern_age = (now - pattern.time_range.end).total_seconds()

            # More recent patterns have higher urgency
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

    def _calculate_final_individual_risk(
        self, pattern: Pattern, base_risk: float
    ) -> float:
        """Calculate final individual risk score."""
        try:
            # Apply all factors
            business_factor = self._get_business_relevance_factor(
                pattern.business_relevance
            )
            strength_factor = self._get_strength_factor(pattern.strength)
            detector_factor = self._calculate_detector_criticality(
                pattern.affected_detectors
            )
            temporal_factor = self._calculate_temporal_urgency(pattern)
            confidence_factor = pattern.confidence

            # Weighted combination
            final_risk = (
                base_risk * 0.3
                + business_factor * base_risk * 0.25
                + strength_factor * base_risk * 0.2
                + detector_factor * base_risk * 0.15
                + temporal_factor * base_risk * 0.1
            ) * confidence_factor

            return min(1.0, max(0.0, final_risk))

        except Exception:
            return base_risk

    async def _analyze_interaction_effects(
        self,
        patterns: List[Pattern],
        relationships: List[Dict[str, any]],
        individual_risks: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """Analyze interaction effects between patterns."""
        interaction_effects = []

        try:
            # Create risk lookup
            risk_lookup = {risk["pattern_id"]: risk for risk in individual_risks}

            for relationship in relationships:
                pattern1_id = relationship["pattern1_id"]
                pattern2_id = relationship["pattern2_id"]

                if pattern1_id in risk_lookup and pattern2_id in risk_lookup:
                    interaction = await self._calculate_interaction_effect(
                        relationship, risk_lookup[pattern1_id], risk_lookup[pattern2_id]
                    )
                    if interaction:
                        interaction_effects.append(interaction)

        except Exception as e:
            logger.error("Interaction effects analysis failed", error=str(e))

        return interaction_effects

    async def _calculate_interaction_effect(
        self, relationship: Dict[str, any], risk1: Dict[str, any], risk2: Dict[str, any]
    ) -> Optional[Dict[str, any]]:
        """Calculate interaction effect between two patterns."""
        try:
            relationship_strength = relationship["strength"]

            if (
                relationship_strength < 0.3
            ):  # Weak relationships have minimal interaction
                return None

            # Determine interaction type
            interaction_type = self._determine_interaction_type(
                relationship, risk1, risk2
            )

            # Calculate interaction magnitude
            interaction_magnitude = self._calculate_interaction_magnitude(
                relationship_strength,
                risk1["final_individual_risk"],
                risk2["final_individual_risk"],
            )

            # Calculate interaction effect
            interaction_effect = self._calculate_interaction_value(
                interaction_type,
                interaction_magnitude,
                risk1["final_individual_risk"],
                risk2["final_individual_risk"],
            )

            return {
                "pattern1_id": risk1["pattern_id"],
                "pattern2_id": risk2["pattern_id"],
                "interaction_type": interaction_type.value,
                "interaction_strength": relationship_strength,
                "interaction_magnitude": interaction_magnitude,
                "interaction_effect": interaction_effect,
                "risk_amplification": interaction_effect > 0,
                "metadata": {
                    "relationship_type": relationship.get(
                        "relationship_type", "unknown"
                    ),
                    "pattern1_risk": risk1["final_individual_risk"],
                    "pattern2_risk": risk2["final_individual_risk"],
                },
            }

        except Exception as e:
            logger.error("Interaction effect calculation failed", error=str(e))
            return None

    def _determine_interaction_type(
        self, relationship: Dict[str, any], risk1: Dict[str, any], risk2: Dict[str, any]
    ) -> RiskInteractionType:
        """Determine the type of risk interaction."""
        try:
            relationship_type = relationship.get("relationship_type", "unknown")
            relationship_strength = relationship["strength"]

            # High strength relationships often have synergistic effects
            if relationship_strength > self.synergy_threshold:
                return RiskInteractionType.SYNERGISTIC

            # Detector-based relationships are often additive
            if relationship_type == "detector_based":
                return RiskInteractionType.ADDITIVE

            # Temporal relationships can be multiplicative
            if relationship_type == "temporal_based":
                return RiskInteractionType.MULTIPLICATIVE

            # Compound relationships are synergistic
            if relationship_type == "compound":
                return RiskInteractionType.SYNERGISTIC

            # Default to additive for moderate relationships
            if relationship_strength > 0.5:
                return RiskInteractionType.ADDITIVE

            return RiskInteractionType.INDEPENDENT

        except Exception:
            return RiskInteractionType.INDEPENDENT

    def _calculate_interaction_magnitude(
        self, relationship_strength: float, risk1: float, risk2: float
    ) -> float:
        """Calculate the magnitude of interaction effect."""
        try:
            # Base magnitude from relationship strength
            base_magnitude = relationship_strength

            # Adjust based on risk levels (higher risks have stronger interactions)
            risk_factor = (risk1 + risk2) / 2

            # Interaction magnitude increases with both relationship strength and risk levels
            magnitude = base_magnitude * (0.5 + risk_factor * 0.5)

            return min(1.0, max(0.0, magnitude))

        except Exception:
            return 0.0

    def _calculate_interaction_value(
        self,
        interaction_type: RiskInteractionType,
        magnitude: float,
        risk1: float,
        risk2: float,
    ) -> float:
        """Calculate the actual interaction effect value."""
        try:
            if interaction_type == RiskInteractionType.ADDITIVE:
                # Simple addition with magnitude scaling
                return magnitude * min(risk1, risk2) * 0.3

            elif interaction_type == RiskInteractionType.MULTIPLICATIVE:
                # Multiplicative effect
                return magnitude * risk1 * risk2 * 0.5

            elif interaction_type == RiskInteractionType.SYNERGISTIC:
                # Synergistic effect (greater than sum of parts)
                base_effect = risk1 + risk2
                synergy_bonus = magnitude * base_effect * 0.2
                return synergy_bonus

            elif interaction_type == RiskInteractionType.MITIGATING:
                # One pattern reduces risk of another
                return -magnitude * min(risk1, risk2) * 0.2

            else:  # INDEPENDENT
                return 0.0

        except Exception:
            return 0.0

    def _calculate_base_compound_risk(
        self, individual_risks: List[Dict[str, any]]
    ) -> float:
        """Calculate base compound risk from individual risks."""
        try:
            if not individual_risks:
                return 0.0

            # Extract individual risk scores
            risk_scores = [risk["final_individual_risk"] for risk in individual_risks]

            # Use root mean square for compound risk (accounts for multiple risks)
            rms_risk = np.sqrt(np.mean([r**2 for r in risk_scores]))

            return min(1.0, rms_risk)

        except Exception:
            return 0.0

    def _apply_interaction_effects(
        self, base_risk: float, interaction_effects: List[Dict[str, any]]
    ) -> float:
        """Apply interaction effects to base compound risk."""
        try:
            total_interaction_effect = sum(
                effect["interaction_effect"] for effect in interaction_effects
            )

            # Apply interaction effects with diminishing returns
            adjusted_risk = base_risk + (total_interaction_effect * 0.5)

            return min(1.0, max(0.0, adjusted_risk))

        except Exception:
            return base_risk

    def _apply_diminishing_returns(
        self, risk_score: float, pattern_count: int
    ) -> float:
        """Apply diminishing returns for multiple patterns."""
        try:
            if pattern_count <= 1:
                return risk_score

            # Diminishing returns factor decreases with more patterns
            diminishing_factor = self.diminishing_returns_factor ** (pattern_count - 1)

            # Apply diminishing returns
            adjusted_risk = risk_score * (0.5 + diminishing_factor * 0.5)

            return min(1.0, max(0.0, adjusted_risk))

        except Exception:
            return risk_score

    def _determine_risk_level(self, compound_risk_score: float) -> RiskLevel:
        """Determine risk level from compound risk score."""
        if compound_risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif compound_risk_score >= 0.6:
            return RiskLevel.HIGH
        elif compound_risk_score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_risk_breakdown(
        self,
        individual_risks: List[Dict[str, any]],
        interaction_effects: List[Dict[str, any]],
        final_compound_risk: float,
    ) -> Dict[str, any]:
        """Calculate detailed risk breakdown."""
        try:
            total_individual_risk = sum(
                risk["final_individual_risk"] for risk in individual_risks
            )
            total_interaction_effect = sum(
                effect["interaction_effect"] for effect in interaction_effects
            )

            return {
                "individual_risk_contribution": (
                    total_individual_risk / len(individual_risks)
                    if individual_risks
                    else 0
                ),
                "interaction_effect_contribution": total_interaction_effect,
                "base_risk_percentage": (
                    (total_individual_risk / final_compound_risk * 100)
                    if final_compound_risk > 0
                    else 0
                ),
                "interaction_percentage": (
                    (abs(total_interaction_effect) / final_compound_risk * 100)
                    if final_compound_risk > 0
                    else 0
                ),
                "top_risk_contributors": sorted(
                    individual_risks,
                    key=lambda x: x["final_individual_risk"],
                    reverse=True,
                )[:3],
                "strongest_interactions": sorted(
                    interaction_effects,
                    key=lambda x: abs(x["interaction_effect"]),
                    reverse=True,
                )[:3],
            }

        except Exception as e:
            logger.error("Risk breakdown calculation failed", error=str(e))
            return {}

    async def _assess_risk_factors(
        self, patterns: List[Pattern], relationships: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """Assess key risk factors."""
        risk_factors = []

        try:
            # Pattern diversity factor
            pattern_types = set(p.pattern_type for p in patterns)
            if len(pattern_types) > 2:
                risk_factors.append(
                    {
                        "factor": "pattern_diversity",
                        "description": "Multiple pattern types detected",
                        "impact": "high",
                        "value": len(pattern_types),
                    }
                )

            # High confidence patterns
            high_conf_patterns = [p for p in patterns if p.confidence > 0.8]
            if len(high_conf_patterns) > 1:
                risk_factors.append(
                    {
                        "factor": "high_confidence_patterns",
                        "description": "Multiple high-confidence patterns",
                        "impact": "high",
                        "value": len(high_conf_patterns),
                    }
                )

            # Strong relationships
            strong_relationships = [r for r in relationships if r["strength"] > 0.7]
            if strong_relationships:
                risk_factors.append(
                    {
                        "factor": "strong_pattern_relationships",
                        "description": "Strong correlations between patterns",
                        "impact": "medium",
                        "value": len(strong_relationships),
                    }
                )

            # Critical business relevance
            critical_patterns = [
                p
                for p in patterns
                if p.business_relevance == BusinessRelevance.CRITICAL
            ]
            if critical_patterns:
                risk_factors.append(
                    {
                        "factor": "critical_business_impact",
                        "description": "Patterns with critical business relevance",
                        "impact": "critical",
                        "value": len(critical_patterns),
                    }
                )

        except Exception as e:
            logger.error("Risk factors assessment failed", error=str(e))

        return risk_factors

    def _generate_risk_scenarios(
        self,
        patterns: List[Pattern],
        individual_risks: List[Dict[str, any]],
        interaction_effects: List[Dict[str, any]],
    ) -> List[Dict[str, any]]:
        """Generate risk scenarios for different conditions."""
        scenarios = []

        try:
            # Best case scenario (minimal interactions)
            best_case_risk = (
                max(risk["final_individual_risk"] for risk in individual_risks)
                if individual_risks
                else 0
            )
            scenarios.append(
                {
                    "scenario": "best_case",
                    "description": "Patterns remain isolated with minimal interaction",
                    "probability": 0.3,
                    "risk_score": best_case_risk,
                    "risk_level": self._determine_risk_level(best_case_risk).value,
                }
            )

            # Worst case scenario (maximum interactions)
            worst_case_risk = min(
                1.0,
                sum(risk["final_individual_risk"] for risk in individual_risks) * 1.2,
            )
            scenarios.append(
                {
                    "scenario": "worst_case",
                    "description": "All patterns interact with maximum synergistic effects",
                    "probability": 0.2,
                    "risk_score": worst_case_risk,
                    "risk_level": self._determine_risk_level(worst_case_risk).value,
                }
            )

            # Most likely scenario (current calculation)
            base_compound_risk = self._calculate_base_compound_risk(individual_risks)
            adjusted_risk = self._apply_interaction_effects(
                base_compound_risk, interaction_effects
            )
            likely_risk = self._apply_diminishing_returns(adjusted_risk, len(patterns))

            scenarios.append(
                {
                    "scenario": "most_likely",
                    "description": "Expected risk based on current pattern relationships",
                    "probability": 0.5,
                    "risk_score": likely_risk,
                    "risk_level": self._determine_risk_level(likely_risk).value,
                }
            )

        except Exception as e:
            logger.error("Risk scenarios generation failed", error=str(e))

        return scenarios

    def _load_base_risk_weights(self) -> Dict[str, float]:
        """Load base risk weights for different pattern types."""
        return self.config.get(
            "base_risk_weights",
            {"temporal": 0.6, "frequency": 0.5, "correlation": 0.7, "anomaly": 0.8},
        )

    def _load_interaction_matrix(self) -> Dict[str, Dict[str, float]]:
        """Load interaction matrix for pattern type combinations."""
        return self.config.get(
            "interaction_matrix",
            {
                "temporal": {"frequency": 0.7, "correlation": 0.6, "anomaly": 0.8},
                "frequency": {"correlation": 0.5, "anomaly": 0.6},
                "correlation": {"anomaly": 0.7},
            },
        )
