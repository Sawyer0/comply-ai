"""
Compound risk calculation engine.

This module provides sophisticated algorithms for calculating compound risks
that arise from interactions between multiple risk factors.
"""

import logging
import math
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskPattern:
    """Risk pattern for compound calculations."""

    pattern_id: str
    risk_factors: List[str]
    interaction_type: str
    amplification_factor: float


class CompoundRiskCalculator:
    """
    Compound risk calculation engine.

    Calculates complex risk interactions and compound effects
    that arise when multiple risk factors are present simultaneously.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interaction_models = self._initialize_interaction_models()

    def _initialize_interaction_models(self) -> Dict[str, Any]:
        """Initialize risk interaction models."""
        return {
            "multiplicative": {"factor": 1.2, "threshold": 0.5},
            "additive": {"factor": 1.1, "threshold": 0.3},
            "exponential": {"factor": 1.5, "threshold": 0.7},
        }

    async def calculate_compound_risk(
        self,
        patterns: List[RiskPattern],
        pattern_relationships: List[Dict[str, Any]],
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calculate compound risk from multiple patterns."""
        try:
            if not patterns:
                return {"risk_level": "low", "compound_score": 0.1}

            # Calculate base compound score
            base_score = self._calculate_base_compound_score(patterns)

            # Apply interaction effects
            interaction_score = self._calculate_interaction_effects(
                patterns, pattern_relationships
            )

            # Apply context adjustments
            context_adjustment = self._calculate_context_adjustment(context_data)

            # Calculate final compound score
            compound_score = min(
                1.0, base_score * interaction_score * context_adjustment
            )

            # Determine risk level
            risk_level = self._determine_compound_risk_level(compound_score)

            return {
                "risk_level": risk_level,
                "compound_score": compound_score,
                "base_score": base_score,
                "interaction_score": interaction_score,
                "context_adjustment": context_adjustment,
                "contributing_patterns": len(patterns),
            }

        except Exception as e:
            logger.error(
                "Compound risk calculation failed", extra={"error": str(e)}
            )
            return {"risk_level": "medium", "compound_score": 0.5, "error": str(e)}

    def _calculate_base_compound_score(self, patterns: List[RiskPattern]) -> float:
        """Calculate base compound score from patterns."""
        if not patterns:
            return 0.0

        # Weighted aggregation that accounts for interaction type and number of factors
        weighted_sum = 0.0
        total_weight = 0.0

        for pattern in patterns:
            model = self.interaction_models.get(
                pattern.interaction_type, self.interaction_models["additive"]
            )

            # Heavier weight for patterns with more contributing factors and
            # higher interaction strength.
            factor_weight = max(1.0, float(len(pattern.risk_factors)))
            interaction_weight = float(model.get("factor", 1.0))
            weight = factor_weight * interaction_weight

            weighted_sum += float(pattern.amplification_factor) * weight
            total_weight += weight

        if total_weight <= 0.0:
            return 0.0

        # Average amplification across patterns.
        avg_amplification = weighted_sum / total_weight

        # Map average amplification to a bounded risk score using a smooth
        # saturation function. This provides diminishing returns for
        # extremely large amplification values while remaining sensitive
        # in the typical range [0, 2].
        base_score = 1.0 - math.exp(-max(0.0, avg_amplification))
        return max(0.0, min(1.0, base_score))

    def _calculate_interaction_effects(
        self, patterns: List[RiskPattern], relationships: List[Dict[str, Any]]
    ) -> float:
        """Calculate interaction effects between patterns."""
        if len(patterns) < 2:
            return 1.0

        # Calculate interaction multiplier based on pattern relationships
        interaction_multiplier = 1.0

        for relationship in relationships:
            interaction_type = relationship.get("type", "additive")
            strength = relationship.get("strength", 0.1)

            if interaction_type == "multiplicative":
                interaction_multiplier *= 1.0 + strength
            elif interaction_type == "exponential":
                interaction_multiplier *= 1.0 + strength * 1.5
            else:  # additive
                interaction_multiplier += strength

        return min(2.0, interaction_multiplier)  # Cap at 2x amplification

    def _calculate_context_adjustment(
        self, context_data: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate context-based risk adjustments."""
        if not context_data:
            return 1.0

        adjustment = 1.0

        # Time-based adjustments
        if context_data.get("time_sensitive", False):
            adjustment *= 1.2

        # Environment-based adjustments
        environment = context_data.get("environment", "production")
        env_multipliers = {
            "production": 1.3,
            "staging": 1.1,
            "development": 0.8,
        }
        adjustment *= env_multipliers.get(environment, 1.0)

        # Data sensitivity adjustments
        sensitivity = context_data.get("data_sensitivity", "medium")
        sensitivity_multipliers = {
            "critical": 1.4,
            "high": 1.2,
            "medium": 1.0,
            "low": 0.8,
        }
        adjustment *= sensitivity_multipliers.get(sensitivity, 1.0)

        return min(2.0, adjustment)  # Cap adjustment

    def _determine_compound_risk_level(self, compound_score: float) -> str:
        """Determine risk level from compound score."""
        if compound_score >= 0.8:
            return "critical"
        if compound_score >= 0.6:
            return "high"
        if compound_score >= 0.3:
            return "medium"
        return "low"
