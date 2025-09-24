"""
Composite Risk Calculator - Sophisticated weighted risk composition.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class CompositeRiskCalculator:
    """Calculator for composite risk scores using weighted algorithms."""

    def __init__(self, calculation_weights: Dict[str, float]):
        self.calculation_weights = calculation_weights

    async def calculate_weighted_composite_score(self, risk_components: Dict[str, float]) -> float:
        """Calculate composite risk score using sophisticated weighted algorithm."""
        if not risk_components:
            return 0.0

        try:
            # Apply weights to each component
            weighted_sum = 0.0
            total_weight = 0.0

            for dimension_name, risk_value in risk_components.items():
                weight = self.calculation_weights.get(dimension_name, 0.0)
                if weight > 0:
                    # Apply diminishing returns for very high risk values
                    if risk_value > 0.8:
                        adjusted_risk = 0.8 + (0.2 * (risk_value - 0.8) / 0.2)
                    else:
                        adjusted_risk = risk_value

                    weighted_sum += adjusted_risk * weight
                    total_weight += weight

            if total_weight == 0:
                logger.warning("No valid weights for risk calculation")
                return 0.5  # Default moderate risk

            composite_score = weighted_sum / total_weight

            # Apply compound risk calculation if multiple high-risk dimensions
            high_risk_dimensions = [v for v in risk_components.values() if v > 0.7]
            if len(high_risk_dimensions) >= 2:
                compound_multiplier = 1.0 + (0.1 * (len(high_risk_dimensions) - 1))
                composite_score = min(1.0, composite_score * compound_multiplier)

            return max(0.0, min(1.0, composite_score))

        except Exception as e:
            logger.error("Error calculating weighted composite score: %s", e)
            # Fallback to simple average
            return sum(risk_components.values()) / len(risk_components)
