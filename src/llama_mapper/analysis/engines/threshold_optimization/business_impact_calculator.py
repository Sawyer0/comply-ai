"""
Business Impact Calculator for threshold changes.
"""

import logging
from typing import Any, Dict

from .types import PerformanceMetrics

logger = logging.getLogger(__name__)


class BusinessImpactCalculator:
    """
    Calculates business impact of threshold changes.

    Responsible for translating performance metrics into business costs,
    productivity impacts, and financial implications.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cost_model = self._load_cost_model()

    def calculate_business_impact(
        self,
        performance_metrics: PerformanceMetrics,
        time_horizon_days: int,
        business_context: Dict[str, Any],
        scenario_name: str,
    ) -> Dict[str, float]:
        """Calculate business impact for given performance metrics."""
        try:
            # Get cost parameters
            fp_cost = business_context.get(
                "false_positive_cost", self.cost_model["fp_cost"]
            )
            fn_cost = business_context.get(
                "false_negative_cost", self.cost_model["fn_cost"]
            )
            investigation_cost = business_context.get(
                "investigation_cost", self.cost_model["investigation_cost"]
            )

            # Estimate daily detection volume
            daily_detections = max(
                50,
                (
                    performance_metrics.true_positives
                    + performance_metrics.false_positives
                ),
            )
            total_detections = daily_detections * time_horizon_days

            # Scale performance metrics to time horizon
            scaled_fp = performance_metrics.false_positive_rate * total_detections
            scaled_fn = performance_metrics.false_negative_rate * total_detections

            # Apply scenario multipliers
            multiplier = self._get_scenario_multiplier(scenario_name)
            scaled_fp *= multiplier
            scaled_fn *= multiplier

            # Calculate costs
            fp_total_cost = scaled_fp * fp_cost
            fn_total_cost = scaled_fn * fn_cost
            investigation_total_cost = scaled_fp * investigation_cost

            total_cost = fp_total_cost + fn_total_cost + investigation_total_cost

            # Calculate productivity impact
            productivity_loss_hours = self._calculate_productivity_loss(
                scaled_fp, business_context
            )

            return {
                "total_cost": total_cost,
                "false_positive_cost": fp_total_cost,
                "false_negative_cost": fn_total_cost,
                "investigation_cost": investigation_total_cost,
                "productivity_loss_hours": productivity_loss_hours,
                "cost_per_day": total_cost / time_horizon_days,
            }

        except Exception as e:
            logger.error("Business impact calculation failed: %s", str(e))
            return {"error": str(e)}

    def _calculate_productivity_loss(
        self, false_positives: float, business_context: Dict[str, Any]
    ) -> float:
        """Calculate productivity loss from false positives."""
        hours_per_fp_investigation = business_context.get(
            "hours_per_fp_investigation", 0.25
        )
        return false_positives * hours_per_fp_investigation

    def _get_scenario_multiplier(self, scenario_name: str) -> float:
        """Get multiplier for different scenarios."""
        multipliers = {
            "optimistic": 0.8,
            "realistic": 1.0,
            "pessimistic": 1.2,
            "worst_case": 1.5,
            "best_case": 0.6,
        }
        return multipliers.get(scenario_name, 1.0)

    def _load_cost_model(self) -> Dict[str, float]:
        """Load business cost model parameters."""
        return self.config.get(
            "cost_model",
            {
                "fp_cost": 25.0,  # Cost per false positive
                "fn_cost": 500.0,  # Cost per false negative
                "investigation_cost": 15.0,  # Cost per investigation
            },
        )
