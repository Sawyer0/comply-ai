"""
Operational Impact Calculator for threshold changes.
"""

import logging
from typing import Any, Dict

from .types import PerformanceMetrics

logger = logging.getLogger(__name__)


class OperationalImpactCalculator:
    """
    Calculates operational impact of threshold changes.

    Responsible for assessing team capacity, workload distribution,
    response times, and operational strain.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def calculate_operational_impact(
        self,
        performance_metrics: PerformanceMetrics,
        time_horizon_days: int,
        business_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate operational impact for given performance metrics."""
        try:
            # Calculate workload impact
            daily_alerts = max(
                20,
                performance_metrics.true_positives
                + performance_metrics.false_positives,
            )
            total_alerts = daily_alerts * time_horizon_days

            # Calculate team capacity impact
            team_size = business_context.get("security_team_size", 5)
            alerts_per_person_per_day = total_alerts / (team_size * time_horizon_days)

            # Assess capacity strain
            capacity_strain = self._assess_capacity_strain(alerts_per_person_per_day)

            # Calculate response time impact
            expected_response_time = self._calculate_response_time_impact(
                capacity_strain, business_context
            )

            # Calculate team utilization
            team_utilization = self._calculate_team_utilization(
                alerts_per_person_per_day
            )

            return {
                "total_alerts": total_alerts,
                "alerts_per_person_per_day": alerts_per_person_per_day,
                "capacity_strain": capacity_strain,
                "expected_response_time_minutes": expected_response_time,
                "team_utilization": team_utilization,
                "overtime_risk": capacity_strain in ["high", "critical"],
            }

        except Exception as e:
            logger.error("Operational impact calculation failed: %s", str(e))
            return {"error": str(e)}

    def _assess_capacity_strain(self, alerts_per_person_per_day: float) -> str:
        """Assess team capacity strain level."""
        if alerts_per_person_per_day > 50:
            return "critical"
        elif alerts_per_person_per_day > 30:
            return "high"
        elif alerts_per_person_per_day > 15:
            return "moderate"
        else:
            return "low"

    def _calculate_response_time_impact(
        self, capacity_strain: str, business_context: Dict[str, Any]
    ) -> float:
        """Calculate expected response time based on capacity strain."""
        base_response_time = business_context.get("base_response_time_minutes", 30)

        multipliers = {
            "critical": 2.0,
            "high": 1.5,
            "moderate": 1.2,
            "low": 1.0,
        }

        multiplier = multipliers.get(capacity_strain, 1.0)
        return base_response_time * multiplier

    def _calculate_team_utilization(self, alerts_per_person_per_day: float) -> float:
        """Calculate team utilization percentage."""
        # Assume 40 alerts/day per person is 100% utilization
        max_alerts_per_day = 40
        return min(1.0, alerts_per_person_per_day / max_alerts_per_day)
