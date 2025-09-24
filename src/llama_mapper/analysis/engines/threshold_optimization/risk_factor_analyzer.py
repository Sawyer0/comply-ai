"""
Risk Factor Analyzer for threshold changes.
"""

import logging
from typing import Any, Dict, List

from .scenario_generator import ImpactScenario

logger = logging.getLogger(__name__)


class RiskFactorAnalyzer:
    """
    Analyzes risk factors from threshold change scenarios.

    Responsible for identifying potential risks, assessing their severity,
    and providing risk categorization.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def identify_risk_factors(
        self,
        scenarios: List[ImpactScenario],
        current_threshold: float,
        proposed_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Identify risk factors from scenario analysis."""
        try:
            risk_factors = []

            # Analyze performance degradation risks
            risk_factors.extend(self._analyze_performance_risks(scenarios))

            # Analyze cost risks
            risk_factors.extend(self._analyze_cost_risks(scenarios))

            # Analyze threshold sensitivity risks
            risk_factors.extend(
                self._analyze_threshold_sensitivity_risks(
                    current_threshold, proposed_threshold
                )
            )

            return risk_factors

        except Exception as e:
            logger.error("Risk factor identification failed: %s", str(e))
            return [
                {"type": "analysis_error", "severity": "unknown", "description": str(e)}
            ]

    def _analyze_performance_risks(
        self, scenarios: List[ImpactScenario]
    ) -> List[Dict[str, Any]]:
        """Analyze performance-related risks."""
        risks = []

        worst_case = next(
            (s for s in scenarios if s.scenario_name == "worst_case"), None
        )
        if worst_case:
            # Check false positive rate risk
            if worst_case.performance_metrics.false_positive_rate > 0.2:
                risks.append(
                    {
                        "type": "high_false_positive_rate",
                        "severity": "high",
                        "probability": worst_case.probability,
                        "description": f"Worst-case scenario shows FPR of {worst_case.performance_metrics.false_positive_rate:.1%}",
                    }
                )

            # Check false negative rate risk
            if worst_case.performance_metrics.false_negative_rate > 0.3:
                risks.append(
                    {
                        "type": "high_false_negative_rate",
                        "severity": "critical",
                        "probability": worst_case.probability,
                        "description": f"Worst-case scenario shows FNR of {worst_case.performance_metrics.false_negative_rate:.1%}",
                    }
                )

        return risks

    def _analyze_cost_risks(
        self, scenarios: List[ImpactScenario]
    ) -> List[Dict[str, Any]]:
        """Analyze cost-related risks."""
        risks = []

        cost_scenarios = [s for s in scenarios if "total_cost" in s.business_impact]
        if cost_scenarios:
            max_cost = max(s.business_impact["total_cost"] for s in cost_scenarios)
            if max_cost > 10000:  # $10k threshold
                risks.append(
                    {
                        "type": "high_cost_impact",
                        "severity": "medium",
                        "probability": 0.1,
                        "description": f"Potential cost impact up to ${max_cost:,.0f}",
                    }
                )

        return risks

    def _analyze_threshold_sensitivity_risks(
        self, current_threshold: float, proposed_threshold: float
    ) -> List[Dict[str, Any]]:
        """Analyze threshold sensitivity risks."""
        risks = []

        threshold_change = abs(proposed_threshold - current_threshold)
        if threshold_change > 0.2:
            risks.append(
                {
                    "type": "large_threshold_change",
                    "severity": "medium",
                    "probability": 1.0,
                    "description": f"Large threshold change of {threshold_change:.2f} may have unpredictable effects",
                }
            )

        return risks
