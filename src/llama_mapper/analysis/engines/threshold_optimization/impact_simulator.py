"""
Impact Simulator for predicting threshold change outcomes.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass

from .scenario_generator import ScenarioGenerator, ImpactScenario
from .business_impact_calculator import BusinessImpactCalculator
from .operational_impact_calculator import OperationalImpactCalculator
from .risk_factor_analyzer import RiskFactorAnalyzer
from .mitigation_strategy_generator import MitigationStrategyGenerator

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Comprehensive simulation result."""

    detector_id: str
    current_threshold: float
    proposed_threshold: float
    scenarios: List[ImpactScenario]
    expected_outcome: ImpactScenario
    confidence_intervals: Dict[str, Dict[str, float]]
    risk_factors: List[Dict[str, Any]]
    mitigation_strategies: List[Dict[str, Any]]
    simulation_metadata: Dict[str, Any]


class ImpactSimulator:
    """
    Impact simulator for predicting threshold change outcomes.

    Orchestrates multiple specialized components to provide comprehensive
    impact analysis for threshold changes.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.confidence_level = self.config.get("confidence_level", 0.95)

        # Initialize specialized components
        self.scenario_generator = ScenarioGenerator(config)
        self.business_impact_calculator = BusinessImpactCalculator(config)
        self.operational_impact_calculator = OperationalImpactCalculator(config)
        self.risk_factor_analyzer = RiskFactorAnalyzer(config)
        self.mitigation_strategy_generator = MitigationStrategyGenerator(config)

    async def simulate_threshold_impact(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        historical_data: List[Dict[str, Any]],
        time_horizon_days: int = 30,
        business_context: Dict[str, Any] = None,
    ) -> SimulationResult:
        """
        Simulate comprehensive impact of threshold change.

        Args:
            detector_id: ID of the detector
            current_threshold: Current threshold value
            proposed_threshold: Proposed new threshold value
            historical_data: Historical detection data
            time_horizon_days: Simulation time horizon
            business_context: Business context for impact assessment

        Returns:
            SimulationResult with comprehensive impact analysis
        """
        try:
            logger.info(
                "Starting threshold impact simulation",
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=proposed_threshold,
            )

            business_context = business_context or {}

            if not historical_data or len(historical_data) < 10:
                return self._create_insufficient_data_result(
                    detector_id, current_threshold, proposed_threshold
                )

            # Extract and validate data
            scores, labels = self._extract_and_validate_data(historical_data)

            if len(scores) < 10:
                return self._create_insufficient_data_result(
                    detector_id, current_threshold, proposed_threshold
                )

            # Generate impact scenarios
            scenarios = await self.scenario_generator.generate_scenarios(
                scores, labels, proposed_threshold, time_horizon_days, business_context
            )

            # Calculate business and operational impacts for each scenario
            for scenario in scenarios:
                scenario.business_impact = (
                    self.business_impact_calculator.calculate_business_impact(
                        scenario.performance_metrics,
                        time_horizon_days,
                        business_context,
                        scenario.scenario_name,
                    )
                )
                scenario.operational_impact = (
                    self.operational_impact_calculator.calculate_operational_impact(
                        scenario.performance_metrics,
                        time_horizon_days,
                        business_context,
                    )
                )

            # Calculate expected outcome
            expected_outcome = self._calculate_expected_outcome(scenarios)

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(scenarios)

            # Identify risk factors
            risk_factors = self.risk_factor_analyzer.identify_risk_factors(
                scenarios, current_threshold, proposed_threshold
            )

            # Generate mitigation strategies
            mitigation_strategies = (
                self.mitigation_strategy_generator.generate_mitigation_strategies(
                    risk_factors, business_context
                )
            )

            # Create simulation metadata
            simulation_metadata = {
                "data_points": len(scores),
                "time_horizon_days": time_horizon_days,
                "confidence_level": self.confidence_level,
                "scenario_count": len(scenarios),
                "simulation_timestamp": datetime.utcnow().isoformat(),
            }

            result = SimulationResult(
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=proposed_threshold,
                scenarios=scenarios,
                expected_outcome=expected_outcome,
                confidence_intervals=confidence_intervals,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies,
                simulation_metadata=simulation_metadata,
            )

            logger.info(
                "Threshold impact simulation completed",
                detector_id=detector_id,
                scenario_count=len(scenarios),
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold impact simulation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_result(
                detector_id, current_threshold, proposed_threshold, str(e)
            )

    def _extract_and_validate_data(
        self, historical_data: List[Dict[str, Any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract and validate scores and labels from historical data."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                # Validate ranges
                if 0.0 <= score <= 1.0 and label in [0, 1]:
                    scores.append(score)
                    labels.append(label)

            except (ValueError, TypeError):
                continue

        return scores, labels

    def _calculate_expected_outcome(
        self, scenarios: List[ImpactScenario]
    ) -> ImpactScenario:
        """Calculate probability-weighted expected outcome."""
        try:
            if not scenarios:
                return self._create_empty_scenario()

            # Calculate weighted averages
            total_probability = sum(s.probability for s in scenarios)
            if total_probability == 0:
                return scenarios[0]  # Return first scenario if no probabilities

            # Weighted performance metrics
            weighted_precision = (
                sum(s.performance_metrics.precision * s.probability for s in scenarios)
                / total_probability
            )
            weighted_recall = (
                sum(s.performance_metrics.recall * s.probability for s in scenarios)
                / total_probability
            )
            weighted_f1 = (
                sum(s.performance_metrics.f1_score * s.probability for s in scenarios)
                / total_probability
            )
            weighted_fpr = (
                sum(
                    s.performance_metrics.false_positive_rate * s.probability
                    for s in scenarios
                )
                / total_probability
            )
            weighted_fnr = (
                sum(
                    s.performance_metrics.false_negative_rate * s.probability
                    for s in scenarios
                )
                / total_probability
            )

            # Weighted business impact
            weighted_total_cost = (
                sum(
                    s.business_impact.get("total_cost", 0) * s.probability
                    for s in scenarios
                )
                / total_probability
            )

            from .types import PerformanceMetrics

            expected_performance = PerformanceMetrics(
                threshold=scenarios[0].performance_metrics.threshold,
                true_positives=int(
                    sum(
                        s.performance_metrics.true_positives * s.probability
                        for s in scenarios
                    )
                    / total_probability
                ),
                false_positives=int(
                    sum(
                        s.performance_metrics.false_positives * s.probability
                        for s in scenarios
                    )
                    / total_probability
                ),
                true_negatives=int(
                    sum(
                        s.performance_metrics.true_negatives * s.probability
                        for s in scenarios
                    )
                    / total_probability
                ),
                false_negatives=int(
                    sum(
                        s.performance_metrics.false_negatives * s.probability
                        for s in scenarios
                    )
                    / total_probability
                ),
                precision=weighted_precision,
                recall=weighted_recall,
                f1_score=weighted_f1,
                accuracy=sum(
                    s.performance_metrics.accuracy * s.probability for s in scenarios
                )
                / total_probability,
                specificity=sum(
                    s.performance_metrics.specificity * s.probability for s in scenarios
                )
                / total_probability,
                false_positive_rate=weighted_fpr,
                false_negative_rate=weighted_fnr,
            )

            expected_business_impact = {
                "total_cost": weighted_total_cost,
                "expected_value": True,
            }

            return ImpactScenario(
                scenario_name="expected",
                probability=1.0,
                performance_metrics=expected_performance,
                business_impact=expected_business_impact,
                operational_impact={"expected_value": True},
            )

        except Exception as e:
            logger.error("Expected outcome calculation failed: %s", str(e))
            return scenarios[0] if scenarios else self._create_empty_scenario()

    def _calculate_confidence_intervals(
        self, scenarios: List[ImpactScenario]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for key metrics."""
        try:
            if not scenarios:
                return {}

            # Extract metric values
            precision_values = [s.performance_metrics.precision for s in scenarios]
            recall_values = [s.performance_metrics.recall for s in scenarios]
            f1_values = [s.performance_metrics.f1_score for s in scenarios]
            cost_values = [s.business_impact.get("total_cost", 0) for s in scenarios]

            # Calculate percentiles for confidence intervals
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            return {
                "precision": {
                    "lower": np.percentile(precision_values, lower_percentile),
                    "upper": np.percentile(precision_values, upper_percentile),
                    "mean": np.mean(precision_values),
                },
                "recall": {
                    "lower": np.percentile(recall_values, lower_percentile),
                    "upper": np.percentile(recall_values, upper_percentile),
                    "mean": np.mean(recall_values),
                },
                "f1_score": {
                    "lower": np.percentile(f1_values, lower_percentile),
                    "upper": np.percentile(f1_values, upper_percentile),
                    "mean": np.mean(f1_values),
                },
                "total_cost": {
                    "lower": np.percentile(cost_values, lower_percentile),
                    "upper": np.percentile(cost_values, upper_percentile),
                    "mean": np.mean(cost_values),
                },
            }

        except Exception as e:
            logger.error("Confidence interval calculation failed: %s", str(e))
            return {"error": str(e)}

    def _create_empty_scenario(self) -> ImpactScenario:
        """Create empty scenario for error cases."""
        from .types import PerformanceMetrics

        return ImpactScenario(
            scenario_name="empty",
            probability=0.0,
            performance_metrics=PerformanceMetrics(
                threshold=0.5,
                true_positives=0,
                false_positives=0,
                true_negatives=0,
                false_negatives=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                accuracy=0.0,
                specificity=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0,
            ),
            business_impact={},
            operational_impact={},
        )

    def _create_insufficient_data_result(
        self, detector_id: str, current_threshold: float, proposed_threshold: float
    ) -> SimulationResult:
        """Create result for insufficient data cases."""
        return SimulationResult(
            detector_id=detector_id,
            current_threshold=current_threshold,
            proposed_threshold=proposed_threshold,
            scenarios=[],
            expected_outcome=ImpactScenario(
                scenario_name="insufficient_data",
                probability=0.0,
                performance_metrics=self._create_empty_scenario().performance_metrics,
                business_impact={"message": "Insufficient data for simulation"},
                operational_impact={"message": "Insufficient data for simulation"},
            ),
            confidence_intervals={},
            risk_factors=[
                {
                    "type": "insufficient_data",
                    "severity": "high",
                    "description": "Insufficient historical data for reliable simulation",
                }
            ],
            mitigation_strategies=[
                {
                    "strategy": "collect_more_data",
                    "description": "Collect more historical data before implementing threshold change",
                }
            ],
            simulation_metadata={"error": "insufficient_data"},
        )

    def _create_error_result(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        error_message: str,
    ) -> SimulationResult:
        """Create result for error cases."""
        return SimulationResult(
            detector_id=detector_id,
            current_threshold=current_threshold,
            proposed_threshold=proposed_threshold,
            scenarios=[],
            expected_outcome=ImpactScenario(
                scenario_name="error",
                probability=0.0,
                performance_metrics=self._create_empty_scenario().performance_metrics,
                business_impact={"error": error_message},
                operational_impact={"error": error_message},
            ),
            confidence_intervals={"error": error_message},
            risk_factors=[{"type": "simulation_error", "description": error_message}],
            mitigation_strategies=[
                {
                    "strategy": "fix_error",
                    "description": f"Resolve simulation error: {error_message}",
                }
            ],
            simulation_metadata={"error": error_message},
        )
