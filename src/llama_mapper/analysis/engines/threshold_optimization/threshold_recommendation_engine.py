"""
Threshold Recommendation Engine with statistical optimization.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from .statistical_optimizer import StatisticalOptimizer
from .impact_simulator import ImpactSimulator
from .types import OptimizationObjective, ThresholdRecommendation, PerformanceMetrics

logger = logging.getLogger(__name__)


class ThresholdRecommendationEngine:
    """
    Threshold recommendation engine with statistical optimization.

    Provides intelligent threshold recommendations based on statistical analysis,
    performance optimization, and impact simulation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.statistical_optimizer = StatisticalOptimizer(config)
        self.impact_simulator = ImpactSimulator(config)
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.7)
        self.recommendation_strategies = self._load_recommendation_strategies()

    async def generate_threshold_recommendation(
        self,
        detector_id: str,
        current_threshold: float,
        historical_data: List[Dict[str, Any]],
        optimization_objective: OptimizationObjective = None,
        constraints: Dict[str, float] = None,
        business_context: Dict[str, Any] = None,
    ) -> ThresholdRecommendation:
        """
        Generate comprehensive threshold recommendation.

        Args:
            detector_id: ID of the detector
            current_threshold: Current threshold value
            historical_data: Historical detection data
            optimization_objective: Optimization goal
            constraints: Performance constraints
            business_context: Business context for recommendations

        Returns:
            ThresholdRecommendation with detailed analysis
        """
        try:
            logger.info(
                "Generating threshold recommendation",
                detector_id=detector_id,
                current_threshold=current_threshold,
            )

            # Set default objective if not provided
            objective = (
                optimization_objective
                or OptimizationObjective.BALANCED_PRECISION_RECALL
            )
            constraints = constraints or {}
            business_context = business_context or {}

            # Perform statistical optimization
            optimization_result = await self.statistical_optimizer.optimize_threshold(
                detector_id=detector_id,
                historical_data=historical_data,
                objective=objective,
                constraints=constraints,
            )

            # Extract recommended threshold
            recommended_threshold = optimization_result.get(
                "optimal_threshold", current_threshold
            )

            # Skip simulation if threshold hasn't changed significantly
            if abs(recommended_threshold - current_threshold) < 0.01:
                return self._create_no_change_recommendation(
                    detector_id, current_threshold, optimization_result
                )

            # Simulate impact of threshold change
            simulation_result = await self.impact_simulator.simulate_threshold_impact(
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=recommended_threshold,
                historical_data=historical_data,
            )

            # Calculate confidence in recommendation
            recommendation_confidence = self._calculate_recommendation_confidence(
                optimization_result, simulation_result, historical_data
            )

            # Generate rationale
            rationale = self._generate_recommendation_rationale(
                optimization_result, simulation_result, objective, business_context
            )

            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(
                optimization_result, simulation_result
            )

            # Create comprehensive impact analysis
            impact_analysis = self._create_impact_analysis(
                optimization_result, simulation_result, business_context
            )

            recommendation = ThresholdRecommendation(
                detector_id=detector_id,
                current_threshold=current_threshold,
                recommended_threshold=recommended_threshold,
                expected_improvement=expected_improvement,
                confidence=recommendation_confidence,
                rationale=rationale,
                impact_analysis=impact_analysis,
            )

            logger.info(
                "Threshold recommendation generated",
                detector_id=detector_id,
                recommended_threshold=recommended_threshold,
                confidence=recommendation_confidence,
            )

            return recommendation

        except Exception as e:
            logger.error(
                "Threshold recommendation generation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_recommendation(
                detector_id, current_threshold, str(e)
            )

    async def evaluate_multiple_strategies(
        self,
        detector_id: str,
        current_threshold: float,
        historical_data: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]] = None,
    ) -> List[ThresholdRecommendation]:
        """
        Evaluate multiple optimization strategies and return ranked recommendations.

        Args:
            detector_id: ID of the detector
            current_threshold: Current threshold value
            historical_data: Historical detection data
            strategies: List of strategy configurations

        Returns:
            List of recommendations ranked by confidence and expected improvement
        """
        try:
            strategies = strategies or self.recommendation_strategies
            recommendations = []

            for strategy in strategies:
                objective = OptimizationObjective(
                    strategy.get("objective", "balanced_pr")
                )
                constraints = strategy.get("constraints", {})
                business_context = strategy.get("business_context", {})

                recommendation = await self.generate_threshold_recommendation(
                    detector_id=detector_id,
                    current_threshold=current_threshold,
                    historical_data=historical_data,
                    optimization_objective=objective,
                    constraints=constraints,
                    business_context=business_context,
                )

                recommendations.append(recommendation)

            # Rank recommendations by confidence and expected improvement
            recommendations.sort(
                key=lambda r: (r.confidence, sum(r.expected_improvement.values())),
                reverse=True,
            )

            return recommendations

        except Exception as e:
            logger.error(
                "Multiple strategy evaluation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return []

    def _calculate_recommendation_confidence(
        self,
        optimization_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> float:
        """Calculate confidence in the threshold recommendation."""
        try:
            confidence_factors = []

            # Data quality factor
            data_quality = min(
                1.0, len(historical_data) / 100.0
            )  # Normalize to 100 samples
            confidence_factors.append(data_quality)

            # Optimization quality factor
            optimization_summary = optimization_result.get("optimization_summary", {})
            optimization_success = optimization_summary.get(
                "optimization_success", False
            )
            if optimization_success:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.3)

            # Statistical significance factor
            optimal_performance = optimization_result.get("optimal_performance", {})
            objective_score = optimal_performance.get("objective_score", 0.0)
            confidence_factors.append(min(1.0, objective_score))

            # Risk assessment factor
            risk_assessment = simulation_result.get("risk_assessment", {})
            recommendation = risk_assessment.get("recommendation", "neutral")

            risk_confidence_map = {
                "strongly_recommended": 0.9,
                "recommended": 0.7,
                "neutral": 0.5,
                "not_recommended": 0.3,
                "strongly_not_recommended": 0.1,
                "insufficient_data": 0.2,
            }

            risk_confidence = risk_confidence_map.get(recommendation, 0.5)
            confidence_factors.append(risk_confidence)

            # Calculate weighted average confidence
            weights = [0.2, 0.3, 0.3, 0.2]  # Data, optimization, stats, risk
            weighted_confidence = sum(
                factor * weight for factor, weight in zip(confidence_factors, weights)
            )

            return max(0.0, min(1.0, weighted_confidence))

        except Exception as e:
            logger.error("Confidence calculation failed: %s", str(e))
            return 0.5

    def _generate_recommendation_rationale(
        self,
        optimization_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
        objective: OptimizationObjective,
        business_context: Dict[str, Any],
    ) -> str:
        """Generate human-readable rationale for the recommendation."""
        try:
            rationale_parts = []

            # Optimization rationale
            optimal_performance = optimization_result.get("optimal_performance", {})
            objective_score = optimal_performance.get("objective_score", 0.0)

            rationale_parts.append(
                f"Statistical optimization for {objective.value} achieved "
                f"objective score of {objective_score:.3f}."
            )

            # Performance impact rationale
            impact_projections = simulation_result.get("impact_projections", {})
            expected_changes = impact_projections.get("expected_changes", {})

            fp_delta = expected_changes.get("false_positives_delta", 0)
            fn_delta = expected_changes.get("false_negatives_delta", 0)

            if fp_delta < -10:
                rationale_parts.append(
                    f"Expected to reduce false positives by approximately {abs(fp_delta):.0f} cases."
                )
            elif fp_delta > 10:
                rationale_parts.append(
                    f"May increase false positives by approximately {fp_delta:.0f} cases."
                )

            if fn_delta < -5:
                rationale_parts.append(
                    f"Expected to reduce false negatives by approximately {abs(fn_delta):.0f} cases."
                )
            elif fn_delta > 5:
                rationale_parts.append(
                    f"May increase false negatives by approximately {fn_delta:.0f} cases."
                )

            # Risk assessment rationale
            risk_assessment = simulation_result.get("risk_assessment", {})
            risks = risk_assessment.get("risks", [])
            benefits = risk_assessment.get("benefits", [])

            if benefits:
                benefit_descriptions = [b.get("description", "") for b in benefits[:2]]
                rationale_parts.append(
                    f"Key benefits: {'; '.join(benefit_descriptions)}"
                )

            if risks:
                risk_descriptions = [r.get("description", "") for r in risks[:2]]
                rationale_parts.append(
                    f"Potential risks: {'; '.join(risk_descriptions)}"
                )

            # Business context rationale
            if business_context.get("cost_sensitivity") == "high":
                rationale_parts.append(
                    "Recommendation considers high cost sensitivity in business context."
                )

            return " ".join(rationale_parts)

        except Exception as e:
            logger.error("Rationale generation failed: %s", str(e))
            return "Recommendation based on statistical optimization and impact simulation."

    def _calculate_expected_improvement(
        self,
        optimization_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
    ) -> Dict[str, float]:
        """Calculate expected improvement metrics."""
        try:
            improvements = {}

            # Extract performance deltas from simulation
            impact_projections = simulation_result.get("impact_projections", {})
            expected_changes = impact_projections.get("expected_changes", {})

            improvements["precision_delta"] = expected_changes.get(
                "precision_delta", 0.0
            )
            improvements["recall_delta"] = expected_changes.get("recall_delta", 0.0)
            improvements["false_positives_delta"] = expected_changes.get(
                "false_positives_delta", 0.0
            )
            improvements["false_negatives_delta"] = expected_changes.get(
                "false_negatives_delta", 0.0
            )

            # Calculate F1 score improvement
            current_perf = simulation_result.get("current_performance", {})
            proposed_perf = simulation_result.get("proposed_performance", {})

            current_f1 = current_perf.get("f1_score", 0.0)
            proposed_f1 = proposed_perf.get("f1_score", 0.0)
            improvements["f1_score_delta"] = proposed_f1 - current_f1

            # Calculate overall performance improvement score
            optimization_summary = optimization_result.get("optimization_summary", {})
            objective_score_range = optimization_summary.get(
                "objective_score_range", {}
            )
            max_score = objective_score_range.get("max", 1.0)

            optimal_performance = optimization_result.get("optimal_performance", {})
            current_score = optimal_performance.get("objective_score", 0.0)

            improvements["objective_score_improvement"] = current_score / max(
                max_score, 0.001
            )

            return improvements

        except Exception as e:
            logger.error("Expected improvement calculation failed: %s", str(e))
            return {"error": str(e)}

    def _create_impact_analysis(
        self,
        optimization_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
        business_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create comprehensive impact analysis."""
        try:
            return {
                "optimization_analysis": {
                    "objective_achieved": optimization_result.get(
                        "optimization_summary", {}
                    ).get("optimization_success", False),
                    "feasible_thresholds": optimization_result.get("feasible_count", 0),
                    "performance_tradeoffs": optimization_result.get(
                        "optimization_summary", {}
                    ).get("performance_tradeoffs", {}),
                },
                "simulation_analysis": {
                    "confidence_intervals": simulation_result.get(
                        "confidence_intervals", {}
                    ),
                    "risk_assessment": simulation_result.get("risk_assessment", {}),
                    "projected_impact": simulation_result.get("impact_projections", {}),
                },
                "business_impact": self._assess_business_impact(
                    simulation_result, business_context
                ),
                "implementation_considerations": {
                    "data_quality": len(optimization_result.get("all_evaluations", [])),
                    "statistical_confidence": simulation_result.get(
                        "simulation_metadata", {}
                    ).get("confidence_level", 0.95),
                    "recommendation_strength": simulation_result.get(
                        "risk_assessment", {}
                    ).get("recommendation", "neutral"),
                },
            }

        except Exception as e:
            logger.error("Impact analysis creation failed: %s", str(e))
            return {"error": str(e)}

    def _assess_business_impact(
        self, simulation_result: Dict[str, Any], business_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess business impact of threshold change."""
        try:
            impact_projections = simulation_result.get("impact_projections", {})
            expected_changes = impact_projections.get("expected_changes", {})

            # Calculate operational impact
            fp_delta = expected_changes.get("false_positives_delta", 0)
            fn_delta = expected_changes.get("false_negatives_delta", 0)

            # Estimate cost impact if business context provides cost information
            fp_cost_per_case = business_context.get(
                "false_positive_cost", 10.0
            )  # Default $10
            fn_cost_per_case = business_context.get(
                "false_negative_cost", 100.0
            )  # Default $100

            cost_impact = (fp_delta * fp_cost_per_case) + (fn_delta * fn_cost_per_case)

            # Estimate resource impact
            investigation_time_per_fp = business_context.get(
                "investigation_time_minutes", 15
            )
            total_time_impact = abs(fp_delta) * investigation_time_per_fp

            return {
                "estimated_cost_impact": cost_impact,
                "investigation_time_impact_minutes": total_time_impact,
                "operational_efficiency_change": -abs(fp_delta)
                * 0.1,  # Rough efficiency estimate
                "compliance_impact": self._assess_compliance_impact(expected_changes),
            }

        except Exception as e:
            logger.error("Business impact assessment failed: %s", str(e))
            return {"error": str(e)}

    def _assess_compliance_impact(self, expected_changes: Dict[str, float]) -> str:
        """Assess impact on compliance posture."""
        fn_delta = expected_changes.get("false_negatives_delta", 0)
        fp_delta = expected_changes.get("false_positives_delta", 0)

        if fn_delta > 10:
            return "negative_high"  # More false negatives = worse compliance
        elif fn_delta > 5:
            return "negative_medium"
        elif fn_delta < -5:
            return "positive_medium"  # Fewer false negatives = better compliance
        elif fn_delta < -10:
            return "positive_high"
        else:
            return "neutral"

    def _load_recommendation_strategies(self) -> List[Dict[str, Any]]:
        """Load predefined recommendation strategies."""
        return [
            {
                "name": "balanced_performance",
                "objective": "balanced_pr",
                "constraints": {"min_recall": 0.7},
                "business_context": {"priority": "balanced"},
            },
            {
                "name": "minimize_false_positives",
                "objective": "minimize_fp",
                "constraints": {"min_recall": 0.6},
                "business_context": {"cost_sensitivity": "high"},
            },
            {
                "name": "maximize_detection",
                "objective": "maximize_recall",
                "constraints": {"min_precision": 0.5},
                "business_context": {"security_priority": "high"},
            },
            {
                "name": "high_precision",
                "objective": "maximize_precision",
                "constraints": {"min_recall": 0.4},
                "business_context": {"accuracy_priority": "high"},
            },
        ]

    def _create_no_change_recommendation(
        self,
        detector_id: str,
        current_threshold: float,
        optimization_result: Dict[str, Any],
    ) -> ThresholdRecommendation:
        """Create recommendation when no threshold change is needed."""
        return ThresholdRecommendation(
            detector_id=detector_id,
            current_threshold=current_threshold,
            recommended_threshold=current_threshold,
            expected_improvement={
                "precision_delta": 0.0,
                "recall_delta": 0.0,
                "f1_score_delta": 0.0,
            },
            confidence=0.8,
            rationale="Current threshold is already optimal based on statistical analysis.",
            impact_analysis={
                "optimization_analysis": optimization_result.get(
                    "optimization_summary", {}
                ),
                "recommendation": "no_change_needed",
            },
        )

    def _create_error_recommendation(
        self, detector_id: str, current_threshold: float, error_message: str
    ) -> ThresholdRecommendation:
        """Create recommendation for error cases."""
        return ThresholdRecommendation(
            detector_id=detector_id,
            current_threshold=current_threshold,
            recommended_threshold=current_threshold,
            expected_improvement={"error": error_message},
            confidence=0.0,
            rationale=f"Recommendation generation failed: {error_message}",
            impact_analysis={"error": error_message},
        )
