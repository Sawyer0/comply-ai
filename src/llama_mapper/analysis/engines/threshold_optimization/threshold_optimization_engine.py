"""
Main Threshold Optimization Engine that orchestrates all components.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..domain import (
    AnalysisConfiguration,
    AnalysisResult,
    BaseAnalysisEngine,
    SecurityFinding,
)
from ..domain.entities import AnalysisRequest

from .roc_analyzer import ROCAnalyzer
from .statistical_optimizer import StatisticalOptimizer
from .threshold_simulator import ThresholdSimulator
from .performance_metrics_calculator import PerformanceMetricsCalculator
from .types import OptimizationObjective, ThresholdRecommendation

logger = logging.getLogger(__name__)


class ThresholdOptimizationEngine(BaseAnalysisEngine):
    """
    Main threshold optimization engine that orchestrates statistical analysis.

    Coordinates ROC analysis, statistical optimization, impact simulation,
    and performance metrics calculation to provide comprehensive threshold
    optimization recommendations.
    """

    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.roc_analyzer = ROCAnalyzer(config.engine_config)
        self.statistical_optimizer = StatisticalOptimizer(config.engine_config)
        self.threshold_simulator = ThresholdSimulator(config.engine_config)
        self.performance_calculator = PerformanceMetricsCalculator(config.engine_config)

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive threshold optimization analysis.

        Args:
            request: Analysis request containing detector data and parameters

        Returns:
            AnalysisResult with threshold optimization recommendations
        """
        try:
            detector_id = request.metadata.get("detector_id", "unknown")
            current_threshold = request.metadata.get("current_threshold", 0.5)
            historical_data = request.metadata.get("historical_data", [])

            logger.info(
                "Starting threshold optimization analysis",
                detector_id=detector_id,
                current_threshold=current_threshold,
                data_points=len(historical_data),
            )

            # Perform ROC analysis
            roc_analysis = await self.roc_analyzer.analyze_roc_curve(
                detector_id, historical_data
            )

            # Perform statistical optimization
            optimization_result = await self.statistical_optimizer.optimize_threshold(
                detector_id=detector_id,
                historical_data=historical_data,
                objective=self._get_optimization_objective(request),
                constraints=request.metadata.get("constraints", {}),
            )

            # Calculate comprehensive performance metrics
            performance_metrics = (
                await self.performance_calculator.calculate_comprehensive_metrics(
                    detector_id=detector_id,
                    historical_data=historical_data,
                    threshold=current_threshold,
                )
            )

            # Generate threshold recommendation
            recommendation = await self._generate_threshold_recommendation(
                detector_id=detector_id,
                current_threshold=current_threshold,
                roc_analysis=roc_analysis,
                optimization_result=optimization_result,
                performance_metrics=performance_metrics,
                historical_data=historical_data,
            )

            # Create analysis result
            result = AnalysisResult(
                analysis_type="threshold_optimization",
                confidence=self._calculate_overall_confidence(
                    roc_analysis, optimization_result, performance_metrics
                ),
                findings=self._create_findings(recommendation),
                metadata={
                    "detector_id": detector_id,
                    "current_threshold": current_threshold,
                    "roc_analysis": roc_analysis,
                    "optimization_result": optimization_result,
                    "performance_metrics": performance_metrics,
                    "recommendation": (
                        recommendation.__dict__ if recommendation else None
                    ),
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                },
                processing_time=0.0,  # Will be set by framework
            )

            logger.info(
                "Threshold optimization analysis completed",
                detector_id=detector_id,
                recommended_threshold=(
                    recommendation.recommended_threshold if recommendation else None
                ),
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold optimization analysis failed",
                error=str(e),
                detector_id=request.metadata.get("detector_id", "unknown"),
            )
            return self._create_error_result(str(e))

    async def _generate_threshold_recommendation(
        self,
        detector_id: str,
        current_threshold: float,
        roc_analysis: Dict[str, Any],
        optimization_result: Dict[str, Any],
        performance_metrics: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> Optional[ThresholdRecommendation]:
        """Generate comprehensive threshold recommendation."""
        try:
            # Get optimal threshold from optimization
            optimal_threshold = optimization_result.get(
                "optimal_threshold", current_threshold
            )

            # Simulate impact of threshold change
            if abs(optimal_threshold - current_threshold) > 0.01:  # Significant change
                simulation_result = (
                    await self.threshold_simulator.simulate_threshold_impact(
                        detector_id=detector_id,
                        current_threshold=current_threshold,
                        proposed_threshold=optimal_threshold,
                        historical_data=historical_data,
                    )
                )
            else:
                simulation_result = {"risk_assessment": {"recommendation": "no_change"}}

            # Calculate expected improvement
            expected_improvement = self._calculate_expected_improvement(
                optimization_result, performance_metrics
            )

            # Generate rationale
            rationale = self._generate_recommendation_rationale(
                current_threshold, optimal_threshold, roc_analysis, optimization_result
            )

            # Calculate recommendation confidence
            confidence = self._calculate_recommendation_confidence(
                roc_analysis, optimization_result, simulation_result
            )

            return ThresholdRecommendation(
                detector_id=detector_id,
                current_threshold=current_threshold,
                recommended_threshold=optimal_threshold,
                expected_improvement=expected_improvement,
                confidence=confidence,
                rationale=rationale,
                impact_analysis=simulation_result,
            )

        except Exception as e:
            logger.error(f"Threshold recommendation generation failed: {e}")
            return None

    def _get_optimization_objective(
        self, request: AnalysisRequest
    ) -> OptimizationObjective:
        """Get optimization objective from request."""
        objective_str = request.metadata.get("optimization_objective", "balanced_pr")
        try:
            return OptimizationObjective(objective_str)
        except ValueError:
            logger.warning(f"Invalid optimization objective: {objective_str}")
            return OptimizationObjective.BALANCED_PRECISION_RECALL

    def _calculate_expected_improvement(
        self, optimization_result: Dict[str, Any], performance_metrics: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate expected improvement metrics."""
        try:
            optimal_performance = optimization_result.get("optimal_performance", {})
            current_metrics = performance_metrics.get("basic_metrics", {})

            return {
                "precision_improvement": (
                    optimal_performance.get("precision", 0)
                    - current_metrics.get("precision", 0)
                ),
                "recall_improvement": (
                    optimal_performance.get("recall", 0)
                    - current_metrics.get("recall", 0)
                ),
                "f1_improvement": (
                    optimal_performance.get("f1_score", 0)
                    - current_metrics.get("f1_score", 0)
                ),
                "false_positive_rate_change": (
                    optimal_performance.get("false_positive_rate", 0)
                    - current_metrics.get("false_positive_rate", 0)
                ),
            }

        except Exception as e:
            logger.error(f"Expected improvement calculation failed: {e}")
            return {}

    def _generate_recommendation_rationale(
        self,
        current_threshold: float,
        optimal_threshold: float,
        roc_analysis: Dict[str, Any],
        optimization_result: Dict[str, Any],
    ) -> str:
        """Generate human-readable rationale for recommendation."""
        try:
            threshold_change = optimal_threshold - current_threshold
            auc_score = roc_analysis.get("auc_score", 0)
            objective = optimization_result.get("optimization_objective", "unknown")

            if abs(threshold_change) < 0.01:
                return f"Current threshold ({current_threshold:.3f}) is already near-optimal for {objective} objective."

            direction = "increase" if threshold_change > 0 else "decrease"
            magnitude = "significant" if abs(threshold_change) > 0.1 else "moderate"

            rationale = f"Recommend {magnitude} {direction} in threshold from {current_threshold:.3f} to {optimal_threshold:.3f} "
            rationale += f"to optimize for {objective}. "

            if auc_score >= 0.8:
                rationale += f"ROC analysis shows good discriminative ability (AUC: {auc_score:.3f}). "
            elif auc_score >= 0.6:
                rationale += f"ROC analysis shows fair discriminative ability (AUC: {auc_score:.3f}). "
            else:
                rationale += f"ROC analysis shows limited discriminative ability (AUC: {auc_score:.3f}). "

            return rationale

        except Exception as e:
            logger.error(f"Rationale generation failed: {e}")
            return "Unable to generate recommendation rationale."

    def _calculate_recommendation_confidence(
        self,
        roc_analysis: Dict[str, Any],
        optimization_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
    ) -> float:
        """Calculate confidence in the threshold recommendation."""
        try:
            confidence_factors = []

            # ROC analysis confidence
            auc_score = roc_analysis.get("auc_score", 0)
            data_quality = roc_analysis.get("data_quality", {})
            sample_count = data_quality.get("total_samples", 0)

            if auc_score >= 0.8 and sample_count >= 100:
                confidence_factors.append(0.9)
            elif auc_score >= 0.7 and sample_count >= 50:
                confidence_factors.append(0.7)
            elif auc_score >= 0.6 and sample_count >= 20:
                confidence_factors.append(0.5)
            else:
                confidence_factors.append(0.3)

            # Optimization confidence
            optimization_success = optimization_result.get(
                "optimization_summary", {}
            ).get("optimization_success", False)
            if optimization_success:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)

            # Simulation confidence
            risk_level = simulation_result.get("risk_assessment", {}).get(
                "risk_level", "high"
            )
            if risk_level == "low":
                confidence_factors.append(0.9)
            elif risk_level == "medium":
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)

            # Calculate overall confidence as weighted average
            return (
                sum(confidence_factors) / len(confidence_factors)
                if confidence_factors
                else 0.5
            )

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _calculate_overall_confidence(
        self,
        roc_analysis: Dict[str, Any],
        optimization_result: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> float:
        """Calculate overall confidence in the analysis."""
        try:
            # Data quality factor
            data_quality = performance_metrics.get("data_quality", {})
            sample_count = data_quality.get("total_samples", 0)

            if sample_count >= 1000:
                data_confidence = 0.9
            elif sample_count >= 100:
                data_confidence = 0.7
            elif sample_count >= 30:
                data_confidence = 0.5
            else:
                data_confidence = 0.3

            # ROC analysis factor
            auc_score = roc_analysis.get("auc_score", 0)
            roc_confidence = min(1.0, auc_score * 1.2)  # Scale AUC to confidence

            # Optimization factor
            optimization_success = optimization_result.get(
                "optimization_summary", {}
            ).get("optimization_success", False)
            opt_confidence = 0.8 if optimization_success else 0.4

            # Weighted average
            weights = [0.4, 0.3, 0.3]  # Data, ROC, Optimization
            confidences = [data_confidence, roc_confidence, opt_confidence]

            return sum(w * c for w, c in zip(weights, confidences))

        except Exception as e:
            logger.error(f"Overall confidence calculation failed: {e}")
            return 0.5

    def _create_findings(
        self, recommendation: Optional[ThresholdRecommendation]
    ) -> List[SecurityFinding]:
        """Create security findings from threshold recommendation."""
        findings = []

        if recommendation:
            # Main recommendation finding
            findings.append(
                SecurityFinding(
                    finding_type="threshold_recommendation",
                    severity="medium",
                    confidence=recommendation.confidence,
                    description=recommendation.rationale,
                    location="detector_threshold",
                    metadata={
                        "current_threshold": recommendation.current_threshold,
                        "recommended_threshold": recommendation.recommended_threshold,
                        "expected_improvement": recommendation.expected_improvement,
                    },
                )
            )

            # Risk assessment findings
            risk_assessment = recommendation.impact_analysis.get("risk_assessment", {})
            risks = risk_assessment.get("risks", [])

            for risk in risks:
                findings.append(
                    SecurityFinding(
                        finding_type="threshold_risk",
                        severity=risk.get("severity", "low"),
                        confidence=0.7,
                        description=risk.get(
                            "description", "Threshold change risk identified"
                        ),
                        location="detector_threshold",
                        metadata={"risk_type": risk.get("type", "unknown")},
                    )
                )

        return findings

    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error analysis result."""
        return AnalysisResult(
            analysis_type="threshold_optimization",
            confidence=0.0,
            findings=[
                SecurityFinding(
                    finding_type="analysis_error",
                    severity="high",
                    confidence=1.0,
                    description=f"Threshold optimization analysis failed: {error_message}",
                    location="analysis_engine",
                    metadata={"error": error_message},
                )
            ],
            metadata={"error": error_message},
            processing_time=0.0,
        )
