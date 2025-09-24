"""
Dynamic Threshold Optimization Engine for statistical threshold analysis.

This engine implements sophisticated statistical methods to optimize detection
thresholds, minimize false positives while maintaining security coverage.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..domain import (
    AnalysisConfiguration,
    AnalysisResult,
    BaseAnalysisEngine,
    ThresholdPerformance,
    ThresholdRecommendation,
    SecurityFinding,
)
from ..domain.entities import AnalysisRequest

logger = logging.getLogger(__name__)


@dataclass
class ROCPoint:
    """Single point on ROC curve."""

    threshold: float
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for threshold analysis."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float


class OptimizationObjective(Enum):
    """Optimization objectives for threshold tuning."""

    MINIMIZE_FALSE_POSITIVES = "minimize_fp"
    MAXIMIZE_F1_SCORE = "maximize_f1"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    BALANCED_PRECISION_RECALL = "balanced_pr"
    MINIMIZE_COST = "minimize_cost"


class ROCAnalyzer:
    """
    Receiver Operating Characteristic curve analysis for threshold optimization.

    Analyzes detector performance across different threshold values to find
    optimal operating points using statistical methods.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.threshold_steps = self.config.get("threshold_steps", 100)
        self.min_threshold = self.config.get("min_threshold", 0.0)
        self.max_threshold = self.config.get("max_threshold", 1.0)

    async def analyze_roc_curve(
        self, detector_id: str, historical_data: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Analyze ROC curve for a detector using historical data.

        Args:
            detector_id: ID of the detector to analyze
            historical_data: Historical detection data with ground truth labels

        Returns:
            Dictionary containing ROC analysis results
        """
        try:
            if not historical_data:
                return self._create_empty_roc_analysis(detector_id)

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 10:  # Need sufficient data
                logger.warning(
                    f"Insufficient data for ROC analysis: {len(scores)} samples"
                )
                return self._create_insufficient_data_analysis(detector_id, len(scores))

            # Generate threshold range
            thresholds = self._generate_threshold_range(scores)

            # Calculate ROC points
            roc_points = []
            for threshold in thresholds:
                roc_point = self._calculate_roc_point(scores, labels, threshold)
                roc_points.append(roc_point)

            # Calculate AUC
            auc_score = self._calculate_auc(roc_points)

            # Find optimal threshold
            optimal_threshold = self._find_optimal_threshold(roc_points)

            # Generate analysis summary
            analysis_summary = self._generate_roc_summary(
                roc_points, auc_score, optimal_threshold
            )

            result = {
                "detector_id": detector_id,
                "roc_points": [self._roc_point_to_dict(point) for point in roc_points],
                "auc_score": auc_score,
                "optimal_threshold": optimal_threshold,
                "analysis_summary": analysis_summary,
                "data_quality": self._assess_data_quality(scores, labels),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "ROC analysis completed",
                detector_id=detector_id,
                auc_score=auc_score,
                optimal_threshold=optimal_threshold,
                data_points=len(scores),
            )

            return result

        except Exception as e:
            logger.error("ROC analysis failed", error=str(e), detector_id=detector_id)
            return self._create_error_analysis(detector_id, str(e))

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract confidence scores and ground truth labels from historical data."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                scores.append(score)
                labels.append(label)

            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid data point: {e}")
                continue

        return scores, labels

    def _generate_threshold_range(self, scores: List[float]) -> List[float]:
        """Generate threshold range based on score distribution."""
        if not scores:
            return np.linspace(
                self.min_threshold, self.max_threshold, self.threshold_steps
            ).tolist()

        # Use actual score range with some padding
        min_score = max(self.min_threshold, min(scores) - 0.1)
        max_score = min(self.max_threshold, max(scores) + 0.1)

        return np.linspace(min_score, max_score, self.threshold_steps).tolist()

    def _calculate_roc_point(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> ROCPoint:
        """Calculate single ROC point for given threshold."""
        try:
            # Make predictions based on threshold
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall/Sensitivity
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tpr
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            return ROCPoint(
                threshold=threshold,
                true_positive_rate=tpr,
                false_positive_rate=fpr,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
            )

        except Exception as e:
            logger.error(f"Error calculating ROC point for threshold {threshold}: {e}")
            return ROCPoint(
                threshold=threshold,
                true_positive_rate=0.0,
                false_positive_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
            )

    def _calculate_auc(self, roc_points: List[ROCPoint]) -> float:
        """Calculate Area Under Curve using trapezoidal rule."""
        try:
            if len(roc_points) < 2:
                return 0.0

            # Sort by false positive rate
            sorted_points = sorted(roc_points, key=lambda p: p.false_positive_rate)

            auc = 0.0
            for i in range(1, len(sorted_points)):
                # Trapezoidal rule
                width = (
                    sorted_points[i].false_positive_rate
                    - sorted_points[i - 1].false_positive_rate
                )
                height = (
                    sorted_points[i].true_positive_rate
                    + sorted_points[i - 1].true_positive_rate
                ) / 2
                auc += width * height

            return max(0.0, min(1.0, auc))

        except Exception as e:
            logger.error(f"AUC calculation failed: {e}")
            return 0.5

    def _find_optimal_threshold(self, roc_points: List[ROCPoint]) -> float:
        """Find optimal threshold using Youden's J statistic."""
        try:
            best_threshold = 0.5
            best_j_score = -1.0

            for point in roc_points:
                # Youden's J = Sensitivity + Specificity - 1
                specificity = 1.0 - point.false_positive_rate
                j_score = point.true_positive_rate + specificity - 1.0

                if j_score > best_j_score:
                    best_j_score = j_score
                    best_threshold = point.threshold

            return best_threshold

        except Exception as e:
            logger.error(f"Optimal threshold calculation failed: {e}")
            return 0.5

    def _generate_roc_summary(
        self, roc_points: List[ROCPoint], auc_score: float, optimal_threshold: float
    ) -> Dict[str, any]:
        """Generate summary of ROC analysis."""
        try:
            # Find point closest to optimal threshold
            optimal_point = min(
                roc_points, key=lambda p: abs(p.threshold - optimal_threshold)
            )

            return {
                "auc_interpretation": self._interpret_auc_score(auc_score),
                "optimal_performance": {
                    "threshold": optimal_threshold,
                    "precision": optimal_point.precision,
                    "recall": optimal_point.recall,
                    "f1_score": optimal_point.f1_score,
                    "false_positive_rate": optimal_point.false_positive_rate,
                },
                "performance_range": {
                    "min_fpr": min(p.false_positive_rate for p in roc_points),
                    "max_fpr": max(p.false_positive_rate for p in roc_points),
                    "min_tpr": min(p.true_positive_rate for p in roc_points),
                    "max_tpr": max(p.true_positive_rate for p in roc_points),
                },
                "threshold_sensitivity": self._calculate_threshold_sensitivity(
                    roc_points
                ),
            }

        except Exception as e:
            logger.error(f"ROC summary generation failed: {e}")
            return {"error": str(e)}

    def _interpret_auc_score(self, auc_score: float) -> str:
        """Interpret AUC score."""
        if auc_score >= 0.9:
            return "excellent"
        elif auc_score >= 0.8:
            return "good"
        elif auc_score >= 0.7:
            return "fair"
        elif auc_score >= 0.6:
            return "poor"
        else:
            return "fail"

    def _calculate_threshold_sensitivity(self, roc_points: List[ROCPoint]) -> float:
        """Calculate how sensitive performance is to threshold changes."""
        try:
            if len(roc_points) < 2:
                return 0.0

            # Calculate variance in F1 scores
            f1_scores = [p.f1_score for p in roc_points]
            f1_variance = np.var(f1_scores)

            # Higher variance = more sensitive to threshold changes
            return min(1.0, f1_variance * 10)  # Scale to 0-1

        except Exception:
            return 0.5

    def _assess_data_quality(
        self, scores: List[float], labels: List[int]
    ) -> Dict[str, any]:
        """Assess quality of data used for ROC analysis."""
        try:
            total_samples = len(scores)
            positive_samples = sum(labels)
            negative_samples = total_samples - positive_samples

            return {
                "total_samples": total_samples,
                "positive_samples": positive_samples,
                "negative_samples": negative_samples,
                "class_balance": (
                    positive_samples / total_samples if total_samples > 0 else 0
                ),
                "data_sufficiency": "sufficient" if total_samples >= 100 else "limited",
                "score_distribution": {
                    "min": min(scores) if scores else 0,
                    "max": max(scores) if scores else 0,
                    "mean": np.mean(scores) if scores else 0,
                    "std": np.std(scores) if scores else 0,
                },
            }

        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return {"error": str(e)}

    def _roc_point_to_dict(self, point: ROCPoint) -> Dict[str, any]:
        """Convert ROC point to dictionary."""
        return {
            "threshold": point.threshold,
            "true_positive_rate": point.true_positive_rate,
            "false_positive_rate": point.false_positive_rate,
            "precision": point.precision,
            "recall": point.recall,
            "f1_score": point.f1_score,
        }

    def _create_empty_roc_analysis(self, detector_id: str) -> Dict[str, any]:
        """Create empty ROC analysis result."""
        return {
            "detector_id": detector_id,
            "roc_points": [],
            "auc_score": 0.0,
            "optimal_threshold": 0.5,
            "analysis_summary": {"message": "No historical data available"},
            "data_quality": {"total_samples": 0},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_analysis(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, any]:
        """Create analysis result for insufficient data."""
        return {
            "detector_id": detector_id,
            "roc_points": [],
            "auc_score": 0.0,
            "optimal_threshold": 0.5,
            "analysis_summary": {
                "message": f"Insufficient data: {sample_count} samples"
            },
            "data_quality": {
                "total_samples": sample_count,
                "data_sufficiency": "insufficient",
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_analysis(
        self, detector_id: str, error_message: str
    ) -> Dict[str, any]:
        """Create analysis result for error cases."""
        return {
            "detector_id": detector_id,
            "roc_points": [],
            "auc_score": 0.0,
            "optimal_threshold": 0.5,
            "analysis_summary": {"error": error_message},
            "data_quality": {"error": error_message},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }


class StatisticalOptimizer:
    """
    Statistical optimizer using precision-recall optimization methods.

    Implements various optimization algorithms to find optimal thresholds
    based on different objectives and constraints.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.default_objective = OptimizationObjective(
            self.config.get("default_objective", "balanced_pr")
        )
        self.cost_matrix = self._load_cost_matrix()

    async def optimize_threshold(
        self,
        detector_id: str,
        historical_data: List[Dict[str, any]],
        objective: OptimizationObjective = None,
        constraints: Dict[str, float] = None,
    ) -> Dict[str, any]:
        """
        Optimize threshold using statistical methods.

        Args:
            detector_id: ID of the detector to optimize
            historical_data: Historical detection data
            objective: Optimization objective
            constraints: Performance constraints (e.g., min_recall=0.8)

        Returns:
            Dictionary containing optimization results
        """
        try:
            objective = objective or self.default_objective
            constraints = constraints or {}

            if not historical_data:
                return self._create_empty_optimization(detector_id)

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 10:
                return self._create_insufficient_data_optimization(
                    detector_id, len(scores)
                )

            # Generate candidate thresholds
            candidate_thresholds = self._generate_candidate_thresholds(scores)

            # Evaluate each threshold
            threshold_evaluations = []
            for threshold in candidate_thresholds:
                evaluation = self._evaluate_threshold(
                    scores, labels, threshold, objective
                )
                threshold_evaluations.append(evaluation)

            # Apply constraints
            feasible_evaluations = self._apply_constraints(
                threshold_evaluations, constraints
            )

            if not feasible_evaluations:
                logger.warning(
                    f"No feasible thresholds found for {detector_id} with given constraints"
                )
                feasible_evaluations = threshold_evaluations  # Use all if none feasible

            # Find optimal threshold
            optimal_evaluation = self._find_optimal_evaluation(
                feasible_evaluations, objective
            )

            # Generate optimization summary
            optimization_summary = self._generate_optimization_summary(
                threshold_evaluations, optimal_evaluation, objective, constraints
            )

            result = {
                "detector_id": detector_id,
                "optimization_objective": objective.value,
                "constraints_applied": constraints,
                "optimal_threshold": optimal_evaluation["threshold"],
                "optimal_performance": optimal_evaluation,
                "all_evaluations": threshold_evaluations,
                "feasible_count": len(feasible_evaluations),
                "optimization_summary": optimization_summary,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Threshold optimization completed",
                detector_id=detector_id,
                optimal_threshold=optimal_evaluation["threshold"],
                objective=objective.value,
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold optimization failed", error=str(e), detector_id=detector_id
            )
            return self._create_error_optimization(detector_id, str(e))

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract confidence scores and ground truth labels."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                scores.append(score)
                labels.append(label)

            except (ValueError, TypeError):
                continue

        return scores, labels

    def _generate_candidate_thresholds(self, scores: List[float]) -> List[float]:
        """Generate candidate thresholds for optimization."""
        if not scores:
            return np.linspace(0.0, 1.0, 50).tolist()

        # Use percentiles of actual scores plus some fixed points
        percentiles = np.percentile(scores, [5, 10, 25, 50, 75, 90, 95])
        fixed_points = [0.1, 0.3, 0.5, 0.7, 0.9]

        candidates = list(percentiles) + fixed_points

        # Remove duplicates and sort
        candidates = sorted(list(set(candidates)))

        return candidates

    def _evaluate_threshold(
        self,
        scores: List[float],
        labels: List[int],
        threshold: float,
        objective: OptimizationObjective,
    ) -> Dict[str, any]:
        """Evaluate a single threshold."""
        try:
            # Make predictions
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            # Calculate objective score
            objective_score = self._calculate_objective_score(
                precision, recall, f1_score, accuracy, fpr, fnr, objective
            )

            return {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "objective_score": objective_score,
            }

        except Exception as e:
            logger.error(f"Threshold evaluation failed for {threshold}: {e}")
            return {"threshold": threshold, "objective_score": 0.0, "error": str(e)}

    def _calculate_objective_score(
        self,
        precision: float,
        recall: float,
        f1_score: float,
        accuracy: float,
        fpr: float,
        fnr: float,
        objective: OptimizationObjective,
    ) -> float:
        """Calculate objective score based on optimization goal."""
        try:
            if objective == OptimizationObjective.MINIMIZE_FALSE_POSITIVES:
                return 1.0 - fpr  # Higher score = lower FPR

            elif objective == OptimizationObjective.MAXIMIZE_F1_SCORE:
                return f1_score

            elif objective == OptimizationObjective.MAXIMIZE_PRECISION:
                return precision

            elif objective == OptimizationObjective.MAXIMIZE_RECALL:
                return recall

            elif objective == OptimizationObjective.BALANCED_PRECISION_RECALL:
                # Harmonic mean of precision and recall (F1 score)
                return f1_score

            elif objective == OptimizationObjective.MINIMIZE_COST:
                # Use cost matrix if available
                fp_cost = self.cost_matrix.get("false_positive", 1.0)
                fn_cost = self.cost_matrix.get("false_negative", 1.0)
                total_cost = fpr * fp_cost + fnr * fn_cost
                return 1.0 / (1.0 + total_cost)  # Higher score = lower cost

            else:
                return f1_score  # Default to F1 score

        except Exception as e:
            logger.error(f"Objective score calculation failed: {e}")
            return 0.0

    def _apply_constraints(
        self, evaluations: List[Dict[str, any]], constraints: Dict[str, float]
    ) -> List[Dict[str, any]]:
        """Apply performance constraints to filter feasible thresholds."""
        if not constraints:
            return evaluations

        feasible = []

        for evaluation in evaluations:
            is_feasible = True

            for constraint_name, constraint_value in constraints.items():
                if constraint_name.startswith("min_"):
                    metric_name = constraint_name[4:]  # Remove 'min_' prefix
                    if evaluation.get(metric_name, 0) < constraint_value:
                        is_feasible = False
                        break

                elif constraint_name.startswith("max_"):
                    metric_name = constraint_name[4:]  # Remove 'max_' prefix
                    if evaluation.get(metric_name, 1) > constraint_value:
                        is_feasible = False
                        break

            if is_feasible:
                feasible.append(evaluation)

        return feasible

    def _find_optimal_evaluation(
        self, evaluations: List[Dict[str, any]], objective: OptimizationObjective
    ) -> Dict[str, any]:
        """Find evaluation with highest objective score."""
        if not evaluations:
            return {"threshold": 0.5, "objective_score": 0.0}

        return max(evaluations, key=lambda e: e.get("objective_score", 0.0))

    def _generate_optimization_summary(
        self,
        all_evaluations: List[Dict[str, any]],
        optimal_evaluation: Dict[str, any],
        objective: OptimizationObjective,
        constraints: Dict[str, float],
    ) -> Dict[str, any]:
        """Generate summary of optimization process."""
        try:
            objective_scores = [e.get("objective_score", 0) for e in all_evaluations]

            return {
                "optimization_success": optimal_evaluation.get("objective_score", 0)
                > 0,
                "objective_score_range": {
                    "min": min(objective_scores) if objective_scores else 0,
                    "max": max(objective_scores) if objective_scores else 0,
                    "mean": np.mean(objective_scores) if objective_scores else 0,
                },
                "threshold_sensitivity": (
                    np.std(objective_scores) if objective_scores else 0
                ),
                "constraints_satisfied": len(constraints) > 0,
                "performance_tradeoffs": self._analyze_performance_tradeoffs(
                    all_evaluations
                ),
            }

        except Exception as e:
            logger.error(f"Optimization summary generation failed: {e}")
            return {"error": str(e)}

    def _analyze_performance_tradeoffs(
        self, evaluations: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Analyze performance tradeoffs across different thresholds."""
        try:
            if not evaluations:
                return {}

            # Find best performance for each metric
            best_precision = max(evaluations, key=lambda e: e.get("precision", 0))
            best_recall = max(evaluations, key=lambda e: e.get("recall", 0))
            best_f1 = max(evaluations, key=lambda e: e.get("f1_score", 0))
            lowest_fpr = min(evaluations, key=lambda e: e.get("false_positive_rate", 1))

            return {
                "best_precision": {
                    "threshold": best_precision["threshold"],
                    "precision": best_precision.get("precision", 0),
                    "recall": best_precision.get("recall", 0),
                },
                "best_recall": {
                    "threshold": best_recall["threshold"],
                    "precision": best_recall.get("precision", 0),
                    "recall": best_recall.get("recall", 0),
                },
                "best_f1": {
                    "threshold": best_f1["threshold"],
                    "f1_score": best_f1.get("f1_score", 0),
                },
                "lowest_false_positive_rate": {
                    "threshold": lowest_fpr["threshold"],
                    "false_positive_rate": lowest_fpr.get("false_positive_rate", 1),
                },
            }

        except Exception as e:
            logger.error(f"Performance tradeoffs analysis failed: {e}")
            return {"error": str(e)}

    def _load_cost_matrix(self) -> Dict[str, float]:
        """Load cost matrix for cost-based optimization."""
        return self.config.get("cost_matrix", {
            "false_positive": 1.0,
            "false_negative": 5.0,  # False negatives typically more costly
            "true_positive": 0.0,
            "true_negative": 0.0,
        })

    def _create_empty_optimization(self, detector_id: str) -> Dict[str, any]:
        """Create empty optimization result."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "all_evaluations": [],
            "feasible_count": 0,
            "optimization_summary": {"message": "No historical data available"},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_optimization(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, any]:
        """Create optimization result for insufficient data."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "all_evaluations": [],
            "feasible_count": 0,
            "optimization_summary": {
                "message": f"Insufficient data: {sample_count} samples"
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_optimization(
        self, detector_id: str, error_message: str
    ) -> Dict[str, any]:
        """Create optimization result for error cases."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "all_evaluations": [],
            "feasible_count": 0,
            "optimization_summary": {"error": error_message},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }


class ThresholdSimulator:
    """
    Threshold simulator for impact prediction using historical data.
    
    Simulates the impact of threshold changes on detection performance
    using historical data and statistical modeling.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.simulation_periods = self.config.get("simulation_periods", 30)  # days
        self.confidence_level = self.config.get("confidence_level", 0.95)

    async def simulate_threshold_impact(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        historical_data: List[Dict[str, any]],
        time_horizon_days: int = 30,
    ) -> Dict[str, any]:
        """
        Simulate impact of changing threshold using historical data.

        Args:
            detector_id: ID of the detector
            current_threshold: Current threshold value
            proposed_threshold: Proposed new threshold value
            historical_data: Historical detection data
            time_horizon_days: Simulation time horizon in days

        Returns:
            Dictionary containing simulation results
        """
        try:
            if not historical_data:
                return self._create_empty_simulation(detector_id)

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 50:  # Need sufficient data for simulation
                return self._create_insufficient_data_simulation(
                    detector_id, len(scores)
                )

            # Calculate current performance
            current_performance = self._calculate_performance_metrics(
                scores, labels, current_threshold
            )

            # Calculate proposed performance
            proposed_performance = self._calculate_performance_metrics(
                scores, labels, proposed_threshold
            )

            # Calculate performance delta
            performance_delta = self._calculate_performance_delta(
                current_performance, proposed_performance
            )

            # Simulate volume impact
            volume_impact = self._simulate_volume_impact(
                scores, current_threshold, proposed_threshold, time_horizon_days
            )

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                scores, labels, proposed_threshold
            )

            # Generate impact scenarios
            impact_scenarios = self._generate_impact_scenarios(
                performance_delta, volume_impact, time_horizon_days
            )

            # Create simulation summary
            simulation_summary = self._create_simulation_summary(
                current_performance,
                proposed_performance,
                performance_delta,
                volume_impact,
            )

            result = {
                "detector_id": detector_id,
                "current_threshold": current_threshold,
                "proposed_threshold": proposed_threshold,
                "time_horizon_days": time_horizon_days,
                "current_performance": current_performance,
                "proposed_performance": proposed_performance,
                "performance_delta": performance_delta,
                "volume_impact": volume_impact,
                "confidence_intervals": confidence_intervals,
                "impact_scenarios": impact_scenarios,
                "simulation_summary": simulation_summary,
                "data_quality": self._assess_simulation_data_quality(scores, labels),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Threshold simulation completed",
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=proposed_threshold,
                expected_alert_change=volume_impact.get("alert_volume_change_pct", 0),
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold simulation failed", error=str(e), detector_id=detector_id
            )
            return self._create_error_simulation(detector_id, str(e))

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract confidence scores and ground truth labels."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                scores.append(score)
                labels.append(label)

            except (ValueError, TypeError):
                continue

        return scores, labels

    def _calculate_performance_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for a threshold."""
        try:
            # Make predictions
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            return PerformanceMetrics(
                threshold=threshold,
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                accuracy=accuracy,
                specificity=specificity,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
            )

        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return PerformanceMetrics(
                threshold=threshold,
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
            )

    def _calculate_performance_delta(
        self,
        current_performance: PerformanceMetrics,
        proposed_performance: PerformanceMetrics,
    ) -> Dict[str, any]:
        """Calculate performance changes between current and proposed thresholds."""
        try:
            return {
                "precision_change": proposed_performance.precision
                - current_performance.precision,
                "recall_change": proposed_performance.recall
                - current_performance.recall,
                "f1_score_change": proposed_performance.f1_score
                - current_performance.f1_score,
                "accuracy_change": proposed_performance.accuracy
                - current_performance.accuracy,
                "false_positive_change": proposed_performance.false_positives
                - current_performance.false_positives,
                "false_negative_change": proposed_performance.false_negatives
                - current_performance.false_negatives,
                "false_positive_rate_change": proposed_performance.false_positive_rate
                - current_performance.false_positive_rate,
                "false_negative_rate_change": proposed_performance.false_negative_rate
                - current_performance.false_negative_rate,
            }

        except Exception as e:
            logger.error(f"Performance delta calculation failed: {e}")
            return {"error": str(e)}

    def _simulate_volume_impact(
        self,
        scores: List[float],
        current_threshold: float,
        proposed_threshold: float,
        time_horizon_days: int,
    ) -> Dict[str, any]:
        """Simulate impact on alert volume."""
        try:
            # Calculate current and proposed alert volumes
            current_alerts = sum(1 for score in scores if score >= current_threshold)
            proposed_alerts = sum(1 for score in scores if score >= proposed_threshold)

            # Calculate volume change
            volume_change = proposed_alerts - current_alerts
            volume_change_pct = (
                (volume_change / current_alerts * 100) if current_alerts > 0 else 0
            )

            # Project to time horizon
            daily_volume_current = current_alerts / len(scores) if scores else 0
            daily_volume_proposed = proposed_alerts / len(scores) if scores else 0

            projected_current = daily_volume_current * time_horizon_days
            projected_proposed = daily_volume_proposed * time_horizon_days
            projected_change = projected_proposed - projected_current

            return {
                "current_alert_volume": current_alerts,
                "proposed_alert_volume": proposed_alerts,
                "alert_volume_change": volume_change,
                "alert_volume_change_pct": volume_change_pct,
                "daily_volume_current": daily_volume_current,
                "daily_volume_proposed": daily_volume_proposed,
                "projected_volume_change": projected_change,
                "time_horizon_days": time_horizon_days,
            }

        except Exception as e:
            logger.error(f"Volume impact simulation failed: {e}")
            return {"error": str(e)}

    def _calculate_confidence_intervals(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate confidence intervals for performance metrics."""
        try:
            # Use bootstrap sampling for confidence intervals
            n_bootstrap = 1000
            bootstrap_metrics = []

            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(scores), len(scores), replace=True)
                bootstrap_scores = [scores[i] for i in indices]
                bootstrap_labels = [labels[i] for i in indices]

                # Calculate metrics for bootstrap sample
                metrics = self._calculate_performance_metrics(
                    bootstrap_scores, bootstrap_labels, threshold
                )
                bootstrap_metrics.append(metrics)

            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            precision_values = [m.precision for m in bootstrap_metrics]
            recall_values = [m.recall for m in bootstrap_metrics]
            f1_values = [m.f1_score for m in bootstrap_metrics]

            return {
                "confidence_level": self.confidence_level,
                "precision_ci": {
                    "lower": np.percentile(precision_values, lower_percentile),
                    "upper": np.percentile(precision_values, upper_percentile),
                },
                "recall_ci": {
                    "lower": np.percentile(recall_values, lower_percentile),
                    "upper": np.percentile(recall_values, upper_percentile),
                },
                "f1_score_ci": {
                    "lower": np.percentile(f1_values, lower_percentile),
                    "upper": np.percentile(f1_values, upper_percentile),
                },
            }

        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return {"error": str(e)}

    def _generate_impact_scenarios(
        self,
        performance_delta: Dict[str, any],
        volume_impact: Dict[str, any],
        time_horizon_days: int,
    ) -> Dict[str, any]:
        """Generate best-case, worst-case, and likely impact scenarios."""
        try:
            # Base scenario (most likely)
            base_scenario = {
                "name": "most_likely",
                "description": "Expected impact based on historical data",
                "precision_change": performance_delta.get("precision_change", 0),
                "recall_change": performance_delta.get("recall_change", 0),
                "alert_volume_change": volume_impact.get("projected_volume_change", 0),
                "probability": 0.6,
            }

            # Best-case scenario (optimistic)
            best_case = {
                "name": "best_case",
                "description": "Optimistic impact scenario",
                "precision_change": performance_delta.get("precision_change", 0) * 1.2,
                "recall_change": max(
                    performance_delta.get("recall_change", 0) * 1.1, 0
                ),
                "alert_volume_change": volume_impact.get("projected_volume_change", 0)
                * 0.8,
                "probability": 0.2,
            }

            # Worst-case scenario (pessimistic)
            worst_case = {
                "name": "worst_case",
                "description": "Pessimistic impact scenario",
                "precision_change": performance_delta.get("precision_change", 0) * 0.8,
                "recall_change": performance_delta.get("recall_change", 0) * 0.9,
                "alert_volume_change": volume_impact.get("projected_volume_change", 0)
                * 1.2,
                "probability": 0.2,
            }

            return {
                "scenarios": [base_scenario, best_case, worst_case],
                "time_horizon_days": time_horizon_days,
                "scenario_methodology": "statistical_projection",
            }

        except Exception as e:
            logger.error(f"Impact scenario generation failed: {e}")
            return {"error": str(e)}

    def _create_simulation_summary(
        self,
        current_performance: PerformanceMetrics,
        proposed_performance: PerformanceMetrics,
        performance_delta: Dict[str, any],
        volume_impact: Dict[str, any],
    ) -> Dict[str, any]:
        """Create summary of simulation results."""
        try:
            # Determine overall impact assessment
            precision_improvement = performance_delta.get("precision_change", 0) > 0.01
            recall_improvement = performance_delta.get("recall_change", 0) > 0.01
            volume_reduction = volume_impact.get("alert_volume_change_pct", 0) < -5

            if precision_improvement and not (
                performance_delta.get("recall_change", 0) < -0.05
            ):
                impact_assessment = "positive"
            elif recall_improvement and not (
                performance_delta.get("precision_change", 0) < -0.05
            ):
                impact_assessment = "positive"
            elif volume_reduction and not (
                performance_delta.get("recall_change", 0) < -0.1
            ):
                impact_assessment = "positive"
            elif (
                abs(performance_delta.get("precision_change", 0)) < 0.02
                and abs(performance_delta.get("recall_change", 0)) < 0.02
            ):
                impact_assessment = "neutral"
            else:
                impact_assessment = "negative"

            return {
                "impact_assessment": impact_assessment,
                "key_changes": {
                    "precision": f"{performance_delta.get('precision_change', 0):.3f}",
                    "recall": f"{performance_delta.get('recall_change', 0):.3f}",
                    "alert_volume": f"{volume_impact.get('alert_volume_change_pct', 0):.1f}%",
                },
                "recommendations": self._generate_threshold_recommendations(
                    impact_assessment, performance_delta, volume_impact
                ),
                "risk_factors": self._identify_risk_factors(
                    performance_delta, volume_impact
                ),
            }

        except Exception as e:
            logger.error(f"Simulation summary creation failed: {e}")
            return {"error": str(e)}

    def _generate_threshold_recommendations(
        self,
        impact_assessment: str,
        performance_delta: Dict[str, any],
        volume_impact: Dict[str, any],
    ) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []

        if impact_assessment == "positive":
            recommendations.append("Threshold change is recommended")
            if volume_impact.get("alert_volume_change_pct", 0) < -10:
                recommendations.append("Significant alert volume reduction expected")
            if performance_delta.get("precision_change", 0) > 0.05:
                recommendations.append("Notable precision improvement expected")

        elif impact_assessment == "neutral":
            recommendations.append("Threshold change has minimal impact")
            recommendations.append("Consider other optimization approaches")

        else:  # negative
            recommendations.append("Threshold change not recommended")
            if performance_delta.get("recall_change", 0) < -0.1:
                recommendations.append("Significant recall degradation risk")
            if volume_impact.get("alert_volume_change_pct", 0) > 20:
                recommendations.append("Alert volume increase may overwhelm analysts")

        return recommendations

    def _identify_risk_factors(
        self, performance_delta: Dict[str, any], volume_impact: Dict[str, any]
    ) -> List[str]:
        """Identify potential risk factors from threshold change."""
        risk_factors = []

        if performance_delta.get("recall_change", 0) < -0.05:
            risk_factors.append("Potential increase in missed threats")

        if volume_impact.get("alert_volume_change_pct", 0) > 15:
            risk_factors.append("Analyst workload may increase significantly")

        if performance_delta.get("precision_change", 0) < -0.05:
            risk_factors.append("Potential increase in false positives")

        if abs(performance_delta.get("f1_score_change", 0)) > 0.1:
            risk_factors.append("Significant change in overall detection performance")

        return risk_factors

    def _assess_simulation_data_quality(
        self, scores: List[float], labels: List[int]
    ) -> Dict[str, any]:
        """Assess quality of data used for simulation."""
        try:
            total_samples = len(scores)
            positive_samples = sum(labels)
            negative_samples = total_samples - positive_samples

            return {
                "total_samples": total_samples,
                "positive_samples": positive_samples,
                "negative_samples": negative_samples,
                "class_balance": (
                    positive_samples / total_samples if total_samples > 0 else 0
                ),
                "data_sufficiency": (
                    "sufficient" if total_samples >= 100 else "limited"
                ),
                "simulation_reliability": (
                    "high"
                    if total_samples >= 500
                    else "medium"
                    if total_samples >= 100
                    else "low"
                ),
            }

        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return {"error": str(e)}

    def _create_empty_simulation(self, detector_id: str) -> Dict[str, any]:
        """Create empty simulation result."""
        return {
            "detector_id": detector_id,
            "current_threshold": 0.5,
            "proposed_threshold": 0.5,
            "simulation_summary": {"message": "No historical data available"},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_simulation(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, any]:
        """Create simulation result for insufficient data."""
        return {
            "detector_id": detector_id,
            "current_threshold": 0.5,
            "proposed_threshold": 0.5,
            "simulation_summary": {
                "message": f"Insufficient data for simulation: {sample_count} samples"
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_simulation(
        self, detector_id: str, error_message: str
    ) -> Dict[str, any]:
        """Create simulation result for error cases."""
        return {
            "detector_id": detector_id,
            "current_threshold": 0.5,
            "proposed_threshold": 0.5,
            "simulation_summary": {"error": error_message},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }


class PerformanceMetricsCalculator:
    """
    Performance metrics calculator for false positive/negative rate analysis.
    
    Provides comprehensive analysis of detector performance metrics with
    statistical significance testing and trend analysis.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.significance_level = self.config.get("significance_level", 0.05)
        self.min_sample_size = self.config.get("min_sample_size", 30)

    async def calculate_comprehensive_metrics(
        self,
        detector_id: str,
        historical_data: List[Dict[str, any]],
        threshold: float,
        time_window_days: int = 30,
    ) -> Dict[str, any]:
        """
        Calculate comprehensive performance metrics for a detector.

        Args:
            detector_id: ID of the detector
            historical_data: Historical detection data with ground truth
            threshold: Threshold value to analyze
            time_window_days: Time window for analysis

        Returns:
            Dictionary containing comprehensive metrics analysis
        """
        try:
            if not historical_data:
                return self._create_empty_metrics(detector_id)

            # Extract scores and labels
            scores, labels, timestamps = self._extract_data_with_timestamps(
                historical_data
            )

            if len(scores) < self.min_sample_size:
                return self._create_insufficient_data_metrics(
                    detector_id, len(scores)
                )

            # Calculate basic performance metrics
            basic_metrics = self._calculate_basic_metrics(scores, labels, threshold)

            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(
                scores, labels, threshold
            )

            # Calculate statistical significance
            statistical_analysis = self._calculate_statistical_significance(
                scores, labels, threshold
            )

            # Analyze temporal trends
            temporal_analysis = self._analyze_temporal_trends(
                scores, labels, timestamps, threshold, time_window_days
            )

            # Calculate confidence intervals
            confidence_intervals = self._calculate_metric_confidence_intervals(
                scores, labels, threshold
            )

            # Generate performance insights
            performance_insights = self._generate_performance_insights(
                basic_metrics, advanced_metrics, temporal_analysis
            )

            result = {
                "detector_id": detector_id,
                "threshold": threshold,
                "time_window_days": time_window_days,
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "statistical_analysis": statistical_analysis,
                "temporal_analysis": temporal_analysis,
                "confidence_intervals": confidence_intervals,
                "performance_insights": performance_insights,
                "data_quality": self._assess_metrics_data_quality(
                    scores, labels, timestamps
                ),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Performance metrics calculation completed",
                detector_id=detector_id,
                threshold=threshold,
                precision=basic_metrics.get("precision", 0),
                recall=basic_metrics.get("recall", 0),
                f1_score=basic_metrics.get("f1_score", 0),
            )

            return result

        except Exception as e:
            logger.error(
                "Performance metrics calculation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_metrics(detector_id, str(e))

    def _extract_data_with_timestamps(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int], List[datetime]]:
        """Extract scores, labels, and timestamps from historical data."""
        scores = []
        labels = []
        timestamps = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                # Parse timestamp
                timestamp_str = item.get("timestamp", item.get("created_at"))
                if timestamp_str:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )
                    else:
                        timestamp = timestamp_str
                else:
                    timestamp = datetime.utcnow()

                scores.append(score)
                labels.append(label)
                timestamps.append(timestamp)

            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid data point: {e}")
                continue

        return scores, labels, timestamps

    def _calculate_basic_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate basic performance metrics."""
        try:
            # Make predictions
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            # Calculate rates
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            tpr = recall  # True positive rate = recall

            return {
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "specificity": specificity,
                "true_positive_rate": tpr,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "positive_predictive_value": precision,
                "negative_predictive_value": tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            }

        except Exception as e:
            logger.error(f"Basic metrics calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_advanced_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate advanced performance metrics."""
        try:
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Matthews Correlation Coefficient
            mcc = self._calculate_mcc(predictions, labels)

            # Balanced accuracy
            balanced_accuracy = self._calculate_balanced_accuracy(predictions, labels)

            # Cohen's Kappa
            cohens_kappa = self._calculate_cohens_kappa(predictions, labels)

            # Informedness (Bookmaker Informedness)
            informedness = self._calculate_informedness(predictions, labels)

            # Markedness
            markedness = self._calculate_markedness(predictions, labels)

            # Diagnostic odds ratio
            diagnostic_odds_ratio = self._calculate_diagnostic_odds_ratio(
                predictions, labels
            )

            return {
                "matthews_correlation_coefficient": mcc,
                "balanced_accuracy": balanced_accuracy,
                "cohens_kappa": cohens_kappa,
                "informedness": informedness,
                "markedness": markedness,
                "diagnostic_odds_ratio": diagnostic_odds_ratio,
                "geometric_mean": self._calculate_geometric_mean(predictions, labels),
                "fowlkes_mallows_index": self._calculate_fowlkes_mallows_index(
                    predictions, labels
                ),
            }

        except Exception as e:
            logger.error(f"Advanced metrics calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_mcc(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate Matthews Correlation Coefficient."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            numerator = (tp * tn) - (fp * fn)
            denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

            return numerator / denominator if denominator > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_balanced_accuracy(
        self, predictions: List[int], labels: List[int]
    ) -> float:
        """Calculate balanced accuracy."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            return (sensitivity + specificity) / 2

        except Exception:
            return 0.0

    def _calculate_cohens_kappa(
        self, predictions: List[int], labels: List[int]
    ) -> float:
        """Calculate Cohen's Kappa coefficient."""
        try:
            n = len(predictions)
            if n == 0:
                return 0.0

            # Observed agreement
            observed_agreement = sum(
                1 for p, l in zip(predictions, labels) if p == l
            ) / n

            # Expected agreement
            pred_pos = sum(predictions) / n
            pred_neg = 1 - pred_pos
            label_pos = sum(labels) / n
            label_neg = 1 - label_pos

            expected_agreement = (pred_pos * label_pos) + (pred_neg * label_neg)

            return (
                (observed_agreement - expected_agreement)
                / (1 - expected_agreement)
                if expected_agreement < 1
                else 0.0
            )

        except Exception:
            return 0.0

    def _calculate_informedness(
        self, predictions: List[int], labels: List[int]
    ) -> float:
        """Calculate Informedness (Bookmaker Informedness)."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            return sensitivity + specificity - 1

        except Exception:
            return 0.0

    def _calculate_markedness(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate Markedness."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive predictive value
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

            return ppv + npv - 1

        except Exception:
            return 0.0

    def _calculate_diagnostic_odds_ratio(
        self, predictions: List[int], labels: List[int]
    ) -> float:
        """Calculate Diagnostic Odds Ratio."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            # Add small constant to avoid division by zero
            epsilon = 0.5
            return ((tp + epsilon) * (tn + epsilon)) / (
                (fp + epsilon) * (fn + epsilon)
            )

        except Exception:
            return 1.0

    def _calculate_geometric_mean(
        self, predictions: List[int], labels: List[int]
    ) -> float:
        """Calculate Geometric Mean of sensitivity and specificity."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            return (sensitivity * specificity) ** 0.5

        except Exception:
            return 0.0

    def _calculate_fowlkes_mallows_index(
        self, predictions: List[int], labels: List[int]
    ) -> float:
        """Calculate Fowlkes-Mallows Index."""
        try:
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            return (precision * recall) ** 0.5

        except Exception:
            return 0.0", 0),
                    "recall": lowest_fpr.get("recall", 0),
                },
            }

        except Exception as e:
            logger.error(f"Performance tradeoffs analysis failed: {e}")
            return {}

    def _load_cost_matrix(self) -> Dict[str, float]:
        """Load cost matrix for cost-based optimization."""
        return self.config.get(
            "cost_matrix",
            {
                "false_positive": 1.0,
                "false_negative": 5.0,  # False negatives typically more costly
                "true_positive": 0.0,
                "true_negative": 0.0,
            },
        )

    def _create_empty_optimization(self, detector_id: str) -> Dict[str, any]:
        """Create empty optimization result."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "optimization_summary": {"message": "No historical data available"},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_optimization(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, any]:
        """Create optimization result for insufficient data."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "optimization_summary": {
                "message": f"Insufficient data: {sample_count} samples"
            },
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_optimization(
        self, detector_id: str, error_message: str
    ) -> Dict[str, any]:
        """Create optimization result for error cases."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "optimization_summary": {"error": error_message},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }


class ThresholdSimulator:
    """
    Threshold simulator for impact prediction using historical data.

    Simulates the impact of threshold changes on system performance
    using historical data and statistical modeling.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.simulation_periods = self.config.get("simulation_periods", 30)  # days
        self.confidence_level = self.config.get("confidence_level", 0.95)

    async def simulate_threshold_impact(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        historical_data: List[Dict[str, any]],
    ) -> Dict[str, any]:
        """
        Simulate impact of changing threshold from current to proposed value.

        Args:
            detector_id: ID of the detector
            current_threshold: Current threshold value
            proposed_threshold: Proposed new threshold value
            historical_data: Historical detection data

        Returns:
            Dictionary containing simulation results
        """
        try:
            if not historical_data:
                return self._create_empty_simulation(
                    detector_id, current_threshold, proposed_threshold
                )

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 20:
                return self._create_insufficient_data_simulation(
                    detector_id, current_threshold, proposed_threshold, len(scores)
                )

            # Calculate current performance
            current_performance = self._calculate_performance_metrics(
                scores, labels, current_threshold
            )

            # Calculate proposed performance
            proposed_performance = self._calculate_performance_metrics(
                scores, labels, proposed_threshold
            )

            # Calculate impact metrics
            impact_metrics = self._calculate_impact_metrics(
                current_performance, proposed_performance
            )

            # Simulate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                scores, labels, proposed_threshold
            )

            # Estimate operational impact
            operational_impact = self._estimate_operational_impact(
                impact_metrics, len(historical_data)
            )

            # Generate simulation summary
            simulation_summary = self._generate_simulation_summary(
                impact_metrics, operational_impact, confidence_intervals
            )

            result = {
                "detector_id": detector_id,
                "current_threshold": current_threshold,
                "proposed_threshold": proposed_threshold,
                "current_performance": current_performance,
                "proposed_performance": proposed_performance,
                "impact_metrics": impact_metrics,
                "confidence_intervals": confidence_intervals,
                "operational_impact": operational_impact,
                "simulation_summary": simulation_summary,
                "simulation_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Threshold impact simulation completed",
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=proposed_threshold,
                fp_reduction=impact_metrics.get("false_positive_reduction", 0),
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold simulation failed", error=str(e), detector_id=detector_id
            )
            return self._create_error_simulation(
                detector_id, current_threshold, proposed_threshold, str(e)
            )

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract confidence scores and ground truth labels."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                scores.append(score)
                labels.append(label)

            except (ValueError, TypeError):
                continue

        return scores, labels

    def _calculate_performance_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate performance metrics for a given threshold."""
        try:
            # Make predictions
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            )
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            return {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "total_predictions": tp + fp,
                "total_samples": len(scores),
            }

        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {"threshold": threshold, "error": str(e)}

    def _calculate_impact_metrics(
        self, current: Dict[str, any], proposed: Dict[str, any]
    ) -> Dict[str, any]:
        """Calculate impact metrics comparing current vs proposed performance."""
        try:
            # Calculate absolute changes
            fp_change = proposed.get("false_positives", 0) - current.get(
                "false_positives", 0
            )
            fn_change = proposed.get("false_negatives", 0) - current.get(
                "false_negatives", 0
            )

            # Calculate relative changes
            current_fp = current.get("false_positives", 1)
            current_fn = current.get("false_negatives", 1)

            fp_reduction = -fp_change / current_fp if current_fp > 0 else 0.0
            fn_increase = fn_change / current_fn if current_fn > 0 else 0.0

            # Calculate metric changes
            precision_change = proposed.get("precision", 0) - current.get(
                "precision", 0
            )
            recall_change = proposed.get("recall", 0) - current.get("recall", 0)
            f1_change = proposed.get("f1_score", 0) - current.get("f1_score", 0)

            return {
                "false_positive_change": fp_change,
                "false_negative_change": fn_change,
                "false_positive_reduction": fp_reduction,
                "false_negative_increase": fn_increase,
                "precision_change": precision_change,
                "recall_change": recall_change,
                "f1_score_change": f1_change,
                "net_benefit": self._calculate_net_benefit(fp_reduction, fn_increase),
                "risk_level": self._assess_change_risk(fp_reduction, fn_increase),
            }

        except Exception as e:
            logger.error(f"Impact metrics calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_net_benefit(self, fp_reduction: float, fn_increase: float) -> float:
        """Calculate net benefit of threshold change."""
        try:
            # Weight false positive reduction vs false negative increase
            fp_weight = 1.0  # Cost of false positive
            fn_weight = 3.0  # Cost of false negative (typically higher)

            net_benefit = (fp_reduction * fp_weight) - (fn_increase * fn_weight)
            return net_benefit

        except Exception:
            return 0.0

    def _assess_change_risk(self, fp_reduction: float, fn_increase: float) -> str:
        """Assess risk level of threshold change."""
        try:
            if fn_increase > 0.2:  # 20% increase in false negatives
                return "high"
            elif fn_increase > 0.1:  # 10% increase in false negatives
                return "medium"
            elif fp_reduction > 0.3:  # 30% reduction in false positives
                return "low"
            else:
                return "medium"

        except Exception:
            return "unknown"

    def _calculate_confidence_intervals(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate confidence intervals for performance metrics."""
        try:
            # Bootstrap sampling for confidence intervals
            n_bootstrap = 1000
            bootstrap_metrics = []

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(scores), size=len(scores), replace=True)
                bootstrap_scores = [scores[i] for i in indices]
                bootstrap_labels = [labels[i] for i in indices]

                # Calculate metrics for bootstrap sample
                metrics = self._calculate_performance_metrics(
                    bootstrap_scores, bootstrap_labels, threshold
                )
                bootstrap_metrics.append(metrics)

            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            confidence_intervals = {}
            for metric in ["precision", "recall", "f1_score", "false_positive_rate"]:
                values = [m.get(metric, 0) for m in bootstrap_metrics if metric in m]
                if values:
                    confidence_intervals[metric] = {
                        "lower": np.percentile(values, lower_percentile),
                        "upper": np.percentile(values, upper_percentile),
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }

            return confidence_intervals

        except Exception as e:
            logger.error(f"Confidence intervals calculation failed: {e}")
            return {}

    def _estimate_operational_impact(
        self, impact_metrics: Dict[str, any], sample_size: int
    ) -> Dict[str, any]:
        """Estimate operational impact of threshold change."""
        try:
            # Estimate daily volumes based on sample size
            daily_volume = (
                sample_size * (24 * 60 * 60) / (self.simulation_periods * 24 * 60 * 60)
            )

            # Calculate daily impact
            fp_reduction = impact_metrics.get("false_positive_reduction", 0)
            fn_increase = impact_metrics.get("false_negative_increase", 0)

            daily_fp_reduction = daily_volume * fp_reduction
            daily_fn_increase = daily_volume * fn_increase

            # Estimate cost impact (placeholder values)
            fp_cost_per_incident = 10  # Cost to investigate false positive
            fn_cost_per_incident = 100  # Cost of missing true positive

            daily_cost_savings = daily_fp_reduction * fp_cost_per_incident
            daily_cost_increase = daily_fn_increase * fn_cost_per_incident
            net_daily_cost_impact = daily_cost_savings - daily_cost_increase

            return {
                "estimated_daily_volume": daily_volume,
                "daily_false_positive_reduction": daily_fp_reduction,
                "daily_false_negative_increase": daily_fn_increase,
                "daily_cost_savings": daily_cost_savings,
                "daily_cost_increase": daily_cost_increase,
                "net_daily_cost_impact": net_daily_cost_impact,
                "monthly_cost_impact": net_daily_cost_impact * 30,
                "payback_period_days": self._calculate_payback_period(
                    net_daily_cost_impact
                ),
            }

        except Exception as e:
            logger.error(f"Operational impact estimation failed: {e}")
            return {}

    def _calculate_payback_period(self, net_daily_impact: float) -> Optional[int]:
        """Calculate payback period for threshold change."""
        try:
            if net_daily_impact <= 0:
                return None  # No payback if not beneficial

            # Assume some implementation cost
            implementation_cost = 1000  # Placeholder

            return int(implementation_cost / net_daily_impact)

        except Exception:
            return None

    def _generate_simulation_summary(
        self,
        impact_metrics: Dict[str, any],
        operational_impact: Dict[str, any],
        confidence_intervals: Dict[str, any],
    ) -> Dict[str, any]:
        """Generate summary of simulation results."""
        try:
            fp_reduction = impact_metrics.get("false_positive_reduction", 0)
            fn_increase = impact_metrics.get("false_negative_increase", 0)
            risk_level = impact_metrics.get("risk_level", "unknown")

            return {
                "recommendation": self._generate_recommendation(
                    fp_reduction, fn_increase, risk_level
                ),
                "key_benefits": self._identify_key_benefits(
                    impact_metrics, operational_impact
                ),
                "key_risks": self._identify_key_risks(
                    impact_metrics, operational_impact
                ),
                "confidence_level": self.confidence_level,
                "simulation_quality": self._assess_simulation_quality(
                    confidence_intervals
                ),
            }

        except Exception as e:
            logger.error(f"Simulation summary generation failed: {e}")
            return {"error": str(e)}

    def _generate_recommendation(
        self, fp_reduction: float, fn_increase: float, risk_level: str
    ) -> str:
        """Generate recommendation based on simulation results."""
        if fp_reduction > 0.2 and fn_increase < 0.1 and risk_level in ["low", "medium"]:
            return "recommended"
        elif fp_reduction > 0.1 and fn_increase < 0.05:
            return "cautiously_recommended"
        elif fn_increase > 0.2 or risk_level == "high":
            return "not_recommended"
        else:
            return "neutral"

    def _identify_key_benefits(
        self, impact_metrics: Dict[str, any], operational_impact: Dict[str, any]
    ) -> List[str]:
        """Identify key benefits of threshold change."""
        benefits = []

        fp_reduction = impact_metrics.get("false_positive_reduction", 0)
        if fp_reduction > 0.1:
            benefits.append(f"Reduces false positives by {fp_reduction:.1%}")

        cost_savings = operational_impact.get("daily_cost_savings", 0)
        if cost_savings > 0:
            benefits.append(f"Saves approximately ${cost_savings:.0f} daily")

        precision_change = impact_metrics.get("precision_change", 0)
        if precision_change > 0.05:
            benefits.append(f"Improves precision by {precision_change:.1%}")

        return benefits

    def _identify_key_risks(
        self, impact_metrics: Dict[str, any], operational_impact: Dict[str, any]
    ) -> List[str]:
        """Identify key risks of threshold change."""
        risks = []

        fn_increase = impact_metrics.get("false_negative_increase", 0)
        if fn_increase > 0.05:
            risks.append(f"Increases false negatives by {fn_increase:.1%}")

        recall_change = impact_metrics.get("recall_change", 0)
        if recall_change < -0.05:
            risks.append(f"Reduces recall by {abs(recall_change):.1%}")

        cost_increase = operational_impact.get("daily_cost_increase", 0)
        if cost_increase > 0:
            risks.append(f"Increases costs by approximately ${cost_increase:.0f} daily")

        return risks

    def _assess_simulation_quality(self, confidence_intervals: Dict[str, any]) -> str:
        """Assess quality of simulation based on confidence intervals."""
        try:
            if not confidence_intervals:
                return "low"

            # Check width of confidence intervals
            avg_width = 0
            count = 0

            for metric, ci in confidence_intervals.items():
                if "lower" in ci and "upper" in ci:
                    width = ci["upper"] - ci["lower"]
                    avg_width += width
                    count += 1

            if count > 0:
                avg_width /= count

                if avg_width < 0.1:
                    return "high"
                elif avg_width < 0.2:
                    return "medium"
                else:
                    return "low"

            return "medium"

        except Exception:
            return "unknown"

    def _create_empty_simulation(
        self, detector_id: str, current_threshold: float, proposed_threshold: float
    ) -> Dict[str, any]:
        """Create empty simulation result."""
        return {
            "detector_id": detector_id,
            "current_threshold": current_threshold,
            "proposed_threshold": proposed_threshold,
            "simulation_summary": {"message": "No historical data available"},
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_simulation(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        sample_count: int,
    ) -> Dict[str, any]:
        """Create simulation result for insufficient data."""
        return {
            "detector_id": detector_id,
            "current_threshold": current_threshold,
            "proposed_threshold": proposed_threshold,
            "simulation_summary": {
                "message": f"Insufficient data: {sample_count} samples"
            },
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_simulation(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        error_message: str,
    ) -> Dict[str, any]:
        """Create simulation result for error cases."""
        return {
            "detector_id": detector_id,
            "current_threshold": current_threshold,
            "proposed_threshold": proposed_threshold,
            "simulation_summary": {"error": error_message},
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }


class PerformanceMetricsCalculator:
    """
    Performance metrics calculator for false positive/negative rate analysis.

    Calculates comprehensive performance metrics and provides detailed
    analysis of detector performance characteristics.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.metrics_to_calculate = self.config.get(
            "metrics",
            [
                "precision",
                "recall",
                "f1_score",
                "accuracy",
                "specificity",
                "false_positive_rate",
                "false_negative_rate",
                "matthews_correlation",
            ],
        )

    async def calculate_comprehensive_metrics(
        self, detector_id: str, historical_data: List[Dict[str, any]], threshold: float
    ) -> Dict[str, any]:
        """
        Calculate comprehensive performance metrics for a detector at given threshold.

        Args:
            detector_id: ID of the detector
            historical_data: Historical detection data with ground truth
            threshold: Threshold value to evaluate

        Returns:
            Dictionary containing comprehensive performance metrics
        """
        try:
            if not historical_data:
                return self._create_empty_metrics(detector_id, threshold)

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 5:
                return self._create_insufficient_data_metrics(
                    detector_id, threshold, len(scores)
                )

            # Calculate basic metrics
            basic_metrics = self._calculate_basic_metrics(scores, labels, threshold)

            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(
                scores, labels, threshold
            )

            # Calculate statistical significance
            statistical_significance = self._calculate_statistical_significance(
                basic_metrics, len(scores)
            )

            # Assess metric reliability
            reliability_assessment = self._assess_metric_reliability(
                basic_metrics, advanced_metrics, len(scores)
            )

            # Generate performance insights
            performance_insights = self._generate_performance_insights(
                basic_metrics, advanced_metrics, statistical_significance
            )

            result = {
                "detector_id": detector_id,
                "threshold": threshold,
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "statistical_significance": statistical_significance,
                "reliability_assessment": reliability_assessment,
                "performance_insights": performance_insights,
                "sample_size": len(scores),
                "calculation_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Performance metrics calculated",
                detector_id=detector_id,
                threshold=threshold,
                f1_score=basic_metrics.get("f1_score", 0),
                sample_size=len(scores),
            )

            return result

        except Exception as e:
            logger.error(
                "Performance metrics calculation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_metrics(detector_id, threshold, str(e))

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract confidence scores and ground truth labels."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                scores.append(score)
                labels.append(label)

            except (ValueError, TypeError):
                continue

        return scores, labels

    def _calculate_basic_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate basic performance metrics."""
        try:
            # Make predictions
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            )
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            return {
                "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "specificity": specificity,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "true_positive_rate": recall,
                "positive_predictions": tp + fp,
                "negative_predictions": tn + fn,
                "total_samples": len(scores),
            }

        except Exception as e:
            logger.error(f"Basic metrics calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_advanced_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate advanced performance metrics."""
        try:
            basic_metrics = self._calculate_basic_metrics(scores, labels, threshold)
            cm = basic_metrics.get("confusion_matrix", {})

            tp, fp, tn, fn = (
                cm.get("tp", 0),
                cm.get("fp", 0),
                cm.get("tn", 0),
                cm.get("fn", 0),
            )

            # Matthews Correlation Coefficient
            mcc = self._calculate_matthews_correlation(tp, fp, tn, fn)

            # Balanced accuracy
            balanced_accuracy = (
                basic_metrics.get("recall", 0) + basic_metrics.get("specificity", 0)
            ) / 2

            # Positive/Negative Predictive Values
            ppv = basic_metrics.get("precision", 0)  # Same as precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            # Likelihood ratios
            lr_positive = (
                basic_metrics.get("recall", 0)
                / basic_metrics.get("false_positive_rate", 1)
                if basic_metrics.get("false_positive_rate", 1) > 0
                else float("inf")
            )
            lr_negative = (
                basic_metrics.get("false_negative_rate", 1)
                / basic_metrics.get("specificity", 1)
                if basic_metrics.get("specificity", 1) > 0
                else 0
            )

            # Diagnostic odds ratio
            dor = lr_positive / lr_negative if lr_negative > 0 else float("inf")

            # Informedness and Markedness
            informedness = (
                basic_metrics.get("recall", 0) + basic_metrics.get("specificity", 0) - 1
            )
            markedness = ppv + npv - 1

            return {
                "matthews_correlation_coefficient": mcc,
                "balanced_accuracy": balanced_accuracy,
                "positive_predictive_value": ppv,
                "negative_predictive_value": npv,
                "positive_likelihood_ratio": (
                    lr_positive if lr_positive != float("inf") else 999.0
                ),
                "negative_likelihood_ratio": lr_negative,
                "diagnostic_odds_ratio": dor if dor != float("inf") else 999.0,
                "informedness": informedness,
                "markedness": markedness,
                "geometric_mean": np.sqrt(
                    basic_metrics.get("recall", 0) * basic_metrics.get("specificity", 0)
                ),
            }

        except Exception as e:
            logger.error(f"Advanced metrics calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_matthews_correlation(
        self, tp: int, fp: int, tn: int, fn: int
    ) -> float:
        """Calculate Matthews Correlation Coefficient."""
        try:
            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

            if denominator == 0:
                return 0.0

            return numerator / denominator

        except Exception:
            return 0.0

    def _calculate_statistical_significance(
        self, basic_metrics: Dict[str, any], sample_size: int
    ) -> Dict[str, any]:
        """Calculate statistical significance of metrics."""
        try:
            # Simple significance assessment based on sample size and performance
            f1_score = basic_metrics.get("f1_score", 0)

            # Larger samples and better performance = higher significance
            significance_score = min(1.0, (sample_size / 100) * f1_score)

            return {
                "significance_score": significance_score,
                "sample_size_adequacy": "adequate" if sample_size >= 100 else "limited",
                "performance_significance": (
                    "significant"
                    if f1_score > 0.7
                    else "moderate" if f1_score > 0.5 else "low"
                ),
                "overall_significance": (
                    "high"
                    if significance_score > 0.7
                    else "medium" if significance_score > 0.4 else "low"
                ),
            }

        except Exception as e:
            logger.error(f"Statistical significance calculation failed: {e}")
            return {"error": str(e)}

    def _assess_metric_reliability(
        self,
        basic_metrics: Dict[str, any],
        advanced_metrics: Dict[str, any],
        sample_size: int,
    ) -> Dict[str, any]:
        """Assess reliability of calculated metrics."""
        try:
            reliability_factors = {}

            # Sample size factor
            reliability_factors["sample_size"] = min(1.0, sample_size / 200)

            # Balance factor (balanced datasets are more reliable)
            cm = basic_metrics.get("confusion_matrix", {})
            total_positive = cm.get("tp", 0) + cm.get("fn", 0)
            total_negative = cm.get("tn", 0) + cm.get("fp", 0)
            total_samples = total_positive + total_negative

            if total_samples > 0:
                balance_ratio = min(total_positive, total_negative) / total_samples
                reliability_factors["class_balance"] = balance_ratio * 2  # Scale to 0-1
            else:
                reliability_factors["class_balance"] = 0.0

            # Performance consistency factor
            f1_score = basic_metrics.get("f1_score", 0)
            mcc = advanced_metrics.get("matthews_correlation_coefficient", 0)
            consistency = 1.0 - abs(f1_score - abs(mcc))  # F1 and MCC should be similar
            reliability_factors["performance_consistency"] = max(0.0, consistency)

            # Overall reliability
            overall_reliability = sum(reliability_factors.values()) / len(
                reliability_factors
            )

            return {
                "reliability_factors": reliability_factors,
                "overall_reliability": overall_reliability,
                "reliability_level": (
                    "high"
                    if overall_reliability > 0.7
                    else "medium" if overall_reliability > 0.4 else "low"
                ),
            }

        except Exception as e:
            logger.error(f"Metric reliability assessment failed: {e}")
            return {"error": str(e)}

    def _generate_performance_insights(
        self,
        basic_metrics: Dict[str, any],
        advanced_metrics: Dict[str, any],
        statistical_significance: Dict[str, any],
    ) -> List[str]:
        """Generate insights about detector performance."""
        insights = []

        try:
            f1_score = basic_metrics.get("f1_score", 0)
            precision = basic_metrics.get("precision", 0)
            recall = basic_metrics.get("recall", 0)
            fpr = basic_metrics.get("false_positive_rate", 0)

            # Performance level insights
            if f1_score > 0.8:
                insights.append("Excellent overall performance")
            elif f1_score > 0.6:
                insights.append("Good overall performance")
            elif f1_score > 0.4:
                insights.append("Moderate performance - optimization recommended")
            else:
                insights.append("Poor performance - significant optimization needed")

            # Precision vs Recall insights
            if precision > recall + 0.2:
                insights.append(
                    "High precision, lower recall - consider lowering threshold"
                )
            elif recall > precision + 0.2:
                insights.append(
                    "High recall, lower precision - consider raising threshold"
                )
            else:
                insights.append("Balanced precision and recall")

            # False positive rate insights
            if fpr > 0.2:
                insights.append(
                    "High false positive rate - threshold adjustment recommended"
                )
            elif fpr < 0.05:
                insights.append("Low false positive rate - good threshold setting")

            # Statistical significance insights
            significance = statistical_significance.get(
                "overall_significance", "unknown"
            )
            if significance == "low":
                insights.append(
                    "Low statistical significance - more data needed for reliable metrics"
                )

        except Exception as e:
            logger.error(f"Performance insights generation failed: {e}")
            insights.append("Error generating insights")

        return insights

    def _create_empty_metrics(
        self, detector_id: str, threshold: float
    ) -> Dict[str, any]:
        """Create empty metrics result."""
        return {
            "detector_id": detector_id,
            "threshold": threshold,
            "basic_metrics": {},
            "advanced_metrics": {},
            "performance_insights": ["No historical data available"],
            "calculation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_metrics(
        self, detector_id: str, threshold: float, sample_count: int
    ) -> Dict[str, any]:
        """Create metrics result for insufficient data."""
        return {
            "detector_id": detector_id,
            "threshold": threshold,
            "basic_metrics": {},
            "advanced_metrics": {},
            "performance_insights": [f"Insufficient data: {sample_count} samples"],
            "calculation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_metrics(
        self, detector_id: str, threshold: float, error_message: str
    ) -> Dict[str, any]:
        """Create metrics result for error cases."""
        return {
            "detector_id": detector_id,
            "threshold": threshold,
            "basic_metrics": {},
            "advanced_metrics": {},
            "performance_insights": [f"Error: {error_message}"],
            "calculation_timestamp": datetime.utcnow().isoformat(),
        }


class ThresholdOptimizationEngine(BaseAnalysisEngine):
    """
    Main threshold optimization engine that coordinates all optimization components.

    Integrates ROC analysis, statistical optimization, simulation, and performance
    metrics calculation to provide comprehensive threshold optimization.
    """

    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.optimization_config = config.parameters.get("threshold_optimization", {})

        # Initialize components
        self.roc_analyzer = ROCAnalyzer(self.optimization_config)
        self.statistical_optimizer = StatisticalOptimizer(self.optimization_config)
        self.threshold_simulator = ThresholdSimulator(self.optimization_config)
        self.performance_calculator = PerformanceMetricsCalculator(
            self.optimization_config
        )

    def get_engine_name(self) -> str:
        """Get the name of this analysis engine."""
        return "threshold_optimization"

    async def _perform_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive threshold optimization analysis.

        Args:
            request: Analysis request containing detector performance data

        Returns:
            AnalysisResult with threshold optimization recommendations
        """
        try:
            # Extract detector performance data from request
            detector_data = self._extract_detector_data(request)

            optimization_results = {}

            # Analyze each detector
            for detector_id, historical_data in detector_data.items():
                detector_result = await self._analyze_detector_threshold(
                    detector_id, historical_data
                )
                optimization_results[detector_id] = detector_result

            # Generate overall recommendations
            overall_recommendations = self._generate_overall_recommendations(
                optimization_results
            )

            # Calculate confidence
            confidence = self._calculate_optimization_confidence(optimization_results)

            # Create analysis result
            result = AnalysisResult(
                analysis_type="threshold_optimization",
                confidence=confidence,
                patterns=[],  # Threshold optimization doesn't detect patterns
                evidence=[
                    {
                        "type": "threshold_analysis",
                        "detectors_analyzed": len(optimization_results),
                        "recommendations_generated": len(overall_recommendations),
                        "analysis_methods": [
                            "roc_analysis",
                            "statistical_optimization",
                            "impact_simulation",
                        ],
                    }
                ],
                metadata={
                    "detector_results": optimization_results,
                    "overall_recommendations": overall_recommendations,
                    "optimization_summary": self._generate_optimization_summary(
                        optimization_results
                    ),
                },
            )

            return result

        except Exception as e:
            logger.error("Threshold optimization analysis failed", error=str(e))
            return self._create_error_result(str(e))

    async def _analyze_detector_threshold(
        self, detector_id: str, historical_data: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Analyze threshold optimization for a single detector."""
        try:
            # Perform ROC analysis
            roc_analysis = await self.roc_analyzer.analyze_roc_curve(
                detector_id, historical_data
            )

            # Perform statistical optimization
            optimization_result = await self.statistical_optimizer.optimize_threshold(
                detector_id, historical_data
            )

            # Simulate impact of recommended threshold
            current_threshold = 0.5  # Would get from detector config
            proposed_threshold = optimization_result.get("optimal_threshold", 0.5)

            simulation_result = (
                await self.threshold_simulator.simulate_threshold_impact(
                    detector_id, current_threshold, proposed_threshold, historical_data
                )
            )

            # Calculate performance metrics for recommended threshold
            performance_metrics = (
                await self.performance_calculator.calculate_comprehensive_metrics(
                    detector_id, historical_data, proposed_threshold
                )
            )

            return {
                "detector_id": detector_id,
                "roc_analysis": roc_analysis,
                "optimization_result": optimization_result,
                "simulation_result": simulation_result,
                "performance_metrics": performance_metrics,
                "recommendation_summary": self._generate_detector_recommendation_summary(
                    roc_analysis, optimization_result, simulation_result
                ),
            }

        except Exception as e:
            logger.error(
                f"Detector threshold analysis failed for {detector_id}", error=str(e)
            )
            return {"detector_id": detector_id, "error": str(e)}

    def _extract_detector_data(
        self, request: AnalysisRequest
    ) -> Dict[str, List[Dict[str, any]]]:
        """Extract detector performance data from analysis request."""
        # This would extract historical performance data for each detector
        # For now, create mock data structure
        detector_data = {}

        # Extract from request data (simplified)
        for detector in request.observed_coverage.keys():
            # In production, this would fetch actual historical data
            detector_data[detector] = []

        return detector_data

    def _generate_detector_recommendation_summary(
        self,
        roc_analysis: Dict[str, any],
        optimization_result: Dict[str, any],
        simulation_result: Dict[str, any],
    ) -> str:
        """Generate recommendation summary for a detector."""
        try:
            auc_score = roc_analysis.get("auc_score", 0)
            optimal_threshold = optimization_result.get("optimal_threshold", 0.5)
            recommendation = simulation_result.get("simulation_summary", {}).get(
                "recommendation", "neutral"
            )

            if recommendation == "recommended":
                return f"Recommend threshold adjustment to {optimal_threshold:.3f} (AUC: {auc_score:.3f})"
            elif recommendation == "not_recommended":
                return f"Current threshold adequate (AUC: {auc_score:.3f})"
            else:
                return f"Monitor performance, consider threshold {optimal_threshold:.3f} (AUC: {auc_score:.3f})"

        except Exception:
            return "Unable to generate recommendation"

    def _generate_overall_recommendations(
        self, optimization_results: Dict[str, any]
    ) -> List[Dict[str, any]]:
        """Generate overall threshold optimization recommendations."""
        recommendations = []

        try:
            for detector_id, result in optimization_results.items():
                if "error" not in result:
                    simulation = result.get("simulation_result", {})
                    recommendation_type = simulation.get("simulation_summary", {}).get(
                        "recommendation", "neutral"
                    )

                    if recommendation_type in ["recommended", "cautiously_recommended"]:
                        recommendations.append(
                            {
                                "detector_id": detector_id,
                                "recommendation_type": recommendation_type,
                                "current_threshold": simulation.get(
                                    "current_threshold", 0.5
                                ),
                                "proposed_threshold": simulation.get(
                                    "proposed_threshold", 0.5
                                ),
                                "expected_benefits": simulation.get(
                                    "simulation_summary", {}
                                ).get("key_benefits", []),
                                "potential_risks": simulation.get(
                                    "simulation_summary", {}
                                ).get("key_risks", []),
                                "priority": (
                                    "high"
                                    if recommendation_type == "recommended"
                                    else "medium"
                                ),
                            }
                        )

        except Exception as e:
            logger.error(f"Overall recommendations generation failed: {e}")

        return recommendations

    def _calculate_optimization_confidence(
        self, optimization_results: Dict[str, any]
    ) -> float:
        """Calculate overall confidence in optimization results."""
        try:
            if not optimization_results:
                return 0.0

            confidences = []

            for result in optimization_results.values():
                if "error" not in result:
                    # Extract confidence indicators
                    roc_auc = result.get("roc_analysis", {}).get("auc_score", 0)
                    simulation_quality = (
                        result.get("simulation_result", {})
                        .get("simulation_summary", {})
                        .get("simulation_quality", "low")
                    )

                    # Convert to numeric confidence
                    quality_score = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(
                        simulation_quality, 0.3
                    )
                    detector_confidence = (roc_auc + quality_score) / 2

                    confidences.append(detector_confidence)

            return sum(confidences) / len(confidences) if confidences else 0.0

        except Exception as e:
            logger.error(f"Optimization confidence calculation failed: {e}")
            return 0.0

    def _generate_optimization_summary(
        self, optimization_results: Dict[str, any]
    ) -> Dict[str, any]:
        """Generate summary of optimization analysis."""
        try:
            total_detectors = len(optimization_results)
            successful_analyses = len(
                [r for r in optimization_results.values() if "error" not in r]
            )

            recommendations = self._generate_overall_recommendations(
                optimization_results
            )
            high_priority_recommendations = [
                r for r in recommendations if r.get("priority") == "high"
            ]

            return {
                "total_detectors_analyzed": total_detectors,
                "successful_analyses": successful_analyses,
                "total_recommendations": len(recommendations),
                "high_priority_recommendations": len(high_priority_recommendations),
                "analysis_success_rate": (
                    successful_analyses / total_detectors if total_detectors > 0 else 0
                ),
                "optimization_methods_used": [
                    "roc_analysis",
                    "statistical_optimization",
                    "impact_simulation",
                    "performance_metrics",
                ],
            }

        except Exception as e:
            logger.error(f"Optimization summary generation failed: {e}")
            return {"error": str(e)}

    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error result for threshold optimization."""
        return AnalysisResult(
            analysis_type="threshold_optimization",
            confidence=0.0,
            patterns=[],
            evidence=[{"type": "error", "message": error_message}],
            metadata={"error": error_message},
        )


class ThresholdSimulator:
    """
    Threshold simulator for impact prediction using historical data.

    Simulates the impact of threshold changes on detection performance
    using historical data and statistical modeling.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.simulation_periods = self.config.get("simulation_periods", 30)  # days
        self.confidence_level = self.config.get("confidence_level", 0.95)

    async def simulate_threshold_impact(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        historical_data: List[Dict[str, any]],
        time_horizon_days: int = 30,
    ) -> Dict[str, any]:
        """
        Simulate impact of changing threshold using historical data.

        Args:
            detector_id: ID of the detector
            current_threshold: Current threshold value
            proposed_threshold: Proposed new threshold value
            historical_data: Historical detection data
            time_horizon_days: Simulation time horizon in days

        Returns:
            Dictionary containing simulation results
        """
        try:
            if not historical_data:
                return self._create_empty_simulation(detector_id)

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 50:  # Need sufficient data for simulation
                return self._create_insufficient_data_simulation(
                    detector_id, len(scores)
                )

            # Simulate current threshold performance
            current_performance = self._simulate_threshold_performance(
                scores, labels, current_threshold
            )

            # Simulate proposed threshold performance
            proposed_performance = self._simulate_threshold_performance(
                scores, labels, proposed_threshold
            )

            # Calculate impact metrics
            impact_analysis = self._calculate_impact_metrics(
                current_performance, proposed_performance
            )

            # Generate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                scores, labels, proposed_threshold, time_horizon_days
            )

            # Predict volume changes
            volume_predictions = self._predict_volume_changes(
                historical_data,
                current_threshold,
                proposed_threshold,
                time_horizon_days,
            )

            # Generate risk assessment
            risk_assessment = self._assess_threshold_change_risk(
                impact_analysis, confidence_intervals, volume_predictions
            )

            result = {
                "detector_id": detector_id,
                "current_threshold": current_threshold,
                "proposed_threshold": proposed_threshold,
                "time_horizon_days": time_horizon_days,
                "current_performance": current_performance,
                "proposed_performance": proposed_performance,
                "impact_analysis": impact_analysis,
                "confidence_intervals": confidence_intervals,
                "volume_predictions": volume_predictions,
                "risk_assessment": risk_assessment,
                "simulation_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Threshold simulation completed",
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=proposed_threshold,
                expected_impact=impact_analysis.get("overall_impact", "unknown"),
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold simulation failed", error=str(e), detector_id=detector_id
            )
            return self._create_error_simulation(detector_id, str(e))

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, any]]
    ) -> Tuple[List[float], List[int]]:
        """Extract confidence scores and ground truth labels."""
        scores = []
        labels = []

        for item in historical_data:
            try:
                score = float(item.get("confidence", item.get("score", 0.5)))
                label = int(item.get("ground_truth", item.get("is_true_positive", 0)))

                scores.append(score)
                labels.append(label)

            except (ValueError, TypeError):
                continue

        return scores, labels

    def _simulate_threshold_performance(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Simulate performance metrics for a given threshold."""
        try:
            # Make predictions based on threshold
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate confusion matrix
            tp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 1
            )
            fp = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 1 and label == 0
            )
            tn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 0
            )
            fn = sum(
                1
                for pred, label in zip(predictions, labels)
                if pred == 0 and label == 1
            )

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            accuracy = (
                (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
            )
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            return {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy,
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "total_detections": tp + fp,
                "total_samples": len(scores),
            }

        except Exception as e:
            logger.error(
                f"Performance simulation failed for threshold {threshold}: {e}"
            )
            return {"threshold": threshold, "error": str(e)}

    def _calculate_impact_metrics(
        self, current_perf: Dict[str, any], proposed_perf: Dict[str, any]
    ) -> Dict[str, any]:
        """Calculate impact metrics between current and proposed thresholds."""
        try:
            # Calculate absolute changes
            precision_change = proposed_perf.get("precision", 0) - current_perf.get(
                "precision", 0
            )
            recall_change = proposed_perf.get("recall", 0) - current_perf.get(
                "recall", 0
            )
            f1_change = proposed_perf.get("f1_score", 0) - current_perf.get(
                "f1_score", 0
            )
            fpr_change = proposed_perf.get("false_positive_rate", 0) - current_perf.get(
                "false_positive_rate", 0
            )

            # Calculate relative changes
            current_fp = current_perf.get("false_positives", 0)
            proposed_fp = proposed_perf.get("false_positives", 0)
            fp_change_pct = (
                (proposed_fp - current_fp) / current_fp * 100 if current_fp > 0 else 0
            )

            current_detections = current_perf.get("total_detections", 0)
            proposed_detections = proposed_perf.get("total_detections", 0)
            detection_change_pct = (
                (proposed_detections - current_detections) / current_detections * 100
                if current_detections > 0
                else 0
            )

            # Determine overall impact
            overall_impact = self._determine_overall_impact(
                precision_change, recall_change, f1_change, fpr_change
            )

            return {
                "precision_change": precision_change,
                "recall_change": recall_change,
                "f1_score_change": f1_change,
                "false_positive_rate_change": fpr_change,
                "false_positive_change_percent": fp_change_pct,
                "detection_volume_change_percent": detection_change_pct,
                "overall_impact": overall_impact,
                "impact_magnitude": abs(f1_change),
                "recommended": f1_change > 0.01,  # Recommend if F1 improves by >1%
            }

        except Exception as e:
            logger.error(f"Impact metrics calculation failed: {e}")
            return {"error": str(e)}

    def _determine_overall_impact(
        self,
        precision_change: float,
        recall_change: float,
        f1_change: float,
        fpr_change: float,
    ) -> str:
        """Determine overall impact classification."""
        if f1_change > 0.05:
            return "significant_improvement"
        elif f1_change > 0.01:
            return "moderate_improvement"
        elif f1_change > -0.01:
            return "minimal_change"
        elif f1_change > -0.05:
            return "moderate_degradation"
        else:
            return "significant_degradation"

    def _calculate_confidence_intervals(
        self,
        scores: List[float],
        labels: List[int],
        threshold: float,
        time_horizon_days: int,
    ) -> Dict[str, any]:
        """Calculate confidence intervals for performance predictions."""
        try:
            # Bootstrap sampling for confidence intervals
            n_bootstrap = 1000
            bootstrap_f1_scores = []
            bootstrap_precisions = []
            bootstrap_recalls = []

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(scores), size=len(scores), replace=True)
                bootstrap_scores = [scores[i] for i in indices]
                bootstrap_labels = [labels[i] for i in indices]

                # Calculate performance
                perf = self._simulate_threshold_performance(
                    bootstrap_scores, bootstrap_labels, threshold
                )

                bootstrap_f1_scores.append(perf.get("f1_score", 0))
                bootstrap_precisions.append(perf.get("precision", 0))
                bootstrap_recalls.append(perf.get("recall", 0))

            # Calculate confidence intervals
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            return {
                "confidence_level": self.confidence_level,
                "f1_score_ci": {
                    "lower": np.percentile(bootstrap_f1_scores, lower_percentile),
                    "upper": np.percentile(bootstrap_f1_scores, upper_percentile),
                    "mean": np.mean(bootstrap_f1_scores),
                },
                "precision_ci": {
                    "lower": np.percentile(bootstrap_precisions, lower_percentile),
                    "upper": np.percentile(bootstrap_precisions, upper_percentile),
                    "mean": np.mean(bootstrap_precisions),
                },
                "recall_ci": {
                    "lower": np.percentile(bootstrap_recalls, lower_percentile),
                    "upper": np.percentile(bootstrap_recalls, upper_percentile),
                    "mean": np.mean(bootstrap_recalls),
                },
            }

        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return {"error": str(e)}

    def _predict_volume_changes(
        self,
        historical_data: List[Dict[str, any]],
        current_threshold: float,
        proposed_threshold: float,
        time_horizon_days: int,
    ) -> Dict[str, any]:
        """Predict volume changes based on threshold adjustment."""
        try:
            # Group data by time periods to understand volume patterns
            daily_volumes = self._calculate_daily_volumes(historical_data)

            if not daily_volumes:
                return {"error": "No volume data available"}

            # Calculate average daily volume
            avg_daily_volume = np.mean(list(daily_volumes.values()))

            # Estimate volume change based on threshold difference
            threshold_diff = proposed_threshold - current_threshold

            # Simple heuristic: higher threshold = fewer detections
            volume_change_factor = 1.0 - (threshold_diff * 0.5)  # Rough estimate
            volume_change_factor = max(0.1, min(2.0, volume_change_factor))  # Bound it

            predicted_daily_volume = avg_daily_volume * volume_change_factor
            predicted_total_volume = predicted_daily_volume * time_horizon_days

            return {
                "current_avg_daily_volume": avg_daily_volume,
                "predicted_daily_volume": predicted_daily_volume,
                "predicted_total_volume": predicted_total_volume,
                "volume_change_factor": volume_change_factor,
                "volume_change_percent": (volume_change_factor - 1.0) * 100,
                "time_horizon_days": time_horizon_days,
            }

        except Exception as e:
            logger.error(f"Volume prediction failed: {e}")
            return {"error": str(e)}

    def _calculate_daily_volumes(
        self, historical_data: List[Dict[str, any]]
    ) -> Dict[str, int]:
        """Calculate daily detection volumes from historical data."""
        daily_volumes = {}

        for item in historical_data:
            try:
                # Extract date from timestamp
                timestamp = item.get("timestamp", item.get("created_at"))
                if timestamp:
                    if isinstance(timestamp, str):
                        date = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        ).date()
                    else:
                        date = timestamp.date()

                    date_str = date.isoformat()
                    daily_volumes[date_str] = daily_volumes.get(date_str, 0) + 1

            except Exception:
                continue

        return daily_volumes

    def _assess_threshold_change_risk(
        self,
        impact_analysis: Dict[str, any],
        confidence_intervals: Dict[str, any],
        volume_predictions: Dict[str, any],
    ) -> Dict[str, any]:
        """Assess risk of threshold change."""
        try:
            risk_factors = []
            risk_score = 0.0

            # Check F1 score confidence interval
            f1_ci = confidence_intervals.get("f1_score_ci", {})
            f1_lower = f1_ci.get("lower", 0)
            if f1_lower < 0.5:
                risk_factors.append("Low F1 score confidence interval")
                risk_score += 0.3

            # Check volume change magnitude
            volume_change_pct = abs(volume_predictions.get("volume_change_percent", 0))
            if volume_change_pct > 50:
                risk_factors.append("Large volume change predicted")
                risk_score += 0.4
            elif volume_change_pct > 25:
                risk_factors.append("Moderate volume change predicted")
                risk_score += 0.2

            # Check performance degradation
            overall_impact = impact_analysis.get("overall_impact", "")
            if "degradation" in overall_impact:
                risk_factors.append("Performance degradation expected")
                risk_score += 0.5

            # Determine risk level
            if risk_score >= 0.7:
                risk_level = "high"
            elif risk_score >= 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"

            return {
                "risk_level": risk_level,
                "risk_score": min(1.0, risk_score),
                "risk_factors": risk_factors,
                "recommendation": self._generate_risk_recommendation(
                    risk_level, risk_factors
                ),
                "mitigation_strategies": self._suggest_mitigation_strategies(
                    risk_factors
                ),
            }

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e)}

    def _generate_risk_recommendation(
        self, risk_level: str, risk_factors: List[str]
    ) -> str:
        """Generate risk-based recommendation."""
        if risk_level == "high":
            return "High risk - consider gradual rollout or additional testing"
        elif risk_level == "medium":
            return "Medium risk - monitor closely during deployment"
        else:
            return "Low risk - safe to proceed with threshold change"

    def _suggest_mitigation_strategies(self, risk_factors: List[str]) -> List[str]:
        """Suggest mitigation strategies based on risk factors."""
        strategies = []

        for factor in risk_factors:
            if "confidence interval" in factor.lower():
                strategies.append("Collect more historical data for better confidence")
            elif "volume change" in factor.lower():
                strategies.append("Implement gradual threshold adjustment")
            elif "degradation" in factor.lower():
                strategies.append("Set up enhanced monitoring during transition")

        if not strategies:
            strategies.append("Monitor performance metrics post-deployment")

        return strategies

    def _create_empty_simulation(self, detector_id: str) -> Dict[str, any]:
        """Create empty simulation result."""
        return {
            "detector_id": detector_id,
            "error": "No historical data available for simulation",
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_simulation(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, any]:
        """Create simulation result for insufficient data."""
        return {
            "detector_id": detector_id,
            "error": f"Insufficient data for simulation: {sample_count} samples",
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_simulation(
        self, detector_id: str, error_message: str
    ) -> Dict[str, any]:
        """Create simulation result for error cases."""
        return {
            "detector_id": detector_id,
            "error": error_message,
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }


class PerformanceMetricsCalculator:
    """
    Performance metrics calculator for false positive/negative rate analysis.

    Provides comprehensive performance analysis including advanced metrics
    and statistical significance testing.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.significance_level = self.config.get("significance_level", 0.05)

    async def calculate_comprehensive_metrics(
        self,
        detector_id: str,
        predictions: List[int],
        ground_truth: List[int],
        confidence_scores: List[float] = None,
    ) -> Dict[str, any]:
        """
        Calculate comprehensive performance metrics.

        Args:
            detector_id: ID of the detector
            predictions: Binary predictions (0 or 1)
            ground_truth: Ground truth labels (0 or 1)
            confidence_scores: Optional confidence scores

        Returns:
            Dictionary containing comprehensive metrics
        """
        try:
            if len(predictions) != len(ground_truth):
                raise ValueError("Predictions and ground truth must have same length")

            if len(predictions) == 0:
                return self._create_empty_metrics(detector_id)

            # Calculate basic confusion matrix
            confusion_matrix = self._calculate_confusion_matrix(
                predictions, ground_truth
            )

            # Calculate basic metrics
            basic_metrics = self._calculate_basic_metrics(confusion_matrix)

            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(
                confusion_matrix, confidence_scores
            )

            # Calculate statistical significance
            statistical_tests = self._perform_statistical_tests(
                predictions, ground_truth, confidence_scores
            )

            # Calculate class-specific metrics
            class_metrics = self._calculate_class_specific_metrics(
                predictions, ground_truth
            )

            # Generate performance summary
            performance_summary = self._generate_performance_summary(
                basic_metrics, advanced_metrics, statistical_tests
            )

            result = {
                "detector_id": detector_id,
                "sample_count": len(predictions),
                "confusion_matrix": confusion_matrix,
                "basic_metrics": basic_metrics,
                "advanced_metrics": advanced_metrics,
                "statistical_tests": statistical_tests,
                "class_metrics": class_metrics,
                "performance_summary": performance_summary,
                "calculation_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Performance metrics calculated",
                detector_id=detector_id,
                sample_count=len(predictions),
                f1_score=basic_metrics.get("f1_score", 0),
            )

            return result

        except Exception as e:
            logger.error(
                "Performance metrics calculation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_metrics(detector_id, str(e))

    def _calculate_confusion_matrix(
        self, predictions: List[int], ground_truth: List[int]
    ) -> Dict[str, int]:
        """Calculate confusion matrix components."""
        tp = sum(
            1
            for pred, true in zip(predictions, ground_truth)
            if pred == 1 and true == 1
        )
        fp = sum(
            1
            for pred, true in zip(predictions, ground_truth)
            if pred == 1 and true == 0
        )
        tn = sum(
            1
            for pred, true in zip(predictions, ground_truth)
            if pred == 0 and true == 0
        )
        fn = sum(
            1
            for pred, true in zip(predictions, ground_truth)
            if pred == 0 and true == 1
        )

        return {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "total_samples": len(predictions),
        }

    def _calculate_basic_metrics(
        self, confusion_matrix: Dict[str, int]
    ) -> Dict[str, float]:
        """Calculate basic performance metrics."""
        tp = confusion_matrix["true_positives"]
        fp = confusion_matrix["false_positives"]
        tn = confusion_matrix["true_negatives"]
        fn = confusion_matrix["false_negatives"]

        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0

        # F-scores
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f2_score = (
            (5 * precision * recall) / (4 * precision + recall)
            if (4 * precision + recall) > 0
            else 0.0
        )

        # Error rates
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        false_discovery_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "accuracy": accuracy,
            "f1_score": f1_score,
            "f2_score": f2_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "false_discovery_rate": false_discovery_rate,
            "sensitivity": recall,  # Same as recall
            "true_negative_rate": specificity,  # Same as specificity
        }

    def _calculate_advanced_metrics(
        self, confusion_matrix: Dict[str, int], confidence_scores: List[float] = None
    ) -> Dict[str, any]:
        """Calculate advanced performance metrics."""
        tp = confusion_matrix["true_positives"]
        fp = confusion_matrix["false_positives"]
        tn = confusion_matrix["true_negatives"]
        fn = confusion_matrix["false_negatives"]

        # Matthews Correlation Coefficient
        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0

        # Balanced Accuracy
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2

        # Youden's J Statistic
        youdens_j = sensitivity + specificity - 1

        # Positive and Negative Predictive Values
        positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Likelihood Ratios
        positive_likelihood_ratio = (
            sensitivity / (1 - specificity) if specificity < 1.0 else float("inf")
        )
        negative_likelihood_ratio = (
            (1 - sensitivity) / specificity if specificity > 0.0 else float("inf")
        )

        # Diagnostic Odds Ratio
        diagnostic_odds_ratio = (
            positive_likelihood_ratio / negative_likelihood_ratio
            if negative_likelihood_ratio > 0
            and negative_likelihood_ratio != float("inf")
            else float("inf")
        )

        advanced_metrics = {
            "matthews_correlation_coefficient": mcc,
            "balanced_accuracy": balanced_accuracy,
            "youdens_j_statistic": youdens_j,
            "positive_predictive_value": positive_predictive_value,
            "negative_predictive_value": negative_predictive_value,
            "positive_likelihood_ratio": positive_likelihood_ratio,
            "negative_likelihood_ratio": negative_likelihood_ratio,
            "diagnostic_odds_ratio": diagnostic_odds_ratio,
        }

        # Add confidence-based metrics if available
        if confidence_scores:
            confidence_metrics = self._calculate_confidence_metrics(confidence_scores)
            advanced_metrics.update(confidence_metrics)

        return advanced_metrics

    def _calculate_confidence_metrics(
        self, confidence_scores: List[float]
    ) -> Dict[str, float]:
        """Calculate metrics based on confidence scores."""
        try:
            return {
                "mean_confidence": np.mean(confidence_scores),
                "median_confidence": np.median(confidence_scores),
                "confidence_std": np.std(confidence_scores),
                "min_confidence": min(confidence_scores),
                "max_confidence": max(confidence_scores),
                "confidence_range": max(confidence_scores) - min(confidence_scores),
            }
        except Exception as e:
            logger.error(f"Confidence metrics calculation failed: {e}")
            return {}

    def _perform_statistical_tests(
        self,
        predictions: List[int],
        ground_truth: List[int],
        confidence_scores: List[float] = None,
    ) -> Dict[str, any]:
        """Perform statistical significance tests."""
        try:
            # McNemar's test for paired predictions (simplified version)
            mcnemar_result = self._mcnemar_test_simplified(predictions, ground_truth)

            # Binomial test for precision
            precision_test = self._binomial_test_precision(predictions, ground_truth)

            # Chi-square test for independence
            chi_square_test = self._chi_square_test(predictions, ground_truth)

            return {
                "mcnemar_test": mcnemar_result,
                "precision_binomial_test": precision_test,
                "chi_square_test": chi_square_test,
                "significance_level": self.significance_level,
            }

        except Exception as e:
            logger.error(f"Statistical tests failed: {e}")
            return {"error": str(e)}

    def _mcnemar_test_simplified(
        self, predictions: List[int], ground_truth: List[int]
    ) -> Dict[str, any]:
        """Simplified McNemar's test for model comparison."""
        try:
            # Count discordant pairs
            b = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 0
            )  # False positives
            c = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 1
            )  # False negatives

            # McNemar's statistic
            if b + c == 0:
                return {"statistic": 0.0, "p_value": 1.0, "significant": False}

            mcnemar_stat = ((abs(b - c) - 1) ** 2) / (b + c)

            # Approximate p-value using chi-square distribution (df=1)
            # For simplicity, use a lookup table approach
            p_value = self._chi_square_p_value_approximation(mcnemar_stat, df=1)

            return {
                "statistic": mcnemar_stat,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "discordant_pairs": {"false_positives": b, "false_negatives": c},
            }

        except Exception as e:
            logger.error(f"McNemar test failed: {e}")
            return {"error": str(e)}

    def _binomial_test_precision(
        self, predictions: List[int], ground_truth: List[int]
    ) -> Dict[str, any]:
        """Binomial test for precision significance."""
        try:
            # Count true positives and total positive predictions
            tp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 1
            )
            total_positive_predictions = sum(predictions)

            if total_positive_predictions == 0:
                return {"precision": 0.0, "significant": False, "p_value": 1.0}

            observed_precision = tp / total_positive_predictions

            # Test against null hypothesis of random guessing (p=0.5)
            # Using normal approximation to binomial
            expected = total_positive_predictions * 0.5
            variance = total_positive_predictions * 0.5 * 0.5

            if variance == 0:
                z_score = 0.0
            else:
                z_score = (tp - expected) / (variance**0.5)

            # Two-tailed test p-value approximation
            p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))

            return {
                "observed_precision": observed_precision,
                "z_score": z_score,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "null_hypothesis": "precision = 0.5 (random guessing)",
            }

        except Exception as e:
            logger.error(f"Binomial test failed: {e}")
            return {"error": str(e)}

    def _chi_square_test(
        self, predictions: List[int], ground_truth: List[int]
    ) -> Dict[str, any]:
        """Chi-square test for independence."""
        try:
            # Create 2x2 contingency table
            tp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 1
            )
            fp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 0
            )
            fn = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 1
            )
            tn = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 0
            )

            # Calculate expected frequencies
            n = tp + fp + fn + tn
            if n == 0:
                return {"statistic": 0.0, "p_value": 1.0, "significant": False}

            row1_total = tp + fp
            row2_total = fn + tn
            col1_total = tp + fn
            col2_total = fp + tn

            expected_tp = (row1_total * col1_total) / n
            expected_fp = (row1_total * col2_total) / n
            expected_fn = (row2_total * col1_total) / n
            expected_tn = (row2_total * col2_total) / n

            # Chi-square statistic
            chi_square = 0.0
            for observed, expected in [
                (tp, expected_tp),
                (fp, expected_fp),
                (fn, expected_fn),
                (tn, expected_tn),
            ]:
                if expected > 0:
                    chi_square += ((observed - expected) ** 2) / expected

            # Approximate p-value
            p_value = self._chi_square_p_value_approximation(chi_square, df=1)

            return {
                "statistic": chi_square,
                "p_value": p_value,
                "significant": p_value < self.significance_level,
                "degrees_of_freedom": 1,
                "contingency_table": {
                    "observed": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
                    "expected": {
                        "tp": expected_tp,
                        "fp": expected_fp,
                        "fn": expected_fn,
                        "tn": expected_tn,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Chi-square test failed: {e}")
            return {"error": str(e)}

    def _chi_square_p_value_approximation(
        self, chi_square_stat: float, df: int
    ) -> float:
        """Approximate p-value for chi-square statistic."""
        # Simple approximation for df=1
        if df == 1:
            if chi_square_stat < 0.016:
                return 0.9
            elif chi_square_stat < 0.064:
                return 0.8
            elif chi_square_stat < 0.148:
                return 0.7
            elif chi_square_stat < 0.275:
                return 0.6
            elif chi_square_stat < 0.455:
                return 0.5
            elif chi_square_stat < 0.708:
                return 0.4
            elif chi_square_stat < 1.074:
                return 0.3
            elif chi_square_stat < 1.642:
                return 0.2
            elif chi_square_stat < 2.706:
                return 0.1
            elif chi_square_stat < 3.841:
                return 0.05
            elif chi_square_stat < 6.635:
                return 0.01
            else:
                return 0.001
        else:
            return 0.05  # Default for other df values

    def _standard_normal_cdf(self, z: float) -> float:
        """Approximate standard normal CDF."""
        # Simple approximation using error function approximation
        if z < -3:
            return 0.001
        elif z < -2:
            return 0.023
        elif z < -1:
            return 0.159
        elif z < 0:
            return 0.5 - (0.5 - 0.159) * abs(z)
        elif z < 1:
            return 0.5 + (0.841 - 0.5) * z
        elif z < 2:
            return 0.841 + (0.977 - 0.841) * (z - 1)
        elif z < 3:
            return 0.977 + (0.999 - 0.977) * (z - 2)
        else:
            return 0.999

    def _calculate_class_specific_metrics(
        self, predictions: List[int], ground_truth: List[int]
    ) -> Dict[str, any]:
        """Calculate metrics for each class."""
        try:
            # Class 0 (negative class) metrics
            class_0_tp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 0
            )
            class_0_fp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 1
            )
            class_0_fn = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 0
            )
            class_0_tn = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 1
            )

            class_0_precision = (
                class_0_tp / (class_0_tp + class_0_fp)
                if (class_0_tp + class_0_fp) > 0
                else 0.0
            )
            class_0_recall = (
                class_0_tp / (class_0_tp + class_0_fn)
                if (class_0_tp + class_0_fn) > 0
                else 0.0
            )
            class_0_f1 = (
                2
                * class_0_precision
                * class_0_recall
                / (class_0_precision + class_0_recall)
                if (class_0_precision + class_0_recall) > 0
                else 0.0
            )

            # Class 1 (positive class) metrics
            class_1_tp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 1
            )
            class_1_fp = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 1 and true == 0
            )
            class_1_fn = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 1
            )
            class_1_tn = sum(
                1
                for pred, true in zip(predictions, ground_truth)
                if pred == 0 and true == 0
            )

            class_1_precision = (
                class_1_tp / (class_1_tp + class_1_fp)
                if (class_1_tp + class_1_fp) > 0
                else 0.0
            )
            class_1_recall = (
                class_1_tp / (class_1_tp + class_1_fn)
                if (class_1_tp + class_1_fn) > 0
                else 0.0
            )
            class_1_f1 = (
                2
                * class_1_precision
                * class_1_recall
                / (class_1_precision + class_1_recall)
                if (class_1_precision + class_1_recall) > 0
                else 0.0
            )

            return {
                "class_0_negative": {
                    "precision": class_0_precision,
                    "recall": class_0_recall,
                    "f1_score": class_0_f1,
                    "support": sum(1 for true in ground_truth if true == 0),
                },
                "class_1_positive": {
                    "precision": class_1_precision,
                    "recall": class_1_recall,
                    "f1_score": class_1_f1,
                    "support": sum(1 for true in ground_truth if true == 1),
                },
                "macro_average": {
                    "precision": (class_0_precision + class_1_precision) / 2,
                    "recall": (class_0_recall + class_1_recall) / 2,
                    "f1_score": (class_0_f1 + class_1_f1) / 2,
                },
            }

        except Exception as e:
            logger.error(f"Class-specific metrics calculation failed: {e}")
            return {"error": str(e)}

    def _generate_performance_summary(
        self,
        basic_metrics: Dict[str, float],
        advanced_metrics: Dict[str, any],
        statistical_tests: Dict[str, any],
    ) -> Dict[str, any]:
        """Generate performance summary and interpretation."""
        try:
            # Overall performance assessment
            f1_score = basic_metrics.get("f1_score", 0)
            precision = basic_metrics.get("precision", 0)
            recall = basic_metrics.get("recall", 0)

            if f1_score >= 0.9:
                performance_level = "excellent"
            elif f1_score >= 0.8:
                performance_level = "good"
            elif f1_score >= 0.7:
                performance_level = "fair"
            elif f1_score >= 0.6:
                performance_level = "poor"
            else:
                performance_level = "very_poor"

            # Identify strengths and weaknesses
            strengths = []
            weaknesses = []

            if precision >= 0.8:
                strengths.append("High precision - low false positive rate")
            elif precision < 0.6:
                weaknesses.append("Low precision - high false positive rate")

            if recall >= 0.8:
                strengths.append("High recall - good detection coverage")
            elif recall < 0.6:
                weaknesses.append("Low recall - missing many true positives")

            # Statistical significance
            is_statistically_significant = any(
                test.get("significant", False)
                for test in statistical_tests.values()
                if isinstance(test, dict)
            )

            return {
                "performance_level": performance_level,
                "f1_score": f1_score,
                "precision": precision,
                "recall": recall,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "statistically_significant": is_statistically_significant,
                "recommendations": self._generate_recommendations(
                    basic_metrics, advanced_metrics
                ),
            }

        except Exception as e:
            logger.error(f"Performance summary generation failed: {e}")
            return {"error": str(e)}

    def _generate_recommendations(
        self, basic_metrics: Dict[str, float], advanced_metrics: Dict[str, any]
    ) -> List[str]:
        """Generate recommendations based on performance metrics."""
        recommendations = []

        precision = basic_metrics.get("precision", 0)
        recall = basic_metrics.get("recall", 0)
        f1_score = basic_metrics.get("f1_score", 0)

        if precision < 0.7:
            recommendations.append(
                "Consider increasing threshold to reduce false positives"
            )

        if recall < 0.7:
            recommendations.append(
                "Consider decreasing threshold to improve detection coverage"
            )

        if f1_score < 0.6:
            recommendations.append(
                "Model performance is poor - consider retraining or feature engineering"
            )

        if abs(precision - recall) > 0.2:
            recommendations.append(
                "Large precision-recall imbalance - review threshold optimization"
            )

        mcc = advanced_metrics.get("matthews_correlation_coefficient", 0)
        if mcc < 0.3:
            recommendations.append(
                "Low Matthews correlation - model may not be better than random"
            )

        if not recommendations:
            recommendations.append(
                "Performance is acceptable - monitor for consistency"
            )

        return recommendations

    def _create_empty_metrics(self, detector_id: str) -> Dict[str, any]:
        """Create empty metrics result."""
        return {
            "detector_id": detector_id,
            "error": "No data provided for metrics calculation",
            "calculation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_metrics(
        self, detector_id: str, error_message: str
    ) -> Dict[str, any]:
        """Create error metrics result."""
        return {
            "detector_id": detector_id,
            "error": error_message,
            "calculation_timestamp": datetime.utcnow().isoformat(),
        }


class ThresholdOptimizationEngine(BaseAnalysisEngine):
    """
    Main threshold optimization engine that coordinates all threshold analysis components.

    Provides comprehensive threshold optimization capabilities including ROC analysis,
    statistical optimization, impact simulation, and performance metrics calculation.
    """

    def __init__(self, config: AnalysisConfiguration):
        super().__init__(config)
        self.roc_analyzer = ROCAnalyzer(config.get_section("roc_analysis"))
        self.statistical_optimizer = StatisticalOptimizer(
            config.get_section("statistical_optimization")
        )
        self.threshold_simulator = ThresholdSimulator(
            config.get_section("threshold_simulation")
        )
        self.performance_calculator = PerformanceMetricsCalculator(
            config.get_section("performance_metrics")
        )

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Perform comprehensive threshold optimization analysis.

        Args:
            request: Analysis request containing detector data and parameters

        Returns:
            AnalysisResult containing threshold optimization recommendations
        """
        try:
            detector_id = request.metadata.get("detector_id", "unknown")
            analysis_type = request.metadata.get("analysis_type", "comprehensive")

            logger.info(
                "Starting threshold optimization analysis",
                detector_id=detector_id,
                analysis_type=analysis_type,
            )

            # Extract historical data
            historical_data = request.data.get("historical_data", [])
            current_threshold = request.metadata.get("current_threshold", 0.5)

            results = {}

            # Perform ROC analysis
            if analysis_type in ["comprehensive", "roc"]:
                roc_results = await self.roc_analyzer.analyze_roc_curve(
                    detector_id, historical_data
                )
                results["roc_analysis"] = roc_results

            # Perform statistical optimization
            if analysis_type in ["comprehensive", "optimization"]:
                optimization_objective = OptimizationObjective(
                    request.metadata.get("optimization_objective", "balanced_pr")
                )
                constraints = request.metadata.get("constraints", {})

                optimization_results = (
                    await self.statistical_optimizer.optimize_threshold(
                        detector_id,
                        historical_data,
                        optimization_objective,
                        constraints,
                    )
                )
                results["threshold_optimization"] = optimization_results

            # Perform threshold simulation
            if analysis_type in ["comprehensive", "simulation"]:
                proposed_threshold = request.metadata.get("proposed_threshold")
                if proposed_threshold is not None:
                    simulation_results = (
                        await self.threshold_simulator.simulate_threshold_impact(
                            detector_id,
                            current_threshold,
                            proposed_threshold,
                            historical_data,
                        )
                    )
                    results["threshold_simulation"] = simulation_results

            # Calculate performance metrics
            if analysis_type in ["comprehensive", "metrics"]:
                predictions = request.data.get("predictions", [])
                ground_truth = request.data.get("ground_truth", [])
                confidence_scores = request.data.get("confidence_scores")

                if predictions and ground_truth:
                    metrics_results = await self.performance_calculator.calculate_comprehensive_metrics(
                        detector_id, predictions, ground_truth, confidence_scores
                    )
                    results["performance_metrics"] = metrics_results

            # Generate final recommendations
            recommendations = self._generate_final_recommendations(results)

            return AnalysisResult(
                analysis_type="threshold_optimization",
                results=results,
                recommendations=recommendations,
                confidence_score=self._calculate_overall_confidence(results),
                metadata={
                    "detector_id": detector_id,
                    "analysis_components": list(results.keys()),
                    "current_threshold": current_threshold,
                },
            )

        except Exception as e:
            logger.error("Threshold optimization analysis failed", error=str(e))
            return AnalysisResult(
                analysis_type="threshold_optimization",
                results={"error": str(e)},
                recommendations=[],
                confidence_score=0.0,
                metadata={"error": str(e)},
            )

    def _generate_final_recommendations(self, results: Dict[str, any]) -> List[str]:
        """Generate final threshold optimization recommendations."""
        recommendations = []

        # ROC analysis recommendations
        roc_results = results.get("roc_analysis", {})
        if roc_results and not roc_results.get("error"):
            auc_score = roc_results.get("auc_score", 0)
            if auc_score < 0.7:
                recommendations.append(
                    "Poor ROC performance - consider model retraining"
                )
            elif auc_score > 0.9:
                recommendations.append(
                    "Excellent ROC performance - model is well-calibrated"
                )

        # Optimization recommendations
        optimization_results = results.get("threshold_optimization", {})
        if optimization_results and not optimization_results.get("error"):
            optimal_threshold = optimization_results.get("optimal_threshold")
            if optimal_threshold is not None:
                recommendations.append(
                    f"Recommended optimal threshold: {optimal_threshold:.3f}"
                )

        # Simulation recommendations
        simulation_results = results.get("threshold_simulation", {})
        if simulation_results and not simulation_results.get("error"):
            risk_assessment = simulation_results.get("risk_assessment", {})
            risk_level = risk_assessment.get("risk_level", "unknown")
            recommendation = risk_assessment.get("recommendation", "")
            if recommendation:
                recommendations.append(
                    f"Threshold change risk: {risk_level} - {recommendation}"
                )

        # Performance metrics recommendations
        metrics_results = results.get("performance_metrics", {})
        if metrics_results and not metrics_results.get("error"):
            performance_summary = metrics_results.get("performance_summary", {})
            metric_recommendations = performance_summary.get("recommendations", [])
            recommendations.extend(metric_recommendations)

        if not recommendations:
            recommendations.append("No specific recommendations - monitor performance")

        return recommendations

    def _calculate_overall_confidence(self, results: Dict[str, any]) -> float:
        """Calculate overall confidence score for the analysis."""
        confidence_scores = []

        # ROC analysis confidence
        roc_results = results.get("roc_analysis", {})
        if roc_results and not roc_results.get("error"):
            auc_score = roc_results.get("auc_score", 0)
            confidence_scores.append(
                min(1.0, auc_score * 1.2)
            )  # Scale AUC to confidence

        # Data quality confidence
        for result_key in results:
            result = results[result_key]
            if isinstance(result, dict):
                data_quality = result.get("data_quality", {})
                if data_quality and not data_quality.get("error"):
                    sample_count = data_quality.get("total_samples", 0)
                    if sample_count >= 1000:
                        confidence_scores.append(0.9)
                    elif sample_count >= 100:
                        confidence_scores.append(0.7)
                    elif sample_count >= 50:
                        confidence_scores.append(0.5)
                    else:
                        confidence_scores.append(0.3)

        return np.mean(confidence_scores) if confidence_scores else 0.5
    def _calculate_statistical_significance(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, any]:
        """Calculate statistical significance of performance metrics."""
    