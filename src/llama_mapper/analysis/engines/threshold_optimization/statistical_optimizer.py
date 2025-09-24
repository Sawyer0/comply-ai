"""
Statistical Optimizer for precision-recall optimization methods.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np

from .types import OptimizationObjective

logger = logging.getLogger(__name__)


class StatisticalOptimizer:
    """
    Statistical optimizer using precision-recall optimization methods.

    Implements various optimization algorithms to find optimal thresholds
    based on different objectives and constraints.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.default_objective = OptimizationObjective(
            self.config.get("default_objective", "balanced_pr")
        )
        self.cost_matrix = self._load_cost_matrix()

    async def optimize_threshold(
        self,
        detector_id: str,
        historical_data: List[Dict[str, Any]],
        objective: OptimizationObjective = None,
        constraints: Dict[str, float] = None,
    ) -> Dict[str, Any]:
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
        self, historical_data: List[Dict[str, Any]]
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
    ) -> Dict[str, Any]:
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
        self, evaluations: List[Dict[str, Any]], constraints: Dict[str, float]
    ) -> List[Dict[str, Any]]:
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
        self, evaluations: List[Dict[str, Any]], objective: OptimizationObjective
    ) -> Dict[str, Any]:
        """Find evaluation with highest objective score."""
        if not evaluations:
            return {"threshold": 0.5, "objective_score": 0.0}

        return max(evaluations, key=lambda e: e.get("objective_score", 0.0))

    def _generate_optimization_summary(
        self,
        all_evaluations: List[Dict[str, Any]],
        optimal_evaluation: Dict[str, Any],
        objective: OptimizationObjective,
        constraints: Dict[str, float],
    ) -> Dict[str, Any]:
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
        self, evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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
                    "false_positive_rate": lowest_fpr.get("false_positive_rate", 0),
                },
            }

        except Exception as e:
            logger.error(f"Performance tradeoff analysis failed: {e}")
            return {"error": str(e)}

    def _load_cost_matrix(self) -> Dict[str, float]:
        """Load cost matrix for cost-based optimization."""
        return self.config.get(
            "cost_matrix",
            {
                "false_positive": 1.0,
                "false_negative": 5.0,  # False negatives typically more costly
            },
        )

    def _create_empty_optimization(self, detector_id: str) -> Dict[str, Any]:
        """Create empty optimization result."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "constraints_applied": {},
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "all_evaluations": [],
            "feasible_count": 0,
            "optimization_summary": {"message": "No historical data available"},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_optimization(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, Any]:
        """Create optimization result for insufficient data."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "constraints_applied": {},
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
    ) -> Dict[str, Any]:
        """Create optimization result for error cases."""
        return {
            "detector_id": detector_id,
            "optimization_objective": self.default_objective.value,
            "constraints_applied": {},
            "optimal_threshold": 0.5,
            "optimal_performance": {"threshold": 0.5, "objective_score": 0.0},
            "all_evaluations": [],
            "feasible_count": 0,
            "optimization_summary": {"error": error_message},
            "analysis_timestamp": datetime.utcnow().isoformat(),
        }
