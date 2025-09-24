"""
ROC Analyzer for receiver operating characteristic curve analysis.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np

from .types import ROCPoint

logger = logging.getLogger(__name__)


class ROCAnalyzer:
    """
    Receiver Operating Characteristic curve analysis for threshold optimization.

    Analyzes detector performance across different threshold values to find
    optimal operating points using statistical methods.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold_steps = self.config.get("threshold_steps", 100)
        self.min_threshold = self.config.get("min_threshold", 0.0)
        self.max_threshold = self.config.get("max_threshold", 1.0)

    async def analyze_roc_curve(
        self, detector_id: str, historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
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
        self, historical_data: List[Dict[str, Any]]
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
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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

    def _roc_point_to_dict(self, point: ROCPoint) -> Dict[str, Any]:
        """Convert ROC point to dictionary."""
        return {
            "threshold": point.threshold,
            "true_positive_rate": point.true_positive_rate,
            "false_positive_rate": point.false_positive_rate,
            "precision": point.precision,
            "recall": point.recall,
            "f1_score": point.f1_score,
        }

    def _create_empty_roc_analysis(self, detector_id: str) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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
