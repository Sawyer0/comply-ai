"""
Performance Metrics Calculator for false positive/negative rate analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .types import PerformanceMetrics

logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """
    Performance metrics calculator for false positive/negative rate analysis.

    Provides comprehensive analysis of detector performance metrics including
    detailed breakdowns and trend analysis.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metric_history_days = self.config.get("metric_history_days", 90)

    async def calculate_comprehensive_metrics(
        self,
        detector_id: str,
        historical_data: List[Dict[str, Any]],
        threshold: float,
        time_window_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics for a detector.

        Args:
            detector_id: ID of the detector
            historical_data: Historical detection data with ground truth
            threshold: Threshold value to analyze
            time_window_days: Optional time window for analysis

        Returns:
            Dictionary containing comprehensive metrics analysis
        """
        try:
            if not historical_data:
                return self._create_empty_metrics(detector_id)

            # Filter data by time window if specified
            if time_window_days:
                cutoff_date = datetime.utcnow() - timedelta(days=time_window_days)
                historical_data = [
                    item
                    for item in historical_data
                    if self._parse_timestamp(item.get("timestamp", "")) >= cutoff_date
                ]

            # Extract scores and labels
            scores, labels = self._extract_scores_and_labels(historical_data)

            if len(scores) < 5:
                return self._create_insufficient_data_metrics(detector_id, len(scores))

            # Calculate basic performance metrics
            basic_metrics = self._calculate_basic_metrics(scores, labels, threshold)

            # Calculate advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(
                scores, labels, threshold
            )

            # Calculate error analysis
            error_analysis = self._analyze_errors(scores, labels, threshold)

            # Calculate temporal trends
            temporal_trends = self._calculate_temporal_trends(
                historical_data, threshold
            )

            # Calculate statistical significance
            statistical_analysis = self._calculate_statistical_significance(
                scores, labels, threshold
            )

            result = {
                "detector_id": detector_id,
                "threshold": threshold,
                "basic_metrics": basic_metrics.__dict__,
                "advanced_metrics": advanced_metrics,
                "error_analysis": error_analysis,
                "temporal_trends": temporal_trends,
                "statistical_analysis": statistical_analysis,
                "data_quality": self._assess_data_quality(scores, labels),
                "analysis_metadata": {
                    "data_points": len(scores),
                    "time_window_days": time_window_days,
                    "analysis_timestamp": datetime.utcnow().isoformat(),
                },
            }

            logger.info(
                "Performance metrics calculation completed",
                detector_id=detector_id,
                threshold=threshold,
                data_points=len(scores),
            )

            return result

        except Exception as e:
            logger.error(
                "Performance metrics calculation failed",
                error=str(e),
                detector_id=detector_id,
            )
            return self._create_error_metrics(detector_id, str(e))

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

    def _calculate_basic_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> PerformanceMetrics:
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
            logger.error(f"Basic metrics calculation failed: {e}")
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

    def _calculate_advanced_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, Any]:
        """Calculate advanced performance metrics."""
        try:
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Matthews Correlation Coefficient
            mcc = self._calculate_mcc(predictions, labels)

            # Balanced Accuracy
            balanced_accuracy = self._calculate_balanced_accuracy(predictions, labels)

            # Cohen's Kappa
            cohens_kappa = self._calculate_cohens_kappa(predictions, labels)

            # Precision-Recall AUC (approximation)
            pr_auc = self._calculate_pr_auc_approximation(scores, labels)

            # Lift and Gain metrics
            lift_metrics = self._calculate_lift_metrics(scores, labels, threshold)

            return {
                "matthews_correlation_coefficient": mcc,
                "balanced_accuracy": balanced_accuracy,
                "cohens_kappa": cohens_kappa,
                "precision_recall_auc": pr_auc,
                "lift_metrics": lift_metrics,
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
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

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
            po = sum(1 for p, l in zip(predictions, labels) if p == l) / n

            # Expected agreement
            p_pos = sum(predictions) / n
            l_pos = sum(labels) / n
            pe = p_pos * l_pos + (1 - p_pos) * (1 - l_pos)

            return (po - pe) / (1 - pe) if pe != 1 else 0.0

        except Exception:
            return 0.0

    def _calculate_pr_auc_approximation(
        self, scores: List[float], labels: List[int]
    ) -> float:
        """Calculate approximation of Precision-Recall AUC."""
        try:
            # Simple approximation using multiple thresholds
            thresholds = np.linspace(0, 1, 20)
            pr_points = []

            for threshold in thresholds:
                predictions = [1 if score >= threshold else 0 for score in scores]
                tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
                fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
                fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                pr_points.append((recall, precision))

            # Sort by recall and calculate AUC using trapezoidal rule
            pr_points.sort()
            auc = 0.0
            for i in range(1, len(pr_points)):
                width = pr_points[i][0] - pr_points[i - 1][0]
                height = (pr_points[i][1] + pr_points[i - 1][1]) / 2
                auc += width * height

            return max(0.0, min(1.0, auc))

        except Exception:
            return 0.0

    def _calculate_lift_metrics(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, float]:
        """Calculate lift and gain metrics."""
        try:
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Calculate lift
            total_positives = sum(labels)
            total_predictions = sum(predictions)
            predicted_positives = sum(
                1 for p, l in zip(predictions, labels) if p == 1 and l == 1
            )

            if total_predictions > 0 and total_positives > 0:
                precision = predicted_positives / total_predictions
                baseline_precision = total_positives / len(labels)
                lift = precision / baseline_precision if baseline_precision > 0 else 0.0
            else:
                lift = 0.0

            # Calculate gain (cumulative lift at top percentile)
            if total_predictions > 0:
                gain = (predicted_positives / total_positives) / (
                    total_predictions / len(labels)
                )
            else:
                gain = 0.0

            return {"lift": lift, "gain": gain}

        except Exception:
            return {"lift": 0.0, "gain": 0.0}

    def _analyze_errors(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, Any]:
        """Analyze false positive and false negative patterns."""
        try:
            predictions = [1 if score >= threshold else 0 for score in scores]

            # Collect error cases
            false_positives = [
                {"score": score, "prediction": pred, "label": label}
                for score, pred, label in zip(scores, predictions, labels)
                if pred == 1 and label == 0
            ]

            false_negatives = [
                {"score": score, "prediction": pred, "label": label}
                for score, pred, label in zip(scores, predictions, labels)
                if pred == 0 and label == 1
            ]

            # Analyze score distributions for errors
            fp_scores = [fp["score"] for fp in false_positives]
            fn_scores = [fn["score"] for fn in false_negatives]

            return {
                "false_positive_analysis": {
                    "count": len(false_positives),
                    "score_distribution": {
                        "mean": np.mean(fp_scores) if fp_scores else 0.0,
                        "std": np.std(fp_scores) if fp_scores else 0.0,
                        "min": min(fp_scores) if fp_scores else 0.0,
                        "max": max(fp_scores) if fp_scores else 0.0,
                    },
                },
                "false_negative_analysis": {
                    "count": len(false_negatives),
                    "score_distribution": {
                        "mean": np.mean(fn_scores) if fn_scores else 0.0,
                        "std": np.std(fn_scores) if fn_scores else 0.0,
                        "min": min(fn_scores) if fn_scores else 0.0,
                        "max": max(fn_scores) if fn_scores else 0.0,
                    },
                },
                "error_patterns": self._identify_error_patterns(
                    false_positives, false_negatives, threshold
                ),
            }

        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            return {"error": str(e)}

    def _identify_error_patterns(
        self,
        false_positives: List[Dict[str, Any]],
        false_negatives: List[Dict[str, Any]],
        threshold: float,
    ) -> Dict[str, Any]:
        """Identify patterns in classification errors."""
        try:
            patterns = []

            # Check for threshold boundary issues
            fp_near_threshold = sum(
                1 for fp in false_positives if abs(fp["score"] - threshold) < 0.1
            )
            fn_near_threshold = sum(
                1 for fn in false_negatives if abs(fn["score"] - threshold) < 0.1
            )

            if fp_near_threshold > len(false_positives) * 0.3:
                patterns.append(
                    {
                        "type": "threshold_boundary_fps",
                        "description": "Many false positives occur near threshold boundary",
                        "severity": "medium",
                    }
                )

            if fn_near_threshold > len(false_negatives) * 0.3:
                patterns.append(
                    {
                        "type": "threshold_boundary_fns",
                        "description": "Many false negatives occur near threshold boundary",
                        "severity": "medium",
                    }
                )

            # Check for score distribution issues
            if false_positives:
                fp_scores = [fp["score"] for fp in false_positives]
                if np.std(fp_scores) < 0.1:
                    patterns.append(
                        {
                            "type": "clustered_false_positives",
                            "description": "False positives clustered in narrow score range",
                            "severity": "high",
                        }
                    )

            return {"identified_patterns": patterns}

        except Exception as e:
            logger.error(f"Error pattern identification failed: {e}")
            return {"error": str(e)}

    def _calculate_temporal_trends(
        self, historical_data: List[Dict[str, Any]], threshold: float
    ) -> Dict[str, Any]:
        """Calculate temporal trends in performance metrics."""
        try:
            # Group data by time periods (daily)
            daily_metrics = {}

            for item in historical_data:
                timestamp = self._parse_timestamp(item.get("timestamp", ""))
                date_key = timestamp.date().isoformat()

                if date_key not in daily_metrics:
                    daily_metrics[date_key] = {"scores": [], "labels": []}

                try:
                    score = float(item.get("confidence", item.get("score", 0.5)))
                    label = int(
                        item.get("ground_truth", item.get("is_true_positive", 0))
                    )

                    daily_metrics[date_key]["scores"].append(score)
                    daily_metrics[date_key]["labels"].append(label)

                except (ValueError, TypeError):
                    continue

            # Calculate daily performance metrics
            daily_performance = []
            for date_key, data in daily_metrics.items():
                if len(data["scores"]) >= 3:  # Minimum data points per day
                    metrics = self._calculate_basic_metrics(
                        data["scores"], data["labels"], threshold
                    )
                    daily_performance.append(
                        {
                            "date": date_key,
                            "precision": metrics.precision,
                            "recall": metrics.recall,
                            "f1_score": metrics.f1_score,
                            "false_positive_rate": metrics.false_positive_rate,
                            "data_points": len(data["scores"]),
                        }
                    )

            # Calculate trends
            if len(daily_performance) >= 3:
                trends = self._calculate_metric_trends(daily_performance)
            else:
                trends = {"insufficient_data": True}

            return {
                "daily_performance": daily_performance[-30:],  # Last 30 days
                "trends": trends,
                "analysis_period_days": len(daily_performance),
            }

        except Exception as e:
            logger.error(f"Temporal trend calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_metric_trends(
        self, daily_performance: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate trends in performance metrics."""
        try:
            if len(daily_performance) < 3:
                return {"insufficient_data": True}

            # Extract metric values
            precision_values = [dp["precision"] for dp in daily_performance]
            recall_values = [dp["recall"] for dp in daily_performance]
            f1_values = [dp["f1_score"] for dp in daily_performance]
            fpr_values = [dp["false_positive_rate"] for dp in daily_performance]

            # Calculate simple linear trends (slope)
            x = np.arange(len(precision_values))

            precision_trend = np.polyfit(x, precision_values, 1)[0]
            recall_trend = np.polyfit(x, recall_values, 1)[0]
            f1_trend = np.polyfit(x, f1_values, 1)[0]
            fpr_trend = np.polyfit(x, fpr_values, 1)[0]

            return {
                "precision_trend": {
                    "slope": precision_trend,
                    "direction": "improving" if precision_trend > 0 else "declining",
                },
                "recall_trend": {
                    "slope": recall_trend,
                    "direction": "improving" if recall_trend > 0 else "declining",
                },
                "f1_trend": {
                    "slope": f1_trend,
                    "direction": "improving" if f1_trend > 0 else "declining",
                },
                "false_positive_rate_trend": {
                    "slope": fpr_trend,
                    "direction": "improving" if fpr_trend < 0 else "declining",
                },
            }

        except Exception as e:
            logger.error(f"Metric trend calculation failed: {e}")
            return {"error": str(e)}

    def _calculate_statistical_significance(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, Any]:
        """Calculate statistical significance of performance metrics."""
        try:
            # Bootstrap confidence intervals for key metrics
            n_bootstrap = 500
            bootstrap_f1_scores = []
            bootstrap_precisions = []
            bootstrap_recalls = []

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(scores), size=len(scores), replace=True)
                bootstrap_scores = [scores[i] for i in indices]
                bootstrap_labels = [labels[i] for i in indices]

                # Calculate metrics
                metrics = self._calculate_basic_metrics(
                    bootstrap_scores, bootstrap_labels, threshold
                )

                bootstrap_f1_scores.append(metrics.f1_score)
                bootstrap_precisions.append(metrics.precision)
                bootstrap_recalls.append(metrics.recall)

            # Calculate confidence intervals (95%)
            confidence_intervals = {
                "f1_score": {
                    "lower": np.percentile(bootstrap_f1_scores, 2.5),
                    "upper": np.percentile(bootstrap_f1_scores, 97.5),
                },
                "precision": {
                    "lower": np.percentile(bootstrap_precisions, 2.5),
                    "upper": np.percentile(bootstrap_precisions, 97.5),
                },
                "recall": {
                    "lower": np.percentile(bootstrap_recalls, 2.5),
                    "upper": np.percentile(bootstrap_recalls, 97.5),
                },
            }

            return {
                "confidence_intervals_95": confidence_intervals,
                "bootstrap_samples": n_bootstrap,
                "statistical_power": self._estimate_statistical_power(len(scores)),
            }

        except Exception as e:
            logger.error(f"Statistical significance calculation failed: {e}")
            return {"error": str(e)}

    def _estimate_statistical_power(self, sample_size: int) -> str:
        """Estimate statistical power based on sample size."""
        if sample_size >= 1000:
            return "high"
        elif sample_size >= 100:
            return "medium"
        elif sample_size >= 30:
            return "low"
        else:
            return "very_low"

    def _assess_data_quality(
        self, scores: List[float], labels: List[int]
    ) -> Dict[str, Any]:
        """Assess quality of data used for metrics calculation."""
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

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime object."""
        try:
            # Try common timestamp formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            # If all formats fail, return current time
            return datetime.utcnow()

        except Exception:
            return datetime.utcnow()

    def _create_empty_metrics(self, detector_id: str) -> Dict[str, Any]:
        """Create empty metrics result."""
        return {
            "detector_id": detector_id,
            "threshold": 0.5,
            "basic_metrics": {},
            "advanced_metrics": {},
            "error_analysis": {"message": "No historical data available"},
            "temporal_trends": {},
            "statistical_analysis": {},
            "data_quality": {"total_samples": 0},
            "analysis_metadata": {
                "data_points": 0,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _create_insufficient_data_metrics(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, Any]:
        """Create metrics result for insufficient data."""
        return {
            "detector_id": detector_id,
            "threshold": 0.5,
            "basic_metrics": {},
            "advanced_metrics": {},
            "error_analysis": {"message": f"Insufficient data: {sample_count} samples"},
            "temporal_trends": {},
            "statistical_analysis": {},
            "data_quality": {
                "total_samples": sample_count,
                "data_sufficiency": "insufficient",
            },
            "analysis_metadata": {
                "data_points": sample_count,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }

    def _create_error_metrics(
        self, detector_id: str, error_message: str
    ) -> Dict[str, Any]:
        """Create metrics result for error cases."""
        return {
            "detector_id": detector_id,
            "threshold": 0.5,
            "basic_metrics": {},
            "advanced_metrics": {"error": error_message},
            "error_analysis": {"error": error_message},
            "temporal_trends": {"error": error_message},
            "statistical_analysis": {"error": error_message},
            "data_quality": {"error": error_message},
            "analysis_metadata": {
                "error": error_message,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }
