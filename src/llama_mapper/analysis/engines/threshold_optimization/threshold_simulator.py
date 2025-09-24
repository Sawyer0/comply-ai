"""
Threshold Simulator for impact prediction using historical data.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import numpy as np

from .types import PerformanceMetrics

logger = logging.getLogger(__name__)


class ThresholdSimulator:
    """
    Threshold simulator for impact prediction using historical data.

    Simulates the impact of threshold changes on detector performance
    using historical data and statistical modeling.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.simulation_periods = self.config.get("simulation_periods", 30)
        self.confidence_level = self.config.get("confidence_level", 0.95)

    async def simulate_threshold_impact(
        self,
        detector_id: str,
        current_threshold: float,
        proposed_threshold: float,
        historical_data: List[Dict[str, Any]],
        time_horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Simulate impact of threshold change using historical data.

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

            if len(scores) < 20:
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

            # Generate impact projections
            impact_projections = self._generate_impact_projections(
                current_performance, proposed_performance, time_horizon_days
            )

            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                scores, labels, proposed_threshold
            )

            # Assess risks and benefits
            risk_assessment = self._assess_risks_and_benefits(
                current_performance, proposed_performance
            )

            result = {
                "detector_id": detector_id,
                "current_threshold": current_threshold,
                "proposed_threshold": proposed_threshold,
                "current_performance": current_performance.__dict__,
                "proposed_performance": proposed_performance.__dict__,
                "impact_projections": impact_projections,
                "confidence_intervals": confidence_intervals,
                "risk_assessment": risk_assessment,
                "simulation_metadata": {
                    "data_points": len(scores),
                    "time_horizon_days": time_horizon_days,
                    "confidence_level": self.confidence_level,
                },
                "simulation_timestamp": datetime.utcnow().isoformat(),
            }

            logger.info(
                "Threshold simulation completed",
                detector_id=detector_id,
                current_threshold=current_threshold,
                proposed_threshold=proposed_threshold,
            )

            return result

        except Exception as e:
            logger.error(
                "Threshold simulation failed", error=str(e), detector_id=detector_id
            )
            return self._create_error_simulation(detector_id, str(e))

    def _extract_scores_and_labels(
        self, historical_data: List[Dict[str, Any]]
    ) -> tuple[List[float], List[int]]:
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
        """Calculate performance metrics for given threshold."""
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
            logger.error(
                f"Performance calculation failed for threshold {threshold}: {e}"
            )
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

    def _generate_impact_projections(
        self,
        current_performance: PerformanceMetrics,
        proposed_performance: PerformanceMetrics,
        time_horizon_days: int,
    ) -> Dict[str, Any]:
        """Generate impact projections over time horizon."""
        try:
            # Calculate daily detection volume (estimated)
            daily_detections = max(
                100,
                current_performance.true_positives
                + current_performance.false_positives,
            )

            # Project changes over time horizon
            total_detections = daily_detections * time_horizon_days

            # Calculate expected changes
            fp_change = (
                proposed_performance.false_positive_rate
                - current_performance.false_positive_rate
            ) * total_detections

            fn_change = (
                proposed_performance.false_negative_rate
                - current_performance.false_negative_rate
            ) * total_detections

            precision_change = (
                proposed_performance.precision - current_performance.precision
            )

            recall_change = proposed_performance.recall - current_performance.recall

            return {
                "time_horizon_days": time_horizon_days,
                "projected_detection_volume": total_detections,
                "expected_changes": {
                    "false_positives_delta": fp_change,
                    "false_negatives_delta": fn_change,
                    "precision_delta": precision_change,
                    "recall_delta": recall_change,
                },
                "percentage_changes": {
                    "false_positive_rate_change": (
                        (
                            proposed_performance.false_positive_rate
                            - current_performance.false_positive_rate
                        )
                        / max(current_performance.false_positive_rate, 0.001)
                        * 100
                    ),
                    "false_negative_rate_change": (
                        (
                            proposed_performance.false_negative_rate
                            - current_performance.false_negative_rate
                        )
                        / max(current_performance.false_negative_rate, 0.001)
                        * 100
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Impact projection generation failed: {e}")
            return {"error": str(e)}

    def _calculate_confidence_intervals(
        self, scores: List[float], labels: List[int], threshold: float
    ) -> Dict[str, Any]:
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

            precision_values = [m.precision for m in bootstrap_metrics]
            recall_values = [m.recall for m in bootstrap_metrics]
            f1_values = [m.f1_score for m in bootstrap_metrics]

            return {
                "confidence_level": self.confidence_level,
                "precision": {
                    "lower": np.percentile(precision_values, lower_percentile),
                    "upper": np.percentile(precision_values, upper_percentile),
                },
                "recall": {
                    "lower": np.percentile(recall_values, lower_percentile),
                    "upper": np.percentile(recall_values, upper_percentile),
                },
                "f1_score": {
                    "lower": np.percentile(f1_values, lower_percentile),
                    "upper": np.percentile(f1_values, upper_percentile),
                },
            }

        except Exception as e:
            logger.error(f"Confidence interval calculation failed: {e}")
            return {"error": str(e)}

    def _assess_risks_and_benefits(
        self,
        current_performance: PerformanceMetrics,
        proposed_performance: PerformanceMetrics,
    ) -> Dict[str, Any]:
        """Assess risks and benefits of threshold change."""
        try:
            risks = []
            benefits = []

            # Analyze false positive rate change
            fp_change = (
                proposed_performance.false_positive_rate
                - current_performance.false_positive_rate
            )
            if fp_change > 0.05:  # 5% increase threshold
                risks.append(
                    {
                        "type": "increased_false_positives",
                        "severity": "high" if fp_change > 0.1 else "medium",
                        "description": f"False positive rate may increase by {fp_change:.1%}",
                    }
                )
            elif fp_change < -0.05:
                benefits.append(
                    {
                        "type": "reduced_false_positives",
                        "impact": "high" if fp_change < -0.1 else "medium",
                        "description": f"False positive rate may decrease by {abs(fp_change):.1%}",
                    }
                )

            # Analyze false negative rate change
            fn_change = (
                proposed_performance.false_negative_rate
                - current_performance.false_negative_rate
            )
            if fn_change > 0.05:
                risks.append(
                    {
                        "type": "increased_false_negatives",
                        "severity": "critical" if fn_change > 0.1 else "high",
                        "description": f"False negative rate may increase by {fn_change:.1%}",
                    }
                )
            elif fn_change < -0.05:
                benefits.append(
                    {
                        "type": "reduced_false_negatives",
                        "impact": "critical" if fn_change < -0.1 else "high",
                        "description": f"False negative rate may decrease by {abs(fn_change):.1%}",
                    }
                )

            # Overall recommendation
            overall_score = proposed_performance.f1_score - current_performance.f1_score

            if overall_score > 0.05:
                recommendation = "strongly_recommended"
            elif overall_score > 0.02:
                recommendation = "recommended"
            elif overall_score > -0.02:
                recommendation = "neutral"
            elif overall_score > -0.05:
                recommendation = "not_recommended"
            else:
                recommendation = "strongly_not_recommended"

            return {
                "risks": risks,
                "benefits": benefits,
                "overall_f1_change": overall_score,
                "recommendation": recommendation,
                "risk_level": self._calculate_overall_risk_level(risks),
            }

        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e)}

    def _calculate_overall_risk_level(self, risks: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level from individual risks."""
        if not risks:
            return "low"

        severity_scores = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }

        max_severity = max(
            severity_scores.get(risk.get("severity", "low"), 1) for risk in risks
        )

        if max_severity >= 4:
            return "critical"
        elif max_severity >= 3:
            return "high"
        elif max_severity >= 2:
            return "medium"
        else:
            return "low"

    def _create_empty_simulation(self, detector_id: str) -> Dict[str, Any]:
        """Create empty simulation result."""
        return {
            "detector_id": detector_id,
            "current_threshold": 0.5,
            "proposed_threshold": 0.5,
            "current_performance": {},
            "proposed_performance": {},
            "impact_projections": {"message": "No historical data available"},
            "confidence_intervals": {},
            "risk_assessment": {"recommendation": "insufficient_data"},
            "simulation_metadata": {"data_points": 0},
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_insufficient_data_simulation(
        self, detector_id: str, sample_count: int
    ) -> Dict[str, Any]:
        """Create simulation result for insufficient data."""
        return {
            "detector_id": detector_id,
            "current_threshold": 0.5,
            "proposed_threshold": 0.5,
            "current_performance": {},
            "proposed_performance": {},
            "impact_projections": {
                "message": f"Insufficient data: {sample_count} samples"
            },
            "confidence_intervals": {},
            "risk_assessment": {"recommendation": "insufficient_data"},
            "simulation_metadata": {"data_points": sample_count},
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }

    def _create_error_simulation(
        self, detector_id: str, error_message: str
    ) -> Dict[str, Any]:
        """Create simulation result for error cases."""
        return {
            "detector_id": detector_id,
            "current_threshold": 0.5,
            "proposed_threshold": 0.5,
            "current_performance": {},
            "proposed_performance": {},
            "impact_projections": {"error": error_message},
            "confidence_intervals": {"error": error_message},
            "risk_assessment": {"error": error_message},
            "simulation_metadata": {"error": error_message},
            "simulation_timestamp": datetime.utcnow().isoformat(),
        }
