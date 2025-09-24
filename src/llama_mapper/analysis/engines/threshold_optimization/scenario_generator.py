"""
Scenario Generator for threshold impact simulation.
"""

import logging
from typing import Any, Dict, List
import numpy as np
from dataclasses import dataclass

from .types import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class ImpactScenario:
    """Impact scenario for threshold change simulation."""

    scenario_name: str
    probability: float
    performance_metrics: PerformanceMetrics
    business_impact: Dict[str, float]
    operational_impact: Dict[str, Any]


class ScenarioGenerator:
    """
    Generates multiple impact scenarios for threshold changes.

    Responsible for creating different scenarios (optimistic, pessimistic, etc.)
    with appropriate noise and probability distributions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scenario_configs = self._load_scenario_configs()

    async def generate_scenarios(
        self,
        scores: List[float],
        labels: List[int],
        proposed_threshold: float,
        time_horizon_days: int,
        business_context: Dict[str, Any],
    ) -> List[ImpactScenario]:
        """Generate multiple impact scenarios."""
        try:
            scenarios = []

            for config in self.scenario_configs:
                scenario = await self._generate_single_scenario(
                    scores,
                    labels,
                    proposed_threshold,
                    time_horizon_days,
                    business_context,
                    config,
                )
                scenarios.append(scenario)

            return scenarios

        except Exception as e:
            logger.error("Scenario generation failed: %s", str(e))
            return []

    async def _generate_single_scenario(
        self,
        scores: List[float],
        labels: List[int],
        proposed_threshold: float,
        time_horizon_days: int,
        business_context: Dict[str, Any],
        scenario_config: Dict[str, Any],
    ) -> ImpactScenario:
        """Generate a single impact scenario."""
        try:
            scenario_name = scenario_config["name"]
            noise_factor = scenario_config["noise_factor"]
            probability = scenario_config["probability"]

            # Add noise to simulate real-world variability
            noisy_scores = self._add_scenario_noise(scores, noise_factor)

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                noisy_scores, labels, proposed_threshold
            )

            return ImpactScenario(
                scenario_name=scenario_name,
                probability=probability,
                performance_metrics=performance_metrics,
                business_impact={},  # Will be calculated by BusinessImpactCalculator
                operational_impact={},  # Will be calculated by OperationalImpactCalculator
            )

        except Exception as e:
            logger.error("Single scenario generation failed: %s", str(e))
            return self._create_error_scenario(
                scenario_config.get("name", "error"), str(e)
            )

    def _add_scenario_noise(
        self, scores: List[float], noise_factor: float
    ) -> List[float]:
        """Add noise to scores to simulate scenario variability."""
        try:
            np.random.seed(42)  # For reproducible results
            noise_std = 0.05 * noise_factor  # 5% base noise scaled by factor

            noisy_scores = []
            for score in scores:
                noise = np.random.normal(0, noise_std)
                noisy_score = max(0.0, min(1.0, score + noise))  # Clamp to [0, 1]
                noisy_scores.append(noisy_score)

            return noisy_scores

        except Exception as e:
            logger.error("Noise addition failed: %s", str(e))
            return scores

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
            logger.error("Performance metrics calculation failed: %s", str(e))
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

    def _load_scenario_configs(self) -> List[Dict[str, Any]]:
        """Load scenario configuration parameters."""
        return [
            {"name": "optimistic", "noise_factor": 0.8, "probability": 0.2},
            {"name": "realistic", "noise_factor": 1.0, "probability": 0.5},
            {"name": "pessimistic", "noise_factor": 1.2, "probability": 0.2},
            {"name": "worst_case", "noise_factor": 1.5, "probability": 0.08},
            {"name": "best_case", "noise_factor": 0.6, "probability": 0.02},
        ]

    def _create_error_scenario(
        self, scenario_name: str, error_message: str
    ) -> ImpactScenario:
        """Create error scenario."""
        return ImpactScenario(
            scenario_name=scenario_name,
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
            business_impact={"error": error_message},
            operational_impact={"error": error_message},
        )
