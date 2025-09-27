"""Routing optimization ML component following SRP.

This module provides ONLY routing optimization capabilities:
- Multi-armed bandit algorithms
- Routing decision optimization
- Exploration vs exploitation balance
- Performance-based routing adaptation
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class BanditArm:
    """Multi-armed bandit arm representing a detector."""

    detector_id: str
    total_pulls: int
    total_reward: float
    success_count: int
    failure_count: int
    last_updated: datetime
    confidence_interval: Tuple[float, float] = (0.0, 1.0)


@dataclass
class RoutingDecision:
    """Routing optimization decision."""

    selected_detectors: List[str]
    selection_probabilities: Dict[str, float]
    exploration_factor: float
    expected_reward: float
    confidence: float
    algorithm_used: str


class RoutingOptimizer:
    """ML-based routing optimizer using multi-armed bandit algorithms following SRP."""

    def __init__(self, exploration_rate: float = 0.1, confidence_level: float = 0.95):
        """Initialize routing optimizer.

        Args:
            exploration_rate: Rate of exploration vs exploitation
            confidence_level: Confidence level for UCB algorithm
        """
        self.exploration_rate = exploration_rate
        self.confidence_level = confidence_level
        self.arms: Dict[str, BanditArm] = {}
        self.total_pulls = 0
        self.reward_history: List[Tuple[str, float, datetime]] = []
        self.algorithm_performance: Dict[str, List[float]] = defaultdict(list)

        # Algorithm parameters
        self.ucb_c = 2.0  # UCB exploration parameter
        self.thompson_alpha = 1.0  # Thompson sampling alpha
        self.thompson_beta = 1.0  # Thompson sampling beta
        self.epsilon = 0.1  # Epsilon-greedy parameter

        # Performance tracking
        self.window_size = 100  # Rolling window for performance calculation
        self.min_pulls_for_confidence = 10

    def add_detector(self, detector_id: str) -> None:
        """Add a new detector to the bandit.

        Args:
            detector_id: Detector identifier
        """
        if detector_id not in self.arms:
            self.arms[detector_id] = BanditArm(
                detector_id=detector_id,
                total_pulls=0,
                total_reward=0.0,
                success_count=0,
                failure_count=0,
                last_updated=datetime.now(),
            )
            logger.info("Added detector to routing optimizer: %s", detector_id)

    def remove_detector(self, detector_id: str) -> None:
        """Remove a detector from the bandit.

        Args:
            detector_id: Detector identifier
        """
        if detector_id in self.arms:
            del self.arms[detector_id]
            logger.info("Removed detector from routing optimizer: %s", detector_id)

    def update_reward(
        self, detector_id: str, reward: float, success: bool = True
    ) -> None:
        """Update reward for a detector based on performance.

        Args:
            detector_id: Detector identifier
            reward: Reward value (typically 0-1 based on performance)
            success: Whether the detection was successful
        """
        if detector_id not in self.arms:
            self.add_detector(detector_id)

        arm = self.arms[detector_id]
        arm.total_pulls += 1
        arm.total_reward += reward
        arm.last_updated = datetime.now()

        if success:
            arm.success_count += 1
        else:
            arm.failure_count += 1

        # Update confidence interval
        arm.confidence_interval = self._calculate_confidence_interval(arm)

        # Track reward history
        self.reward_history.append((detector_id, reward, datetime.now()))

        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.reward_history = [
            (det_id, rew, timestamp)
            for det_id, rew, timestamp in self.reward_history
            if timestamp > cutoff_time
        ]

        self.total_pulls += 1

        logger.debug(
            "Updated detector reward",
            extra={
                "detector_id": detector_id,
                "reward": reward,
                "success": success,
                "total_pulls": arm.total_pulls,
                "avg_reward": arm.total_reward / arm.total_pulls,
            },
        )

    def _calculate_confidence_interval(self, arm: BanditArm) -> Tuple[float, float]:
        """Calculate confidence interval for an arm.

        Args:
            arm: Bandit arm

        Returns:
            Confidence interval tuple (lower, upper)
        """
        if arm.total_pulls < self.min_pulls_for_confidence:
            return (0.0, 1.0)

        mean_reward = arm.total_reward / arm.total_pulls

        # Use Wilson score interval for binomial confidence interval
        n = arm.total_pulls
        p = mean_reward
        z = 1.96  # 95% confidence

        denominator = 1 + z**2 / n
        centre_adjusted_probability = p + z**2 / (2 * n)
        adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)

        lower_bound = (
            centre_adjusted_probability - z * adjusted_standard_deviation
        ) / denominator
        upper_bound = (
            centre_adjusted_probability + z * adjusted_standard_deviation
        ) / denominator

        return (max(0.0, lower_bound), min(1.0, upper_bound))

    def select_detectors_ucb(
        self, available_detectors: List[str], num_detectors: int = 3
    ) -> RoutingDecision:
        """Select detectors using Upper Confidence Bound algorithm.

        Args:
            available_detectors: List of available detector IDs
            num_detectors: Number of detectors to select

        Returns:
            Routing decision
        """
        if not available_detectors:
            return self._empty_decision("ucb")

        # Ensure all detectors are in arms
        for detector_id in available_detectors:
            if detector_id not in self.arms:
                self.add_detector(detector_id)

        # Calculate UCB values
        ucb_values = {}
        for detector_id in available_detectors:
            arm = self.arms[detector_id]

            if arm.total_pulls == 0:
                # Unplayed arms get infinite UCB value
                ucb_values[detector_id] = float("inf")
            else:
                mean_reward = arm.total_reward / arm.total_pulls
                confidence_bonus = self.ucb_c * math.sqrt(
                    math.log(self.total_pulls) / arm.total_pulls
                )
                ucb_values[detector_id] = mean_reward + confidence_bonus

        # Select top UCB detectors
        sorted_detectors = sorted(ucb_values.items(), key=lambda x: x[1], reverse=True)
        selected = [detector_id for detector_id, _ in sorted_detectors[:num_detectors]]

        # Calculate selection probabilities (softmax of UCB values)
        selected_ucb_values = [ucb_values[det_id] for det_id in selected]
        probabilities = self._softmax(selected_ucb_values)
        selection_probabilities = dict(zip(selected, probabilities))

        # Calculate expected reward
        expected_reward = np.mean(
            [
                self.arms[det_id].total_reward / max(1, self.arms[det_id].total_pulls)
                for det_id in selected
            ]
        )

        # Calculate exploration factor
        exploration_factor = np.mean(
            [
                self.ucb_c
                * math.sqrt(
                    math.log(self.total_pulls) / max(1, self.arms[det_id].total_pulls)
                )
                for det_id in selected
            ]
        )

        return RoutingDecision(
            selected_detectors=selected,
            selection_probabilities=selection_probabilities,
            exploration_factor=exploration_factor,
            expected_reward=expected_reward,
            confidence=0.9,
            algorithm_used="ucb",
        )

    def select_detectors_thompson(
        self, available_detectors: List[str], num_detectors: int = 3
    ) -> RoutingDecision:
        """Select detectors using Thompson Sampling algorithm.

        Args:
            available_detectors: List of available detector IDs
            num_detectors: Number of detectors to select

        Returns:
            Routing decision
        """
        if not available_detectors:
            return self._empty_decision("thompson")

        # Ensure all detectors are in arms
        for detector_id in available_detectors:
            if detector_id not in self.arms:
                self.add_detector(detector_id)

        # Sample from Beta distribution for each detector
        sampled_values = {}
        for detector_id in available_detectors:
            arm = self.arms[detector_id]

            # Beta distribution parameters
            alpha = self.thompson_alpha + arm.success_count
            beta = self.thompson_beta + arm.failure_count

            # Sample from Beta distribution
            sampled_values[detector_id] = np.random.beta(alpha, beta)

        # Select detectors with highest sampled values
        sorted_detectors = sorted(
            sampled_values.items(), key=lambda x: x[1], reverse=True
        )
        selected = [detector_id for detector_id, _ in sorted_detectors[:num_detectors]]

        # Calculate selection probabilities based on sampled values
        selected_values = [sampled_values[det_id] for det_id in selected]
        probabilities = self._softmax(selected_values)
        selection_probabilities = dict(zip(selected, probabilities))

        # Calculate expected reward
        expected_reward = np.mean(
            [
                self.arms[det_id].total_reward / max(1, self.arms[det_id].total_pulls)
                for det_id in selected
            ]
        )

        # Exploration factor based on uncertainty
        exploration_factor = np.mean(
            [1.0 / max(1, self.arms[det_id].total_pulls) for det_id in selected]
        )

        return RoutingDecision(
            selected_detectors=selected,
            selection_probabilities=selection_probabilities,
            exploration_factor=exploration_factor,
            expected_reward=expected_reward,
            confidence=0.85,
            algorithm_used="thompson",
        )

    def select_detectors_epsilon_greedy(
        self, available_detectors: List[str], num_detectors: int = 3
    ) -> RoutingDecision:
        """Select detectors using Epsilon-Greedy algorithm.

        Args:
            available_detectors: List of available detector IDs
            num_detectors: Number of detectors to select

        Returns:
            Routing decision
        """
        if not available_detectors:
            return self._empty_decision("epsilon_greedy")

        # Ensure all detectors are in arms
        for detector_id in available_detectors:
            if detector_id not in self.arms:
                self.add_detector(detector_id)

        selected = []
        selection_probabilities = {}

        for _ in range(min(num_detectors, len(available_detectors))):
            if random.random() < self.epsilon:
                # Explore: select random detector
                remaining_detectors = [
                    d for d in available_detectors if d not in selected
                ]
                if remaining_detectors:
                    detector_id = random.choice(remaining_detectors)
                    selected.append(detector_id)
                    selection_probabilities[detector_id] = self.epsilon / len(
                        remaining_detectors
                    )
            else:
                # Exploit: select best detector
                remaining_detectors = [
                    d for d in available_detectors if d not in selected
                ]
                if remaining_detectors:
                    best_detector = max(
                        remaining_detectors,
                        key=lambda d: self.arms[d].total_reward
                        / max(1, self.arms[d].total_pulls),
                    )
                    selected.append(best_detector)
                    selection_probabilities[best_detector] = 1.0 - self.epsilon

        # Normalize probabilities
        total_prob = sum(selection_probabilities.values())
        if total_prob > 0:
            selection_probabilities = {
                k: v / total_prob for k, v in selection_probabilities.items()
            }

        # Calculate expected reward
        expected_reward = np.mean(
            [
                self.arms[det_id].total_reward / max(1, self.arms[det_id].total_pulls)
                for det_id in selected
            ]
        )

        return RoutingDecision(
            selected_detectors=selected,
            selection_probabilities=selection_probabilities,
            exploration_factor=self.epsilon,
            expected_reward=expected_reward,
            confidence=0.8,
            algorithm_used="epsilon_greedy",
        )

    def select_detectors_adaptive(
        self,
        available_detectors: List[str],
        num_detectors: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """Adaptive detector selection that chooses the best algorithm.

        Args:
            available_detectors: List of available detector IDs
            num_detectors: Number of detectors to select
            context: Optional context information

        Returns:
            Routing decision
        """
        # Choose algorithm based on current performance and context
        algorithm = self._choose_best_algorithm(context)

        if algorithm == "ucb":
            return self.select_detectors_ucb(available_detectors, num_detectors)
        elif algorithm == "thompson":
            return self.select_detectors_thompson(available_detectors, num_detectors)
        else:
            return self.select_detectors_epsilon_greedy(
                available_detectors, num_detectors
            )

    def _choose_best_algorithm(self, context: Optional[Dict[str, Any]]) -> str:
        """Choose the best algorithm based on current performance.

        Args:
            context: Optional context information

        Returns:
            Best algorithm name
        """
        # Simple algorithm selection based on recent performance
        if len(self.algorithm_performance["ucb"]) < 10:
            return "ucb"  # Default to UCB initially

        # Calculate recent performance for each algorithm
        recent_performance = {}
        for algorithm in ["ucb", "thompson", "epsilon_greedy"]:
            if self.algorithm_performance[algorithm]:
                recent_perf = self.algorithm_performance[algorithm][-10:]
                recent_performance[algorithm] = np.mean(recent_perf)
            else:
                recent_performance[algorithm] = 0.0

        # Choose algorithm with best recent performance
        best_algorithm = max(recent_performance.items(), key=lambda x: x[1])[0]

        # Add some exploration
        if random.random() < 0.1:  # 10% chance to explore other algorithms
            algorithms = ["ucb", "thompson", "epsilon_greedy"]
            algorithms.remove(best_algorithm)
            return random.choice(algorithms)

        return best_algorithm

    def _softmax(self, values: List[float], temperature: float = 1.0) -> List[float]:
        """Apply softmax function to values.

        Args:
            values: List of values
            temperature: Temperature parameter

        Returns:
            Softmax probabilities
        """
        if not values:
            return []

        # Handle infinite values
        finite_values = [v if not math.isinf(v) else 1000.0 for v in values]

        exp_values = np.exp(np.array(finite_values) / temperature)
        return (exp_values / np.sum(exp_values)).tolist()

    def _empty_decision(self, algorithm: str) -> RoutingDecision:
        """Create empty routing decision.

        Args:
            algorithm: Algorithm name

        Returns:
            Empty routing decision
        """
        return RoutingDecision(
            selected_detectors=[],
            selection_probabilities={},
            exploration_factor=0.0,
            expected_reward=0.0,
            confidence=0.0,
            algorithm_used=algorithm,
        )

    def get_detector_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all detectors.

        Returns:
            Detector statistics
        """
        stats = {}

        for detector_id, arm in self.arms.items():
            avg_reward = arm.total_reward / max(1, arm.total_pulls)
            success_rate = arm.success_count / max(1, arm.total_pulls)

            stats[detector_id] = {
                "total_pulls": arm.total_pulls,
                "average_reward": avg_reward,
                "success_rate": success_rate,
                "confidence_interval": arm.confidence_interval,
                "last_updated": arm.last_updated.isoformat(),
            }

        return stats

    def get_optimizer_stats(self) -> Dict[str, Any]:
        """Get routing optimizer statistics.

        Returns:
            Optimizer statistics
        """
        return {
            "total_pulls": self.total_pulls,
            "tracked_detectors": len(self.arms),
            "exploration_rate": self.exploration_rate,
            "confidence_level": self.confidence_level,
            "reward_history_size": len(self.reward_history),
            "algorithm_performance": {
                alg: np.mean(perfs) if perfs else 0.0
                for alg, perfs in self.algorithm_performance.items()
            },
        }

    def reset_statistics(self) -> None:
        """Reset all statistics and start fresh."""
        self.arms.clear()
        self.total_pulls = 0
        self.reward_history.clear()
        self.algorithm_performance.clear()
        logger.info("Reset routing optimizer statistics")

    def export_model(self) -> Dict[str, Any]:
        """Export model state for persistence.

        Returns:
            Model state dictionary
        """
        return {
            "arms": {
                detector_id: {
                    "detector_id": arm.detector_id,
                    "total_pulls": arm.total_pulls,
                    "total_reward": arm.total_reward,
                    "success_count": arm.success_count,
                    "failure_count": arm.failure_count,
                    "last_updated": arm.last_updated.isoformat(),
                    "confidence_interval": arm.confidence_interval,
                }
                for detector_id, arm in self.arms.items()
            },
            "total_pulls": self.total_pulls,
            "exploration_rate": self.exploration_rate,
            "confidence_level": self.confidence_level,
            "parameters": {
                "ucb_c": self.ucb_c,
                "thompson_alpha": self.thompson_alpha,
                "thompson_beta": self.thompson_beta,
                "epsilon": self.epsilon,
            },
        }

    def import_model(self, model_state: Dict[str, Any]) -> None:
        """Import model state from persistence.

        Args:
            model_state: Model state dictionary
        """
        try:
            # Import arms
            self.arms = {}
            for detector_id, arm_data in model_state.get("arms", {}).items():
                self.arms[detector_id] = BanditArm(
                    detector_id=arm_data["detector_id"],
                    total_pulls=arm_data["total_pulls"],
                    total_reward=arm_data["total_reward"],
                    success_count=arm_data["success_count"],
                    failure_count=arm_data["failure_count"],
                    last_updated=datetime.fromisoformat(arm_data["last_updated"]),
                    confidence_interval=tuple(arm_data["confidence_interval"]),
                )

            # Import other state
            self.total_pulls = model_state.get("total_pulls", 0)
            self.exploration_rate = model_state.get("exploration_rate", 0.1)
            self.confidence_level = model_state.get("confidence_level", 0.95)

            # Import parameters
            params = model_state.get("parameters", {})
            self.ucb_c = params.get("ucb_c", 2.0)
            self.thompson_alpha = params.get("thompson_alpha", 1.0)
            self.thompson_beta = params.get("thompson_beta", 1.0)
            self.epsilon = params.get("epsilon", 0.1)

            logger.info("Imported routing optimizer model state")

        except Exception as e:
            logger.error("Failed to import model state: %s", str(e))
            raise
