"""Adaptive load balancing ML component following SRP.

This module provides ONLY load balancing capabilities:
- Real-time load monitoring
- Adaptive detector selection
- Performance-based routing
- Dynamic capacity management
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    ADAPTIVE = "adaptive"


@dataclass
class DetectorLoad:
    """Current load information for a detector."""

    detector_id: str
    active_requests: int
    avg_response_time_ms: float
    success_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    queue_length: int
    last_updated: datetime
    capacity_score: float = field(default=1.0)


@dataclass
class LoadBalancingDecision:
    """Load balancing decision result."""

    selected_detectors: List[str]
    load_distribution: Dict[str, float]
    strategy_used: str
    confidence: float
    reasoning: str
    estimated_response_time: float


class AdaptiveLoadBalancer:
    """ML-enhanced adaptive load balancer following SRP."""

    def __init__(
        self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    ):
        """Initialize adaptive load balancer.

        Args:
            strategy: Load balancing strategy to use
        """
        self.strategy = strategy
        self.detector_loads: Dict[str, DetectorLoad] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self.round_robin_index = 0
        self.load_update_interval = 5  # seconds
        self.last_load_update = datetime.now()

        # Adaptive parameters
        self.response_time_weight = 0.4
        self.success_rate_weight = 0.3
        self.load_weight = 0.2
        self.availability_weight = 0.1

        # Performance thresholds
        self.max_response_time_ms = 5000
        self.min_success_rate = 0.8
        self.max_error_rate = 0.2
        self.max_queue_length = 10

    def update_detector_load(self, detector_id: str, load_info: Dict[str, Any]) -> None:
        """Update load information for a detector.

        Args:
            detector_id: Detector identifier
            load_info: Load information dictionary
        """
        try:
            # Calculate capacity score based on multiple factors
            capacity_score = self._calculate_capacity_score(load_info)

            detector_load = DetectorLoad(
                detector_id=detector_id,
                active_requests=load_info.get("active_requests", 0),
                avg_response_time_ms=load_info.get("avg_response_time_ms", 100.0),
                success_rate=load_info.get("success_rate", 1.0),
                error_rate=load_info.get("error_rate", 0.0),
                cpu_usage=load_info.get("cpu_usage", 0.5),
                memory_usage=load_info.get("memory_usage", 0.5),
                queue_length=load_info.get("queue_length", 0),
                last_updated=datetime.now(),
                capacity_score=capacity_score,
            )

            self.detector_loads[detector_id] = detector_load

            # Update performance history
            self.performance_history[detector_id].append(
                {
                    "timestamp": datetime.now(),
                    "response_time": detector_load.avg_response_time_ms,
                    "success_rate": detector_load.success_rate,
                    "capacity_score": capacity_score,
                }
            )

            logger.debug(
                "Updated detector load",
                extra={
                    "detector_id": detector_id,
                    "capacity_score": capacity_score,
                    "active_requests": detector_load.active_requests,
                    "response_time": detector_load.avg_response_time_ms,
                },
            )

        except Exception as e:
            logger.error("Failed to update detector load: %s", str(e))

    def _calculate_capacity_score(self, load_info: Dict[str, Any]) -> float:
        """Calculate capacity score for a detector.

        Args:
            load_info: Load information

        Returns:
            Capacity score between 0 and 1 (higher = better capacity)
        """
        # Response time factor (lower is better)
        response_time = load_info.get("avg_response_time_ms", 100.0)
        response_time_factor = max(0, 1 - (response_time / self.max_response_time_ms))

        # Success rate factor (higher is better)
        success_rate = load_info.get("success_rate", 1.0)
        success_rate_factor = success_rate

        # Load factor (lower is better)
        active_requests = load_info.get("active_requests", 0)
        queue_length = load_info.get("queue_length", 0)
        load_factor = max(0, 1 - ((active_requests + queue_length) / 20.0))

        # Resource utilization factor
        cpu_usage = load_info.get("cpu_usage", 0.5)
        memory_usage = load_info.get("memory_usage", 0.5)
        resource_factor = max(0, 1 - max(cpu_usage, memory_usage))

        # Weighted combination
        capacity_score = (
            response_time_factor * self.response_time_weight
            + success_rate_factor * self.success_rate_weight
            + load_factor * self.load_weight
            + resource_factor * self.availability_weight
        )

        return max(0.0, min(1.0, capacity_score))

    async def select_detectors(
        self,
        available_detectors: List[str],
        num_detectors: int = 3,
        content_features: Optional[Dict[str, Any]] = None,
    ) -> LoadBalancingDecision:
        """Select optimal detectors based on current load and performance.

        Args:
            available_detectors: List of available detector IDs
            num_detectors: Number of detectors to select
            content_features: Optional content features for optimization

        Returns:
            Load balancing decision
        """
        try:
            # Update load information if needed
            await self._update_load_if_needed()

            # Filter available detectors
            healthy_detectors = self._filter_healthy_detectors(available_detectors)

            if not healthy_detectors:
                return self._fallback_selection(available_detectors, num_detectors)

            # Select based on strategy
            if self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return await self._adaptive_selection(
                    healthy_detectors, num_detectors, content_features
                )
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._response_time_selection(healthy_detectors, num_detectors)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(
                    healthy_detectors, num_detectors
                )
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(
                    healthy_detectors, num_detectors
                )
            else:  # ROUND_ROBIN
                return self._round_robin_selection(healthy_detectors, num_detectors)

        except Exception as e:
            logger.error("Detector selection failed: %s", str(e))
            return self._fallback_selection(available_detectors, num_detectors)

    def _filter_healthy_detectors(self, detectors: List[str]) -> List[str]:
        """Filter detectors based on health criteria.

        Args:
            detectors: List of detector IDs

        Returns:
            List of healthy detector IDs
        """
        healthy = []

        for detector_id in detectors:
            load_info = self.detector_loads.get(detector_id)

            if not load_info:
                # No load info available, assume healthy
                healthy.append(detector_id)
                continue

            # Check health criteria
            is_healthy = (
                load_info.success_rate >= self.min_success_rate
                and load_info.error_rate <= self.max_error_rate
                and load_info.queue_length <= self.max_queue_length
                and load_info.avg_response_time_ms <= self.max_response_time_ms
            )

            if is_healthy:
                healthy.append(detector_id)

        return healthy

    async def _adaptive_selection(
        self,
        detectors: List[str],
        num_detectors: int,
        content_features: Optional[Dict[str, Any]],
    ) -> LoadBalancingDecision:
        """Adaptive detector selection using ML-enhanced scoring.

        Args:
            detectors: Available detector IDs
            num_detectors: Number to select
            content_features: Content features for optimization

        Returns:
            Load balancing decision
        """
        # Calculate adaptive scores for each detector
        detector_scores = {}

        for detector_id in detectors:
            score = await self._calculate_adaptive_score(detector_id, content_features)
            detector_scores[detector_id] = score

        # Select top scoring detectors
        sorted_detectors = sorted(
            detector_scores.items(), key=lambda x: x[1], reverse=True
        )
        selected = [detector_id for detector_id, _ in sorted_detectors[:num_detectors]]

        # Calculate load distribution
        total_score = sum(score for _, score in sorted_detectors[:num_detectors])
        load_distribution = {
            detector_id: score / total_score
            for detector_id, score in sorted_detectors[:num_detectors]
        }

        # Estimate response time
        estimated_response_time = self._estimate_response_time(selected)

        return LoadBalancingDecision(
            selected_detectors=selected,
            load_distribution=load_distribution,
            strategy_used="adaptive",
            confidence=0.9,
            reasoning="Selected based on adaptive ML scoring",
            estimated_response_time=estimated_response_time,
        )

    async def _calculate_adaptive_score(
        self, detector_id: str, content_features: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate adaptive score for a detector.

        Args:
            detector_id: Detector ID
            content_features: Content features

        Returns:
            Adaptive score
        """
        load_info = self.detector_loads.get(detector_id)

        if not load_info:
            return 0.5  # Default score for unknown detectors

        # Base capacity score
        base_score = load_info.capacity_score

        # Historical performance factor
        history_factor = self._calculate_history_factor(detector_id)

        # Content affinity factor (if content features available)
        content_factor = 1.0
        if content_features:
            content_factor = self._calculate_content_affinity(
                detector_id, content_features
            )

        # Time-based factor (consider time of day patterns)
        time_factor = self._calculate_time_factor(detector_id)

        # Combine factors
        adaptive_score = (
            base_score * 0.4
            + history_factor * 0.3
            + content_factor * 0.2
            + time_factor * 0.1
        )

        return max(0.0, min(1.0, adaptive_score))

    def _calculate_history_factor(self, detector_id: str) -> float:
        """Calculate historical performance factor.

        Args:
            detector_id: Detector ID

        Returns:
            History factor between 0 and 1
        """
        history = self.performance_history.get(detector_id, deque())

        if not history:
            return 0.5  # Default for no history

        # Calculate trend in performance
        recent_scores = [entry["capacity_score"] for entry in list(history)[-10:]]

        if len(recent_scores) < 2:
            return recent_scores[0] if recent_scores else 0.5

        # Calculate trend (improving vs declining)
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        # Base score from recent average
        avg_score = np.mean(recent_scores)

        # Adjust for trend
        trend_adjustment = min(0.2, max(-0.2, trend * 10))

        return max(0.0, min(1.0, avg_score + trend_adjustment))

    def _calculate_content_affinity(
        self, detector_id: str, content_features: Dict[str, Any]
    ) -> float:
        """Calculate content affinity factor.

        Args:
            detector_id: Detector ID
            content_features: Content features

        Returns:
            Content affinity factor
        """
        # Simple content-detector affinity mapping
        # In production, this could be learned from historical data

        content_type = content_features.get("content_type", "text")
        complexity = content_features.get("complexity_score", 0.5)

        # Detector type affinity (simplified)
        type_affinities = {
            "pii-detector": {"email": 1.0, "text": 0.8, "json": 0.6},
            "json-validator": {"json": 1.0, "text": 0.3, "xml": 0.4},
            "text-analyzer": {"text": 1.0, "email": 0.8, "log": 0.7},
            "security-scanner": {"code": 1.0, "sql": 0.9, "text": 0.6},
        }

        base_affinity = type_affinities.get(detector_id, {}).get(content_type, 0.7)

        # Adjust for complexity
        if complexity > 0.7:
            # High complexity content might need specialized detectors
            complexity_factor = 0.9 if "advanced" in detector_id else 0.8
        else:
            complexity_factor = 1.0

        return base_affinity * complexity_factor

    def _calculate_time_factor(self, detector_id: str) -> float:
        """Calculate time-based performance factor.

        Args:
            detector_id: Detector ID

        Returns:
            Time factor between 0 and 1
        """
        # Simple time-based factor
        # Could be enhanced with historical time-of-day performance data

        current_hour = datetime.now().hour

        # Assume some detectors perform better at certain times
        # This is a simplified example
        if 9 <= current_hour <= 17:  # Business hours
            return 1.0  # Peak performance
        elif 18 <= current_hour <= 22:  # Evening
            return 0.9
        else:  # Night/early morning
            return 0.8

    def _response_time_selection(
        self, detectors: List[str], num_detectors: int
    ) -> LoadBalancingDecision:
        """Select detectors based on response time.

        Args:
            detectors: Available detectors
            num_detectors: Number to select

        Returns:
            Load balancing decision
        """
        detector_times = []

        for detector_id in detectors:
            load_info = self.detector_loads.get(detector_id)
            response_time = load_info.avg_response_time_ms if load_info else 100.0
            detector_times.append((detector_id, response_time))

        # Sort by response time (ascending)
        detector_times.sort(key=lambda x: x[1])
        selected = [detector_id for detector_id, _ in detector_times[:num_detectors]]

        # Equal distribution
        load_distribution = {
            detector_id: 1.0 / len(selected) for detector_id in selected
        }

        estimated_response_time = np.mean(
            [time for _, time in detector_times[:num_detectors]]
        )

        return LoadBalancingDecision(
            selected_detectors=selected,
            load_distribution=load_distribution,
            strategy_used="response_time",
            confidence=0.8,
            reasoning="Selected based on lowest response times",
            estimated_response_time=estimated_response_time,
        )

    def _least_connections_selection(
        self, detectors: List[str], num_detectors: int
    ) -> LoadBalancingDecision:
        """Select detectors with least active connections.

        Args:
            detectors: Available detectors
            num_detectors: Number to select

        Returns:
            Load balancing decision
        """
        detector_loads = []

        for detector_id in detectors:
            load_info = self.detector_loads.get(detector_id)
            active_requests = load_info.active_requests if load_info else 0
            detector_loads.append((detector_id, active_requests))

        # Sort by active requests (ascending)
        detector_loads.sort(key=lambda x: x[1])
        selected = [detector_id for detector_id, _ in detector_loads[:num_detectors]]

        # Equal distribution
        load_distribution = {
            detector_id: 1.0 / len(selected) for detector_id in selected
        }

        estimated_response_time = self._estimate_response_time(selected)

        return LoadBalancingDecision(
            selected_detectors=selected,
            load_distribution=load_distribution,
            strategy_used="least_connections",
            confidence=0.7,
            reasoning="Selected based on least active connections",
            estimated_response_time=estimated_response_time,
        )

    def _weighted_round_robin_selection(
        self, detectors: List[str], num_detectors: int
    ) -> LoadBalancingDecision:
        """Weighted round robin selection based on capacity.

        Args:
            detectors: Available detectors
            num_detectors: Number to select

        Returns:
            Load balancing decision
        """
        # Calculate weights based on capacity scores
        weights = {}
        for detector_id in detectors:
            load_info = self.detector_loads.get(detector_id)
            weights[detector_id] = load_info.capacity_score if load_info else 0.5

        # Select detectors with highest weights
        sorted_detectors = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        selected = [detector_id for detector_id, _ in sorted_detectors[:num_detectors]]

        # Weighted distribution
        total_weight = sum(weight for _, weight in sorted_detectors[:num_detectors])
        load_distribution = {
            detector_id: weight / total_weight
            for detector_id, weight in sorted_detectors[:num_detectors]
        }

        estimated_response_time = self._estimate_response_time(selected)

        return LoadBalancingDecision(
            selected_detectors=selected,
            load_distribution=load_distribution,
            strategy_used="weighted_round_robin",
            confidence=0.8,
            reasoning="Selected using weighted round robin based on capacity",
            estimated_response_time=estimated_response_time,
        )

    def _round_robin_selection(
        self, detectors: List[str], num_detectors: int
    ) -> LoadBalancingDecision:
        """Simple round robin selection.

        Args:
            detectors: Available detectors
            num_detectors: Number to select

        Returns:
            Load balancing decision
        """
        selected = []

        for i in range(num_detectors):
            if not detectors:
                break

            index = (self.round_robin_index + i) % len(detectors)
            selected.append(detectors[index])

        self.round_robin_index = (self.round_robin_index + num_detectors) % len(
            detectors
        )

        # Equal distribution
        load_distribution = {
            detector_id: 1.0 / len(selected) for detector_id in selected
        }

        estimated_response_time = self._estimate_response_time(selected)

        return LoadBalancingDecision(
            selected_detectors=selected,
            load_distribution=load_distribution,
            strategy_used="round_robin",
            confidence=0.6,
            reasoning="Selected using round robin",
            estimated_response_time=estimated_response_time,
        )

    def _fallback_selection(
        self, detectors: List[str], num_detectors: int
    ) -> LoadBalancingDecision:
        """Fallback selection when other methods fail.

        Args:
            detectors: Available detectors
            num_detectors: Number to select

        Returns:
            Fallback load balancing decision
        """
        selected = detectors[:num_detectors]
        load_distribution = {
            detector_id: 1.0 / len(selected) for detector_id in selected
        }

        return LoadBalancingDecision(
            selected_detectors=selected,
            load_distribution=load_distribution,
            strategy_used="fallback",
            confidence=0.3,
            reasoning="Fallback selection due to insufficient data",
            estimated_response_time=200.0,
        )

    def _estimate_response_time(self, selected_detectors: List[str]) -> float:
        """Estimate response time for selected detectors.

        Args:
            selected_detectors: List of selected detector IDs

        Returns:
            Estimated response time in milliseconds
        """
        response_times = []

        for detector_id in selected_detectors:
            load_info = self.detector_loads.get(detector_id)
            if load_info:
                response_times.append(load_info.avg_response_time_ms)
            else:
                response_times.append(100.0)  # Default estimate

        # Return average response time
        return np.mean(response_times) if response_times else 100.0

    async def _update_load_if_needed(self) -> None:
        """Update load information if needed."""
        now = datetime.now()
        if (now - self.last_load_update).seconds >= self.load_update_interval:
            # In a real implementation, this would fetch current load from detectors
            # For now, we just update the timestamp
            self.last_load_update = now

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics.

        Returns:
            Load balancer statistics
        """
        return {
            "strategy": self.strategy.value,
            "tracked_detectors": len(self.detector_loads),
            "total_requests_tracked": sum(
                len(history) for history in self.request_history.values()
            ),
            "last_load_update": self.last_load_update.isoformat(),
            "performance_thresholds": {
                "max_response_time_ms": self.max_response_time_ms,
                "min_success_rate": self.min_success_rate,
                "max_error_rate": self.max_error_rate,
                "max_queue_length": self.max_queue_length,
            },
        }
