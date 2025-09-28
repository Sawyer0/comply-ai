"""ML Feedback Service following SRP.

This module provides ONLY ML feedback functionality:
- Performance feedback collection
- Reward calculation for routing optimization
- Model performance tracking
- Learning from orchestration results
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from shared.interfaces.orchestration import DetectorResult
from shared.utils.performance import calculate_performance_reward
from shared.utils.correlation import get_correlation_id

from .performance_predictor import PerformancePredictor, PerformanceMetrics
from .routing_optimizer import RoutingOptimizer
from .load_balancer import AdaptiveLoadBalancer

logger = logging.getLogger(__name__)


class MLFeedbackService:
    """ML feedback service following SRP - ONLY handles ML model feedback."""

    def __init__(
        self,
        performance_predictor: PerformancePredictor,
        routing_optimizer: RoutingOptimizer,
        load_balancer: AdaptiveLoadBalancer,
    ):
        """Initialize ML feedback service.

        Args:
            performance_predictor: Performance prediction model
            routing_optimizer: Routing optimization model
            load_balancer: Adaptive load balancer
        """
        self.performance_predictor = performance_predictor
        self.routing_optimizer = routing_optimizer
        self.load_balancer = load_balancer

    async def update_models_with_feedback(
        self,
        detector_results: List[DetectorResult],
        routing_decision: Any,
        _content_features: Any,
    ) -> None:
        """Update all ML models with performance feedback.

        Args:
            detector_results: Results from detector execution
            routing_decision: Routing decision that was made
            content_features: Content analysis features
        """
        correlation_id = get_correlation_id()

        try:
            # Update performance predictor
            await self._update_performance_predictor(detector_results)

            # Update routing optimizer
            await self._update_routing_optimizer(detector_results, routing_decision)

            # Update load balancer
            await self._update_load_balancer(detector_results)

            logger.debug(
                "ML models updated with feedback",
                extra={
                    "correlation_id": correlation_id,
                    "detectors_count": len(detector_results),
                    "routing_confidence": getattr(routing_decision, "confidence", 0.0),
                },
            )

        except (ValueError, RuntimeError, TypeError) as exc:
            logger.error(
                "Failed to update ML models with feedback: %s",
                exc,
                extra={"correlation_id": correlation_id},
            )

    async def _update_performance_predictor(
        self, detector_results: List[DetectorResult]
    ) -> None:
        """Update performance predictor with actual results following SRP."""
        for result in detector_results:
            if hasattr(result, "processing_time_ms") and result.processing_time_ms:
                metrics = PerformanceMetrics(
                    detector_id=result.detector_id,
                    response_time_ms=result.processing_time_ms,
                    success_rate=1.0 if result.confidence > 0.0 else 0.0,
                    confidence_score=result.confidence,
                    error_rate=0.0 if result.confidence > 0.0 else 1.0,
                    throughput=1.0,
                    timestamp=datetime.utcnow(),
                )
                self.performance_predictor.add_performance_data(metrics)

    async def _update_routing_optimizer(
        self, detector_results: List[DetectorResult], routing_decision: Any
    ) -> None:
        """Update routing optimizer with reward feedback following SRP."""
        if not hasattr(routing_decision, "selected_detectors"):
            return

        for detector_id in routing_decision.selected_detectors:
            detector_result = next(
                (r for r in detector_results if r.detector_id == detector_id), None
            )
            if detector_result:
                # Use shared utility for reward calculation (DRY principle)
                reward = calculate_performance_reward(
                    confidence=detector_result.confidence,
                    response_time_ms=getattr(
                        detector_result, "processing_time_ms", None
                    ),
                    success=detector_result.confidence > 0.5,
                )

                self.routing_optimizer.update_reward(
                    detector_id, reward, detector_result.confidence > 0.5
                )

    async def _update_load_balancer(
        self, detector_results: List[DetectorResult]
    ) -> None:
        """Update load balancer with performance data following SRP."""
        for result in detector_results:
            load_info = {
                "active_requests": 1,
                "avg_response_time_ms": getattr(result, "processing_time_ms", 100.0),
                "success_rate": 1.0 if result.confidence > 0.0 else 0.0,
                "error_rate": 0.0 if result.confidence > 0.0 else 1.0,
                "cpu_usage": 0.5,  # Default estimate
                "memory_usage": 0.5,  # Default estimate
                "queue_length": 0,
            }
            self.load_balancer.update_detector_load(result.detector_id, load_info)

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get ML feedback statistics.

        Returns:
            Dictionary with feedback statistics
        """
        return {
            "performance_predictor_trained": self.performance_predictor.is_trained,
            "routing_optimizer_stats": self.routing_optimizer.get_optimizer_stats(),
            "load_balancer_stats": self.load_balancer.get_load_balancer_stats(),
            "last_updated": datetime.utcnow().isoformat(),
        }
