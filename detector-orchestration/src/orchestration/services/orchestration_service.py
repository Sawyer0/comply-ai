"""Main orchestration service following SRP.

This service has a SINGLE responsibility: coordinate detector orchestration requests.
It delegates specific concerns to other focused services.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from shared.interfaces.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    DetectorResult,
    AggregationSummary,
)
from shared.utils.correlation import get_correlation_id

from ..core import DetectorCoordinator, ContentRouter, ResponseAggregator
from ..ml import ContentAnalyzer, AdaptiveLoadBalancer, RoutingOptimizer
from .detector_management_service import DetectorManagementService
from .health_management_service import HealthManagementService
from .security_service import SecurityService

logger = logging.getLogger(__name__)


class OrchestrationService:
    """Main orchestration service following SRP.

    Single Responsibility: Coordinate detector orchestration requests.
    """

    def __init__(
        self,
        detector_service: DetectorManagementService,
        health_service: HealthManagementService,
        security_service: SecurityService,
    ):
        """Initialize orchestration service with injected dependencies.

        Args:
            detector_service: Detector management service
            health_service: Health management service
            security_service: Security service
        """
        # Injected services (dependency injection)
        self.detector_service = detector_service
        self.health_service = health_service
        self.security_service = security_service

        # Core orchestration components
        self.content_router = ContentRouter()
        self.detector_coordinator = DetectorCoordinator({})
        self.response_aggregator = ResponseAggregator()

        # ML components
        self.content_analyzer = ContentAnalyzer()
        self.load_balancer = AdaptiveLoadBalancer()
        self.routing_optimizer = RoutingOptimizer()

        # Metrics
        self.total_requests = 0
        self.successful_requests = 0

    async def orchestrate(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        correlation_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> OrchestrationResponse:
        """Orchestrate detector execution for a request.

        Args:
            request: Orchestration request
            tenant_id: Tenant identifier
            correlation_id: Request correlation ID
            api_key: API key for authentication
            user_id: User identifier

        Returns:
            Orchestration response

        Raises:
            ValueError: Invalid request parameters
            RuntimeError: Orchestration execution failed
        """
        start_time = datetime.utcnow()
        correlation_id = correlation_id or get_correlation_id()

        try:
            # Validate security
            await self.security_service.validate_request_security(
                request, tenant_id, api_key, user_id
            )

            # Analyze content
            content_features = self.content_analyzer.analyze_content(
                request.content, correlation_id
            )

            # Get available detectors
            available_detectors = await self.detector_service.get_available_detectors(
                tenant_id
            )

            # Route request to detectors
            routing_decision = self.content_router.route_request(request)

            # Execute detectors
            detector_results = await self.detector_coordinator.execute_routing_plan(
                request.content, routing_decision, correlation_id
            )

            # Aggregate results
            aggregated_output = self.response_aggregator.aggregate_results(
                detector_results, tenant_id
            )

            # Create response
            response = OrchestrationResponse(
                detector_results=detector_results,
                aggregation_summary=AggregationSummary(
                    total_detectors=len(detector_results),
                    successful_detectors=len(
                        [r for r in detector_results if r.confidence > 0]
                    ),
                    failed_detectors=len(
                        [r for r in detector_results if r.confidence == 0]
                    ),
                    average_confidence=aggregated_output.confidence_score,
                ),
                coverage_achieved=1.0,  # Calculate actual coverage
                policy_violations=[],
                recommendations=[],
            )

            # Update metrics
            self.total_requests += 1
            self.successful_requests += 1

            # Record performance for ML
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._record_performance_metrics(
                detector_results, processing_time, correlation_id
            )

            return response

        except (ValueError, TypeError) as e:
            logger.error(
                "Orchestration validation failed",
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            self.total_requests += 1
            raise

        except RuntimeError as e:
            logger.error(
                "Orchestration execution failed",
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            self.total_requests += 1
            raise

    async def _record_performance_metrics(
        self,
        detector_results: List[DetectorResult],
        processing_time: float,
        correlation_id: str,
    ) -> None:
        """Record performance metrics for ML learning.

        Args:
            detector_results: Results from detectors
            processing_time: Total processing time in milliseconds
            correlation_id: Request correlation ID
        """
        try:
            for result in detector_results:
                if hasattr(result, "processing_time_ms") and result.processing_time_ms:
                    # Update routing optimizer with performance feedback
                    reward = self._calculate_performance_reward(result)
                    self.routing_optimizer.update_reward(
                        result.detector_id, reward, result.confidence > 0.5
                    )

        except (AttributeError, ValueError) as e:
            logger.warning(
                "Failed to record performance metrics",
                extra={"correlation_id": correlation_id, "error": str(e)},
            )

    def _calculate_performance_reward(self, result: DetectorResult) -> float:
        """Calculate performance reward for ML feedback.

        Args:
            result: Detector result

        Returns:
            Reward value between 0.0 and 1.0
        """
        # Simple reward calculation based on confidence and response time
        confidence_reward = result.confidence

        # Time penalty (prefer faster responses)
        time_penalty = 0.0
        if hasattr(result, "processing_time_ms") and result.processing_time_ms:
            # Penalty increases with response time (max 0.3 penalty for 5+ seconds)
            time_penalty = min(0.3, result.processing_time_ms / 5000.0 * 0.3)

        reward = confidence_reward - time_penalty
        return max(0.0, min(1.0, reward))

    async def get_service_status(self) -> Dict[str, Any]:
        """Get orchestration service status.

        Returns:
            Service status information
        """
        return {
            "status": "running",
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "success_rate": (self.successful_requests / max(1, self.total_requests)),
            "components": {
                "content_analyzer": self.content_analyzer.get_analyzer_stats(),
                "routing_optimizer": self.routing_optimizer.get_optimizer_stats(),
                "detector_service": await self.detector_service.get_service_status(),
                "health_service": await self.health_service.get_service_status(),
                "security_service": await self.security_service.get_service_status(),
            },
        }
