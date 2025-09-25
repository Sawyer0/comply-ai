"""Main orchestration service integrating all SRP components.

This service coordinates all the single-responsibility components to provide
comprehensive detector orchestration functionality following the requirements.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from shared.interfaces.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    DetectorResult,
    AggregationSummary,
    PolicyViolation,
)
from shared.utils.correlation import get_correlation_id, set_correlation_id
from shared.exceptions.base import ServiceUnavailableError, ValidationError

# Import SRP-organized components
from .core import (
    DetectorCoordinator,
    ContentRouter,
    ResponseAggregator,
    RoutingPlan,
    RoutingDecision,
    DetectorConfig,
    AggregatedOutput,
)
from .monitoring import (
    HealthMonitor,
    HealthCheck,
    HealthStatus,
    PrometheusMetricsCollector,
)
from .discovery import ServiceDiscoveryManager, ServiceEndpoint
from .policy import PolicyManager, PolicyDecision
from .resilience import CircuitBreaker, RateLimiter, RateLimitStrategy
from .cache import RedisCache, IdempotencyManager
from .pipelines import AsyncJobProcessor
from .security import (
    ApiKeyManager,
    RBACManager,
    AttackDetector,
    InputSanitizer,
    Permission,
    Role,
)
from .tenancy import TenantManager, TenantIsolationManager, TenantContext

logger = logging.getLogger(__name__)


class OrchestrationService:
    """Main orchestration service that coordinates all SRP components.

    This service integrates:
    - DetectorCoordinator: Execute detectors according to routing plans
    - ContentRouter: Route content to appropriate detectors
    - ResponseAggregator: Aggregate detector results
    - HealthMonitor: Monitor detector health
    - ServiceDiscoveryManager: Manage detector registry
    - PolicyManager: Enforce policies

    Requirements satisfied:
    - 2.1: Preserves all existing detector coordination capabilities
    - 2.2: Maintains registry, health monitoring, and circuit breaker implementations
    - 2.3: Consolidates OPA policy management and conflict resolution logic
    """

    def __init__(
        self,
        enable_health_monitoring: bool = True,
        enable_service_discovery: bool = True,
        enable_policy_management: bool = True,
        health_check_interval: int = 30,
        service_ttl_minutes: int = 30,
    ):
        """Initialize orchestration service with SRP components.

        Args:
            enable_health_monitoring: Enable health monitoring
            enable_service_discovery: Enable service discovery
            enable_policy_management: Enable policy management
            health_check_interval: Health check interval in seconds
            service_ttl_minutes: Service TTL in minutes
        """
        self.enable_health_monitoring = enable_health_monitoring
        self.enable_service_discovery = enable_service_discovery
        self.enable_policy_management = enable_policy_management

        # Initialize SRP components
        self._initialize_components(
            health_check_interval=health_check_interval,
            service_ttl_minutes=service_ttl_minutes,
        )

        # Service state
        self._is_running = False
        self._background_tasks = set()

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
        }

    def _initialize_components(
        self, health_check_interval: int, service_ttl_minutes: int
    ):
        """Initialize all SRP components."""

        # Core components (always enabled)
        self.detector_coordinator = DetectorCoordinator(detector_clients={})
        self.content_router = ContentRouter(detector_configs={})
        self.response_aggregator = ResponseAggregator()

        # Security components (always enabled for security)
        self.api_key_manager = ApiKeyManager()
        self.rbac_manager = RBACManager()
        self.attack_detector = AttackDetector()
        self.input_sanitizer = InputSanitizer(strict_mode=True)

        # Multi-tenancy components (always enabled)
        self.tenant_manager = TenantManager()
        self.tenant_isolation = TenantIsolationManager()

        # Monitoring and resilience components (always enabled)
        self.metrics_collector = PrometheusMetricsCollector()
        self.rate_limiter = RateLimiter(strategy=RateLimitStrategy.SLIDING_WINDOW)
        self.redis_cache = RedisCache()
        self.idempotency_manager = IdempotencyManager(self.redis_cache)
        self.job_processor = AsyncJobProcessor()

        # Optional components
        if self.enable_health_monitoring:
            self.health_monitor = HealthMonitor(
                health_check_interval=health_check_interval
            )
        else:
            self.health_monitor = None

        if self.enable_service_discovery:
            self.service_discovery = ServiceDiscoveryManager(
                service_ttl_minutes=service_ttl_minutes
            )
        else:
            self.service_discovery = None

        if self.enable_policy_management:
            self.policy_manager = PolicyManager()
        else:
            self.policy_manager = None

    async def validate_request_security(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Validate request security including input sanitization and authorization.

        Args:
            request: Orchestration request
            tenant_id: Tenant identifier
            api_key: API key for authentication (optional)
            user_id: User identifier (optional)

        Returns:
            True if request is valid and authorized
        """
        correlation_id = get_correlation_id()

        try:
            # 1. Validate tenant exists and is active
            if not self.tenant_manager.is_tenant_active(tenant_id):
                logger.warning(
                    "Request denied: tenant %s is not active",
                    tenant_id,
                    extra={"correlation_id": correlation_id, "tenant_id": tenant_id},
                )
                return False

            # 2. Validate API key if provided
            if api_key:
                api_key_obj = self.api_key_manager.validate_api_key(api_key)
                if not api_key_obj or api_key_obj.tenant_id != tenant_id:
                    logger.warning(
                        "Request denied: invalid API key for tenant %s",
                        tenant_id,
                        extra={
                            "correlation_id": correlation_id,
                            "tenant_id": tenant_id,
                        },
                    )
                    return False

            # 3. Check for attack patterns in content
            attack_detections = self.attack_detector.detect_attacks(request.content)
            if self.attack_detector.has_high_severity_attacks(request.content):
                logger.warning(
                    "Request denied: high severity attack patterns detected for tenant %s",
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "attack_types": [
                            d.attack_type.value for d in attack_detections
                        ],
                    },
                )
                return False

            # 4. Sanitize input content
            sanitized_content = self.input_sanitizer.sanitize(request.content)
            request.content = sanitized_content

            # 5. Check RBAC permissions if user provided
            if user_id:
                has_permission = self.rbac_manager.check_permission(
                    user_id, tenant_id, Permission.ORCHESTRATE_DETECTORS
                )
                if not has_permission:
                    logger.warning(
                        "Request denied: user %s lacks orchestration permission for tenant %s",
                        user_id,
                        tenant_id,
                        extra={
                            "correlation_id": correlation_id,
                            "tenant_id": tenant_id,
                            "user_id": user_id,
                        },
                    )
                    return False

            return True

        except Exception as e:
            logger.error(
                "Security validation failed: %s",
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            return False

    async def orchestrate(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        correlation_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> OrchestrationResponse:
        """Main orchestration method that coordinates all components.

        This method implements the core orchestration workflow:
        1. Validate security and tenant access
        2. Route request to determine which detectors to use
        3. Execute detectors according to routing plan
        4. Aggregate results into unified response
        5. Apply policies and generate recommendations

        Args:
            request: Orchestration request
            tenant_id: Tenant identifier
            correlation_id: Optional correlation ID
            api_key: Optional API key for authentication
            user_id: Optional user identifier

        Returns:
            Orchestration response with aggregated results
        """
        # Set correlation ID for tracing
        if correlation_id:
            set_correlation_id(correlation_id)
        else:
            correlation_id = get_correlation_id()

        # Validate security and tenant access
        is_valid = await self.validate_request_security(
            request=request, tenant_id=tenant_id, api_key=api_key, user_id=user_id
        )

        if not is_valid:
            return OrchestrationResponse(
                request_id=correlation_id,
                success=False,
                timestamp=datetime.utcnow(),
                processing_time_ms=0,
                correlation_id=correlation_id,
                detector_results=[],
                aggregation_summary=AggregationSummary(
                    total_detectors=0,
                    successful_detectors=0,
                    failed_detectors=0,
                    average_confidence=0.0,
                ),
                coverage_achieved=0.0,
                policy_violations=[],
                recommendations=["Request failed security validation"],
            )

        # Use tenant isolation context for the entire request
        async with self.tenant_isolation.tenant_context(
            tenant_id, user_id
        ) as tenant_context:
            start_time = datetime.utcnow()
            self._metrics["total_requests"] += 1

            logger.info(
                "Starting orchestration for tenant %s",
                tenant_id,
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "detector_types": getattr(request, "detector_types", []),
                    "processing_mode": getattr(request, "processing_mode", "standard"),
                },
            )

            try:
                # Step 1: Route request to determine which detectors to use
                routing_plan, routing_decision = (
                    await self.content_router.route_request(request)
                )

                logger.info(
                    "Routing completed: %d detectors selected",
                    len(routing_decision.selected_detectors),
                    extra={
                        "correlation_id": correlation_id,
                        "selected_detectors": routing_decision.selected_detectors,
                        "routing_reason": routing_decision.routing_reason,
                    },
                )

                # Step 2: Execute detectors according to routing plan
                detector_results = await self.detector_coordinator.execute_routing_plan(
                    content=request.content,
                    routing_plan=routing_plan,
                    request_id=correlation_id,
                    metadata={"tenant_id": tenant_id},
                )

                logger.info(
                    "Detector execution completed: %d results",
                    len(detector_results),
                    extra={
                        "correlation_id": correlation_id,
                        "total_results": len(detector_results),
                        "successful_results": len(
                            [r for r in detector_results if r.confidence > 0.0]
                        ),
                    },
                )

                # Step 3: Aggregate results
                aggregated_output, coverage = (
                    await self.response_aggregator.aggregate_results(
                        detector_results=detector_results, tenant_id=tenant_id
                    )
                )

                # Step 4: Create aggregation summary
                aggregation_summary = AggregationSummary(
                    total_detectors=len(detector_results),
                    successful_detectors=len(
                        [r for r in detector_results if r.confidence > 0.0]
                    ),
                    failed_detectors=len(
                        [r for r in detector_results if r.confidence == 0.0]
                    ),
                    average_confidence=(
                        sum(r.confidence for r in detector_results)
                        / len(detector_results)
                        if detector_results
                        else 0.0
                    ),
                )

                # Step 5: Check for policy violations (if policy management enabled)
                policy_violations = []
                if self.policy_manager:
                    # This would integrate with actual policy checking
                    # For now, we'll leave it empty
                    pass

                # Step 6: Generate recommendations
                recommendations = self._generate_recommendations(
                    detector_results=detector_results,
                    aggregated_output=aggregated_output,
                    coverage=coverage,
                )

                # Calculate processing time
                processing_time = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                # Create response
                response = OrchestrationResponse(
                    request_id=correlation_id,
                    success=True,
                    timestamp=datetime.utcnow(),
                    processing_time_ms=processing_time,
                    correlation_id=correlation_id,
                    detector_results=detector_results,
                    aggregation_summary=aggregation_summary,
                    coverage_achieved=coverage,
                    policy_violations=policy_violations,
                    recommendations=recommendations,
                )

                # Update metrics
                self._metrics["successful_requests"] += 1
                self._update_average_response_time(processing_time)

                logger.info(
                    "Orchestration completed successfully in %.2fms",
                    processing_time,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "processing_time_ms": processing_time,
                        "coverage_achieved": coverage,
                        "total_detectors": len(detector_results),
                    },
                )

                return response

            except Exception as e:
                self._metrics["failed_requests"] += 1
                processing_time = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                logger.error(
                    "Orchestration failed: %s",
                    str(e),
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "processing_time_ms": processing_time,
                        "error": str(e),
                    },
                )

                # Return error response
                return OrchestrationResponse(
                    request_id=correlation_id,
                    success=False,
                    timestamp=datetime.utcnow(),
                    processing_time_ms=processing_time,
                    correlation_id=correlation_id,
                    detector_results=[],
                    aggregation_summary=AggregationSummary(
                        total_detectors=0,
                        successful_detectors=0,
                        failed_detectors=0,
                        average_confidence=0.0,
                    ),
                    coverage_achieved=0.0,
                    policy_violations=[],
                    recommendations=["Check service logs for error details"],
                )

    def _generate_recommendations(
        self,
        detector_results: List[DetectorResult],
        aggregated_output: AggregatedOutput,
        coverage: float,
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []

        # Coverage-based recommendations
        if coverage < 0.5:
            recommendations.append(
                "Low detector coverage - consider adding more detectors"
            )

        # Confidence-based recommendations
        if aggregated_output.confidence_score < 0.7:
            recommendations.append("Low confidence results - consider manual review")

        # Failure-based recommendations
        failed_count = len([r for r in detector_results if r.confidence == 0.0])
        if failed_count > len(detector_results) * 0.3:
            recommendations.append("High detector failure rate - check detector health")

        return recommendations

    def _update_average_response_time(self, processing_time: float):
        """Update average response time metric."""
        current_avg = self._metrics["average_response_time_ms"]
        total_requests = self._metrics["successful_requests"]

        if total_requests == 1:
            self._metrics["average_response_time_ms"] = processing_time
        else:
            # Calculate running average
            self._metrics["average_response_time_ms"] = (
                current_avg * (total_requests - 1) + processing_time
            ) / total_requests

    async def register_detector(
        self,
        detector_id: str,
        endpoint: str,
        detector_type: str,
        timeout_ms: int = 5000,
        max_retries: int = 3,
        supported_content_types: Optional[List[str]] = None,
    ) -> bool:
        """Register a new detector with all relevant components.

        Args:
            detector_id: Unique detector identifier
            endpoint: Detector endpoint URL
            detector_type: Type of detector
            timeout_ms: Timeout in milliseconds
            max_retries: Maximum retry attempts
            supported_content_types: Supported content types

        Returns:
            True if registration successful
        """
        correlation_id = get_correlation_id()

        logger.info(
            "Registering detector %s at %s",
            detector_id,
            endpoint,
            extra={
                "correlation_id": correlation_id,
                "detector_id": detector_id,
                "endpoint": endpoint,
                "detector_type": detector_type,
            },
        )

        try:
            # Register with content router
            router_success = self.content_router.register_detector(
                name=detector_id,
                endpoint=endpoint,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                supported_content_types=supported_content_types,
            )

            if not router_success:
                logger.error("Failed to register detector with router: %s", detector_id)
                return False

            # Register with service discovery (if enabled)
            if self.service_discovery:
                discovery_success = self.service_discovery.register_service(
                    service_id=detector_id,
                    endpoint_url=endpoint,
                    service_type=detector_type,
                    metadata={
                        "timeout_ms": timeout_ms,
                        "max_retries": max_retries,
                        "supported_content_types": supported_content_types or [],
                    },
                )

                if not discovery_success:
                    logger.error(
                        "Failed to register detector with service discovery: %s",
                        detector_id,
                    )
                    # Clean up router registration
                    self.content_router.unregister_detector(detector_id)
                    return False

            # Register with health monitor (if enabled)
            if self.health_monitor:
                # Create a mock health check client for now
                class MockHealthClient:
                    async def health_check(self):
                        return True

                health_success = self.health_monitor.register_service(
                    service_id=detector_id, health_check_client=MockHealthClient()
                )

                if not health_success:
                    logger.warning(
                        "Failed to register detector with health monitor: %s",
                        detector_id,
                    )
                    # Continue anyway - health monitoring is optional

            logger.info(
                "Successfully registered detector %s",
                detector_id,
                extra={"correlation_id": correlation_id, "detector_id": detector_id},
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to register detector %s: %s",
                detector_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "detector_id": detector_id,
                    "error": str(e),
                },
            )
            return False

    async def unregister_detector(self, detector_id: str) -> bool:
        """Unregister a detector from all components.

        Args:
            detector_id: Detector identifier to unregister

        Returns:
            True if unregistration successful
        """
        correlation_id = get_correlation_id()

        logger.info(
            "Unregistering detector %s",
            detector_id,
            extra={"correlation_id": correlation_id, "detector_id": detector_id},
        )

        success = True

        # Unregister from router
        if not self.content_router.unregister_detector(detector_id):
            logger.error("Failed to unregister detector from router: %s", detector_id)
            success = False

        # Unregister from service discovery
        if self.service_discovery:
            if not self.service_discovery.unregister_service(detector_id):
                logger.error(
                    "Failed to unregister detector from service discovery: %s",
                    detector_id,
                )
                success = False

        # Unregister from health monitor
        if self.health_monitor:
            if not self.health_monitor.unregister_service(detector_id):
                logger.error(
                    "Failed to unregister detector from health monitor: %s", detector_id
                )
                success = False

        return success

    async def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status.

        Returns:
            Dictionary with service status information
        """
        status = {
            "service": "orchestration",
            "status": "running" if self._is_running else "stopped",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._metrics.copy(),
            "components": {
                "detector_coordinator": "enabled",
                "content_router": "enabled",
                "response_aggregator": "enabled",
                "health_monitor": "enabled" if self.health_monitor else "disabled",
                "service_discovery": (
                    "enabled" if self.service_discovery else "disabled"
                ),
                "policy_manager": "enabled" if self.policy_manager else "disabled",
                "api_key_manager": "enabled",
                "rbac_manager": "enabled",
                "attack_detector": "enabled",
                "input_sanitizer": "enabled",
                "tenant_manager": "enabled",
                "tenant_isolation": "enabled",
                "metrics_collector": "enabled",
                "rate_limiter": "enabled",
                "redis_cache": "enabled",
                "idempotency_manager": "enabled",
                "job_processor": "enabled",
            },
        }

        # Add component-specific status
        if self.health_monitor:
            status["health_summary"] = self.health_monitor.get_health_summary()

        if self.service_discovery:
            status["registry_status"] = self.service_discovery.get_registry_status()

        status["routing_statistics"] = self.content_router.get_routing_statistics()

        # Add security and tenancy status
        status["security_status"] = {
            "attack_patterns_loaded": len(self.attack_detector._patterns),
            "input_sanitizer_strict_mode": self.input_sanitizer.strict_mode,
            "rbac_roles_available": len(self.rbac_manager.list_available_roles()),
        }

        status["tenancy_status"] = {
            "tenant_stats": self.tenant_manager.get_tenant_stats(),
            "isolation_stats": self.tenant_isolation.get_tenant_isolation_stats(),
        }

        # Add monitoring and resilience status
        status["monitoring_status"] = {
            "metrics_summary": self.metrics_collector.get_metrics_summary(),
            "cache_statistics": self.redis_cache.get_statistics(),
        }

        status["resilience_status"] = {
            "rate_limiter_stats": self.rate_limiter.get_statistics(),
            "job_processor_stats": self.job_processor.get_statistics(),
        }

        return status

    async def start(self):
        """Start the orchestration service."""
        if self._is_running:
            logger.warning("Service is already running")
            return

        logger.info("Starting orchestration service")

        # Start background tasks if health monitoring is enabled
        if self.health_monitor:
            task = asyncio.create_task(self._health_check_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        # Start job processor
        await self.job_processor.start()

        self._is_running = True
        logger.info("Orchestration service started successfully")

    async def stop(self):
        """Stop the orchestration service."""
        if not self._is_running:
            logger.warning("Service is not running")
            return

        logger.info("Stopping orchestration service")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Stop job processor
        await self.job_processor.stop()

        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._is_running = False
        logger.info("Orchestration service stopped")

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while self._is_running:
            try:
                if self.health_monitor:
                    await self.health_monitor.check_all_services()
                await asyncio.sleep(self.health_monitor.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error: %s", str(e))
                await asyncio.sleep(5)  # Wait before retrying


# Export the main service class
__all__ = [
    "OrchestrationService",
]
