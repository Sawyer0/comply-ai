"""Component initialization helpers for the orchestration service."""

# pylint: disable=protected-access
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import redis.asyncio as redis

from shared.utils.cache import create_idempotency_cache, create_response_cache
from shared.utils.correlation import get_correlation_id

from ..core import (
    ContentRouter,
    CustomerDetectorClient,
    DetectorClientConfig,
    DetectorConfig,
    DetectorCoordinator,
    ResponseAggregator,
)
from ..discovery import ServiceDiscoveryManager
from ..discovery.service_discovery import ServiceMetadata
from ..ml import (
    AdaptiveLoadBalancer,
    ContentAnalyzer,
    PerformancePredictor,
    RiskScorer,
    RoutingOptimizer,
)
from ..ml.feedback_service import MLFeedbackService
from ..monitoring import HealthMonitor, PrometheusMetricsCollector
from ..pipelines import AsyncJobProcessor
from ..policy import PolicyManager
from ..repository import (
    DetectorMappingConfigRepository,
    DetectorRepository,
    RiskAnalysisRepository,
)
from ..resilience import RateLimiter
from ..security import ApiKeyManager, AttackDetector, InputSanitizer, RBACManager
from ..tenancy.tenant_isolation import TenantIsolationManager
from ..tenancy.tenant_manager import TenantManager
from ..utils.detector_response import get_response_parser

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .orchestration_service import OrchestrationService


def initialize_components(service: "OrchestrationService") -> None:
    """Populate the service with all core and optional components."""
    components = service.components

    # Core
    components.detector_coordinator = DetectorCoordinator(
        detector_clients=components.detector_clients
    )
    components.content_router = ContentRouter(detector_configs={})
    components.response_aggregator = ResponseAggregator()
    components.detector_repository = DetectorRepository()
    components.detector_mapping_repository = DetectorMappingConfigRepository()
    components.risk_repository = RiskAnalysisRepository()

    # ML
    components.performance_predictor = PerformancePredictor()
    components.content_analyzer = ContentAnalyzer()
    components.load_balancer = AdaptiveLoadBalancer()
    components.routing_optimizer = RoutingOptimizer()
    components.ml_feedback = MLFeedbackService(
        components.performance_predictor,
        components.routing_optimizer,
        components.load_balancer,
    )
    components.risk_scorer = RiskScorer()

    # Security
    components.api_key_manager = ApiKeyManager()
    components.rbac_manager = RBACManager()
    components.attack_detector = AttackDetector()
    components.input_sanitizer = InputSanitizer(strict_mode=True)

    # Tenancy
    components.tenant_manager = TenantManager()
    components.tenant_isolation = TenantIsolationManager()

    # Monitoring & jobs
    components.metrics_collector = PrometheusMetricsCollector()

    # Rate limiting: use Redis-backed storage when available so counters
    # survive process restarts and are shared across instances.
    cache_backend = os.getenv("ORCHESTRATION_CACHE_BACKEND", "redis")
    redis_url = os.getenv("ORCHESTRATION_REDIS_URL", "redis://localhost:6379")
    redis_client = None
    if cache_backend.lower() == "redis":
        redis_client = redis.from_url(redis_url, decode_responses=True)

    components.rate_limiter = RateLimiter(
        redis_client=redis_client,
        tenant_limit=service.config.rate_limit_tenant_limit,
        window_seconds=service.config.rate_limit_window_seconds,
        tenant_overrides=service.config.rate_limit_tenant_overrides,
    )
    components.job_processor = AsyncJobProcessor()

    if service.config.enable_health_monitoring:
        components.health_monitor = HealthMonitor(
            health_check_interval=service.config.health_check_interval
        )
        if components.detector_coordinator:
            components.detector_coordinator.health_monitor = components.health_monitor

    if service.config.enable_service_discovery:
        components.service_discovery = ServiceDiscoveryManager(
            service_ttl_minutes=service.config.service_ttl_minutes
        )
        bootstrap_detectors_from_discovery(service)

    if service.config.enable_policy_management:
        components.policy_manager = PolicyManager()


def bootstrap_detectors_from_discovery(service: "OrchestrationService") -> None:
    """Hydrate router and detector clients using discovered metadata."""
    discovery = service.components.service_discovery
    router = service.components.content_router
    if not discovery or not router:
        return

    for endpoint in discovery.discover_services():
        metadata = ServiceMetadata.from_dict(endpoint.metadata)

        try:
            parser_callable = get_response_parser(metadata.response_parser)
        except ValueError:
            logger.warning(
                "Unknown response parser for %s: %s; falling back to default",
                endpoint.service_id,
                metadata.response_parser,
                extra=service._log_extra(  # noqa: SLF001
                    get_correlation_id(),
                    detector_id=endpoint.service_id,
                    response_parser=metadata.response_parser,
                ),
            )
            parser_callable = get_response_parser(None)

        router.register_detector(
            DetectorConfig(
                name=endpoint.service_id,
                endpoint=endpoint.endpoint_url,
                timeout_ms=metadata.timeout_ms,
                max_retries=metadata.max_retries,
                supported_content_types=metadata.supported_content_types or ["text"],
            )
        )

        if endpoint.service_id in service.components.detector_clients:
            continue

        analyze_endpoint = endpoint.endpoint_url.rstrip("/")
        if metadata.analyze_path:
            analyze_endpoint = f"{analyze_endpoint}/{metadata.analyze_path.lstrip('/')}"

        timeout_seconds = max(metadata.timeout_ms / 1000.0, 0.1)
        client_config = DetectorClientConfig(
            name=endpoint.service_id,
            endpoint=analyze_endpoint,
            timeout=timeout_seconds,
            max_retries=metadata.max_retries,
            default_headers=metadata.auth_headers or {},
            response_parser=parser_callable,
        )
        service.components.detector_clients[endpoint.service_id] = CustomerDetectorClient(
            client_config
        )


def initialize_cache_components(service: "OrchestrationService") -> None:
    """Initialize caches for idempotency and responses."""
    backend = os.getenv("ORCHESTRATION_CACHE_BACKEND", "redis")
    redis_url = os.getenv("ORCHESTRATION_REDIS_URL", "redis://localhost:6379")

    cache_kwargs = {"redis_url": redis_url} if backend.lower() == "redis" else {}

    service.components.idempotency_cache = create_idempotency_cache(
        backend_type=backend,
        ttl_seconds=int(os.getenv("ORCHESTRATION_IDEMPOTENCY_TTL", "3600")),
        **cache_kwargs,
    )
    service.components.response_cache = create_response_cache(
        backend_type=backend,
        ttl_seconds=int(os.getenv("ORCHESTRATION_RESPONSE_CACHE_TTL", "1800")),
        **cache_kwargs,
    )
