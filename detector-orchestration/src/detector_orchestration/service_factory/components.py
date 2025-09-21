"""Component creation helpers for the orchestration service factory."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

from detector_orchestration import __version__ as ORCH_VERSION  # noqa: F401
from detector_orchestration.aggregator import ResponseAggregator
from detector_orchestration.cache import (
    IdempotencyCache,
    RedisIdempotencyCache,
    RedisResponseCache,
    ResponseCache,
)
from detector_orchestration.circuit_breaker import CircuitBreakerManager
from detector_orchestration.clients import DetectorClient
from detector_orchestration.conflict import ConflictResolver
from detector_orchestration.coordinator import DetectorCoordinator
from detector_orchestration.health_monitor import HealthMonitor
from detector_orchestration.jobs import JobManager
from detector_orchestration.mapper_client import MapperClient
from detector_orchestration.metrics import OrchestrationMetricsCollector
from detector_orchestration.policy import OPAPolicyEngine, PolicyManager, PolicyStore
from detector_orchestration.registry import DetectorRegistry
from detector_orchestration.router import ContentRouter

logger = logging.getLogger(__name__)


class ServiceFactoryComponentsMixin:
    """Constructs concrete service dependencies used by orchestration."""

    # pylint: disable=too-many-instance-attributes

    settings: any
    metrics: OrchestrationMetricsCollector
    detector_clients: Dict[str, DetectorClient]
    circuit_breaker: Optional[CircuitBreakerManager]
    health_monitor: Optional[HealthMonitor]
    idempotency_cache: Optional[IdempotencyCache]
    response_cache: Optional[ResponseCache]
    policy_store: Optional[PolicyStore]
    policy_manager: Optional[PolicyManager]
    policy_engine: Optional[OPAPolicyEngine]
    mapper_client: Optional[MapperClient]
    aggregator: Optional[ResponseAggregator]
    conflict_resolver: Optional[ConflictResolver]
    router: Optional[ContentRouter]
    coordinator: Optional[DetectorCoordinator]
    job_manager: Optional[JobManager]
    registry: Optional[DetectorRegistry]
    container: any

    def create_detector_clients(self) -> Dict[str, DetectorClient]:
        """Create detector client instances."""
        clients: Dict[str, DetectorClient] = {}
        for name, det in self.settings.detectors.items():
            clients[name] = DetectorClient(
                name=name,
                endpoint=det.endpoint,
                timeout_ms=det.timeout_ms,
                max_retries=det.max_retries,
                auth=det.auth,
            )
        return clients

    def create_circuit_breaker(self) -> CircuitBreakerManager:
        """Create circuit breaker manager."""
        return CircuitBreakerManager(
            failure_threshold=self.settings.config.circuit_breaker_failure_threshold,
            recovery_timeout_seconds=self.settings.config.circuit_breaker_recovery_timeout_seconds,
        )

    def create_health_monitor(
        self, clients: Dict[str, DetectorClient]
    ) -> HealthMonitor:
        """Create health monitor."""
        return HealthMonitor(
            clients,
            interval_seconds=self.settings.config.health_check_interval_seconds,
            metrics=self.metrics,
            unhealthy_threshold=self.settings.config.unhealthy_threshold,
        )

    def create_caches(self) -> Tuple[IdempotencyCache, ResponseCache]:
        """Create cache instances based on configuration."""
        if (
            self.settings.config.cache_backend == "redis"
            and self.settings.config.redis_url
        ):
            redis_idem = RedisIdempotencyCache(
                self.settings.config.redis_url,
                ttl_seconds=60 * 60 * 24,
                key_prefix=f"idem:{self.settings.config.redis_prefix}",
            )
            redis_resp = RedisResponseCache(
                self.settings.config.redis_url,
                ttl_seconds=self.settings.config.response_cache_ttl_seconds,
                key_prefix=f"resp:{self.settings.config.redis_prefix}",
            )

            if redis_idem.is_healthy() and redis_resp.is_healthy():
                self.metrics.set_redis_backend_up("idempotency", True)
                self.metrics.set_redis_backend_up("response", True)
                return redis_idem, redis_resp

            self.metrics.set_redis_backend_up("idempotency", False)
            self.metrics.set_redis_backend_up("response", False)
            self.metrics.inc_redis_fallback("idempotency")
            self.metrics.inc_redis_fallback("response")

        return IdempotencyCache(), ResponseCache(
            ttl_seconds=self.settings.config.response_cache_ttl_seconds
        )

    def create_policy_components(
        self,
    ) -> Tuple[PolicyStore, PolicyManager, OPAPolicyEngine]:
        """Create policy store, policy engine, and manager."""
        policy_store = PolicyStore(self.settings.config.policy_dir)
        policy_engine = OPAPolicyEngine(self.settings)
        policy_manager = PolicyManager(policy_store, policy_engine)
        return policy_store, policy_manager, policy_engine

    def create_mapper_client(self) -> MapperClient:
        """Create mapper client."""
        return MapperClient(self.settings)

    def create_job_manager(self) -> JobManager:
        """Create job manager wired to the orchestration pipeline."""
        return JobManager(run_fn=self.run_async_job)

    def create_registry(
        self, clients: Dict[str, DetectorClient], health_monitor: HealthMonitor
    ) -> DetectorRegistry:
        """Create detector registry."""
        return DetectorRegistry(self.settings, clients, health_monitor)

    def initialize_services(self) -> None:
        """Initialize all service components."""
        logger.info("Initializing orchestration services...")

        self.detector_clients = self.create_detector_clients()
        self.circuit_breaker = self.create_circuit_breaker()
        self.health_monitor = self.create_health_monitor(self.detector_clients)
        self.idempotency_cache, self.response_cache = self.create_caches()
        (
            self.policy_store,
            self.policy_manager,
            self.policy_engine,
        ) = self.create_policy_components()
        self.mapper_client = self.create_mapper_client()
        self.aggregator = ResponseAggregator()
        self.conflict_resolver = ConflictResolver(self.policy_engine)
        self.router = ContentRouter(
            self.settings,
            self.health_monitor,
            self.policy_manager,
        )
        self.coordinator = DetectorCoordinator(
            self.detector_clients,
            self.circuit_breaker,
            self.metrics,
            retry_on_timeouts=bool(
                getattr(self.settings.config, "retry_on_timeouts", True)
            ),
            retry_on_failures=bool(
                getattr(self.settings.config, "retry_on_failures", True)
            ),
        )
        self.job_manager = self.create_job_manager()
        self.registry = self.create_registry(self.detector_clients, self.health_monitor)

        if self.health_monitor:
            self.container.register_service("health_monitor", self.health_monitor)
        if self.job_manager:
            self.container.register_service("job_manager", self.job_manager)

        logger.info("Orchestration services initialized successfully")


__all__ = ["ServiceFactoryComponentsMixin"]
