"""Main orchestration service integrating shared components."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from shared.exceptions.base import BaseServiceException
from shared.interfaces.base import BaseService
from shared.interfaces.orchestration import (
    AggregationSummary,
    OrchestrationRequest,
    OrchestrationResponse,
)
from shared.interfaces.common import HealthStatus
from shared.utils.correlation import get_correlation_id, set_correlation_id

from ..core import (
    AggregatedOutput,
    RoutingDecision,
    RoutingPlan as CoordinatorRoutingPlan,
)
from ..monitoring import HealthMonitor, HTTPHealthCheckClient
from ..repository import DetectorRecord
from ..tenancy.tenant_isolation import TenantContext
from . import (
    initialization,
    lifecycle,
    pipeline,
    registration as registration_helpers,
    security,
)
from .models import (
    AggregationContext,
    DetectorRegistrationConfig,
    OrchestrationArtifacts,
    OrchestrationComponents,
    OrchestrationConfig,
    OrchestrationRequestContext,
    PipelineContext,
)

logger = logging.getLogger(__name__)


class OrchestrationService(BaseService):
    """Main orchestration service that coordinates all SRP components."""

    def __init__(
        self,
        config: Optional[OrchestrationConfig] = None,
        *,
        components: Optional[OrchestrationComponents] = None,
    ):
        super().__init__("detector-orchestration", "1.0.0")

        self.config = config or OrchestrationConfig()
        self.components = components or OrchestrationComponents()

        self._is_running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._metrics: Dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time_ms": 0.0,
        }

        initialization.initialize_components(self)
        initialization.initialize_cache_components(self)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def orchestrate(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        context_input: Optional[OrchestrationRequestContext] = None,
    ) -> OrchestrationResponse:
        """Coordinate detector orchestration for a single tenant request."""
        context_input = context_input or OrchestrationRequestContext()
        correlation_id = self._ensure_correlation_id(context_input.correlation_id)
        context_input.correlation_id = correlation_id
        context_input.processing_mode = getattr(request, "processing_mode", "standard")

        cached_response = await self._get_cached_response(context_input.idempotency_key)
        if cached_response:
            logger.info(
                "Returning cached response for idempotency key",
                extra=self._log_extra(
                    correlation_id,
                    tenant_id=tenant_id,
                    idempotency_key=context_input.idempotency_key,
                ),
            )
            return cached_response

        if not await security.validate_request_security(
            self,
            request=request,
            tenant_id=tenant_id,
            api_key=context_input.api_key,
            user_id=context_input.user_id,
        ):
            return self._build_security_failure_response(correlation_id)

        start_time = datetime.utcnow()
        self._metrics["total_requests"] += 1

        logger.info(
            "Starting orchestration for tenant %s",
            tenant_id,
            extra=self._log_extra(
                correlation_id,
                tenant_id=tenant_id,
                detector_types=getattr(request, "detector_types", []),
                processing_mode=context_input.processing_mode,
            ),
        )

        artifacts: Optional[OrchestrationArtifacts] = None
        pipeline_context: Optional[PipelineContext] = None
        risk_score = None

        try:
            pipeline_context, artifacts = await pipeline.run_pipeline(
                self,
                request=request,
                tenant_id=tenant_id,
                context_input=context_input,
            )

            # Compute risk score for this request if the scorer is available.
            if self.components.risk_scorer and artifacts is not None and pipeline_context is not None:
                try:
                    risk_score = self.components.risk_scorer.score(
                        detector_results=artifacts.detector_results,
                        aggregated_output=artifacts.aggregated_output,
                        policy_violations=artifacts.policy_violations,
                        coverage=artifacts.coverage,
                        canonical_outputs=artifacts.canonical_outputs,
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(
                        "Failed to compute risk score: %s",
                        exc,
                        extra=self._log_extra(
                            correlation_id,
                            tenant_id=tenant_id,
                        ),
                    )
                    risk_score = None

            processing_time = lifecycle.get_processing_time_ms(start_time)
            response = lifecycle.build_success_response(
                self,
                context=pipeline_context,
                processing_time=processing_time,
                artifacts=artifacts,
                risk_score=risk_score,
            )

            await lifecycle.cache_idempotent_response(
                self,
                context_input.idempotency_key,
                response,
            )
            lifecycle.record_success(
                self,
                context=pipeline_context,
                processing_time=processing_time,
                artifacts=artifacts,
            )

            # Persist risk analysis for this request if a repository is configured
            if (
                self.components.risk_repository
                and pipeline_context is not None
                and artifacts is not None
                and risk_score is not None
            ):
                try:
                    detector_ids = [
                        result.detector_id for result in artifacts.detector_results
                    ]

                    await self.components.risk_repository.create_risk_analysis(
                        fields={
                            "tenant_id": pipeline_context.tenant_id,
                            "request_correlation_id": pipeline_context.correlation_id,
                            "risk_level": risk_score.level.value,
                            "risk_score": risk_score.score,
                            "rules_evaluation": risk_score.rules_evaluation,
                            "model_features": risk_score.model_features,
                            "detector_ids": detector_ids,
                            "requested_by": context_input.user_id,
                            "requested_via_api_key": context_input.api_key,
                        },
                    )
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.error(
                        "Failed to persist risk analysis: %s",
                        exc,
                        extra=self._log_extra(
                            correlation_id,
                            tenant_id=tenant_id,
                        ),
                    )

            # Feed detector results back into ML components
            if (
                self.components.ml_feedback
                and pipeline_context is not None
                and artifacts is not None
            ):
                await self.components.ml_feedback.update_models_with_feedback(
                    detector_results=artifacts.detector_results,
                    routing_decision=pipeline_context.routing_decision,
                    _content_features=pipeline_context.content_features,
                )
            return response

        except BaseServiceException as exc:
            processing_time = lifecycle.get_processing_time_ms(start_time)
            fallback_context = pipeline_context or self._build_fallback_context(
                tenant_id=tenant_id,
                context_input=context_input,
            )
            lifecycle.record_failure(
                self,
                context=fallback_context,
                processing_time=processing_time,
                exc=exc,
            )
            return lifecycle.build_failure_response(
                self,
                context=fallback_context,
                processing_time=processing_time,
                artifacts=artifacts,
                exc=exc,
            )

    async def register_detector(
        self,
        registration: Optional[DetectorRegistrationConfig] = None,
        **kwargs: Any,
    ) -> bool:
        """Register an external detector across discovery, routing, and client registries.

        Accepts either a DetectorRegistrationConfig instance or the individual keyword
        arguments required to construct one. The kwargs path keeps backward compatibility
        with older call sites that passed detector_id/endpoint directly.
        """

        if registration is None:
            if not kwargs:
                raise ValueError("registration details are required")
            registration = DetectorRegistrationConfig(**kwargs)

        await self._persist_detector_registration(registration)
        success = await registration_helpers.register_detector(self, registration)
        if success:
            await self._register_health_client(
                registration.detector_id,
                registration.endpoint,
                registration.timeout_ms,
                health_check_url=None,
            )
        return success

    async def unregister_detector(
        self,
        detector_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> bool:
        """Remove detector from persistence and runtime components."""

        repo = self.components.detector_repository
        if not repo:
            return False

        record: Optional[DetectorRecord]
        if tenant_id:
            record = await repo.get_detector_by_identity(
                tenant_id=tenant_id,
                detector_name=detector_id,
                detector_type=detector_id,
            )
            if not record:
                record = await repo.get_detector_by_name(
                    tenant_id=tenant_id,
                    detector_name=detector_id,
                )
        else:
            record = await repo.get_detector_by_name(
                tenant_id="default",
                detector_name=detector_id,
            )

        if not record:
            existing = await repo.get_detector(detector_id)
            record = existing

        if not record:
            return False

        await self._unregister_runtime_detector(record.detector_name)
        await repo.delete_detector(record.id)
        return True

    async def list_detectors(
        self,
        *,
        tenant_id: Optional[str] = None,
        detector_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[DetectorRecord]:
        repo = self.components.detector_repository
        if not repo:
            return []
        return await repo.list_detectors(
            tenant_id=tenant_id,
            detector_type=detector_type,
            status=status,
        )

    async def get_detector(
        self,
        detector_id: str,
        *,
        tenant_id: Optional[str] = None,
    ) -> Optional[DetectorRecord]:
        repo = self.components.detector_repository
        if not repo:
            return None
        if tenant_id:
            record = await repo.get_detector_by_name(
                tenant_id=tenant_id,
                detector_name=detector_id,
            )
            if record:
                return record
        return await repo.get_detector(detector_id)

    async def update_detector(
        self,
        detector_id: str,
        *,
        tenant_id: str,
        fields: Dict[str, Any],
    ) -> Optional[DetectorRecord]:
        repo = self.components.detector_repository
        if not repo:
            return None

        record = await repo.get_detector_by_name(
            tenant_id=tenant_id,
            detector_name=detector_id,
        )
        if not record:
            return None

        await repo.update_detector(record.id, fields=fields)
        updated = await repo.get_detector(record.id)
        if updated:
            await self._register_runtime_from_record(updated)
        return updated

    async def get_service_status(self) -> Dict[str, Any]:
        """Return a structured snapshot of service health, metrics, and enabled components."""
        return {
            "service": "orchestration",
            "status": "running" if self._is_running else "stopped",
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._metrics.copy(),
            "components": {
                "detector_coordinator": (
                    "enabled" if self.components.detector_coordinator else "disabled"
                ),
                "content_router": (
                    "enabled" if self.components.content_router else "disabled"
                ),
                "response_aggregator": (
                    "enabled" if self.components.response_aggregator else "disabled"
                ),
                "health_monitor": (
                    "enabled" if self.components.health_monitor else "disabled"
                ),
                "service_discovery": (
                    "enabled" if self.components.service_discovery else "disabled"
                ),
                "policy_manager": (
                    "enabled" if self.components.policy_manager else "disabled"
                ),
            },
        }

    async def start(self) -> None:
        """Start background orchestration tasks and supporting processors."""
        if self._is_running:
            logger.warning("Service is already running")
            return

        logger.info("Starting orchestration service")

        await self._hydrate_detectors_from_repository()

        if self.components.health_monitor:
            task = asyncio.create_task(self._health_check_loop())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        if self.components.job_processor:
            await self.components.job_processor.start()

        self._is_running = True
        logger.info("Orchestration service started successfully")

    async def stop(self) -> None:
        """Gracefully stop background tasks and release detector resources."""
        if not self._is_running:
            logger.warning("Service is not running")
            return

        logger.info("Stopping orchestration service")

        for task in self._background_tasks:
            task.cancel()

        if self.components.job_processor:
            await self.components.job_processor.stop()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        if self.components.detector_clients:
            await self._close_detector_clients()

        self._is_running = False
        logger.info("Orchestration service stopped")

    async def _close_detector_clients(self) -> None:
        """Close all registered detector HTTP clients."""
        tasks = [client.close() for client in self.components.detector_clients.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.components.detector_clients.clear()

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._is_running:
            try:
                monitor = self.components.health_monitor
                if isinstance(monitor, HealthMonitor):
                    checks = await monitor.check_all_services()
                    await self._update_detector_health_from_checks(checks)
                    await asyncio.sleep(monitor.health_check_interval)
            except asyncio.CancelledError:  # pragma: no cover - cancellation flow
                break
            except (BaseServiceException, asyncio.TimeoutError) as exc:
                logger.error("Health check loop error: %s", str(exc))
                await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Supporting utilities
    # ------------------------------------------------------------------
    # Supporting utilities
    # ------------------------------------------------------------------
    @property
    def content_router(self):
        """Return the configured content router."""
        return self.components.content_router

    @property
    def health_monitor(self):
        """Return the service health monitor, when enabled."""
        return self.components.health_monitor

    @property
    def metrics_collector(self):
        """Return the metrics collector used for observability."""
        return self.components.metrics_collector

    @staticmethod
    def _log_extra(correlation_id: str, **details: Any) -> Dict[str, Any]:
        extra = {"correlation_id": correlation_id}
        extra.update(
            {key: value for key, value in details.items() if value is not None}
        )
        return extra

    def _ensure_correlation_id(self, correlation_id: Optional[str]) -> str:
        if correlation_id:
            set_correlation_id(correlation_id)
            return correlation_id

        generated = get_correlation_id()
        set_correlation_id(generated)
        return generated

    async def _get_cached_response(
        self, idempotency_key: Optional[str]
    ) -> Optional[OrchestrationResponse]:
        if not idempotency_key or not self.components.idempotency_cache:
            return None
        return await self.components.idempotency_cache.get(idempotency_key)

    def _build_security_failure_response(
        self, correlation_id: str
    ) -> OrchestrationResponse:
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

    @asynccontextmanager
    async def _fallback_tenant_context(
        self,
        tenant_id: str,
        user_id: Optional[str],
        correlation_id: str,
    ):
        yield TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            request_id=correlation_id,
        )

    def _build_fallback_context(
        self,
        *,
        tenant_id: str,
        context_input: OrchestrationRequestContext,
    ) -> PipelineContext:
        correlation_id = context_input.correlation_id or get_correlation_id()
        return PipelineContext(
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            tenant_context=TenantContext(
                tenant_id=tenant_id,
                user_id=context_input.user_id,
                request_id=correlation_id,
            ),
            processing_mode=context_input.processing_mode,
        )

    # Compatibility wrappers for refactored helpers
    # pylint: disable=missing-function-docstring
    async def validate_request_security(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        return await security.validate_request_security(
            self,
            request=request,
            tenant_id=tenant_id,
            api_key=api_key,
            user_id=user_id,
        )

    def _initialize_components(self) -> None:
        """Compatibility wrapper for component initialization."""
        initialization.initialize_components(self)

    def _bootstrap_detectors_from_discovery(self) -> None:
        """Compatibility wrapper for discovery bootstrap logic."""
        initialization.bootstrap_detectors_from_discovery(self)

    def _initialize_cache_components(self) -> None:
        """Compatibility wrapper for cache initialization."""
        initialization.initialize_cache_components(self)

    async def _run_pipeline(
        self,
        *,
        request: OrchestrationRequest,
        tenant_id: str,
        context_input: OrchestrationRequestContext,
    ) -> Tuple[PipelineContext, OrchestrationArtifacts]:
        return await pipeline.run_pipeline(
            self,
            request=request,
            tenant_id=tenant_id,
            context_input=context_input,
        )

    async def _execute_pipeline(
        self,
        *,
        request: OrchestrationRequest,
        context: PipelineContext,
    ) -> OrchestrationArtifacts:
        return await pipeline.execute_pipeline(self, request=request, context=context)

    def _analyze_content(
        self, content: str, correlation_id: str
    ) -> Optional[Any]:
        return pipeline.analyze_content(self, content, correlation_id)

    async def _determine_routing(
        self,
        request: OrchestrationRequest,
        tenant_id: str,
        correlation_id: str,
    ) -> Tuple[Optional[CoordinatorRoutingPlan], Optional[RoutingDecision]]:
        return await pipeline.determine_routing(
            self,
            request=request,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
        )

    def _to_coordinator_routing_plan(
        self, router_plan: Any
    ) -> Optional[CoordinatorRoutingPlan]:
        return pipeline.to_coordinator_routing_plan(router_plan)

    async def _execute_detectors(
        self,
        request: OrchestrationRequest,
        context: PipelineContext,
    ) -> List:
        return await pipeline.execute_detectors(self, request=request, context=context)

    def _aggregate_results(
        self,
        detector_results,
        *,
        context: PipelineContext,
    ) -> Tuple[Optional[AggregatedOutput], float]:
        return pipeline.aggregate_results(self, detector_results, context=context)

    async def _evaluate_policy_violations(
        self,
        *,
        context: PipelineContext,
        policy_bundle: Optional[str],
        aggregation: AggregationContext,
    ) -> List:
        return await pipeline.evaluate_policy_violations(
            self,
            context=context,
            policy_bundle=policy_bundle,
            aggregation=aggregation,
        )

    def _generate_recommendations(self, aggregation: AggregationContext) -> List[str]:
        return pipeline.generate_recommendations(aggregation)

    async def _record_success(
        self,
        *,
        context: PipelineContext,
        processing_time: float,
        artifacts: OrchestrationArtifacts,
    ) -> None:
        lifecycle.record_success(
            self,
            context=context,
            processing_time=processing_time,
            artifacts=artifacts,
        )

    async def _cache_idempotent_response(
        self,
        idempotency_key: Optional[str],
        response: OrchestrationResponse,
    ) -> None:
        await lifecycle.cache_idempotent_response(self, idempotency_key, response)

    def _get_processing_time_ms(self, start_time: datetime) -> float:
        return lifecycle.get_processing_time_ms(start_time)

    def _update_average_response_time(self, processing_time: float) -> None:
        lifecycle.update_average_response_time(self, processing_time)

    def _record_failure(
        self,
        *,
        context: PipelineContext,
        processing_time: float,
        exc: BaseServiceException,
    ) -> None:
        lifecycle.record_failure(
            self,
            context=context,
            processing_time=processing_time,
            exc=exc,
        )

    def _build_success_response(
        self,
        *,
        context: PipelineContext,
        processing_time: float,
        artifacts: OrchestrationArtifacts,
    ) -> OrchestrationResponse:
        return lifecycle.build_success_response(
            self,
            context=context,
            processing_time=processing_time,
            artifacts=artifacts,
        )

    def _build_failure_response(
        self,
        *,
        context: PipelineContext,
        processing_time: float,
        artifacts: Optional[OrchestrationArtifacts],
        exc: BaseServiceException,
    ) -> OrchestrationResponse:
        return lifecycle.build_failure_response(
            self,
            context=context,
            processing_time=processing_time,
            artifacts=artifacts,
            exc=exc,
        )
    async def _persist_detector_registration(
        self, registration: DetectorRegistrationConfig
    ) -> None:
        """Persist detector registration details to the detectors table.

        This keeps the database as the source of truth while the runtime
        router/service-discovery/client layers are hydrated separately.
        """

        repo = self.components.detector_repository
        if not repo:
            return

        tenant_id = registration.tenant_id or "default"
        config_payload: Dict[str, Any] = {
            "analyze_path": registration.analyze_path,
            "response_parser": registration.response_parser,
            "auth_headers": registration.auth_headers or {},
            "timeout_ms": registration.timeout_ms,
            "max_retries": registration.max_retries,
        }

        existing = await repo.get_detector_by_identity(
            tenant_id=tenant_id,
            detector_name=registration.detector_id,
            detector_type=registration.detector_type,
        )

        fields: Dict[str, Any] = {
            "detector_type": registration.detector_type,
            "detector_name": registration.detector_id,
            "endpoint_url": registration.endpoint,
            "health_check_url": registration.endpoint.rstrip("/") + "/health",
            "status": "active",
            "version": "1.0.0",
            "capabilities": registration.supported_content_types or ["text"],
            "configuration": config_payload,
            "tenant_id": tenant_id,
        }

        if existing:
            await repo.update_detector(existing.id, fields=fields)
        else:
            await repo.create_detector(fields=fields)

    async def _register_runtime_from_record(self, record: DetectorRecord) -> bool:
        """Hydrate routing, discovery, and client layers from a DB record."""

        config = dict(record.configuration or {})
        registration = DetectorRegistrationConfig(
            detector_id=record.detector_name,
            endpoint=record.endpoint_url,
            detector_type=record.detector_type,
            tenant_id=record.tenant_id,
            timeout_ms=int(config.get("timeout_ms", 5000)),
            max_retries=int(config.get("max_retries", 3)),
            supported_content_types=record.capabilities or ["text"],
            auth_headers=config.get("auth_headers") or {},
            analyze_path=config.get("analyze_path", "/analyze"),
            response_parser=config.get("response_parser"),
        )

        success = await registration_helpers.register_detector(self, registration)
        if success:
            timeout_ms = int(config.get("timeout_ms", 5000))
            await self._register_health_client(
                record.detector_name,
                record.endpoint_url,
                timeout_ms,
                health_check_url=record.health_check_url,
            )
        return success

    async def _hydrate_detectors_from_repository(self) -> None:
        """Hydrate runtime components from all persisted detector records."""

        repo = self.components.detector_repository
        if not repo:
            return

        try:
            records = await repo.list_detectors()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to hydrate detectors from repository: %s", exc)
            return

        for record in records:
            try:
                await self._register_runtime_from_record(record)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Failed to hydrate detector %s: %s",
                    record.detector_name,
                    exc,
                )

    async def _unregister_runtime_detector(self, detector_name: str) -> None:
        """Remove detector from router, discovery, and close its client."""

        router = self.components.content_router
        if router:
            router.unregister_detector(detector_name)

        discovery = self.components.service_discovery
        if discovery:
            discovery.unregister_service(detector_name)

        monitor = self.components.health_monitor
        if monitor:
            monitor.unregister_service(detector_name)

        client = self.components.detector_clients.pop(detector_name, None)
        if client:
            try:
                await client.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.warning("Failed to close detector client for %s", detector_name)

    async def _register_health_client(
        self,
        service_id: str,
        endpoint_url: str,
        timeout_ms: int,
        *,
        health_check_url: Optional[str] = None,
    ) -> None:
        monitor = self.components.health_monitor
        if not monitor:
            return

        url = health_check_url or endpoint_url.rstrip("/") + "/health"
        timeout_seconds = max(timeout_ms / 1000.0, 0.1)
        client = HTTPHealthCheckClient(url, timeout_seconds)
        monitor.register_service(service_id, client)

    async def _update_detector_health_from_checks(self, checks: Dict[str, Any]) -> None:
        repo = self.components.detector_repository
        if not repo:
            return

        metrics = self.components.metrics_collector
        healthy_count = 0

        for detector_name, health_check in checks.items():
            try:
                if metrics:
                    metrics.update_detector_health(detector_name, health_check.status)
                if health_check.status == HealthStatus.HEALTHY:
                    healthy_count += 1

                await repo.update_health_by_name(
                    detector_name,
                    health_status=health_check.status.value,
                    response_time_ms=health_check.response_time_ms,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(
                    "Failed to persist health for detector %s: %s",
                    detector_name,
                    exc,
                )

        if metrics is not None:
            metrics.update_active_detectors(healthy_count)

    # pylint: enable=missing-function-docstring


__all__ = [
    "OrchestrationService",
]
