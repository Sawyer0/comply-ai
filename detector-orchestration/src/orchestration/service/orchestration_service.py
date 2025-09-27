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
from shared.utils.correlation import get_correlation_id, set_correlation_id

from ..core import (
    AggregatedOutput,
    RoutingDecision,
    RoutingPlan as CoordinatorRoutingPlan,
)
from ..monitoring import HealthMonitor
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

        try:
            pipeline_context, artifacts = await pipeline.run_pipeline(
                self,
                request=request,
                tenant_id=tenant_id,
                context_input=context_input,
            )

            processing_time = lifecycle.get_processing_time_ms(start_time)
            response = lifecycle.build_success_response(
                self,
                context=pipeline_context,
                processing_time=processing_time,
                artifacts=artifacts,
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

    async def register_detector(self, registration: DetectorRegistrationConfig) -> bool:
        """Register an external detector across discovery, routing, and client registries."""
        return await registration_helpers.register_detector(self, registration)

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
                if isinstance(self.components.health_monitor, HealthMonitor):
                    await self.components.health_monitor.check_all_services()
                    await asyncio.sleep(
                        self.components.health_monitor.health_check_interval
                    )
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

    async def _close_detector_clients(self) -> None:
        tasks = [client.close() for client in self.components.detector_clients.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.components.detector_clients.clear()

    async def _health_check_loop(self) -> None:
        while self._is_running:
            try:
                if isinstance(self.components.health_monitor, HealthMonitor):
                    await self.components.health_monitor.check_all_services()
                    await asyncio.sleep(
                        self.components.health_monitor.health_check_interval
                    )
            except asyncio.CancelledError:  # pragma: no cover - cancellation flow
                break
            except (BaseServiceException, asyncio.TimeoutError) as exc:
                logger.error("Health check loop error: %s", str(exc))
                await asyncio.sleep(5)

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
    ) -> Optional[Dict[str, Any]]:
        return pipeline.analyze_content(self, content, correlation_id)

    async def _determine_routing(
        self, request: OrchestrationRequest, correlation_id: str
    ) -> Tuple[Optional[CoordinatorRoutingPlan], Optional[RoutingDecision]]:
        return await pipeline.determine_routing(
            self, request=request, correlation_id=correlation_id
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

    # pylint: enable=missing-function-docstring


__all__ = [
    "OrchestrationService",
]
