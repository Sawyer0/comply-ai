"""Core orchestration pipeline helpers for the service factory."""

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-locals,too-many-branches,too-many-statements,too-many-positional-arguments

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

from detector_orchestration.conflict import ConflictResolutionRequest
from detector_orchestration.cache import ResponseCache
from detector_orchestration.models import (
    DetectorStatus,
    MappingResponse,
    OrchestrationRequest,
    OrchestrationResponse,
    PolicyContext,
    RoutingDecision,
    RoutingPlan,
)

logger = logging.getLogger(__name__)


class ServiceFactoryPipelineMixin:
    """Provides orchestration execution helpers and caching utilities."""

    # pylint: disable=too-few-public-methods

    settings: Any
    metrics: Any
    conflict_resolver: Any
    aggregator: Any
    router: Any
    coordinator: Any
    mapper_client: Any
    idempotency_cache: Any
    response_cache: Any
    pending_idempotent_jobs: Dict[str, Tuple[str, str]]

    @staticmethod
    def build_idempotency_cache_key(tenant_id: str, idempotency_key: str) -> str:
        return f"{tenant_id}::orchestrate::{idempotency_key}"

    @staticmethod
    def request_fingerprint(request: OrchestrationRequest) -> str:
        payload = request.model_dump(exclude_none=True)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def estimate_processing_time_ms(self, routing_plan: RoutingPlan) -> int:
        total = 0
        for detector in routing_plan.primary_detectors:
            det_cfg = self.settings.detectors.get(detector)
            if det_cfg:
                total += int(getattr(det_cfg, "timeout_ms", 0))
        return total

    def response_cache_key(
        self, request: OrchestrationRequest, routing_plan: RoutingPlan
    ) -> Optional[str]:
        if not routing_plan.primary_detectors:
            return None
        try:
            detector_tuple = tuple(routing_plan.primary_detectors)
            return ResponseCache.build_key(
                request.content, detector_tuple, request.policy_bundle
            )
        except Exception:  # pylint: disable=broad-exception-caught
            return None

    @staticmethod
    def clone_response(
        response: OrchestrationResponse,
        *,
        request_id: Optional[str] = None,
        job_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> OrchestrationResponse:
        payload = response.model_dump()
        if request_id is not None:
            payload["request_id"] = request_id
        if job_id is not None:
            payload["job_id"] = job_id
        if idempotency_key is not None:
            payload["idempotency_key"] = idempotency_key
        return OrchestrationResponse(**payload)

    async def run_pipeline(
        self,
        request: OrchestrationRequest,
        *,
        routing_plan: RoutingPlan,
        decision: RoutingDecision,
        request_id: str,
        raw_idempotency_key: Optional[str],
        idempotency_cache_key: Optional[str],
        fingerprint: Optional[str],
        job_id: Optional[str] = None,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> OrchestrationResponse:
        if not self.coordinator or not self.aggregator or not self.router:
            raise RuntimeError("Orchestration services not initialized")

        start = time.perf_counter()
        tenant = request.tenant_id
        processing_mode = request.processing_mode.value

        try:
            self.metrics.record_request_start(
                request_id,
                tenant,
                decision.policy_applied,
                processing_mode=processing_mode,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        if progress_cb:
            progress_cb(0.05)

        detector_results = await self.coordinator.execute_routing_plan(
            request.content,
            routing_plan,
            request_id,
            request.metadata or {},
        )

        if progress_cb:
            progress_cb(0.45)

        try:
            for result in detector_results:
                self.metrics.record_detector_latency(
                    result.detector,
                    result.status == DetectorStatus.SUCCESS,
                    float(result.processing_time_ms),
                )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        detectors_attempted = len(detector_results)
        detectors_succeeded = sum(
            1 for r in detector_results if r.status == DetectorStatus.SUCCESS
        )
        detectors_failed = sum(
            1
            for r in detector_results
            if r.status
            in {
                DetectorStatus.FAILED,
                DetectorStatus.TIMEOUT,
                DetectorStatus.UNAVAILABLE,
            }
        )

        coverage_achieved = 0.0
        payload = None
        conflict_outcome = None
        if self.conflict_resolver:
            conflict_request = ConflictResolutionRequest(
                tenant_id=request.tenant_id,
                policy_bundle=request.policy_bundle,
                content_type=request.content_type,
                detector_results=detector_results,
                weights=routing_plan.weights or {},
            )
            conflict_outcome = await self.conflict_resolver.resolve(conflict_request)

        payload, coverage_achieved = self.aggregator.aggregate(
            detector_results,
            routing_plan,
            request.tenant_id,
            conflict_outcome=conflict_outcome,
        )
        coverage_target = float(
            decision.coverage_requirements.get("min_success_fraction", 1.0)
        )

        if progress_cb:
            progress_cb(0.7)

        fallback_used = payload.output == "none"
        fallback_reason = "no_detector_output" if fallback_used else None
        mapping_result: Optional[MappingResponse] = None
        error_code: Optional[str] = None

        if decision.auto_map_results and self.mapper_client:
            try:
                mapping_result, mapper_error = await self.mapper_client.map(
                    tenant_id=request.tenant_id,
                    policy_bundle=request.policy_bundle,
                    aggregated_payload=payload,
                    detector_results=detector_results,
                    routing_plan=routing_plan,
                    expected_detectors=routing_plan.primary_detectors,
                    request_metadata=request.metadata or {},
                )
                if mapper_error:
                    error_code = mapper_error
                    fallback_used = True
                    fallback_reason = mapper_error
                elif mapping_result:
                    try:
                        mapping_result.policy_context = PolicyContext(
                            expected_detectors=routing_plan.primary_detectors,
                            environment=request.environment,
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
            except Exception:  # pylint: disable=broad-exception-caught
                error_code = "DETECTOR_COMMUNICATION_FAILED"
                fallback_used = True
                fallback_reason = "mapper_exception"
                mapping_result = None

        if progress_cb:
            progress_cb(0.9)

        total_processing_time_ms = max(
            int((time.perf_counter() - start) * 1000),
            0,
        )

        response_obj = OrchestrationResponse(
            request_id=request_id,
            job_id=job_id,
            processing_mode=request.processing_mode,
            detector_results=detector_results,
            aggregated_payload=payload,
            mapping_result=mapping_result,
            total_processing_time_ms=total_processing_time_ms,
            detectors_attempted=detectors_attempted,
            detectors_succeeded=detectors_succeeded,
            detectors_failed=detectors_failed,
            coverage_achieved=coverage_achieved,
            routing_decision=decision,
            fallback_used=fallback_used,
            timestamp=datetime.now(timezone.utc),
            error_code=error_code,
            idempotency_key=raw_idempotency_key,
        )

        try:
            self.metrics.record_coverage(
                tenant,
                decision.policy_applied,
                coverage_achieved,
            )
            self.metrics.record_request_end(
                request_id,
                success=error_code is None,
                duration_ms=float(total_processing_time_ms),
                tenant=tenant,
                policy=decision.policy_applied,
                processing_mode=processing_mode,
            )
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        cache_key = self.response_cache_key(request, routing_plan)
        if cache_key and self.response_cache:
            try:
                cached_copy = OrchestrationResponse(**response_obj.model_dump())
                self.response_cache.set(cache_key, cached_copy)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

        if idempotency_cache_key and self.idempotency_cache:
            try:
                stored_copy = OrchestrationResponse(**response_obj.model_dump())
                self.idempotency_cache.set(
                    idempotency_cache_key,
                    stored_copy,
                    fingerprint=fingerprint,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            self.pending_idempotent_jobs.pop(idempotency_cache_key, None)

        event_payload: Dict[str, Any] = {
            "type": "orchestration_result",
            "request_id": request_id,
            "job_id": job_id,
            "tenant_id": request.tenant_id,
            "policy_bundle": request.policy_bundle,
            "processing_mode": processing_mode,
            "coverage_achieved": coverage_achieved,
            "coverage_target": coverage_target,
            "detectors_attempted": detectors_attempted,
            "detectors_failed": detectors_failed,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "error_code": error_code,
        }
        self.publish_event(event_payload)

        if fallback_used or coverage_achieved < coverage_target:
            severity = "high" if fallback_used else "medium"
            incident_event = {
                "type": "incident",
                "request_id": request_id,
                "job_id": job_id,
                "tenant_id": request.tenant_id,
                "policy_bundle": request.policy_bundle,
                "coverage_achieved": coverage_achieved,
                "coverage_target": coverage_target,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "detectors_failed": detectors_failed,
                "error_code": error_code,
                "severity": severity,
            }
            self.publish_event(incident_event)

        if progress_cb:
            progress_cb(1.0)

        return response_obj

    async def run_async_job(
        self,
        request: OrchestrationRequest,
        idempotency_key: Optional[str],
        decision: RoutingDecision,
        routing_plan: RoutingPlan,
        progress_cb: Optional[Callable[[float], None]] = None,
    ) -> OrchestrationResponse:
        request_id = str(uuid.uuid4())
        fingerprint = self.request_fingerprint(request)
        cache_key = (
            self.build_idempotency_cache_key(request.tenant_id, idempotency_key)
            if idempotency_key
            else None
        )
        return await self.run_pipeline(
            request,
            routing_plan=routing_plan,
            decision=decision,
            request_id=request_id,
            raw_idempotency_key=idempotency_key,
            idempotency_cache_key=cache_key,
            fingerprint=fingerprint,
            job_id=None,
            progress_cb=progress_cb,
        )


__all__ = ["ServiceFactoryPipelineMixin"]
