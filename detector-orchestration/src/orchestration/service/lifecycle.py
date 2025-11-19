"""Lifecycle helpers for orchestration metrics and responses."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING

from shared.exceptions.base import BaseServiceException
from shared.interfaces.common import Severity
from shared.interfaces.orchestration import (
    AggregationSummary,
    DetectorResult,
    OrchestrationResponse,
    PolicyViolation,
    RiskSummary,
)

if TYPE_CHECKING:
    from .models import OrchestrationArtifacts, PipelineContext


logger = logging.getLogger(__name__)


def get_processing_time_ms(start_time: datetime) -> float:
    return (datetime.utcnow() - start_time).total_seconds() * 1000


async def cache_idempotent_response(service, idempotency_key: Optional[str], response):
    if not idempotency_key or not service.components.idempotency_cache:
        return
    await service.components.idempotency_cache.set(idempotency_key, response)


def update_average_response_time(service, processing_time: float) -> None:
    current_avg = service._metrics["average_response_time_ms"]  # noqa: SLF001
    total_requests = service._metrics["successful_requests"]  # noqa: SLF001

    if total_requests == 1:
        service._metrics["average_response_time_ms"] = processing_time  # noqa: SLF001
    else:
        service._metrics["average_response_time_ms"] = (  # noqa: SLF001
            current_avg * (total_requests - 1) + processing_time
        ) / total_requests


def record_success(
    service,
    *,
    context: "PipelineContext",
    processing_time: float,
    artifacts: "OrchestrationArtifacts",
) -> None:
    service._metrics["successful_requests"] += 1  # noqa: SLF001
    update_average_response_time(service, processing_time)

    metrics = service.components.metrics_collector
    if metrics:
        # Request-level metrics
        metrics.record_request(
            tenant_id=context.tenant_id,
            processing_mode=context.processing_mode,
            status="success",
            duration_seconds=processing_time / 1000.0,
        )

        # Per-detector execution metrics
        for result in artifacts.detector_results:
            status = "success" if result.confidence > 0.0 else "failure"
            duration_ms = getattr(result, "processing_time_ms", None)
            duration_seconds = (duration_ms or 0) / 1000.0
            if isinstance(result, DetectorResult):
                metrics.record_detector_execution(
                    detector_id=result.detector_id,
                    detector_type=result.detector_type,
                    status=status,
                    duration_seconds=duration_seconds,
                )

    logger.info(
        "Orchestration completed successfully in %.2fms",
        processing_time,
        extra=service._log_extra(  # noqa: SLF001
            context.correlation_id,
            tenant_id=context.tenant_id,
            processing_time_ms=processing_time,
            coverage_achieved=artifacts.coverage,
            total_detectors=len(artifacts.detector_results),
            processing_mode=context.processing_mode,
        ),
    )


def record_failure(
    service,
    *,
    context: "PipelineContext",
    processing_time: float,
    exc: BaseServiceException,
) -> None:
    service._metrics["failed_requests"] += 1  # noqa: SLF001
    update_average_response_time(service, processing_time)

    metrics = service.components.metrics_collector
    if metrics:
        metrics.record_request(
            tenant_id=context.tenant_id,
            processing_mode=context.processing_mode,
            status="failure",
            duration_seconds=processing_time / 1000.0,
        )

    logger.error(
        "Orchestration failed in %.2fms: %s",
        processing_time,
        str(exc),
        extra=service._log_extra(  # noqa: SLF001
            context.correlation_id,
            tenant_id=context.tenant_id,
            processing_time_ms=processing_time,
            error=str(exc),
            processing_mode=context.processing_mode,
        ),
    )


def build_success_response(
    service,
    *,
    context: "PipelineContext",
    processing_time: float,
    artifacts: "OrchestrationArtifacts",
    risk_score: Optional[Any] = None,
) -> OrchestrationResponse:
    detector_results = artifacts.detector_results
    total_detectors = len(detector_results)
    successful_detectors = len([r for r in detector_results if r.confidence > 0.0])
    failed_detectors = total_detectors - successful_detectors
    average_confidence = (
        sum(r.confidence for r in detector_results) / total_detectors
        if detector_results
        else 0.0
    )

    risk_summary: Optional[RiskSummary] = None
    if risk_score is not None:
        # RiskScore comes from orchestration.ml.risk_scorer and is intentionally kept
        # as a loose dependency here to avoid tight coupling between service layers.
        try:
            level = getattr(risk_score, "level", None)
            score = float(getattr(risk_score, "score", 0.0))
            rules = getattr(risk_score, "rules_evaluation", {}) or {}
            features = getattr(risk_score, "model_features", {}) or {}
            if level is not None:
                risk_summary = RiskSummary(
                    level=level,
                    score=score,
                    rules_evaluation=rules,
                    model_features=features,
                )
        except Exception:
            # Defensive: if anything goes wrong building risk summary,
            # we still want to return a valid orchestration response.
            logger.exception("Failed to build risk summary for response")

    canonical_outputs_dict: Optional[dict] = None
    if getattr(artifacts, "canonical_outputs", None) is not None:
        canonical = artifacts.canonical_outputs
        try:
            # CanonicalDetectorOutputs is a Pydantic model; use model_dump when available.
            if hasattr(canonical, "model_dump"):
                canonical_outputs_dict = canonical.model_dump()
            elif hasattr(canonical, "dict"):
                canonical_outputs_dict = canonical.dict()
            else:
                canonical_outputs_dict = canonical  # type: ignore[assignment]
        except Exception:
            logger.exception("Failed to serialize canonical outputs for response")

    correlation_id = context.correlation_id

    return OrchestrationResponse(
        request_id=correlation_id,
        success=True,
        timestamp=datetime.utcnow(),
        processing_time_ms=processing_time,
        correlation_id=correlation_id,
        detector_results=detector_results,
        aggregation_summary=AggregationSummary(
            total_detectors=total_detectors,
            successful_detectors=successful_detectors,
            failed_detectors=failed_detectors,
            average_confidence=average_confidence,
        ),
        coverage_achieved=artifacts.coverage,
        policy_violations=artifacts.policy_violations,
        recommendations=artifacts.recommendations,
        canonical_outputs=canonical_outputs_dict,
        risk_summary=risk_summary,
    )


def build_failure_response(
    service,
    *,
    context: "PipelineContext",
    processing_time: float,
    artifacts: Optional["OrchestrationArtifacts"],
    exc: BaseServiceException,
) -> OrchestrationResponse:
    policy_violations = list(artifacts.policy_violations if artifacts else [])
    policy_violations.append(
        PolicyViolation(
            policy_id="system:orchestration-error",
            violation_type="system_error",
            message=str(exc),
            severity=Severity.HIGH,
        )
    )

    correlation_id = context.correlation_id

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
        policy_violations=policy_violations,
        recommendations=["Check service logs for error details"],
    )
