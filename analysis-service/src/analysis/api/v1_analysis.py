"""Canonical /api/v1/analyze endpoint for Analysis Service.

This endpoint accepts the shared AnalysisRequest model (with an embedded
OrchestrationResponse that carries canonical_outputs) and returns the
shared AnalysisResponse model with canonical_results populated from the
canonical detector outputs.
"""

import time
from typing import List

from fastapi import APIRouter, HTTPException

from ..shared_integration import (
    get_shared_logger,
    get_correlation_id,
    set_correlation_id,
    SharedAnalysisRequest,
    AnalysisResponse,
    CanonicalTaxonomyResult,
    QualityMetrics,
)

router = APIRouter()

logger = get_shared_logger(__name__)


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_v1(request: SharedAnalysisRequest) -> AnalysisResponse:
    """Canonical analysis entrypoint for orchestrated detector outputs.

    This endpoint is designed for the orchestrator/clients that already have
    an OrchestrationResponse with canonical_outputs. It:
    - Extracts canonical taxonomy results from canonical_outputs
    - Normalizes them into CanonicalTaxonomyResult instances
    - Builds minimal QualityMetrics
    - Returns a shared AnalysisResponse suitable for downstream mapping.
    """

    start_time = time.time()

    # Respect incoming correlation ID if present
    if getattr(request, "correlation_id", None):
        set_correlation_id(request.correlation_id)  # type: ignore[arg-type]

    try:
        # Extract canonical results from orchestration response helper
        orchestration_response = request.orchestration_response
        canonical_dicts = orchestration_response.canonical_results_dict()

        canonical_results: List[CanonicalTaxonomyResult] = []
        for item in canonical_dicts:
            try:
                canonical_results.append(CanonicalTaxonomyResult(**item))
            except Exception:
                # Skip any malformed entries rather than failing the whole request
                continue

        confidences = [cr.confidence for cr in canonical_results] or [0.0]
        max_conf = max(confidences)
        min_conf = min(confidences)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        processing_time_ms = (time.time() - start_time) * 1000

        quality_metrics = QualityMetrics(
            accuracy_score=max_conf,
            confidence_distribution={
                "max": max_conf,
                "min": min_conf,
                "avg": avg_conf,
                "count": len(canonical_results),
            },
            processing_time_ms=processing_time_ms,
            model_version="canonical-orchestration",
            fallback_used=False,
        )

        response = AnalysisResponse(
            request_id=orchestration_response.request_id,
            success=True,
            processing_time_ms=processing_time_ms,
            correlation_id=get_correlation_id(),
            canonical_results=canonical_results,
            quality_metrics=quality_metrics,
            pattern_analysis=None,
            risk_scores=None,
            compliance_mappings=None,
            rag_insights=None,
        )

        logger.info(
            "Canonical /api/v1/analyze completed",
            tenant_id=getattr(request, "tenant_id", None),
            correlation_id=get_correlation_id(),
            canonical_results_count=len(canonical_results),
        )

        return response

    except HTTPException:
        raise
    except Exception as e:  # pragma: no cover - defensive
        processing_time_ms = (time.time() - start_time) * 1000

        logger.error(
            "Canonical /api/v1/analyze failed",
            tenant_id=getattr(request, "tenant_id", None),
            correlation_id=get_correlation_id(),
            processing_time_ms=processing_time_ms,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
