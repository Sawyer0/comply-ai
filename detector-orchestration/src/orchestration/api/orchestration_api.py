"""Orchestration API endpoints following SRP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from shared.exceptions.base import BaseServiceException
from shared.interfaces.base import ApiResponse
from shared.interfaces.orchestration import (
    BatchOrchestrationRequest,
    BatchOrchestrationResponse,
    OrchestrationRequest,
    OrchestrationResponse,
)

from ..service import OrchestrationRequestContext, OrchestrationService
from .dependencies import (
    get_correlation_id_header,
    get_idempotency_key,
    get_orchestration_service,
    get_tenant_id,
    validate_request_auth,
)
from .utils import make_api_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orchestrate", tags=["orchestration"])


@dataclass
class OrchestrationEndpointContext:
    """Common dependencies shared by orchestration endpoints."""

    tenant_id: str
    correlation_id: str
    api_key: Optional[str]
    idempotency_key: Optional[str]
    service: OrchestrationService


async def build_orchestration_context(
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    api_key: Optional[str] = Depends(validate_request_auth),
    idempotency_key: Optional[str] = Depends(get_idempotency_key),
    service: OrchestrationService = Depends(get_orchestration_service),
) -> OrchestrationEndpointContext:
    """Assemble reusable context information for orchestration handlers."""

    return OrchestrationEndpointContext(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        api_key=api_key,
        idempotency_key=idempotency_key,
        service=service,
    )


@router.post("", response_model=ApiResponse[OrchestrationResponse])
async def orchestrate_detectors(
    request: OrchestrationRequest,
    context: OrchestrationEndpointContext = Depends(build_orchestration_context),
) -> ApiResponse[OrchestrationResponse]:
    """Main orchestration endpoint for detector coordination."""

    start_time = datetime.utcnow()
    request_context = OrchestrationRequestContext(
        correlation_id=context.correlation_id,
        api_key=context.api_key,
        idempotency_key=context.idempotency_key,
    )

    try:
        logger.info(
            "Processing orchestration request",
            extra={
                "correlation_id": context.correlation_id,
                "tenant_id": context.tenant_id,
                "content_length": len(request.content) if request.content else 0,
                "processing_mode": request.processing_mode,
            },
        )

        response = await context.service.orchestrate(
            request=request,
            tenant_id=context.tenant_id,
            context_input=request_context,
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return make_api_response(
            data=response,
            request_id=context.correlation_id,
            tenant_id=context.tenant_id,
            extra_metadata={"processingTime": f"{processing_time:.3f}s"},
        )

    except BaseServiceException as exc:
        logger.error(
            "Orchestration failed with service exception",
            extra={
                "correlation_id": context.correlation_id,
                "tenant_id": context.tenant_id,
                "error": str(exc),
            },
        )
        status_code = getattr(exc, "http_status_code", 500)
        raise HTTPException(status_code=status_code, detail=exc.to_dict()) from exc


@router.post("/batch", response_model=ApiResponse[BatchOrchestrationResponse])
async def orchestrate_batch(
    request: BatchOrchestrationRequest,
    context: OrchestrationEndpointContext = Depends(build_orchestration_context),
) -> ApiResponse[BatchOrchestrationResponse]:
    """Batch orchestration endpoint supporting synchronous and async workflows."""

    logger.info(
        "Processing batch orchestration request",
        extra={
            "correlation_id": context.correlation_id,
            "tenant_id": context.tenant_id,
            "batch_size": len(request.requests),
            "async_processing": request.async_processing,
        },
    )

    try:
        if request.async_processing:
            job_id = await context.service.submit_batch_job(
                requests=request.requests,
                tenant_id=context.tenant_id,
                correlation_id=context.correlation_id,
                api_key=context.api_key,
            )

            return make_api_response(
                data=BatchOrchestrationResponse(
                    job_id=job_id,
                    status="submitted",
                    total_requests=len(request.requests),
                    completed_requests=0,
                    results=[],
                ),
                request_id=context.correlation_id,
                tenant_id=context.tenant_id,
                extra_metadata={"processing": "async"},
            )

        response = await context.service.orchestrate_batch(
            requests=request.requests,
            tenant_id=context.tenant_id,
            correlation_id=context.correlation_id,
            api_key=context.api_key,
        )

        return make_api_response(
            data=response,
            request_id=context.correlation_id,
            tenant_id=context.tenant_id,
            extra_metadata={"processing": "sync"},
        )

    except BaseServiceException as exc:
        logger.error(
            "Batch orchestration failed",
            extra={
                "correlation_id": context.correlation_id,
                "tenant_id": context.tenant_id,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    context: OrchestrationEndpointContext = Depends(build_orchestration_context),
) -> ApiResponse[BatchOrchestrationResponse]:
    """Retrieve the status of an asynchronous batch orchestration job."""

    response = await context.service.get_batch_job_status(
        job_id=job_id,
        tenant_id=context.tenant_id,
        correlation_id=context.correlation_id,
    )

    if not response:
        raise HTTPException(status_code=404, detail="Job not found")

    return make_api_response(
        data=response,
        request_id=context.correlation_id,
        tenant_id=context.tenant_id,
    )
