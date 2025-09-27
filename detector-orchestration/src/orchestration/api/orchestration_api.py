"""Orchestration API endpoints following SRP.

This module handles ONLY orchestration-related endpoints:
- Main orchestration endpoint
- Batch orchestration
- Job status tracking
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from datetime import datetime
import logging

from shared.interfaces.orchestration import (
    OrchestrationRequest,
    OrchestrationResponse,
    BatchOrchestrationRequest,
    BatchOrchestrationResponse,
)
from shared.interfaces.base import ApiResponse
from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import BaseServiceException

from ..service import OrchestrationService, OrchestrationRequestContext
from .dependencies import (
    get_orchestration_service,
    get_tenant_id,
    get_correlation_id_header,
    validate_request_auth,
    get_idempotency_key,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestration"])


@router.post("", response_model=ApiResponse[OrchestrationResponse])
async def orchestrate_detectors(
    request: OrchestrationRequest,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    api_key: Optional[str] = Depends(validate_request_auth),
    idempotency_key: Optional[str] = Depends(get_idempotency_key),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Main orchestration endpoint for detector coordination.

    Coordinates detector execution with:
    - Intelligent routing based on content and policies
    - Circuit breaker protection and retry logic
    - Result aggregation and confidence scoring
    - Policy enforcement and compliance validation
    """
    start_time = datetime.utcnow()

    try:
        logger.info(
            "Processing orchestration request",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "content_length": len(request.content) if request.content else 0,
                "processing_mode": request.processing_mode,
            },
        )

        # Execute orchestration
        request_context = OrchestrationRequestContext(
            correlation_id=correlation_id,
            api_key=api_key,
            idempotency_key=idempotency_key,
        )
        response = await service.orchestrate(
            request=request, tenant_id=tenant_id, context_input=request_context
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return ApiResponse(
            data=response,
            metadata={
                "requestId": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "processingTime": f"{processing_time:.3f}s",
                "tenantId": tenant_id,
            },
        )

    except BaseServiceException as e:
        logger.error(
            "Orchestration failed with service exception",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error_code": e.error_code,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=e.http_status_code, detail=e.to_dict())

    except Exception as e:
        logger.error(
            "Orchestration failed with unexpected error",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch", response_model=ApiResponse[BatchOrchestrationResponse])
async def orchestrate_batch(
    request: BatchOrchestrationRequest,
    background_tasks: BackgroundTasks,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    api_key: Optional[str] = Depends(validate_request_auth),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Batch orchestration endpoint for processing multiple requests.

    Supports:
    - Async processing for large batches
    - Progress tracking and status updates
    - Partial result retrieval
    - Configurable batch size limits
    """
    try:
        logger.info(
            "Processing batch orchestration request",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "batch_size": len(request.requests),
                "async_processing": request.async_processing,
            },
        )

        if request.async_processing:
            # Submit as background job
            job_id = await service.submit_batch_job(
                requests=request.requests,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                api_key=api_key,
            )

            return ApiResponse(
                data=BatchOrchestrationResponse(
                    job_id=job_id,
                    status="submitted",
                    total_requests=len(request.requests),
                    completed_requests=0,
                    results=[],
                ),
                metadata={
                    "requestId": correlation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tenantId": tenant_id,
                    "processing": "async",
                },
            )
        else:
            # Process synchronously
            response = await service.orchestrate_batch(
                requests=request.requests,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
                api_key=api_key,
            )

            return ApiResponse(
                data=response,
                metadata={
                    "requestId": correlation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tenantId": tenant_id,
                    "processing": "sync",
                },
            )

    except Exception as e:
        logger.error(
            "Batch orchestration failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/status")
async def get_job_status(
    job_id: str,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Get status of an async batch job."""
    try:
        status = await service.get_job_status(job_id, tenant_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")

        return ApiResponse(
            data=status,
            metadata={
                "requestId": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "tenantId": tenant_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get job status",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "job_id": job_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))
