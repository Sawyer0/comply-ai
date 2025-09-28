"""Health and monitoring API endpoints following SRP."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from shared.exceptions.base import BaseServiceException
from shared.interfaces.base import ApiResponse, HealthResponse

from ..config import settings
from ..service import OrchestrationService
from .dependencies import (
    get_correlation_id_header,
    get_orchestration_service,
    get_tenant_id,
)
from .utils import make_api_response

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: OrchestrationService = Depends(get_orchestration_service),
) -> HealthResponse:
    """Report basic service health information."""

    try:
        status = await service.get_service_status()
    except BaseServiceException as exc:
        logger.error("Health check failed", extra={"error": str(exc)})
        return HealthResponse(
            status="unhealthy",
            version=settings.service_version,
            metadata={"error": str(exc)},
        )

    return HealthResponse(
        status="healthy" if status.get("status") == "running" else "unhealthy",
        version=settings.service_version,
        uptime_seconds=status.get("uptime_seconds"),
        metadata={
            "service_name": settings.service_name,
            "environment": settings.environment,
            "components": status.get("components", {}),
        },
    )


@router.get("/status", response_model=ApiResponse[Dict[str, Any]])
async def get_service_status(
    context_tenant: str = Depends(get_tenant_id),
    context_correlation: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
) -> ApiResponse[Dict[str, Any]]:
    """Return internal status and component metadata."""

    try:
        status = await service.get_service_status()
    except BaseServiceException as exc:
        logger.error(
            "Get service status failed",
            extra={
                "correlation_id": context_correlation,
                "tenant_id": context_tenant,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return make_api_response(
        data=status,
        request_id=context_correlation,
        tenant_id=context_tenant,
    )


@router.get("/metrics")
async def get_metrics(
    context_tenant: str = Depends(get_tenant_id),
    context_correlation: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
) -> Response:
    """Expose Prometheus metrics."""

    if not service.metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector unavailable")

    try:
        metrics_payload = service.metrics_collector.get_metrics()
    except BaseServiceException as exc:
        logger.error(
            "Get metrics failed",
            extra={
                "correlation_id": context_correlation,
                "tenant_id": context_tenant,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(content=metrics_payload, media_type="text/plain; charset=utf-8")


@router.get("/readiness")
async def readiness_check(
    service: OrchestrationService = Depends(get_orchestration_service),
) -> Dict[str, str]:
    """Kubernetes readiness probe endpoint."""

    status = await service.get_service_status()
    if status.get("status") != "running":
        raise HTTPException(status_code=503, detail="Service not ready")

    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}


@router.get("/liveness")
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint."""

    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}
