"""Health and monitoring API endpoints following SRP.

This module handles ONLY health and monitoring endpoints:
- Service health checks
- Service status
- Metrics endpoints
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from shared.interfaces.base import HealthResponse, ApiResponse
from shared.utils.correlation import get_correlation_id

from ..service import OrchestrationService
from ..config import settings
from .dependencies import (
    get_orchestration_service,
    get_tenant_id,
    get_correlation_id_header,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for service monitoring."""
    try:
        service = await get_orchestration_service()
        status = await service.get_service_status()

        return HealthResponse(
            status="healthy" if status["status"] == "running" else "unhealthy",
            version=settings.service_version,
            uptime_seconds=status.get("uptime_seconds"),
            metadata={
                "service_name": settings.service_name,
                "environment": settings.environment,
                "components": status.get("components", {}),
            },
        )
    except Exception as e:
        logger.error("Health check failed: %s", str(e))
        return HealthResponse(
            status="unhealthy",
            version=settings.service_version,
            metadata={"error": str(e)},
        )


@router.get("/status", response_model=ApiResponse[Dict[str, Any]])
async def get_service_status(
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Get comprehensive service status and metrics."""
    try:
        status = await service.get_service_status()

        return ApiResponse(
            data=status,
            metadata={
                "requestId": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "tenantId": tenant_id,
            },
        )

    except Exception as e:
        logger.error(
            "Get service status failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_metrics(
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Get Prometheus metrics for monitoring."""
    try:
        metrics = service.metrics_collector.get_metrics()
        return JSONResponse(
            content=metrics, headers={"Content-Type": "text/plain; charset=utf-8"}
        )

    except Exception as e:
        logger.error(
            "Get metrics failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readiness")
async def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        service = await get_orchestration_service()

        # Check if service is ready to handle requests
        status = await service.get_service_status()
        if status["status"] != "running":
            raise HTTPException(status_code=503, detail="Service not ready")

        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error("Readiness check failed: %s", str(e))
        raise HTTPException(status_code=503, detail="Service not ready")


@router.get("/liveness")
async def liveness_check():
    """Kubernetes liveness probe endpoint."""
    try:
        # Basic liveness check - service is alive if it can respond
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error("Liveness check failed: %s", str(e))
        raise HTTPException(status_code=503, detail="Service not alive")
