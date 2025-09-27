"""Detector management API endpoints following SRP.

This module handles ONLY detector management:
- Detector registration
- Detector unregistration
- Detector listing with pagination
- Detector health status
"""

from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
import logging

from shared.interfaces.orchestration import (
    DetectorInfo,
    DetectorRegistration,
)
from shared.interfaces.base import ApiResponse, PaginatedResponse
from shared.utils.correlation import get_correlation_id

from ..service import OrchestrationService, DetectorRegistrationConfig
from .dependencies import (
    get_orchestration_service,
    get_tenant_id,
    get_correlation_id_header,
    validate_request_auth,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/detectors", tags=["detectors"])


@router.post("/register", response_model=ApiResponse[Dict[str, str]])
async def register_detector(
    registration: DetectorRegistration,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    api_key: Optional[str] = Depends(validate_request_auth),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Register a new detector with the orchestration service."""
    try:
        logger.info(
            "Registering detector",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_type": registration.detector_type,
                "endpoint": registration.endpoint_url,
            },
        )

        detector_config = DetectorRegistrationConfig(
            detector_id=registration.detector_id or registration.detector_type,
            endpoint=registration.endpoint_url,
            detector_type=registration.detector_type,
            timeout_ms=registration.timeout_ms or 5000,
            max_retries=registration.max_retries or 3,
            supported_content_types=registration.supported_content_types or ["text"],
            auth_headers=registration.auth_headers,
            analyze_path=registration.analyze_path,
            response_parser=registration.response_parser,
        )
        success = await service.register_detector(detector_config)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to register detector")

        return ApiResponse(
            data={
                "message": f"Detector {registration.detector_type} registered successfully"
            },
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
            "Detector registration failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_type": registration.detector_type,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{detector_id}", response_model=ApiResponse[Dict[str, str]])
async def unregister_detector(
    detector_id: str,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    api_key: Optional[str] = Depends(validate_request_auth),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Unregister a detector from the orchestration service."""
    try:
        success = await service.unregister_detector(detector_id)

        if not success:
            raise HTTPException(status_code=404, detail="Detector not found")

        return ApiResponse(
            data={"message": f"Detector {detector_id} unregistered successfully"},
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
            "Detector unregistration failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_id": detector_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=PaginatedResponse[DetectorInfo])
async def list_detectors(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=1000, description="Items per page"),
    detector_type: Optional[str] = Query(None, description="Filter by detector type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """List available detectors with pagination and filtering."""
    try:
        detector_names = service.content_router.get_available_detectors()

        # Apply filters and build detector info
        detector_infos = []
        for detector_name in detector_names:
            config = service.content_router.get_detector_config(detector_name)
            if not config:
                continue

            # Apply detector_type filter
            if detector_type and config.detector_type != detector_type:
                continue

            # Get health status
            is_healthy = service.health_monitor.is_service_healthy(detector_name)
            detector_status = "active" if is_healthy else "inactive"

            # Apply status filter
            if status and detector_status != status:
                continue

            detector_infos.append(
                DetectorInfo(
                    detector_id=detector_name,
                    detector_type=config.detector_type,
                    status=detector_status,
                    endpoint_url=config.endpoint,
                    supported_content_types=config.supported_content_types,
                    timeout_ms=config.timeout_ms,
                    max_retries=config.max_retries,
                    metadata={
                        "enabled": config.enabled,
                        "last_health_check": service.health_monitor.get_service_health(
                            detector_name
                        ),
                    },
                )
            )

        # Apply pagination
        total = len(detector_infos)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_items = detector_infos[start_idx:end_idx]

        return PaginatedResponse(
            data=paginated_items,
            metadata={
                "requestId": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "tenantId": tenant_id,
            },
            pagination={
                "page": page,
                "limit": limit,
                "total": total,
                "hasNext": end_idx < total,
                "hasPrev": page > 1,
            },
        )

    except Exception as e:
        logger.error(
            "List detectors failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{detector_id}/health")
async def get_detector_health(
    detector_id: str,
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    service: OrchestrationService = Depends(get_orchestration_service),
):
    """Get health status for a specific detector."""
    try:
        health_status = service.health_monitor.get_service_health(detector_id)
        if not health_status:
            raise HTTPException(status_code=404, detail="Detector not found")

        return ApiResponse(
            data=health_status,
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
            "Get detector health failed",
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "detector_id": detector_id,
                "error": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))
