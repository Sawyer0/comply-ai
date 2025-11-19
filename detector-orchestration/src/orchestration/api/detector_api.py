"""Detector management API endpoints following SRP."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from shared.exceptions.base import BaseServiceException
from shared.interfaces.base import ApiResponse, PaginatedResponse
from shared.interfaces.orchestration import DetectorInfo, DetectorRegistration

from ..service import DetectorRegistrationConfig, OrchestrationService
from .dependencies import (
    get_correlation_id_header,
    get_orchestration_service,
    get_tenant_id,
    validate_request_auth,
)
from .utils import build_metadata, make_api_response

if TYPE_CHECKING:
    from ..core.router import ContentRouter
    from ..monitoring.health_monitor import HealthMonitor


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detectors", tags=["detectors"])


@dataclass
class DetectorRequestContext:
    """Common dependency bundle for detector endpoints."""

    tenant_id: str
    correlation_id: str
    service: OrchestrationService
    api_key: Optional[str]


async def build_detector_context(
    tenant_id: str = Depends(get_tenant_id),
    correlation_id: str = Depends(get_correlation_id_header),
    api_key: Optional[str] = Depends(validate_request_auth),
    service: OrchestrationService = Depends(get_orchestration_service),
) -> DetectorRequestContext:
    """Assemble the shared context required by detector endpoints."""

    return DetectorRequestContext(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        service=service,
        api_key=api_key,
    )


@router.post("/register", response_model=ApiResponse[Dict[str, str]])
async def register_detector(
    registration: DetectorRegistration,
    context: DetectorRequestContext = Depends(build_detector_context),
) -> ApiResponse[Dict[str, str]]:
    """Register a detector with service discovery, routing, and client registry."""

    logger.info(
        "Registering detector",
        extra={
            "correlation_id": context.correlation_id,
            "tenant_id": context.tenant_id,
            "detector_type": registration.detector_type,
            "endpoint": registration.endpoint_url,
        },
    )

    detector_config = DetectorRegistrationConfig(
        detector_id=registration.detector_id or registration.detector_type,
        endpoint=registration.endpoint_url,
        detector_type=registration.detector_type,
        tenant_id=context.tenant_id,
        timeout_ms=registration.timeout_ms or 5000,
        max_retries=registration.max_retries or 3,
        supported_content_types=registration.supported_content_types or ["text"],
    )

    try:
        was_registered = await context.service.register_detector(detector_config)
    except (BaseServiceException, ValueError) as exc:
        logger.error(
            "Detector registration failed",
            extra={
                "correlation_id": context.correlation_id,
                "tenant_id": context.tenant_id,
                "detector_type": registration.detector_type,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not was_registered:
        raise HTTPException(status_code=400, detail="Failed to register detector")

    return make_api_response(
        data={"message": f"Detector {registration.detector_type} registered successfully"},
        request_id=context.correlation_id,
        tenant_id=context.tenant_id,
    )


@router.delete("/{detector_id}", response_model=ApiResponse[Dict[str, str]])
async def unregister_detector(
    detector_id: str,
    context: DetectorRequestContext = Depends(build_detector_context),
) -> ApiResponse[Dict[str, str]]:
    """Remove a detector from the orchestration service."""

    try:
        was_removed = await context.service.unregister_detector(
            detector_id, tenant_id=context.tenant_id
        )
    except BaseServiceException as exc:
        logger.error(
            "Detector unregistration failed",
            extra={
                "correlation_id": context.correlation_id,
                "tenant_id": context.tenant_id,
                "detector_id": detector_id,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not was_removed:
        raise HTTPException(status_code=404, detail="Detector not found")

    return make_api_response(
        data={"message": f"Detector {detector_id} unregistered successfully"},
        request_id=context.correlation_id,
        tenant_id=context.tenant_id,
    )


def _record_to_detector_info(record: Any) -> DetectorInfo:
    """Map a DetectorRecord (or similar) to DetectorInfo."""

    config = dict(getattr(record, "configuration", {}) or {})
    return DetectorInfo(
        detector_id=record.detector_name,
        detector_type=record.detector_type,
        status=record.status,
        endpoint_url=record.endpoint_url,
        supported_content_types=list(record.capabilities or []),
        timeout_ms=int(config.get("timeout_ms", 5000)),
        max_retries=int(config.get("max_retries", 3)),
        metadata={
            "tenant_id": record.tenant_id,
            "health_status": record.health_status,
            "last_health_check": record.last_health_check,
            "capabilities": record.capabilities,
        },
    )


def _paginate(items: List[DetectorInfo], *, page: int, limit: int) -> List[DetectorInfo]:
    """Return the requested pagination slice."""

    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    return items[start_idx:end_idx]


@router.get("", response_model=PaginatedResponse[DetectorInfo])
async def list_detectors(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=1000, description="Items per page"),
    detector_type: Optional[str] = Query(None, description="Filter by detector type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    context: DetectorRequestContext = Depends(build_detector_context),
) -> PaginatedResponse[DetectorInfo]:
    """List detectors registered with the service."""

    # Fetch from persistent registry via service, then project to DetectorInfo
    try:
        records = await context.service.list_detectors(
            tenant_id=context.tenant_id,
            detector_type=detector_type,
            status=status,
        )
    except BaseServiceException as exc:  # pragma: no cover - defensive
        logger.error(
            "Detector listing failed",
            extra={
                "correlation_id": context.correlation_id,
                "tenant_id": context.tenant_id,
                "error": str(exc),
            },
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    detector_infos: List[DetectorInfo] = [
        _record_to_detector_info(record) for record in records
    ]

    paginated_items = _paginate(detector_infos, page=page, limit=limit)

    total_items = len(detector_infos)
    return PaginatedResponse(
        data=paginated_items,
        metadata=build_metadata(
            request_id=context.correlation_id,
            tenant_id=context.tenant_id,
        ),
        pagination={
            "page": page,
            "limit": limit,
            "total": total_items,
            "hasNext": (page * limit) < total_items,
            "hasPrev": page > 1,
        },
    )


class DetectorUpdate(BaseModel):
    """Fields allowed to be modified via PATCH /detectors/{detector_id}."""

    status: Optional[str] = Field(None, description="Detector status")
    endpoint_url: Optional[str] = Field(
        None, description="Detector endpoint URL (will update health check URL too)"
    )
    capabilities: Optional[List[str]] = Field(
        None, description="Updated detector capabilities (supported content types)"
    )


@router.get("/{detector_id}", response_model=ApiResponse[DetectorInfo])
async def get_detector(
    detector_id: str,
    context: DetectorRequestContext = Depends(build_detector_context),
) -> ApiResponse[DetectorInfo]:
    """Get a single detector from the persistent registry."""

    record = await context.service.get_detector(
        detector_id, tenant_id=context.tenant_id
    )
    if not record:
        raise HTTPException(status_code=404, detail="Detector not found")

    info = _record_to_detector_info(record)
    return make_api_response(
        data=info,
        request_id=context.correlation_id,
        tenant_id=context.tenant_id,
    )


@router.patch("/{detector_id}", response_model=ApiResponse[DetectorInfo])
async def update_detector(
    detector_id: str,
    update: DetectorUpdate,
    context: DetectorRequestContext = Depends(build_detector_context),
) -> ApiResponse[DetectorInfo]:
    """Patch mutable fields of a detector in the persistent registry."""

    fields: Dict[str, Any] = {}
    if update.status is not None:
        fields["status"] = update.status
    if update.endpoint_url is not None:
        fields["endpoint_url"] = update.endpoint_url
        fields["health_check_url"] = update.endpoint_url.rstrip("/") + "/health"
    if update.capabilities is not None:
        fields["capabilities"] = list(update.capabilities)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = await context.service.update_detector(
        detector_id,
        tenant_id=context.tenant_id,
        fields=fields,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Detector not found")

    info = _record_to_detector_info(updated)
    return make_api_response(
        data=info,
        request_id=context.correlation_id,
        tenant_id=context.tenant_id,
    )


@router.get("/{detector_id}/health", response_model=ApiResponse[Dict[str, Any]])
async def get_detector_health(
    detector_id: str,
    context: DetectorRequestContext = Depends(build_detector_context),
) -> ApiResponse[Dict[str, Any]]:
    """Report the health information for a specific detector."""

    monitor = getattr(context.service, "health_monitor", None)
    if not monitor:
        raise HTTPException(status_code=503, detail="Health monitoring not enabled")

    health_status = monitor.get_service_health(detector_id)
    if not health_status:
        raise HTTPException(status_code=404, detail="Detector not found")

    return make_api_response(
        data=health_status,
        request_id=context.correlation_id,
        tenant_id=context.tenant_id,
    )
