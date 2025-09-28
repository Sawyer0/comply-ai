"""Detector management API endpoints following SRP."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query

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
        timeout_ms=registration.timeout_ms or 5000,
        max_retries=registration.max_retries or 3,
        supported_content_types=registration.supported_content_types or ["text"],
        auth_headers=registration.auth_headers,
        analyze_path=registration.analyze_path,
        response_parser=registration.response_parser,
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
        was_removed = await context.service.unregister_detector(detector_id)
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


def _build_detector_info(
    detector_name: str,
    *,
    router_component: "ContentRouter",
    monitor: Optional["HealthMonitor"],
    type_filter: Optional[str],
    status_filter: Optional[str],
) -> Optional[DetectorInfo]:
    """Create DetectorInfo for a registered detector if it matches filters."""

    config = router_component.get_detector_config(detector_name)
    if not config:
        return None

    if type_filter and config.detector_type != type_filter:
        return None

    is_healthy = bool(monitor and monitor.is_service_healthy(detector_name))
    detector_status = "active" if is_healthy else "inactive"

    if status_filter and detector_status != status_filter:
        return None

    health_metadata = (
        monitor.get_service_health(detector_name) if monitor else None
    )

    return DetectorInfo(
        detector_id=detector_name,
        detector_type=config.detector_type,
        status=detector_status,
        endpoint_url=config.endpoint,
        supported_content_types=config.supported_content_types,
        timeout_ms=config.timeout_ms,
        max_retries=config.max_retries,
        metadata={
            "enabled": config.enabled,
            "last_health_check": health_metadata,
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

    router_component = getattr(context.service, "content_router", None)
    if not router_component:
        raise HTTPException(status_code=503, detail="Routing not configured")

    monitor = getattr(context.service, "health_monitor", None)

    detector_infos = [
        info
        for info in (
            _build_detector_info(
                detector_name,
                router_component=router_component,
                monitor=monitor,
                type_filter=detector_type,
                status_filter=status,
            )
            for detector_name in router_component.get_available_detectors()
        )
        if info is not None
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
