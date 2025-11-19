"""
API endpoints for the Mapper Service.

Single responsibility: Handle ALL HTTP endpoints for the mapper service.

This includes:
- Core mapping operations
- API key management
- Health checks
- Authentication and authorization
- Rate limiting
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import shared components for consistency
from ..shared_integration import (
    get_shared_logger,
    set_correlation_id,
    get_correlation_id,
    track_request_metrics,
    ValidationError,
    BaseServiceException,
    ServiceUnavailableError,
)

from ..core.mapper import CoreMapper
from ..schemas.models import (
    MappingRequest as LegacyMappingRequest,
    BatchMappingRequest,
    BatchMappingResponse,
)
from ..shared_lib.interfaces.mapper import (
    MappingRequest as CanonicalMappingRequest,
)
from ..security.api_key_manager import APIKeyInfo, APIKeyCreateRequest
from ..security.authorization import Permission
from ..main import get_rate_limiting_service
from .dependencies import (
    get_mapper,
    authenticate_request,
    require_permission,
    get_database_health,
    check_rate_limit,
    check_rate_limit_optional,
    get_api_key_manager_dependency as get_api_key_manager,
)

logger = get_shared_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["mapper"])


# =============================================================================
# ROOT ENDPOINT
# =============================================================================


@router.get("/", response_model="APIResponse")
async def get_service_info() -> "APIResponse":
    """
    Get service information.

    Public endpoint - no authentication required.
    """
    return APIResponse(
        data={
            "service": "Mapper Service",
            "version": "1.0.0",
            "description": (
                "Core mapping functionality, model serving, "
                "and response generation"
            ),
            "endpoints": {
                "health": "/api/v1/health",
                "mapping": "/api/v1/map",
                "batch_mapping": "/api/v1/map/batch",
                "detectors": "/api/v1/detectors",
                "frameworks": "/api/v1/frameworks",
                "api_keys": "/api/v1/api-keys",
            },
        },
        metadata={"timestamp": datetime.utcnow().isoformat()},
    )


# =============================================================================
# RESPONSE MODELS FOLLOWING API GUIDELINES
# =============================================================================


class APIResponse(BaseModel):
    """Base API response model following API guidelines."""

    data: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for API response serialization."""

        json_encoders = {datetime: lambda v: v.isoformat()}


class ErrorResponse(BaseModel):
    """Error response model following API guidelines."""

    error: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PaginatedResponse(BaseModel):
    """Paginated response model following API guidelines."""

    data: List[Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    pagination: Dict[str, Any]


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================


class APIKeyCreateResponse(BaseModel):
    """API key creation response."""

    api_key: str = Field(..., description="Generated API key")
    key_id: str = Field(..., description="Key identifier")
    tenant_id: str = Field(..., description="Tenant ID")
    name: str = Field(..., description="Key name")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    permissions: List[str] = Field(..., description="Granted permissions")
    scopes: List[str] = Field(..., description="Granted scopes")


class APIKeyListResponse(BaseModel):
    """API key list item response."""

    key_id: str
    name: str
    description: Optional[str]
    permissions: List[str]
    scopes: List[str]
    is_active: bool
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    created_at: datetime


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Overall health status")
    timestamp: str = Field(..., description="Check timestamp")
    services: Dict[str, Any] = Field(..., description="Individual service status")
    version: str = Field(..., description="Service version")


class UsageStatsResponse(BaseModel):
    """Usage statistics response model."""

    period_days: int
    total_requests: int
    avg_response_time_ms: float
    total_tokens: int
    total_cost_cents: float
    active_days: int
    top_endpoints: List[Dict[str, Any]]


class TenantCreateRequest(BaseModel):
    """Request model for creating a new tenant."""

    tenant_name: str = Field(..., description="Human-readable tenant name")
    tier: str = Field(..., description="Tenant tier (free, basic, premium, enterprise)")
    model_preferences: Dict[str, Any] = Field(
        default_factory=dict, description="Model preferences"
    )
    feature_flags: Dict[str, bool] = Field(
        default_factory=dict, description="Feature flags"
    )


class TenantResponse(BaseModel):
    """Response model for tenant information."""

    tenant_id: str
    tenant_name: str
    tier: str
    quotas: Dict[str, Any]
    model_preferences: Dict[str, Any]
    feature_flags: Dict[str, bool]
    created_at: datetime
    updated_at: datetime


class CostTrackingRequest(BaseModel):
    """Request model for cost tracking."""

    operation_type: str
    model_name: str
    input_tokens: int
    output_tokens: int
    inference_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


# =============================================================================
# CORE MAPPING ENDPOINTS
# =============================================================================


@router.post("/map", response_model=APIResponse)
@track_request_metrics("map_detector_output")
async def map_detector_output(
    mapping_request: CanonicalMappingRequest,
    request: Request,
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.MAP_CANONICAL)),
    mapper: CoreMapper = Depends(get_mapper),
    _rate_limit_check: APIKeyInfo = Depends(check_rate_limit),
) -> APIResponse:
    """
    Map detector output to canonical taxonomy.

    Requires: map:canonical permission
    """
    # Set correlation ID from request headers or generate new one
    correlation_id = request.headers.get("X-Correlation-ID")
    if correlation_id:
        set_correlation_id(correlation_id)
    else:
        correlation_id = get_correlation_id()

    start_time = datetime.utcnow()

    try:
        response = await mapper.map_canonical(mapping_request)

        # Track usage
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return APIResponse(
            data=response.dict(),
            metadata={
                "request_id": correlation_id,
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": processing_time,
                "tenant_id": api_key_info.tenant_id,
            },
        )

    except ValidationError as e:
        logger.error(
            "Validation error in canonical mapping request",
            error=str(e),
            tenant_id=api_key_info.tenant_id,
            detector="canonical_mapping",
            correlation_id=correlation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}",
        ) from e
    except ServiceUnavailableError as e:
        logger.error(
            "Service unavailable during canonical mapping",
            error=str(e),
            tenant_id=api_key_info.tenant_id,
            detector=getattr(mapping_request, "detector", "unknown"),
            correlation_id=correlation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service temporarily unavailable: {str(e)}",
        ) from e
    except BaseServiceException as e:
        logger.error(
            "Service error in canonical mapping request",
            error=str(e),
            tenant_id=api_key_info.tenant_id,
            detector=getattr(mapping_request, "detector", "unknown"),
            correlation_id=correlation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Service error: {str(e)}",
        ) from e
    except Exception as e:
        logger.error(
            "Unexpected error in canonical mapping request",
            error=str(e),
            error_type=type(e).__name__,
            tenant_id=api_key_info.tenant_id,
            detector=getattr(mapping_request, "detector", "unknown"),
            correlation_id=correlation_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error occurred: {str(e)}",
        ) from e


@router.post("/map/batch", response_model=APIResponse)
async def batch_map_detector_outputs(
    batch_request: BatchMappingRequest,
    mapper=Depends(get_mapper),  # type: ignore[assignment]
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.MAP_BATCH)),
    _rate_limit_check: APIKeyInfo = Depends(check_rate_limit),
) -> APIResponse:
    """
    Map multiple detector outputs in batch.

    Requires: map:batch permission
    """
    start_time = datetime.utcnow()

    try:
        responses = await mapper.batch_map(batch_request.requests)

        success_count = sum(1 for r in responses if r.confidence > 0)
        error_count = len(responses) - success_count
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        batch_response = BatchMappingResponse(
            responses=responses,
            total_processed=len(responses),
            success_count=success_count,
            error_count=error_count,
        )

        return APIResponse(
            data=batch_response.dict(),
            metadata={
                "request_id": getattr(batch_request, "correlation_id", None),
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": processing_time,
                "tenant_id": api_key_info.tenant_id,
                "batch_size": len(batch_request.requests),
            },
        )

    except Exception as e:
        logger.error(
            "Batch mapping failed",
            error=str(e),
            tenant_id=api_key_info.tenant_id,
            batch_size=len(batch_request.requests),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch mapping failed: {str(e)}",
        ) from e


@router.get("/detectors", response_model=APIResponse)
async def get_supported_detectors(
    api_key_info: APIKeyInfo = Depends(check_rate_limit_optional),
    mapper=Depends(get_mapper),
) -> APIResponse:
    """
    Get list of supported detector types.

    Public endpoint - no authentication required.
    """
    try:
        detectors = await mapper.get_supported_detectors()

        return APIResponse(
            data=detectors,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(detectors),
                "authenticated": api_key_info is not None,
                "tenant_id": api_key_info.tenant_id if api_key_info else None,
            },
        )

    except Exception as e:
        logger.error("Failed to get supported detectors", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get detectors: {str(e)}",
        ) from e


@router.get("/frameworks", response_model=APIResponse)
async def get_supported_frameworks(
    api_key_info: APIKeyInfo = Depends(check_rate_limit_optional),
    mapper=Depends(get_mapper),
) -> APIResponse:
    """
    Get list of supported compliance frameworks.

    Public endpoint - no authentication required.
    """
    try:
        frameworks = await mapper.get_supported_frameworks()

        return APIResponse(
            data=frameworks,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "count": len(frameworks),
                "authenticated": api_key_info is not None,
                "tenant_id": api_key_info.tenant_id if api_key_info else None,
            },
        )

    except Exception as e:
        logger.error("Failed to get supported frameworks", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get frameworks: {str(e)}",
        ) from e


# =============================================================================
# HEALTH AND STATUS ENDPOINTS
# =============================================================================


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> JSONResponse:
    """
    Comprehensive health check endpoint.

    Public endpoint - no authentication required.
    """
    try:
        # Get database health
        db_health = await get_database_health()

        # Determine overall status
        overall_status = "healthy"
        if db_health.status != "healthy":
            overall_status = (
                "degraded" if db_health.status == "degraded" else "unhealthy"
            )

        services = {
            "database": {
                "status": db_health.status,
                "response_time_ms": db_health.response_time_ms,
                "details": db_health.details,
            }
        }

        health_response = HealthCheckResponse(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            services=services,
            version="1.0.0",
        )

        status_code = (
            status.HTTP_200_OK
            if overall_status == "healthy"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )

        return JSONResponse(content=health_response.dict(), status_code=status_code)

    except (RuntimeError, OSError, ConnectionError) as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "version": "1.0.0",
            },
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


# =============================================================================
# API KEY MANAGEMENT ENDPOINTS
# =============================================================================


@router.post("/api-keys", response_model=APIResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.ADMIN_KEYS)),
) -> APIResponse:
    """
    Create a new API key.

    Requires: admin:keys permission
    """
    try:
        api_key_manager = get_api_key_manager()
        if not api_key_manager:
            raise HTTPException(status_code=500, detail="API key manager not available")

        result = await api_key_manager.create_api_key(
            request, created_by=api_key_info.key_id
        )

        response_data = APIKeyCreateResponse(**result)

        return APIResponse(
            data=response_data.dict(),
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "created_by": api_key_info.key_id,
                "tenant_id": api_key_info.tenant_id,
            },
        )

    except Exception as e:
        logger.error(
            "Failed to create API key", error=str(e), tenant_id=request.tenant_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        ) from e


@router.get("/api-keys", response_model=PaginatedResponse)
async def list_api_keys(
    include_inactive: bool = Query(False, description="Include inactive keys"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=1000, description="Items per page"),
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.ADMIN_KEYS)),
) -> PaginatedResponse:
    """
    List API keys for the authenticated tenant.

    Requires: admin:keys permission
    """
    try:
        api_key_manager = get_api_key_manager()
        if not api_key_manager:
            raise HTTPException(status_code=500, detail="API key manager not available")

        keys = await api_key_manager.list_api_keys(
            api_key_info.tenant_id, include_inactive=include_inactive
        )

        # Apply pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_keys = keys[start_idx:end_idx]

        response_data = [
            APIKeyListResponse(
                key_id=key.key_id,
                name=key.name,
                description=key.description,
                permissions=key.permissions,
                scopes=key.scopes,
                is_active=key.is_active,
                expires_at=key.expires_at,
                last_used_at=key.last_used_at,
                usage_count=key.usage_count,
                created_at=key.created_at,
            ).dict()
            for key in paginated_keys
        ]

        return PaginatedResponse(
            data=response_data,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": api_key_info.tenant_id,
            },
            pagination={
                "page": page,
                "limit": limit,
                "total": len(keys),
                "has_next": end_idx < len(keys),
                "has_prev": page > 1,
            },
        )

    except Exception as e:
        logger.error(
            "Failed to list API keys", error=str(e), tenant_id=api_key_info.tenant_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list API keys: {str(e)}",
        ) from e


@router.delete("/api-keys/{key_id}", response_model=APIResponse)
async def revoke_api_key(
    key_id: str = Path(..., description="API key ID to revoke"),
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.ADMIN_KEYS)),
) -> APIResponse:
    """
    Revoke an API key.

    Requires: admin:keys permission
    """
    try:
        api_key_manager = get_api_key_manager()
        if not api_key_manager:
            raise HTTPException(status_code=500, detail="API key manager not available")

        success = await api_key_manager.revoke_api_key(key_id, api_key_info.tenant_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
            )

        return APIResponse(
            data={"key_id": key_id, "status": "revoked"},
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "revoked_by": api_key_info.key_id,
                "tenant_id": api_key_info.tenant_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to revoke API key", error=str(e), key_id=key_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke API key: {str(e)}",
        ) from e


@router.get("/api-keys/{key_id}/usage", response_model=APIResponse)
async def get_api_key_usage(
    key_id: str = Path(..., description="API key ID"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.ADMIN_KEYS)),
) -> APIResponse:
    """
    Get usage statistics for an API key.

    Requires: admin:keys permission
    """
    try:
        api_key_manager = get_api_key_manager()
        if not api_key_manager:
            raise HTTPException(status_code=500, detail="API key manager not available")

        stats = await api_key_manager.get_usage_statistics(
            api_key_info.tenant_id, key_id=key_id, days=days
        )

        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found or no usage data",
            )

        response_data = UsageStatsResponse(**stats)

        return APIResponse(
            data=response_data.dict(),
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "key_id": key_id,
                "tenant_id": api_key_info.tenant_id,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get API key usage", error=str(e), key_id=key_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage statistics: {str(e)}",
        ) from e


# =============================================================================
# RATE LIMITING ENDPOINTS
# =============================================================================


@router.get("/rate-limit/status", response_model=APIResponse)
async def get_rate_limit_status(
    request: Request, api_key_info: APIKeyInfo = Depends(authenticate_request)
) -> APIResponse:
    """
    Get current rate limit status for the authenticated user.

    Requires: Authentication
    """
    try:
        rate_limiting_service = get_rate_limiting_service()
        if not rate_limiting_service:
            raise HTTPException(
                status_code=500, detail="Rate limiting service not available"
            )

        # Extract client IP using public method
        ip_address = None
        if hasattr(request, "client") and request.client:
            ip_address = request.client.host

        # Get rate limit status
        rate_limit_status = await rate_limiting_service.get_rate_limit_status(
            api_key_info=api_key_info, ip_address=ip_address
        )

        return APIResponse(
            data=rate_limit_status,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": api_key_info.tenant_id,
                "api_key_id": api_key_info.key_id,
            },
        )

    except Exception as e:
        logger.error(
            "Failed to get rate limit status",
            error=str(e),
            tenant_id=api_key_info.tenant_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get rate limit status: {str(e)}",
        ) from e


# =============================================================================
# USAGE ANALYTICS ENDPOINTS
# =============================================================================


@router.get("/usage", response_model=APIResponse)
async def get_tenant_usage(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    api_key_info: APIKeyInfo = Depends(require_permission(Permission.COST_VIEW)),
) -> APIResponse:
    """
    Get usage analytics for the authenticated tenant.

    Requires: cost:view permission
    """
    try:
        api_key_manager = get_api_key_manager()
        if not api_key_manager:
            raise HTTPException(status_code=500, detail="API key manager not available")

        stats = await api_key_manager.get_usage_statistics(
            api_key_info.tenant_id, days=days
        )

        response_data = (
            UsageStatsResponse(**stats)
            if stats
            else UsageStatsResponse(
                period_days=days,
                total_requests=0,
                avg_response_time_ms=0.0,
                total_tokens=0,
                total_cost_cents=0.0,
                active_days=0,
                top_endpoints=[],
            )
        )

        return APIResponse(
            data=response_data.dict(),
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "tenant_id": api_key_info.tenant_id,
            },
        )

    except Exception as e:
        logger.error(
            "Failed to get usage analytics",
            error=str(e),
            tenant_id=api_key_info.tenant_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get usage analytics: {str(e)}",
        ) from e
