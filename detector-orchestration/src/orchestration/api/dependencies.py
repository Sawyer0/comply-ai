"""Shared API dependencies following SRP.

This module provides ONLY dependency injection functions:
- Service instance retrieval
- Authentication and authorization
- Request context extraction
"""

from typing import Optional
from fastapi import HTTPException, Header, Depends
import logging

from shared.utils.correlation import get_correlation_id, set_correlation_id
from shared.validation.auth import validate_api_key, get_tenant_from_api_key

from ..service import OrchestrationService
from ..app_state import service_container
from ..config import settings

logger = logging.getLogger(__name__)


async def get_orchestration_service() -> OrchestrationService:
    """Get the orchestration service instance.

    This dependency provides access to the main orchestration service.
    In a production setup, this would use proper dependency injection.
    """
    service = service_container.get_orchestration_service()

    if not service:
        raise HTTPException(status_code=503, detail="Service not available")

    return service


async def get_tenant_id(
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """Extract tenant ID from headers or API key.

    Priority:
    1. X-Tenant-ID header
    2. Tenant ID from API key
    3. Default tenant ID (if configured)
    4. Raise error if tenant ID is required
    """
    # First try explicit tenant ID header
    if x_tenant_id:
        return x_tenant_id

    # Try to extract from API key
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]  # Remove "Bearer " prefix
        try:
            tenant_id = await get_tenant_from_api_key(api_key)
            if tenant_id:
                return tenant_id
        except Exception as e:
            logger.warning(
                "Failed to extract tenant ID from API key", extra={"error": str(e)}
            )

    # Use default tenant ID if configured
    if not settings.require_tenant_id and settings.default_tenant_id:
        return settings.default_tenant_id

    # Raise error if tenant ID is required
    if settings.require_tenant_id:
        raise HTTPException(
            status_code=400,
            detail="Tenant ID required in X-Tenant-ID header or API key",
        )

    return "default"


async def get_correlation_id_header(
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID")
) -> str:
    """Extract or generate correlation ID for request tracing.

    If a correlation ID is provided in the header, it will be used.
    Otherwise, a new UUID will be generated.
    """
    if x_correlation_id:
        set_correlation_id(x_correlation_id)
        return x_correlation_id

    return get_correlation_id()


async def get_idempotency_key(
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
) -> Optional[str]:
    """Extract idempotency key from headers."""
    return idempotency_key


async def validate_request_auth(
    authorization: Optional[str] = Header(None, alias="Authorization")
) -> Optional[str]:
    """Validate API key if provided.

    This dependency validates the API key format and authenticity.
    Returns the API key if valid, None if not provided, or raises HTTPException if invalid.
    """
    if not authorization:
        return None

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use 'Bearer <api_key>'",
        )

    api_key = authorization[7:]  # Remove "Bearer " prefix

    try:
        is_valid = await validate_api_key(api_key)
        if not is_valid:
            raise HTTPException(status_code=401, detail="Invalid API key")

        return api_key

    except Exception as e:
        logger.error("API key validation failed", extra={"error": str(e)})
        raise HTTPException(status_code=401, detail="API key validation failed")


async def require_api_key(
    api_key: Optional[str] = Depends(validate_request_auth),
) -> str:
    """Require a valid API key for the request.

    This dependency ensures that a valid API key is provided.
    Use this for endpoints that require authentication.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide in Authorization header as 'Bearer <api_key>'",
        )

    return api_key
