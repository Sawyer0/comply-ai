"""
FastAPI dependencies for the Mapper Service.

Single responsibility: Dependency injection for API endpoints.

Following SRP, each dependency function has a single responsibility:
- get_settings(): Provide application settings
- get_database_manager(): Provide database manager
- get_mapper(): Provide core mapper instance
- Authentication and authorization dependencies
"""

from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, Request
import structlog

from ..main import (
    get_database_manager,
    get_auth_service,
    get_authz_service,
    get_health_checker,
    get_rate_limiting_service,
    get_api_key_manager,
)
from ..core.mapper import CoreMapper
from ..config.settings import MapperSettings
from ..security.api_key_manager import APIKeyInfo
from ..security.authorization import Permission, Scope

logger = structlog.get_logger(__name__)


# Module-level instances (avoiding global statement)
class _DependencyState:
    """Container for dependency instances to avoid global statements."""

    def __init__(self):
        self.mapper_instance: Optional[CoreMapper] = None
        self.settings_instance: Optional[MapperSettings] = None
        self.tenant_manager = None
        self.cost_tracker = None
        self.billing_manager = None
        self.resource_manager = None


_state = _DependencyState()


@lru_cache()
def get_settings() -> MapperSettings:
    """
    Get mapper settings instance.

    Single responsibility: Settings dependency injection.
    """
    if _state.settings_instance is None:
        _state.settings_instance = MapperSettings()
    return _state.settings_instance


# Authentication and Authorization Dependencies
async def authenticate_request(request: Request) -> APIKeyInfo:
    """
    Authenticate incoming request using API key.

    Single responsibility: Request authentication.
    """
    auth_service = get_auth_service()
    if not auth_service:
        raise HTTPException(
            status_code=500, detail="Authentication service not available"
        )

    result = await auth_service.authenticate_request(request)

    if not result.authenticated:
        raise HTTPException(
            status_code=401, detail=result.error or "Authentication required"
        )

    return result.api_key_info


async def optional_authenticate_request(request: Request) -> Optional[APIKeyInfo]:
    """
    Optionally authenticate incoming request.

    Single responsibility: Optional request authentication.
    """
    try:
        return await authenticate_request(request)
    except HTTPException:
        return None


def require_permission(permission: Permission):
    """
    Create dependency that requires specific permission.

    Single responsibility: Permission-based authorization.
    """

    def permission_dependency(
        api_key_info: APIKeyInfo = Depends(authenticate_request),
    ) -> APIKeyInfo:
        authz_service = get_authz_service()
        if not authz_service:
            raise HTTPException(
                status_code=500, detail="Authorization service not available"
            )

        result = authz_service.check_permission(api_key_info, permission)

        if not result.authorized:
            raise HTTPException(status_code=403, detail=result.reason)

        return api_key_info

    return permission_dependency


def require_scopes(scopes: list[Scope]):
    """
    Create dependency that requires specific scopes.

    Single responsibility: Scope-based authorization.
    """

    def scope_dependency(
        api_key_info: APIKeyInfo = Depends(authenticate_request),
    ) -> APIKeyInfo:
        authz_service = get_authz_service()
        if not authz_service:
            raise HTTPException(
                status_code=500, detail="Authorization service not available"
            )

        result = authz_service.check_scope(api_key_info, scopes)

        if not result.authorized:
            raise HTTPException(status_code=403, detail=result.reason)

        return api_key_info

    return scope_dependency


def require_tenant_access(resource_tenant_id: str):
    """
    Create dependency that requires tenant access.

    Single responsibility: Tenant-based authorization.
    """

    def tenant_dependency(
        api_key_info: APIKeyInfo = Depends(authenticate_request),
    ) -> APIKeyInfo:
        authz_service = get_authz_service()
        if not authz_service:
            raise HTTPException(
                status_code=500, detail="Authorization service not available"
            )

        result = authz_service.check_tenant_access(api_key_info, resource_tenant_id)

        if not result.authorized:
            raise HTTPException(status_code=403, detail=result.reason)

        return api_key_info

    return tenant_dependency


async def get_tenant_manager():
    """
    Get tenant manager instance.

    Single responsibility: Tenant management dependency injection.
    """
    if _state.tenant_manager is None:
        database_manager = get_database_manager()
        if not database_manager:
            raise HTTPException(
                status_code=500, detail="Database manager not available"
            )

        # Get the underlying asyncpg pool from database manager
        if hasattr(database_manager, "_pool") and database_manager._pool:
            from ..tenancy import MapperTenantManager

            _state.tenant_manager = MapperTenantManager(database_manager._pool)
        else:
            raise HTTPException(status_code=500, detail="Database pool not available")

    return _state.tenant_manager


async def get_cost_tracker():
    """
    Get cost tracker instance.

    Single responsibility: Cost tracking dependency injection.
    """
    if _state.cost_tracker is None:
        database_manager = get_database_manager()
        if not database_manager:
            raise HTTPException(
                status_code=500, detail="Database manager not available"
            )

        # Get the underlying asyncpg pool from database manager
        if hasattr(database_manager, "_pool") and database_manager._pool:
            from ..tenancy import MapperCostTracker

            _state.cost_tracker = MapperCostTracker(database_manager._pool)
        else:
            raise HTTPException(status_code=500, detail="Database pool not available")

    return _state.cost_tracker


async def get_billing_manager():
    """
    Get billing manager instance.

    Single responsibility: Billing management dependency injection.
    """
    if _state.billing_manager is None:
        database_manager = get_database_manager()
        if not database_manager:
            raise HTTPException(
                status_code=500, detail="Database manager not available"
            )

        # Get the underlying asyncpg pool from database manager
        if hasattr(database_manager, "_pool") and database_manager._pool:
            from ..tenancy import BillingManager

            tenant_manager = await get_tenant_manager()
            cost_tracker = await get_cost_tracker()

            _state.billing_manager = BillingManager(
                db_pool=database_manager._pool,
                tenant_manager=tenant_manager,
                cost_tracker=cost_tracker,
            )
        else:
            raise HTTPException(status_code=500, detail="Database pool not available")

    return _state.billing_manager


async def get_resource_manager():
    """
    Get resource manager instance.

    Single responsibility: Resource management dependency injection.
    """
    if _state.resource_manager is None:
        database_manager = get_database_manager()
        if not database_manager:
            raise HTTPException(
                status_code=500, detail="Database manager not available"
            )

        # Get the underlying asyncpg pool from database manager
        if hasattr(database_manager, "_pool") and database_manager._pool:
            from ..tenancy import ResourceManager

            tenant_manager = await get_tenant_manager()

            _state.resource_manager = ResourceManager(
                db_pool=database_manager._pool, tenant_manager=tenant_manager
            )
        else:
            raise HTTPException(status_code=500, detail="Database pool not available")

    return _state.resource_manager


async def get_mapper() -> CoreMapper:
    """
    Get mapper instance.

    Single responsibility: Mapper dependency injection.
    """
    if _state.mapper_instance is None:
        settings = get_settings()
        database_manager = get_database_manager()
        if not database_manager:
            raise HTTPException(
                status_code=500, detail="Database manager not available"
            )

        # CoreMapper constructor takes only settings parameter
        _state.mapper_instance = CoreMapper(settings)
        await _state.mapper_instance.initialize()

    return _state.mapper_instance


# Health check dependencies
async def get_database_health():
    """
    Get database health status.

    Single responsibility: Database health check dependency.
    """
    health_checker = get_health_checker()
    if not health_checker:
        raise HTTPException(status_code=500, detail="Health checker not available")

    return await health_checker.check_database_health()


# Rate limiting dependencies
async def check_rate_limit(
    request: Request, api_key_info: APIKeyInfo = Depends(authenticate_request)
):
    """
    Check rate limits for authenticated requests.

    Single responsibility: Rate limiting enforcement.
    """
    rate_limiting_service = get_rate_limiting_service()
    if not rate_limiting_service:
        # If rate limiting service is not available, allow the request
        logger.warning("Rate limiting service not available, allowing request")
        return api_key_info

    # Check rate limits
    result = await rate_limiting_service.check_rate_limits(request, api_key_info)

    if not result.allowed:
        # Add rate limit headers to the exception
        headers = result.to_headers()
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded", headers=headers
        )

    return api_key_info


async def check_rate_limit_optional(
    request: Request,
    api_key_info: Optional[APIKeyInfo] = Depends(optional_authenticate_request),
):
    """
    Check rate limits for optionally authenticated requests.

    Single responsibility: Rate limiting enforcement for public endpoints.
    """
    rate_limiting_service = get_rate_limiting_service()
    if not rate_limiting_service:
        # If rate limiting service is not available, allow the request
        return api_key_info

    # Check rate limits (will use IP-based limiting if no API key)
    result = await rate_limiting_service.check_rate_limits(request, api_key_info)

    if not result.allowed:
        # Add rate limit headers to the exception
        headers = result.to_headers()
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded", headers=headers
        )

    return api_key_info


def get_api_key_manager_dependency():
    """
    Get API key manager from application state.

    Single responsibility: API key manager dependency injection.
    """
    from ..main import get_api_key_manager as get_manager

    return get_manager()


async def shutdown_dependencies() -> None:
    """
    Shutdown all dependency instances.

    Single responsibility: Cleanup on shutdown.
    """
    if _state.mapper_instance is not None:
        await _state.mapper_instance.shutdown()
        _state.mapper_instance = None

    # Reset settings instance
    _state.settings_instance = None
