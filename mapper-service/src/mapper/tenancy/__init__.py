"""
Tenancy components for mapper service.

This package provides tenant isolation and management functionality
that integrates with shared components for consistent tenant handling.
"""

from .shared_tenant_manager import (
    MapperTenantManager,
    get_shared_tenant_manager,
)

def get_tenant_service() -> MapperTenantManager:
    """Get tenant service instance (alias for get_shared_tenant_manager)."""
    return get_shared_tenant_manager()

__all__ = [
    "MapperTenantManager",
    "get_shared_tenant_manager",
    "get_tenant_service",
]