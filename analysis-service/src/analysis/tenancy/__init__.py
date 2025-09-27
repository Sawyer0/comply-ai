"""
Tenancy components for analysis service.

This package provides tenant isolation and management functionality
that integrates with shared components for consistent tenant handling.
"""

from .shared_tenant_manager import (
    AnalysisTenantManager,
    get_shared_tenant_manager,
)

__all__ = [
    "AnalysisTenantManager",
    "get_shared_tenant_manager",
]