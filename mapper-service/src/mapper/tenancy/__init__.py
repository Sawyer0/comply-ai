"""
Tenancy module for mapper service.

Provides comprehensive multi-tenancy support including:
- Tenant management and isolation
- Cost monitoring and billing
- Resource management and auto-scaling
- Usage analytics and reporting
"""

from typing import Optional
from .tenant_manager import (
    MapperTenantManager,
    MapperTenantConfig,
    MapperResourceType,
    MapperResourceQuota,
)
from .cost_tracker import (
    MapperCostTracker,
    MappingCostEvent,
)
from .billing_manager import (
    BillingManager,
    BillingTier,
    Invoice,
    InvoiceStatus,
    PaymentMethod,
    UsageLineItem,
)
from .resource_manager import (
    ResourceManager,
    ResourceAllocation,
    ResourceMetrics,
    ScalingRecommendation,
    ScalingAction,
)

# Import service adapter
from .service_adapter import TenantServiceAdapter, TenantInfo, TenantUsage

# Global tenant service instance
_tenant_service: Optional[TenantServiceAdapter] = None


def get_tenant_service() -> Optional[TenantServiceAdapter]:
    """Get the global tenant service instance."""
    return _tenant_service


def set_tenant_service(service: TenantServiceAdapter) -> None:
    """Set the global tenant service instance."""
    global _tenant_service
    _tenant_service = service


__all__ = [
    # Tenant Management
    "MapperTenantManager",
    "MapperTenantConfig",
    "MapperResourceType",
    "MapperResourceQuota",
    "TenantServiceAdapter",
    "TenantInfo",
    "TenantUsage",
    "get_tenant_service",
    "set_tenant_service",
    # Cost Tracking
    "MapperCostTracker",
    "MappingCostEvent",
    # Billing
    "BillingManager",
    "BillingTier",
    "Invoice",
    "InvoiceStatus",
    "PaymentMethod",
    "UsageLineItem",
    # Resource Management
    "ResourceManager",
    "ResourceAllocation",
    "ResourceMetrics",
    "ScalingRecommendation",
    "ScalingAction",
]
