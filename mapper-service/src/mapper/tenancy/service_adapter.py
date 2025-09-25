"""
Tenant Service Adapter for API endpoints.

This adapter provides a simplified interface for the deployment API endpoints
while wrapping the more complex MapperTenantManager functionality.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from .tenant_manager import (
    MapperTenantManager,
    MapperTenantConfig,
    MapperResourceType,
    MapperResourceQuota,
)


@dataclass
class TenantInfo:
    """Simplified tenant information for API responses."""

    tenant_id: str
    name: str
    description: Optional[str]
    settings: Dict[str, Any]
    enabled: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class TenantUsage:
    """Tenant usage statistics."""

    tenant_id: str
    period_start: datetime
    period_end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    total_cost: float
    usage_by_detector: Dict[str, int]
    usage_by_framework: Dict[str, int]


class TenantServiceAdapter:
    """
    Adapter that provides a simplified tenant service interface for API endpoints.

    This wraps the MapperTenantManager and provides the methods expected by
    the deployment API endpoints.
    """

    def __init__(self, tenant_manager: MapperTenantManager):
        self.tenant_manager = tenant_manager

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> TenantInfo:
        """Create a new tenant."""

        # Create default quotas for new tenant
        default_quotas = {
            MapperResourceType.MAPPING_REQUESTS: MapperResourceQuota(
                resource_type=MapperResourceType.MAPPING_REQUESTS,
                limit=1000,
                current_usage=0,
                reset_period="daily",
            ),
            MapperResourceType.MODEL_INFERENCE_TIME: MapperResourceQuota(
                resource_type=MapperResourceType.MODEL_INFERENCE_TIME,
                limit=3600,  # 1 hour in seconds
                current_usage=0,
                reset_period="daily",
            ),
        }

        tenant_config = MapperTenantConfig(
            tenant_id=tenant_id,
            tenant_name=name,
            tier="basic",
            quotas=default_quotas,
            model_preferences=settings or {},
            feature_flags={"enabled": enabled},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        await self.tenant_manager.create_mapper_tenant(tenant_config)

        return TenantInfo(
            tenant_id=tenant_id,
            name=name,
            description=description,
            settings=settings or {},
            enabled=enabled,
            created_at=tenant_config.created_at,
            updated_at=tenant_config.updated_at,
        )

    async def list_tenants(
        self, enabled: Optional[bool] = None, limit: int = 50, offset: int = 0
    ) -> List[TenantInfo]:
        """List tenants with optional filtering."""

        # For now, return empty list as the underlying manager doesn't have list method
        # In a real implementation, this would query the database
        return []

    async def get_tenant(self, tenant_id: str) -> Optional[TenantInfo]:
        """Get tenant by ID."""

        tenant_config = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not tenant_config:
            return None

        return TenantInfo(
            tenant_id=tenant_config.tenant_id,
            name=tenant_config.tenant_name,
            description=None,  # Not stored in MapperTenantConfig
            settings=tenant_config.model_preferences,
            enabled=tenant_config.feature_flags.get("enabled", True),
            created_at=tenant_config.created_at,
            updated_at=tenant_config.updated_at,
        )

    async def update_tenant(
        self,
        tenant_id: str,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> Optional[TenantInfo]:
        """Update tenant configuration."""

        # Get existing tenant
        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return None

        # Update the configuration
        updated_config = MapperTenantConfig(
            tenant_id=tenant_id,
            tenant_name=name,
            tier=existing.tier,
            quotas=existing.quotas,
            model_preferences=settings or existing.model_preferences,
            feature_flags={"enabled": enabled},
            created_at=existing.created_at,
            updated_at=datetime.utcnow(),
        )

        await self.tenant_manager.update_mapper_tenant(updated_config)

        return TenantInfo(
            tenant_id=tenant_id,
            name=name,
            description=description,
            settings=settings or {},
            enabled=enabled,
            created_at=existing.created_at,
            updated_at=updated_config.updated_at,
        )

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant."""

        # Check if tenant exists
        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return False

        # For now, just return True as the underlying manager doesn't have delete method
        # In a real implementation, this would delete from the database
        return True

    async def get_tenant_usage(
        self, tenant_id: str, start_date: datetime, end_date: datetime
    ) -> Optional[TenantUsage]:
        """Get tenant usage statistics."""

        # Check if tenant exists
        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return None

        # For now, return mock usage data
        # In a real implementation, this would query usage analytics
        return TenantUsage(
            tenant_id=tenant_id,
            period_start=start_date,
            period_end=end_date,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            avg_response_time_ms=0.0,
            total_cost=0.0,
            usage_by_detector={},
            usage_by_framework={},
        )

    async def enable_tenant(self, tenant_id: str) -> bool:
        """Enable a tenant."""

        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return False

        # Update feature flags to enable
        updated_config = MapperTenantConfig(
            tenant_id=existing.tenant_id,
            tenant_name=existing.tenant_name,
            tier=existing.tier,
            quotas=existing.quotas,
            model_preferences=existing.model_preferences,
            feature_flags={"enabled": True},
            created_at=existing.created_at,
            updated_at=datetime.utcnow(),
        )

        await self.tenant_manager.update_mapper_tenant(updated_config)
        return True

    async def disable_tenant(self, tenant_id: str) -> bool:
        """Disable a tenant."""

        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return False

        # Update feature flags to disable
        updated_config = MapperTenantConfig(
            tenant_id=existing.tenant_id,
            tenant_name=existing.tenant_name,
            tier=existing.tier,
            quotas=existing.quotas,
            model_preferences=existing.model_preferences,
            feature_flags={"enabled": False},
            created_at=existing.created_at,
            updated_at=datetime.utcnow(),
        )

        await self.tenant_manager.update_mapper_tenant(updated_config)
        return True
