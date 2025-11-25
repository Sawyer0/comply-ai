"""
Tenant Service Adapter for API endpoints.

This adapter provides a simplified interface for the deployment API endpoints
while wrapping the more complex MapperTenantManager functionality.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List, Dict, Any

import asyncpg

from .tenant_manager import (
    MapperTenantManager,
    MapperTenantConfig,
    MapperResourceType,
    MapperResourceQuota,
)
from ..infrastructure.database_manager import (
    DatabaseManager,
    create_database_manager_from_env,
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

    def __init__(
        self,
        tenant_manager: Optional[MapperTenantManager] = None,
        database_manager: Optional[DatabaseManager] = None,
    ):
        self.tenant_manager = tenant_manager
        self._database_manager = database_manager or create_database_manager_from_env()
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        """Ensure the database pool and tenant manager are ready."""

        if self.tenant_manager is not None and self._database_manager.is_connected:
            return

        async with self._init_lock:
            if not self._database_manager.is_connected:
                await self._database_manager.initialize()

            if self.tenant_manager is None:
                pool = self._get_pool()
                self.tenant_manager = MapperTenantManager(pool)

    def _get_pool(self) -> asyncpg.Pool:
        pool = getattr(self._database_manager, "_pool", None)
        if pool is None:
            raise RuntimeError("Database pool is not available. Ensure initialization succeeded.")
        return pool

    async def create_tenant(
        self,
        tenant_id: str,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> TenantInfo:
        """Create a new tenant."""

        await self._ensure_initialized()
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

        await self._ensure_initialized()

        clauses = []
        params: List[Any] = []

        if enabled is not None:
            clauses.append("COALESCE((feature_flags->>'enabled')::boolean, TRUE) = $1")
            params.append(enabled)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        param_index = len(params) + 1
        query = f"""
            SELECT tenant_id,
                   tenant_name,
                   model_preferences,
                   feature_flags,
                   created_at,
                   updated_at
            FROM mapper_tenants
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_index} OFFSET ${param_index + 1}
        """

        params.extend([limit, offset])

        async with self._database_manager.get_connection() as conn:
            rows = await conn.fetch(query, *params)

        return [self._build_tenant_info(row) for row in rows]

    async def get_tenant(self, tenant_id: str) -> Optional[TenantInfo]:
        """Get tenant by ID."""

        await self._ensure_initialized()
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

        await self._ensure_initialized()
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

        await self._ensure_initialized()

        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return False

        tables_to_purge = [
            "mapper_cost_events",
            "resource_metrics",
            "resource_allocations",
            "scaling_recommendations",
            "model_inferences",
            "mapping_results",
            "mapping_requests",
        ]

        async with self._database_manager.get_connection() as conn:
            async with conn.transaction():
                for table in tables_to_purge:
                    await conn.execute(f"DELETE FROM {table} WHERE tenant_id = $1", tenant_id)

                result = await conn.execute("DELETE FROM mapper_tenants WHERE tenant_id = $1", tenant_id)

        self.tenant_manager.tenant_cache.pop(tenant_id, None)
        return result != "DELETE 0"

    async def get_tenant_usage(
        self, tenant_id: str, start_date: datetime, end_date: datetime
    ) -> Optional[TenantUsage]:
        """Get tenant usage statistics."""

        await self._ensure_initialized()
        existing = await self.tenant_manager.get_mapper_tenant(tenant_id)
        if not existing:
            return None

        usage = await self._collect_usage(tenant_id, start_date, end_date)
        cost = await self._collect_costs(tenant_id, start_date, end_date)

        return TenantUsage(
            tenant_id=tenant_id,
            period_start=start_date,
            period_end=end_date,
            total_requests=usage["total_requests"],
            successful_requests=usage["successful_requests"],
            failed_requests=usage["failed_requests"],
            avg_response_time_ms=usage["avg_response_time_ms"],
            total_cost=cost["total_cost"],
            usage_by_detector=usage["by_detector"],
            usage_by_framework=usage["by_framework"],
        )

    async def enable_tenant(self, tenant_id: str) -> bool:
        """Enable a tenant."""

        await self._ensure_initialized()
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

    def _build_tenant_info(self, row: asyncpg.Record) -> TenantInfo:
        feature_flags = row["feature_flags"] or {}
        return TenantInfo(
            tenant_id=row["tenant_id"],
            name=row["tenant_name"],
            description=feature_flags.get("description"),
            settings=row["model_preferences"] or {},
            enabled=feature_flags.get("enabled", True),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def _collect_usage(
        self, tenant_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        stats_query = """
            SELECT COUNT(*) AS total_requests,
                   COUNT(*) FILTER (WHERE status = 'completed') AS successful_requests,
                   COUNT(*) FILTER (WHERE status != 'completed') AS failed_requests,
                   AVG(processing_time_ms) AS avg_response_time_ms
            FROM mapping_requests
            WHERE tenant_id = $1 AND started_at BETWEEN $2 AND $3
        """

        detector_query = """
            SELECT COALESCE(detector_type, 'unknown') AS detector,
                   COUNT(*) AS count
            FROM mapping_requests
            WHERE tenant_id = $1 AND started_at BETWEEN $2 AND $3
            GROUP BY detector
        """

        framework_query = """
            SELECT jsonb_object_keys(framework_mappings) AS framework,
                   COUNT(*)
            FROM mapping_results
            WHERE tenant_id = $1 AND created_at BETWEEN $2 AND $3
            GROUP BY framework
        """

        async with self._database_manager.get_connection() as conn:
            stats = await conn.fetchrow(stats_query, tenant_id, start_date, end_date)
            detector_rows = await conn.fetch(detector_query, tenant_id, start_date, end_date)
            framework_rows = await conn.fetch(framework_query, tenant_id, start_date, end_date)

        return {
            "total_requests": stats["total_requests"] or 0,
            "successful_requests": stats["successful_requests"] or 0,
            "failed_requests": stats["failed_requests"] or 0,
            "avg_response_time_ms": float(stats["avg_response_time_ms"] or 0.0),
            "by_detector": {row["detector"]: row["count"] for row in detector_rows},
            "by_framework": {row["framework"]: row["count"] for row in framework_rows},
        }

    async def _collect_costs(
        self, tenant_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        cost_query = """
            SELECT COALESCE(SUM(cost_amount::decimal), 0) AS total_cost
            FROM mapper_cost_events
            WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
        """

        async with self._database_manager.get_connection() as conn:
            row = await conn.fetchrow(cost_query, tenant_id, start_date, end_date)

        total_cost = row["total_cost"] if row and row["total_cost"] is not None else Decimal("0")
        return {"total_cost": float(total_cost)}

    async def disable_tenant(self, tenant_id: str) -> bool:
        """Disable a tenant."""

        await self._ensure_initialized()
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
