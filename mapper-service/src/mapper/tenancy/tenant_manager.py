"""
Mapper Service Tenant Management

Single Responsibility: Manage mapper-specific tenant configurations and quotas.

This is part of the microservice refactoring from the monolithic llama-mapper system.
The mapper service handles its own tenant management following SRP principles.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
import json
from decimal import Decimal

# Import from shared interfaces (these define contracts between services)
from shared.interfaces.tenant_isolation import (
    TenantContext,
    TenantConfig,
    TenantAccessLevel,
)
from shared.interfaces.cost_monitoring import CostEvent, CostCategory

logger = logging.getLogger(__name__)


class MapperResourceType(str, Enum):
    """Mapper-specific resource types"""

    MAPPING_REQUESTS = "mapping_requests"
    MODEL_INFERENCE_TIME = "model_inference_time"
    TRAINING_JOBS = "training_jobs"
    CUSTOM_MODELS = "custom_models"
    STORAGE_MODELS = "storage_models"
    VALIDATION_REQUESTS = "validation_requests"


@dataclass
class MapperResourceQuota:
    """Mapper-specific resource quota"""

    resource_type: MapperResourceType
    limit: int
    current_usage: int = 0
    reset_period: str = "daily"  # daily, weekly, monthly

    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage"""
        if self.limit == 0:
            return 0.0
        return min(100.0, (self.current_usage / self.limit) * 100)

    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded"""
        return self.current_usage >= self.limit


@dataclass
class MapperTenantConfig:
    """Mapper-specific tenant configuration"""

    tenant_id: str
    tenant_name: str
    tier: str  # free, basic, premium, enterprise
    quotas: Dict[MapperResourceType, MapperResourceQuota]
    model_preferences: Dict[str, Any]
    feature_flags: Dict[str, bool]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            **asdict(self),
            "quotas": {k.value: asdict(v) for k, v in self.quotas.items()},
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class MapperTenantManager:
    """
    Mapper Service Tenant Management

    Single Responsibility: Manage mapper-specific tenant configurations and quotas.

    This class is responsible ONLY for:
    - Mapper tenant configuration management
    - Mapper-specific resource quotas
    - Model preferences per tenant
    - Feature flags per tenant

    It does NOT handle:
    - Cross-service tenant isolation (handled by shared interfaces)
    - Cost calculation (handled by cost monitoring service)
    - Billing (handled by billing manager)
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.tenant_cache = {}

    async def create_mapper_tenant(self, tenant_config: MapperTenantConfig) -> str:
        """
        Create a new mapper tenant configuration.

        Single Responsibility: Store mapper-specific tenant data only.
        """
        query = """
        INSERT INTO mapper_tenants (
            tenant_id, tenant_name, tier, quotas, 
            model_preferences, feature_flags, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING tenant_id
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow(
                query,
                tenant_config.tenant_id,
                tenant_config.tenant_name,
                tenant_config.tier,
                json.dumps(
                    {k.value: asdict(v) for k, v in tenant_config.quotas.items()}
                ),
                json.dumps(tenant_config.model_preferences),
                json.dumps(tenant_config.feature_flags),
                tenant_config.created_at,
                tenant_config.updated_at,
            )

        # Clear cache
        self.tenant_cache.pop(tenant_config.tenant_id, None)

        logger.info(f"Created mapper tenant {tenant_config.tenant_id}")
        return result["tenant_id"]

    async def get_mapper_tenant(self, tenant_id: str) -> Optional[MapperTenantConfig]:
        """
        Get mapper tenant configuration.

        Single Responsibility: Retrieve mapper-specific tenant data only.
        """
        # Check cache first
        if tenant_id in self.tenant_cache:
            return self.tenant_cache[tenant_id]

        query = """
        SELECT * FROM mapper_tenants WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, tenant_id)

        if not row:
            return None

        # Parse quotas
        quotas = {}
        for resource_type, quota_data in json.loads(row["quotas"]).items():
            quotas[MapperResourceType(resource_type)] = MapperResourceQuota(
                **quota_data
            )

        tenant_config = MapperTenantConfig(
            tenant_id=row["tenant_id"],
            tenant_name=row["tenant_name"],
            tier=row["tier"],
            quotas=quotas,
            model_preferences=json.loads(row["model_preferences"]),
            feature_flags=json.loads(row["feature_flags"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

        # Cache the result
        self.tenant_cache[tenant_id] = tenant_config
        return tenant_config

    async def update_mapper_tenant(self, tenant_config: MapperTenantConfig) -> bool:
        """
        Update mapper tenant configuration.

        Single Responsibility: Update mapper-specific tenant data only.
        """
        query = """
        UPDATE mapper_tenants SET
            tenant_name = $2,
            tier = $3,
            quotas = $4,
            model_preferences = $5,
            feature_flags = $6,
            updated_at = $7
        WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                query,
                tenant_config.tenant_id,
                tenant_config.tenant_name,
                tenant_config.tier,
                json.dumps(
                    {k.value: asdict(v) for k, v in tenant_config.quotas.items()}
                ),
                json.dumps(tenant_config.model_preferences),
                json.dumps(tenant_config.feature_flags),
                tenant_config.updated_at,
            )

        # Clear cache
        self.tenant_cache.pop(tenant_config.tenant_id, None)

        logger.info(f"Updated mapper tenant {tenant_config.tenant_id}")
        return result != "UPDATE 0"

    async def check_mapping_quota(
        self, tenant_id: str, resource_type: MapperResourceType, amount: int = 1
    ) -> bool:
        """
        Check if tenant has quota available for mapping resource.

        Single Responsibility: Check mapper-specific quotas only.
        """
        tenant = await self.get_mapper_tenant(tenant_id)
        if not tenant:
            return False

        quota = tenant.quotas.get(resource_type)
        if not quota:
            return True  # No quota limit set

        return quota.current_usage + amount <= quota.limit

    async def consume_mapping_quota(
        self, tenant_id: str, resource_type: MapperResourceType, amount: int = 1
    ) -> bool:
        """
        Consume quota for a mapping resource.

        Single Responsibility: Update mapper-specific quota usage only.
        """
        if not await self.check_mapping_quota(tenant_id, resource_type, amount):
            return False

        # Update quota usage
        query = """
        UPDATE mapper_tenants 
        SET quotas = jsonb_set(
            quotas, 
            $2, 
            (COALESCE(quotas->$3->>'current_usage', '0')::int + $4)::text::jsonb
        ),
        updated_at = $5
        WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                tenant_id,
                f"{{{resource_type.value},current_usage}}",
                resource_type.value,
                amount,
                datetime.utcnow(),
            )

        # Clear cache
        self.tenant_cache.pop(tenant_id, None)

        return True

    async def get_tenant_model_config(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get tenant-specific model configuration.

        Single Responsibility: Return mapper-specific model preferences only.
        """
        tenant = await self.get_mapper_tenant(tenant_id)
        if not tenant:
            return {}

        return tenant.model_preferences

    async def update_tenant_model_config(
        self, tenant_id: str, model_config: Dict[str, Any]
    ) -> bool:
        """
        Update tenant-specific model configuration.

        Single Responsibility: Update mapper-specific model preferences only.
        """
        query = """
        UPDATE mapper_tenants 
        SET model_preferences = $2, updated_at = $3
        WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                query, tenant_id, json.dumps(model_config), datetime.utcnow()
            )

        # Clear cache
        self.tenant_cache.pop(tenant_id, None)

        return result != "UPDATE 0"

    async def update_feature_flags(
        self, tenant_id: str, feature_flags: Dict[str, bool]
    ) -> bool:
        """
        Update tenant feature flags.

        Single Responsibility: Update mapper-specific feature flags only.
        """
        query = """
        UPDATE mapper_tenants 
        SET feature_flags = $2, updated_at = $3
        WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            result = await conn.execute(
                query, tenant_id, json.dumps(feature_flags), datetime.utcnow()
            )

        # Clear cache
        self.tenant_cache.pop(tenant_id, None)

        return result != "UPDATE 0"

    async def reset_tenant_quotas(
        self, tenant_id: str, resource_types: List[MapperResourceType] = None
    ) -> None:
        """
        Reset quotas for tenant.

        Single Responsibility: Reset mapper-specific quotas only.
        """
        tenant = await self.get_mapper_tenant(tenant_id)
        if not tenant:
            return

        if resource_types is None:
            resource_types = list(tenant.quotas.keys())

        for resource_type in resource_types:
            if resource_type in tenant.quotas:
                tenant.quotas[resource_type].current_usage = 0

        query = """
        UPDATE mapper_tenants 
        SET quotas = $2, updated_at = $3
        WHERE tenant_id = $1
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query,
                tenant_id,
                json.dumps({k.value: asdict(v) for k, v in tenant.quotas.items()}),
                datetime.utcnow(),
            )

        # Clear cache
        self.tenant_cache.pop(tenant_id, None)

        logger.info(
            f"Reset mapper quotas for tenant {tenant_id}: {[rt.value for rt in resource_types]}"
        )

    async def get_tenant_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get tenant usage summary.

        Single Responsibility: Return mapper-specific usage data only.
        """
        tenant = await self.get_mapper_tenant(tenant_id)
        if not tenant:
            return {}

        quota_status = {}
        for resource_type, quota in tenant.quotas.items():
            quota_status[resource_type.value] = {
                "limit": quota.limit,
                "current_usage": quota.current_usage,
                "usage_percentage": quota.usage_percentage,
                "is_exceeded": quota.is_exceeded,
                "reset_period": quota.reset_period,
            }

        return {
            "tenant_id": tenant_id,
            "tier": tenant.tier,
            "quota_status": quota_status,
            "feature_flags": tenant.feature_flags,
            "generated_at": datetime.utcnow().isoformat(),
        }

    async def initialize_database_schema(self) -> None:
        """
        Initialize database schema for mapper tenant management.

        Single Responsibility: Create mapper-specific tenant tables only.
        """
        schema_sql = """
        CREATE TABLE IF NOT EXISTS mapper_tenants (
            tenant_id VARCHAR(255) PRIMARY KEY,
            tenant_name VARCHAR(255) NOT NULL,
            tier VARCHAR(50) NOT NULL,
            quotas JSONB NOT NULL DEFAULT '{}',
            model_preferences JSONB NOT NULL DEFAULT '{}',
            feature_flags JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_mapper_tenants_tier ON mapper_tenants(tier);
        CREATE INDEX IF NOT EXISTS idx_mapper_tenants_created_at ON mapper_tenants(created_at);
        """

        async with self.db_pool.acquire() as conn:
            await conn.execute(schema_sql)
