"""
Tenant management for the Analysis Service.

This module provides tenant configuration management and resource quota enforcement.
Single Responsibility: Manage tenant configurations and quotas only.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .models import (
    TenantConfiguration,
    TenantStatus,
    ResourceType,
    ResourceQuota,
    TenantRequest,
    QuotaRequest,
)
from shared.database.connection_manager import get_service_db

logger = logging.getLogger(__name__)


class TenantManager:
    """
    Tenant manager for multi-tenancy support in Analysis Service.

    Single Responsibility: Manage tenant configurations and resource quotas.
    """

    def __init__(
        self, redis_client: Optional[object] = None, db_service_name: str = "analysis"
    ):
        """
        Initialize the tenant manager.

        Args:
            redis_client: Redis client for caching
            db_service_name: Database service name for tenant data
        """
        self.redis_client = redis_client
        self.db_service_name = db_service_name
        self.tenant_prefix = "analysis_tenant:"

        # In-memory fallback storage
        self._memory_storage: Dict[str, TenantConfiguration] = {}

    def _get_tenant_key(self, tenant_id: str) -> str:
        """Get Redis key for tenant configuration."""
        return f"{self.tenant_prefix}{tenant_id}"

    async def create_tenant(
        self, tenant_id: str, request: TenantRequest
    ) -> TenantConfiguration:
        """
        Create a new tenant configuration.

        Args:
            tenant_id: Unique tenant identifier
            request: Tenant creation request

        Returns:
            Created tenant configuration
        """
        try:
            # Create default quotas
            default_quotas = {
                ResourceType.ANALYSIS_REQUESTS: ResourceQuota(
                    resource_type=ResourceType.ANALYSIS_REQUESTS,
                    limit=1000,
                    period_hours=24,
                ),
                ResourceType.BATCH_REQUESTS: ResourceQuota(
                    resource_type=ResourceType.BATCH_REQUESTS,
                    limit=100,
                    period_hours=24,
                ),
                ResourceType.STORAGE_MB: ResourceQuota(
                    resource_type=ResourceType.STORAGE_MB,
                    limit=1024,  # 1GB
                    period_hours=24,
                ),
                ResourceType.CPU_MINUTES: ResourceQuota(
                    resource_type=ResourceType.CPU_MINUTES,
                    limit=60,  # 1 hour
                    period_hours=24,
                ),
                ResourceType.ML_INFERENCE_CALLS: ResourceQuota(
                    resource_type=ResourceType.ML_INFERENCE_CALLS,
                    limit=500,
                    period_hours=24,
                ),
            }

            # Create tenant configuration
            tenant_config = TenantConfiguration(
                tenant_id=tenant_id,
                name=request.name,
                status=TenantStatus(request.status),
                default_confidence_threshold=request.default_confidence_threshold,
                enable_ml_analysis=request.enable_ml_analysis,
                enable_statistical_analysis=request.enable_statistical_analysis,
                enable_pattern_recognition=request.enable_pattern_recognition,
                quality_alert_threshold=request.quality_alert_threshold,
                enable_quality_monitoring=request.enable_quality_monitoring,
                enable_content_scrubbing=request.enable_content_scrubbing,
                log_level=request.log_level,
                custom_engines=request.custom_engines,
                preferred_frameworks=request.preferred_frameworks,
                quotas=default_quotas,
            )

            # Store tenant configuration
            await self._store_tenant_config(tenant_config)
            await self._store_tenant_in_db(tenant_config)

            logger.info("Created tenant configuration", tenant_id=tenant_id)
            return tenant_config

        except Exception as e:
            logger.error("Failed to create tenant", tenant_id=tenant_id, error=str(e))
            raise

    async def get_tenant_config(self, tenant_id: str) -> Optional[TenantConfiguration]:
        """
        Get tenant configuration.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant configuration if found, None otherwise
        """
        try:
            # Try Redis first
            if self.redis_client:
                key = self._get_tenant_key(tenant_id)
                data = await self.redis_client.hgetall(key)
                if data:
                    # Convert bytes to strings
                    data = {k.decode(): v.decode() for k, v in data.items()}
                    config_data = json.loads(data.get("config", "{}"))
                    return TenantConfiguration.from_dict(config_data)

            # Try database
            config = await self._load_tenant_from_db(tenant_id)
            if config:
                # Cache in Redis
                if self.redis_client:
                    await self._store_tenant_config(config)
                return config

            # Try memory fallback
            return self._memory_storage.get(tenant_id)

        except Exception as e:
            logger.error(
                "Failed to get tenant config", tenant_id=tenant_id, error=str(e)
            )
            return None

    async def update_tenant_config(
        self, tenant_id: str, updates: Dict[str, Any]
    ) -> Optional[TenantConfiguration]:
        """
        Update tenant configuration.

        Args:
            tenant_id: Tenant identifier
            updates: Configuration updates

        Returns:
            Updated tenant configuration
        """
        try:
            config = await self.get_tenant_config(tenant_id)
            if not config:
                return None

            # Apply updates
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            config.updated_at = datetime.now(timezone.utc)

            # Store updated configuration
            await self._store_tenant_config(config)
            await self._store_tenant_in_db(config)

            logger.info("Updated tenant configuration", tenant_id=tenant_id)
            return config

        except Exception as e:
            logger.error(
                "Failed to update tenant config", tenant_id=tenant_id, error=str(e)
            )
            return None

    async def set_resource_quota(
        self, tenant_id: str, quota_request: QuotaRequest
    ) -> bool:
        """
        Set resource quota for a tenant.

        Args:
            tenant_id: Tenant identifier
            quota_request: Quota configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            config = await self.get_tenant_config(tenant_id)
            if not config:
                return False

            resource_type = ResourceType(quota_request.resource_type)
            quota = ResourceQuota(
                resource_type=resource_type,
                limit=quota_request.limit,
                period_hours=quota_request.period_hours,
            )

            config.quotas[resource_type] = quota
            config.updated_at = datetime.now(timezone.utc)

            await self._store_tenant_config(config)
            await self._store_tenant_in_db(config)

            logger.info(
                "Set resource quota",
                tenant_id=tenant_id,
                resource_type=resource_type.value,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to set resource quota", tenant_id=tenant_id, error=str(e)
            )
            return False

    async def check_quota(
        self, tenant_id: str, resource_type: ResourceType, amount: int = 1
    ) -> bool:
        """
        Check if tenant has sufficient quota for resource usage.

        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource
            amount: Amount to consume

        Returns:
            True if quota is available, False otherwise
        """
        try:
            config = await self.get_tenant_config(tenant_id)
            if not config:
                return False

            quota = config.quotas.get(resource_type)
            if not quota:
                return True  # No quota set, allow usage

            # Reset quota if expired
            quota.reset_if_expired()

            # Check if usage would exceed quota
            if quota.current_usage + amount > quota.limit:
                logger.warning(
                    "Quota exceeded",
                    tenant_id=tenant_id,
                    resource_type=resource_type.value,
                    current=quota.current_usage,
                    limit=quota.limit,
                    requested=amount,
                )
                return False

            return True

        except Exception as e:
            logger.error("Failed to check quota", tenant_id=tenant_id, error=str(e))
            return False

    async def consume_quota(
        self, tenant_id: str, resource_type: ResourceType, amount: int = 1
    ) -> bool:
        """
        Consume tenant resource quota.

        Args:
            tenant_id: Tenant identifier
            resource_type: Type of resource
            amount: Amount to consume

        Returns:
            True if successful, False otherwise
        """
        try:
            config = await self.get_tenant_config(tenant_id)
            if not config:
                return False

            quota = config.quotas.get(resource_type)
            if not quota:
                return True  # No quota set, allow usage

            # Reset quota if expired
            quota.reset_if_expired()

            # Check and consume quota
            if quota.current_usage + amount <= quota.limit:
                quota.current_usage += amount
                config.updated_at = datetime.now(timezone.utc)

                # Store updated configuration
                await self._store_tenant_config(config)
                await self._update_quota_in_db(tenant_id, resource_type, quota)

                return True
            else:
                return False

        except Exception as e:
            logger.error("Failed to consume quota", tenant_id=tenant_id, error=str(e))
            return False

    async def list_tenants(self) -> List[TenantConfiguration]:
        """
        List all tenant configurations.

        Returns:
            List of tenant configurations
        """
        try:
            tenants = await self._list_tenants_from_db()
            if not tenants:
                tenants = list(self._memory_storage.values())

            return tenants

        except Exception as e:
            logger.error("Failed to list tenants", error=str(e))
            return []

    # Private helper methods

    async def _store_tenant_config(self, config: TenantConfiguration) -> bool:
        """Store tenant configuration in Redis."""
        try:
            if self.redis_client:
                key = self._get_tenant_key(config.tenant_id)
                await self.redis_client.hset(
                    key,
                    mapping={
                        "config": json.dumps(config.to_dict()),
                        "updated_at": config.updated_at.isoformat(),
                    },
                )
                return True
            else:
                self._memory_storage[config.tenant_id] = config
                return True

        except Exception as e:
            logger.error(
                "Failed to store tenant config",
                tenant_id=config.tenant_id,
                error=str(e),
            )
            return False

    async def _store_tenant_in_db(self, config: TenantConfiguration) -> bool:
        """Store tenant configuration in database."""
        try:
            db = get_service_db(self.db_service_name, config.tenant_id)

            # Store tenant configuration
            config_query = """
            INSERT INTO tenant_configurations (
                tenant_id, name, status, configuration, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (tenant_id) DO UPDATE SET
                name = EXCLUDED.name,
                status = EXCLUDED.status,
                configuration = EXCLUDED.configuration,
                updated_at = EXCLUDED.updated_at
            """

            await db.execute(
                config_query,
                config.tenant_id,
                config.name,
                config.status.value,
                json.dumps(config.to_dict()),
                config.created_at,
                config.updated_at,
            )

            # Store resource quotas
            for resource_type, quota in config.quotas.items():
                quota_query = """
                INSERT INTO tenant_resource_quotas (
                    tenant_id, resource_type, quota_limit, current_usage, 
                    period_hours, reset_at, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (tenant_id, resource_type) DO UPDATE SET
                    quota_limit = EXCLUDED.quota_limit,
                    current_usage = EXCLUDED.current_usage,
                    period_hours = EXCLUDED.period_hours,
                    reset_at = EXCLUDED.reset_at,
                    updated_at = EXCLUDED.updated_at
                """

                await db.execute(
                    quota_query,
                    config.tenant_id,
                    resource_type.value,
                    quota.limit,
                    quota.current_usage,
                    quota.period_hours,
                    quota.reset_at,
                    config.created_at,
                    config.updated_at,
                )

            return True

        except Exception as e:
            logger.error(
                "Failed to store tenant in DB", tenant_id=config.tenant_id, error=str(e)
            )
            return False

    async def _load_tenant_from_db(
        self, tenant_id: str
    ) -> Optional[TenantConfiguration]:
        """Load tenant configuration from database."""
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            # Load tenant configuration
            config_query = """
            SELECT tenant_id, name, status, configuration, created_at, updated_at
            FROM tenant_configurations 
            WHERE tenant_id = $1
            """

            config_row = await db.fetchrow(config_query, tenant_id)
            if not config_row:
                return None

            # Load resource quotas
            quota_query = """
            SELECT resource_type, quota_limit, current_usage, period_hours, reset_at
            FROM tenant_resource_quotas
            WHERE tenant_id = $1
            """

            quota_rows = await db.fetch(quota_query, tenant_id)

            # Parse configuration
            config_data = (
                json.loads(config_row["configuration"])
                if isinstance(config_row["configuration"], str)
                else config_row["configuration"]
            )

            # Build quotas from database
            quotas = {}
            for quota_row in quota_rows:
                resource_type = ResourceType(quota_row["resource_type"])
                quota = ResourceQuota(
                    resource_type=resource_type,
                    limit=quota_row["quota_limit"],
                    period_hours=quota_row["period_hours"],
                    current_usage=quota_row["current_usage"],
                    reset_at=quota_row["reset_at"],
                )
                quotas[resource_type] = quota

            # Update config data with database quotas
            config_data["quotas"] = {
                rt.value: {
                    "limit": quota.limit,
                    "period_hours": quota.period_hours,
                    "current_usage": quota.current_usage,
                    "reset_at": quota.reset_at.isoformat() if quota.reset_at else None,
                }
                for rt, quota in quotas.items()
            }

            return TenantConfiguration.from_dict(config_data)

        except Exception as e:
            logger.error(
                "Failed to load tenant from DB", tenant_id=tenant_id, error=str(e)
            )
            return None

    async def _update_quota_in_db(
        self, tenant_id: str, resource_type: ResourceType, quota: ResourceQuota
    ) -> bool:
        """Update resource quota in database."""
        try:
            db = get_service_db(self.db_service_name, tenant_id)

            query = """
            UPDATE tenant_resource_quotas 
            SET current_usage = $3, reset_at = $4, updated_at = NOW()
            WHERE tenant_id = $1 AND resource_type = $2
            """

            await db.execute(
                query,
                tenant_id,
                resource_type.value,
                quota.current_usage,
                quota.reset_at,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to update quota in DB", tenant_id=tenant_id, error=str(e)
            )
            return False

    async def _list_tenants_from_db(self) -> List[TenantConfiguration]:
        """List all tenants from database."""
        try:
            db = get_service_db(self.db_service_name)

            query = """
            SELECT tenant_id, name, status, configuration, created_at, updated_at
            FROM tenant_configurations 
            WHERE status != 'inactive'
            ORDER BY created_at DESC
            """

            rows = await db.fetch(query)
            tenants = []

            for row in rows:
                try:
                    # Load quotas for each tenant
                    quota_query = """
                    SELECT resource_type, quota_limit, current_usage, period_hours, reset_at
                    FROM tenant_resource_quotas
                    WHERE tenant_id = $1
                    """

                    quota_rows = await db.fetch(quota_query, row["tenant_id"])

                    # Parse configuration
                    config_data = (
                        json.loads(row["configuration"])
                        if isinstance(row["configuration"], str)
                        else row["configuration"]
                    )

                    # Build quotas from database
                    quotas = {}
                    for quota_row in quota_rows:
                        resource_type = ResourceType(quota_row["resource_type"])
                        quota = ResourceQuota(
                            resource_type=resource_type,
                            limit=quota_row["quota_limit"],
                            period_hours=quota_row["period_hours"],
                            current_usage=quota_row["current_usage"],
                            reset_at=quota_row["reset_at"],
                        )
                        quotas[resource_type] = quota

                    # Update config data with database quotas
                    config_data["quotas"] = {
                        rt.value: {
                            "limit": quota.limit,
                            "period_hours": quota.period_hours,
                            "current_usage": quota.current_usage,
                            "reset_at": (
                                quota.reset_at.isoformat() if quota.reset_at else None
                            ),
                        }
                        for rt, quota in quotas.items()
                    }

                    tenant_config = TenantConfiguration.from_dict(config_data)
                    tenants.append(tenant_config)

                except Exception as e:
                    logger.error(
                        "Failed to parse tenant config",
                        tenant_id=row["tenant_id"],
                        error=str(e),
                    )
                    continue

            return tenants

        except Exception as e:
            logger.error("Failed to list tenants from DB", error=str(e))
            return []
