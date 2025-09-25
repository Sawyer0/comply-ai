"""
Dependency injection for Analysis Service.

This module provides dependency injection for service components.
Single Responsibility: Manage service dependencies only.
"""

import logging
from typing import Optional

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .config import settings, Settings
from .tenancy import TenantManager, AnalyticsManager
from .plugins import PluginManager, PluginDatabaseManager
from .engines.risk_scoring.database import RiskScoringDatabaseManager
from .privacy.database import PrivacyDatabaseManager
from .engines.risk_scoring.validator import RiskScoringValidator
from .privacy.privacy_validator import PrivacyValidator

logger = logging.getLogger(__name__)

# Type annotation to help linters understand settings is an instance
_settings: Settings = settings


class ServiceDependencies:
    """Container for service dependencies."""

    def __init__(self):
        # Core managers
        self._tenant_manager: Optional[TenantManager] = None
        self._analytics_manager: Optional[AnalyticsManager] = None
        self._plugin_manager: Optional[PluginManager] = None

        # Database managers
        self._db_managers: dict = {}

        # Validators and clients
        self._redis_client: Optional[object] = None
        self._validators: dict = {}

    async def get_redis_client(self) -> Optional[object]:
        """Get Redis client instance."""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, using memory fallback")
            return None

        if self._redis_client is None:
            try:
                # Access Redis configuration
                redis_config = _settings.redis

                self._redis_client = redis.from_url(
                    getattr(redis_config, "url"),
                    password=getattr(redis_config, "password", None),
                    decode_responses=True,
                )
                # Test connection
                await self._redis_client.ping()
                logger.info("Redis client connected successfully")
            except (ConnectionError, redis.RedisError) as e:
                logger.error("Failed to connect to Redis", error=str(e))
                self._redis_client = None

        return self._redis_client

    async def get_tenant_manager(self) -> TenantManager:
        """Get tenant manager instance."""
        if self._tenant_manager is None:
            redis_client = await self.get_redis_client()
            self._tenant_manager = TenantManager(
                redis_client=redis_client,
                db_service_name=getattr(_settings, "service_name"),
            )
        return self._tenant_manager

    async def get_analytics_manager(self) -> AnalyticsManager:
        """Get analytics manager instance."""
        if self._analytics_manager is None:
            self._analytics_manager = AnalyticsManager(
                db_service_name=getattr(_settings, "service_name")
            )
        return self._analytics_manager

    async def get_plugin_manager(self) -> PluginManager:
        """Get plugin manager instance."""
        if self._plugin_manager is None:
            tenant_manager = await self.get_tenant_manager()
            plugin_config = _settings.plugins

            self._plugin_manager = PluginManager(
                plugin_directories=getattr(plugin_config, "directories"),
                tenant_manager=tenant_manager,
                db_service_name=getattr(_settings, "service_name"),
            )

            # Initialize plugin manager if auto-discovery is enabled
            if getattr(plugin_config, "auto_discover", True):
                await self._plugin_manager.initialize()

        return self._plugin_manager

    async def get_plugin_db_manager(self) -> PluginDatabaseManager:
        """Get plugin database manager instance."""
        if "plugin" not in self._db_managers:
            self._db_managers["plugin"] = PluginDatabaseManager(
                db_service_name=getattr(_settings, "service_name")
            )
        return self._db_managers["plugin"]

    async def get_risk_scoring_db_manager(self) -> RiskScoringDatabaseManager:
        """Get risk scoring database manager instance."""
        if "risk_scoring" not in self._db_managers:
            self._db_managers["risk_scoring"] = RiskScoringDatabaseManager(
                db_service_name=getattr(_settings, "service_name")
            )
        return self._db_managers["risk_scoring"]

    async def get_privacy_db_manager(self) -> PrivacyDatabaseManager:
        """Get privacy database manager instance."""
        if "privacy" not in self._db_managers:
            self._db_managers["privacy"] = PrivacyDatabaseManager(
                db_service_name=getattr(_settings, "service_name")
            )
        return self._db_managers["privacy"]

    async def get_risk_scoring_validator(self) -> RiskScoringValidator:
        """Get risk scoring validator instance."""
        if "risk_scoring" not in self._validators:
            self._validators["risk_scoring"] = RiskScoringValidator()
        return self._validators["risk_scoring"]

    async def get_privacy_validator(self) -> PrivacyValidator:
        """Get privacy validator instance."""
        if "privacy" not in self._validators:
            self._validators["privacy"] = PrivacyValidator()
        return self._validators["privacy"]

    async def cleanup(self):
        """Cleanup service dependencies."""
        if self._plugin_manager:
            await self._plugin_manager.shutdown()

        if self._redis_client:
            await self._redis_client.close()


# Global dependencies instance
dependencies = ServiceDependencies()


# FastAPI dependency functions
async def get_tenant_manager() -> TenantManager:
    """FastAPI dependency for tenant manager."""
    return await dependencies.get_tenant_manager()


async def get_analytics_manager() -> AnalyticsManager:
    """FastAPI dependency for analytics manager."""
    return await dependencies.get_analytics_manager()


async def get_plugin_manager() -> PluginManager:
    """FastAPI dependency for plugin manager."""
    return await dependencies.get_plugin_manager()


async def get_plugin_db_manager() -> PluginDatabaseManager:
    """FastAPI dependency for plugin database manager."""
    return await dependencies.get_plugin_db_manager()


async def get_risk_scoring_db_manager() -> RiskScoringDatabaseManager:
    """FastAPI dependency for risk scoring database manager."""
    return await dependencies.get_risk_scoring_db_manager()


async def get_privacy_db_manager() -> PrivacyDatabaseManager:
    """FastAPI dependency for privacy database manager."""
    return await dependencies.get_privacy_db_manager()


async def get_risk_scoring_validator() -> RiskScoringValidator:
    """FastAPI dependency for risk scoring validator."""
    return await dependencies.get_risk_scoring_validator()


async def get_privacy_validator() -> PrivacyValidator:
    """FastAPI dependency for privacy validator."""
    return await dependencies.get_privacy_validator()


# Cleanup function for application shutdown
async def cleanup_dependencies():
    """Cleanup function for application shutdown."""
    await dependencies.cleanup()
