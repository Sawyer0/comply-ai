"""Database service registry following SRP.

This module provides ONLY service registration and lookup functionality.
Single Responsibility: Manage registration of database services.
"""

import logging
from typing import Dict, List, Optional
from .connection_pool import ConnectionPool, DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseServiceRegistry:
    """Registry for database services and their connection pools.

    Single Responsibility: Register and manage database services.
    Does NOT handle: connection pooling, health checking, migrations.
    """

    def __init__(self):
        self.pools: Dict[str, ConnectionPool] = {}
        self.configs: Dict[str, DatabaseConfig] = {}

    def register_service(self, config: DatabaseConfig) -> bool:
        """Register a database service."""
        try:
            if config.service_name in self.pools:
                logger.warning("Service %s already registered", config.service_name)
                return False

            pool = ConnectionPool(config)
            self.pools[config.service_name] = pool
            self.configs[config.service_name] = config

            logger.info("Registered database service: %s", config.service_name)
            return True

        except Exception as e:
            logger.error(
                "Failed to register service %s: %s", config.service_name, str(e)
            )
            return False

    def unregister_service(self, service_name: str) -> bool:
        """Unregister a database service."""
        try:
            if service_name not in self.pools:
                logger.warning("Service %s not registered", service_name)
                return False

            # Note: This doesn't close the pool - that's the caller's responsibility
            del self.pools[service_name]
            del self.configs[service_name]

            logger.info("Unregistered database service: %s", service_name)
            return True

        except Exception as e:
            logger.error("Failed to unregister service %s: %s", service_name, str(e))
            return False

    def get_pool(self, service_name: str) -> Optional[ConnectionPool]:
        """Get connection pool for a service."""
        return self.pools.get(service_name)

    def get_config(self, service_name: str) -> Optional[DatabaseConfig]:
        """Get configuration for a service."""
        return self.configs.get(service_name)

    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self.pools.keys())

    def is_service_registered(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self.pools

    def get_service_count(self) -> int:
        """Get number of registered services."""
        return len(self.pools)

    async def initialize_all_services(self) -> Dict[str, bool]:
        """Initialize all registered services."""
        results = {}

        for service_name, pool in self.pools.items():
            try:
                await pool.initialize()
                results[service_name] = True
            except Exception as e:
                logger.error(
                    "Failed to initialize service %s: %s", service_name, str(e)
                )
                results[service_name] = False

        return results

    async def close_all_services(self) -> Dict[str, bool]:
        """Close all registered services."""
        results = {}

        for service_name, pool in self.pools.items():
            try:
                await pool.close()
                results[service_name] = True
            except Exception as e:
                logger.error("Failed to close service %s: %s", service_name, str(e))
                results[service_name] = False

        return results


# Export only the service registry functionality
__all__ = [
    "DatabaseServiceRegistry",
]
