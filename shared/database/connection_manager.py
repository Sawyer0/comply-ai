"""Multi-database connection management for microservices following SRP.

This module provides ONLY high-level coordination of database services.
Single Responsibility: Coordinate database services using SRP components.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from .connection_pool import DatabaseConfig
from .service_registry import DatabaseServiceRegistry
from .health_checker import DatabaseHealthChecker

logger = logging.getLogger(__name__)


class DatabaseConnectionManager:
    """Coordinates database services using SRP components.

    Single Responsibility: Coordinate database services and provide unified interface.
    Uses SRP components: ServiceRegistry, HealthChecker, ConnectionPool.
    """

    def __init__(self):
        self.service_registry = DatabaseServiceRegistry()
        self.health_checker = DatabaseHealthChecker(self.service_registry.pools)
        self._initialized = False

    async def initialize(self, configs: List[DatabaseConfig]) -> None:
        """Initialize database connections for all services."""
        if self._initialized:
            return

        # Register all services
        for config in configs:
            if not self.service_registry.register_service(config):
                raise RuntimeError(f"Failed to register service: {config.service_name}")

        # Initialize all connection pools
        results = await self.service_registry.initialize_all_services()

        # Check if any initialization failed
        failed_services = [
            service for service, success in results.items() if not success
        ]
        if failed_services:
            raise RuntimeError(f"Failed to initialize services: {failed_services}")

        # Update health checker with current pools
        self.health_checker = DatabaseHealthChecker(self.service_registry.pools)
        self._initialized = True

    @asynccontextmanager
    async def get_connection(self, service_name: str, tenant_id: Optional[str] = None):
        """Get a database connection for a specific service."""
        if not self._initialized:
            raise RuntimeError("DatabaseConnectionManager not initialized")

        pool = self.service_registry.get_pool(service_name)
        if not pool:
            raise ValueError(f"No database pool configured for service: {service_name}")

        async with pool.get_connection(tenant_id) as conn:
            yield conn

    async def execute_query(
        self, service_name: str, query: str, *args, tenant_id: Optional[str] = None
    ) -> Any:
        """Execute a query on a specific service database."""
        pool = self.service_registry.get_pool(service_name)
        if not pool:
            raise ValueError(f"No database pool configured for service: {service_name}")
        return await pool.execute(query, *args, tenant_id=tenant_id)

    async def fetch_query(
        self, service_name: str, query: str, *args, tenant_id: Optional[str] = None
    ) -> List[Any]:
        """Fetch results from a query on a specific service database."""
        pool = self.service_registry.get_pool(service_name)
        if not pool:
            raise ValueError(f"No database pool configured for service: {service_name}")
        return await pool.fetch(query, *args, tenant_id=tenant_id)

    async def fetchrow_query(
        self, service_name: str, query: str, *args, tenant_id: Optional[str] = None
    ) -> Optional[Any]:
        """Fetch a single row from a query on a specific service database."""
        pool = self.service_registry.get_pool(service_name)
        if not pool:
            raise ValueError(f"No database pool configured for service: {service_name}")
        return await pool.fetchrow(query, *args, tenant_id=tenant_id)

    async def fetchval_query(
        self, service_name: str, query: str, *args, tenant_id: Optional[str] = None
    ) -> Any:
        """Fetch a single value from a query on a specific service database."""
        pool = self.service_registry.get_pool(service_name)
        if not pool:
            raise ValueError(f"No database pool configured for service: {service_name}")
        return await pool.fetchval(query, *args, tenant_id=tenant_id)

    @asynccontextmanager
    async def transaction(self, service_name: str, tenant_id: Optional[str] = None):
        """Start a transaction on a specific service database."""
        pool = self.service_registry.get_pool(service_name)
        if not pool:
            raise ValueError(f"No database pool configured for service: {service_name}")
        async with pool.transaction(tenant_id) as conn:
            yield conn

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all database connections."""
        return await self.health_checker.check_all_pools()

    async def close_all(self) -> None:
        """Close all database connections."""
        await self.service_registry.close_all_services()
        self._initialized = False

    def list_services(self) -> List[str]:
        """List all registered database services."""
        return self.service_registry.list_services()

    def get_service_count(self) -> int:
        """Get number of registered services."""
        return self.service_registry.get_service_count()


# Global connection manager instance
db_manager = DatabaseConnectionManager()


async def initialize_databases() -> None:
    """Initialize database connections from environment variables."""
    configs = []

    # Orchestration service database
    orchestration_url = os.getenv(
        "ORCHESTRATION_DATABASE_URL",
        "postgresql://orchestration:password@localhost:5432/orchestration_db",
    )
    configs.append(
        DatabaseConfig(
            service_name="orchestration",
            database_url=orchestration_url,
            pool_min_size=int(os.getenv("ORCHESTRATION_DB_POOL_MIN", "5")),
            pool_max_size=int(os.getenv("ORCHESTRATION_DB_POOL_MAX", "20")),
        )
    )

    # Analysis service database
    analysis_url = os.getenv(
        "ANALYSIS_DATABASE_URL",
        "postgresql://analysis:password@localhost:5432/analysis_db",
    )
    configs.append(
        DatabaseConfig(
            service_name="analysis",
            database_url=analysis_url,
            pool_min_size=int(os.getenv("ANALYSIS_DB_POOL_MIN", "5")),
            pool_max_size=int(os.getenv("ANALYSIS_DB_POOL_MAX", "20")),
        )
    )

    # Mapper service database
    mapper_url = os.getenv(
        "MAPPER_DATABASE_URL", "postgresql://mapper:password@localhost:5432/mapper_db"
    )
    configs.append(
        DatabaseConfig(
            service_name="mapper",
            database_url=mapper_url,
            pool_min_size=int(os.getenv("MAPPER_DB_POOL_MIN", "5")),
            pool_max_size=int(os.getenv("MAPPER_DB_POOL_MAX", "20")),
        )
    )

    await db_manager.initialize(configs)


async def get_service_connection(service_name: str, tenant_id: Optional[str] = None):
    """Convenience function to get a service database connection."""
    return db_manager.get_connection(service_name, tenant_id)


async def close_all_databases() -> None:
    """Close all database connections."""
    await db_manager.close_all()


# Context manager for service-specific database operations
class ServiceDatabase:
    """Context manager for service-specific database operations.

    Single Responsibility: Provide convenient interface for service database operations.
    """

    def __init__(self, service_name: str, tenant_id: Optional[str] = None):
        self.service_name = service_name
        self.tenant_id = tenant_id

    async def execute(self, query: str, *args) -> Any:
        """Execute a query."""
        return await db_manager.execute_query(
            self.service_name, query, *args, tenant_id=self.tenant_id
        )

    async def fetch(self, query: str, *args) -> List[Any]:
        """Fetch multiple rows."""
        return await db_manager.fetch_query(
            self.service_name, query, *args, tenant_id=self.tenant_id
        )

    async def fetchrow(self, query: str, *args) -> Optional[Any]:
        """Fetch a single row."""
        return await db_manager.fetchrow_query(
            self.service_name, query, *args, tenant_id=self.tenant_id
        )

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        return await db_manager.fetchval_query(
            self.service_name, query, *args, tenant_id=self.tenant_id
        )

    @asynccontextmanager
    async def transaction(self):
        """Start a transaction."""
        async with db_manager.transaction(self.service_name, self.tenant_id) as conn:
            yield conn


def get_service_db(
    service_name: str, tenant_id: Optional[str] = None
) -> ServiceDatabase:
    """Get a ServiceDatabase instance for a specific service."""
    return ServiceDatabase(service_name, tenant_id)
