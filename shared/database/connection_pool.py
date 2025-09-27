"""Database connection pool management following SRP.

This module provides ONLY connection pool management functionality.
Single Responsibility: Manage database connection pools for services.
"""

import asyncpg
from typing import Dict, Any, Optional
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration for a service."""

    service_name: str
    database_url: str
    pool_min_size: int = 5
    pool_max_size: int = 20
    pool_timeout: int = 30
    enable_ssl: bool = True
    tenant_isolation: bool = True


class ConnectionPool:
    """Manages a single database connection pool.

    Single Responsibility: Manage one database connection pool.
    Does NOT handle: migrations, health checks, multi-service coordination.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return

        try:
            self.pool = await asyncpg.create_pool(
                self.config.database_url,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.pool_timeout,
                server_settings={
                    "application_name": f"llama-mapper-{self.config.service_name}",
                    "timezone": "UTC",
                },
            )

            # Test connection
            if self.pool:
                async with self.pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")

            self._initialized = True
            logger.info("Database pool initialized for %s", self.config.service_name)

        except Exception as e:
            logger.error(
                "Failed to initialize database pool for %s: %s",
                self.config.service_name,
                str(e),
            )
            raise

    @asynccontextmanager
    async def get_connection(self, tenant_id: Optional[str] = None):
        """Get a database connection with optional tenant context."""
        if not self._initialized or not self.pool:
            raise RuntimeError("Connection pool not initialized")

        async with self.pool.acquire() as conn:
            # Set tenant context for RLS if enabled
            if self.config.tenant_isolation and tenant_id:
                await conn.execute("SET app.current_tenant_id = $1", tenant_id)

            try:
                yield conn
            finally:
                # Clear tenant context
                if self.config.tenant_isolation and tenant_id:
                    await conn.execute("RESET app.current_tenant_id")

    async def execute(self, query: str, *args, tenant_id: Optional[str] = None) -> Any:
        """Execute a query."""
        async with self.get_connection(tenant_id) as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args, tenant_id: Optional[str] = None) -> list:
        """Fetch results from a query."""
        async with self.get_connection(tenant_id) as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(
        self, query: str, *args, tenant_id: Optional[str] = None
    ) -> Optional[Any]:
        """Fetch a single row from a query."""
        async with self.get_connection(tenant_id) as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args, tenant_id: Optional[str] = None) -> Any:
        """Fetch a single value from a query."""
        async with self.get_connection(tenant_id) as conn:
            return await conn.fetchval(query, *args)

    @asynccontextmanager
    async def transaction(self, tenant_id: Optional[str] = None):
        """Start a transaction."""
        async with self.get_connection(tenant_id) as conn:
            async with conn.transaction():
                yield conn

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if not self.pool:
            return {"status": "not_initialized"}

        return {
            "status": "healthy",
            "pool_size": self.pool.get_size(),
            "pool_max_size": self.pool.get_max_size(),
            "pool_min_size": self.pool.get_min_size(),
            "idle_connections": self.pool.get_idle_size(),
        }

    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            try:
                await self.pool.close()
                logger.info("Closed database pool for %s", self.config.service_name)
            except Exception as e:
                logger.error(
                    "Error closing pool for %s: %s", self.config.service_name, str(e)
                )
            finally:
                self.pool = None
                self._initialized = False


# Export only the connection pool functionality
__all__ = [
    "ConnectionPool",
    "DatabaseConfig",
]
