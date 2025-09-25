"""
Database Manager for Mapper Service

Handles database connections, connection pooling, and database operations
following Single Responsibility Principle.
"""

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import asyncpg
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration model."""

    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    ssl_mode: str = Field("require", description="SSL mode")
    pool_min_size: int = Field(5, description="Minimum pool size")
    pool_max_size: int = Field(20, description="Maximum pool size")
    command_timeout: int = Field(60, description="Command timeout in seconds")
    connection_timeout: int = Field(30, description="Connection timeout in seconds")


class DatabaseManager:
    """
    Database manager responsible for connection management and basic operations.

    Follows SRP by focusing solely on database connectivity and connection lifecycle.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logger.bind(component="database_manager")
        self._pool: Optional[asyncpg.Pool] = None
        self._is_connected = False

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        try:
            self.logger.info("Initializing database connection pool")

            # Build connection string
            connection_string = self._build_connection_string()

            # Create connection pool
            self._pool = await asyncpg.create_pool(
                connection_string,
                min_size=self.config.pool_min_size,
                max_size=self.config.pool_max_size,
                command_timeout=self.config.command_timeout,
                server_settings={
                    "application_name": "mapper-service",
                    "timezone": "UTC",
                },
            )

            self._is_connected = True
            self.logger.info(
                "Database connection pool initialized successfully",
                pool_size=f"{self.config.pool_min_size}-{self.config.pool_max_size}",
            )

        except Exception as e:
            self.logger.error(
                "Failed to initialize database connection pool", error=str(e)
            )
            raise

    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool:
            self.logger.info("Closing database connection pool")
            await self._pool.close()
            self._pool = None
            self._is_connected = False
            self.logger.info("Database connection pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self._pool:
            raise RuntimeError("Database not initialized")

        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                self.logger.error("Database operation failed", error=str(e))
                raise

    async def execute_query(self, query: str, *args) -> str:
        """Execute a query that doesn't return data."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)

    async def fetch_one(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch single record."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetch_many(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple records."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def fetch_value(self, query: str, *args) -> Any:
        """Fetch single value."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)

    async def execute_transaction(self, queries: List[tuple]) -> None:
        """Execute multiple queries in a transaction."""
        async with self.get_connection() as conn:
            async with conn.transaction():
                for query, args in queries:
                    await conn.execute(query, *args)

    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            if not self._pool:
                return {
                    "status": "unhealthy",
                    "error": "Database not initialized",
                    "timestamp": datetime.utcnow().isoformat(),
                }

            # Test connection with simple query
            start_time = datetime.utcnow()
            result = await self.fetch_value("SELECT 1")
            end_time = datetime.utcnow()

            response_time = (end_time - start_time).total_seconds() * 1000

            return {
                "status": "healthy" if result == 1 else "unhealthy",
                "response_time_ms": response_time,
                "pool_size": self._pool.get_size(),
                "pool_idle": self._pool.get_idle_size(),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            self.logger.error("Database health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"postgresql://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
            f"?sslmode={self.config.ssl_mode}"
        )

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._is_connected and self._pool is not None


def create_database_manager_from_env() -> DatabaseManager:
    """Create database manager from environment variables."""
    config = DatabaseConfig(
        host=os.getenv("MAPPER_DATABASE_HOST", "localhost"),
        port=int(os.getenv("MAPPER_DATABASE_PORT", "5432")),
        database=os.getenv("MAPPER_DATABASE_NAME", "mapper_service"),
        username=os.getenv("MAPPER_DATABASE_USER", "mapper_user"),
        password=os.getenv("MAPPER_DATABASE_PASSWORD", ""),
        ssl_mode=os.getenv("MAPPER_DATABASE_SSL_MODE", "require"),
        pool_min_size=int(os.getenv("MAPPER_DATABASE_POOL_MIN", "5")),
        pool_max_size=int(os.getenv("MAPPER_DATABASE_POOL_MAX", "20")),
        command_timeout=int(os.getenv("MAPPER_DATABASE_COMMAND_TIMEOUT", "60")),
        connection_timeout=int(os.getenv("MAPPER_DATABASE_CONNECTION_TIMEOUT", "30")),
    )

    return DatabaseManager(config)
