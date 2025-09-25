"""
Connection Pool Manager for Mapper Service

Manages multiple database connections and connection pooling strategies
following Single Responsibility Principle.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel

from .database_manager import DatabaseManager, DatabaseConfig

# Import Redis at module level with availability check
try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ConnectionPoolConfig(BaseModel):
    """Connection pool configuration."""

    primary_db: DatabaseConfig
    read_replicas: List[DatabaseConfig] = []
    redis_config: Optional[Dict[str, str]] = None
    connection_retry_attempts: int = 3
    connection_retry_delay: float = 1.0
    health_check_interval: int = 60


class ConnectionPoolManager:
    """
    Manages multiple database connections and connection pooling.

    Responsible for:
    - Managing primary and read replica connections
    - Connection health monitoring
    - Load balancing across read replicas
    - Connection failover and recovery
    """

    def __init__(self, config: ConnectionPoolConfig):
        self.config = config
        self.logger = logger.bind(component="connection_pool_manager")

        # Database managers
        self.primary_db: Optional[DatabaseManager] = None
        self.read_replicas: List[DatabaseManager] = []
        self.redis_manager: Optional[Any] = None

        # Connection state (combine related attributes to reduce count)
        self._state = {
            "replica_index": 0,
            "health_check_task": None,
            "is_initialized": False,
        }

    async def initialize(self) -> None:
        """Initialize all database connections."""
        try:
            self.logger.info("Initializing connection pool manager")

            # Initialize primary database
            self.primary_db = DatabaseManager(self.config.primary_db)
            await self.primary_db.initialize()

            # Initialize read replicas
            for i, replica_config in enumerate(self.config.read_replicas):
                replica_manager = DatabaseManager(replica_config)
                await replica_manager.initialize()
                self.read_replicas.append(replica_manager)
                self.logger.info(f"Initialized read replica {i+1}")

            # Initialize Redis if configured
            if self.config.redis_config:
                await self._initialize_redis()

            # Start health check task
            self._state["health_check_task"] = asyncio.create_task(
                self._health_check_loop()
            )

            self._state["is_initialized"] = True
            self.logger.info(
                "Connection pool manager initialized successfully",
                primary_db=True,
                read_replicas=len(self.read_replicas),
                redis_enabled=self.redis_manager is not None,
            )

        except (RuntimeError, OSError, ConnectionError) as e:
            self.logger.error(
                "Failed to initialize connection pool manager", error=str(e)
            )
            await self.close()
            raise

    async def close(self) -> None:
        """Close all database connections."""
        self.logger.info("Closing connection pool manager")

        # Cancel health check task
        if self._state["health_check_task"]:
            self._state["health_check_task"].cancel()
            try:
                await self._state["health_check_task"]
            except asyncio.CancelledError:
                pass

        # Close primary database
        if self.primary_db:
            await self.primary_db.close()

        # Close read replicas
        for replica in self.read_replicas:
            await replica.close()

        # Close Redis
        if self.redis_manager:
            await self._close_redis()

        self._state["is_initialized"] = False
        self.logger.info("Connection pool manager closed")

    def get_write_db(self) -> DatabaseManager:
        """Get database manager for write operations."""
        if not self.primary_db:
            raise RuntimeError("Primary database not initialized")
        return self.primary_db

    def get_read_db(self, preferred_replica: Optional[int] = None) -> DatabaseManager:
        """Get database manager for read operations with load balancing."""
        # Use primary if no read replicas available
        if not self.read_replicas:
            return self.get_write_db()

        # Use specific replica if requested and available
        if preferred_replica is not None and 0 <= preferred_replica < len(
            self.read_replicas
        ):
            replica = self.read_replicas[preferred_replica]
            if replica.is_connected:
                return replica

        # Round-robin load balancing
        for _ in range(len(self.read_replicas)):
            replica = self.read_replicas[self._state["replica_index"]]
            self._state["replica_index"] = (self._state["replica_index"] + 1) % len(
                self.read_replicas
            )

            if replica.is_connected:
                return replica

        # Fallback to primary if no read replicas are healthy
        self.logger.warning(
            "No healthy read replicas available, falling back to primary"
        )
        return self.get_write_db()

    async def execute_with_retry(
        self, operation, *args, max_retries: int = None, **kwargs
    ):
        """Execute database operation with retry logic."""
        max_retries = max_retries or self.config.connection_retry_attempts

        for attempt in range(max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except (RuntimeError, OSError, ConnectionError, ValueError) as e:
                if attempt == max_retries:
                    self.logger.error(
                        "Database operation failed after all retries",
                        error=str(e),
                        attempts=attempt + 1,
                    )
                    raise

                self.logger.warning(
                    "Database operation failed, retrying",
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )

                await asyncio.sleep(self.config.connection_retry_delay * (2**attempt))

    async def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all database connections."""
        status: Dict[str, Any] = {
            "primary_db": None,
            "read_replicas": [],
            "redis": None,
            "overall_health": "unknown",
        }

        # Check primary database
        if self.primary_db:
            status["primary_db"] = await self.primary_db.health_check()

        # Check read replicas
        for i, replica in enumerate(self.read_replicas):
            replica_status = await replica.health_check()
            replica_status["replica_id"] = i
            status["read_replicas"].append(replica_status)

        # Check Redis
        if self.redis_manager:
            status["redis"] = await self._redis_health_check()

        # Determine overall health
        primary_healthy = (
            status["primary_db"] is not None
            and isinstance(status["primary_db"], dict)
            and status["primary_db"].get("status") == "healthy"
        )
        replicas_healthy = all(
            r.get("status") == "healthy" for r in status["read_replicas"]
        )
        redis_healthy = not self.redis_manager or (
            status["redis"] is not None
            and isinstance(status["redis"], dict)
            and status["redis"].get("status") == "healthy"
        )

        if primary_healthy and replicas_healthy and redis_healthy:
            status["overall_health"] = "healthy"
        elif primary_healthy:
            status["overall_health"] = "degraded"
        else:
            status["overall_health"] = "unhealthy"

        return status

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection."""
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis not available: redis package not installed")
            return

        try:
            redis_config = self.config.redis_config
            if not redis_config:
                return

            self.redis_manager = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=int(redis_config.get("port", 6379)),
                db=int(redis_config.get("db", 0)),
                password=redis_config.get("password"),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            await self.redis_manager.ping()
            self.logger.info("Redis connection initialized successfully")

        except (ConnectionError, OSError, ValueError) as e:
            self.logger.error("Failed to initialize Redis connection", error=str(e))
            self.redis_manager = None

    async def _close_redis(self) -> None:
        """Close Redis connection."""
        if self.redis_manager:
            try:
                await self.redis_manager.close()
                self.logger.info("Redis connection closed")
            except (ConnectionError, OSError) as e:
                self.logger.error("Error closing Redis connection", error=str(e))

    async def _redis_health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            if not self.redis_manager:
                return {"status": "not_configured"}

            start_time = datetime.utcnow()
            await self.redis_manager.ping()
            end_time = datetime.utcnow()

            response_time = (end_time - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "timestamp": datetime.utcnow().isoformat(),
            }

        except (ConnectionError, OSError, ValueError) as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def _health_check_loop(self) -> None:
        """Periodic health check for all connections."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                status = await self.get_connection_status()

                # Log health status
                if status["overall_health"] != "healthy":
                    primary_status = "unknown"
                    if status["primary_db"] and isinstance(status["primary_db"], dict):
                        primary_status = status["primary_db"].get("status", "unknown")

                    healthy_replicas = [
                        r
                        for r in status["read_replicas"]
                        if isinstance(r, dict) and r.get("status") == "healthy"
                    ]

                    self.logger.warning(
                        "Database health check detected issues",
                        overall_health=status["overall_health"],
                        primary_status=primary_status,
                        replica_count=len(healthy_replicas),
                    )
                else:
                    self.logger.debug("Database health check passed")

            except asyncio.CancelledError:
                break
            except (RuntimeError, OSError, ConnectionError) as e:
                self.logger.error("Health check loop error", error=str(e))

    @property
    def is_initialized(self) -> bool:
        """Check if connection pool manager is initialized."""
        return self._state["is_initialized"]
