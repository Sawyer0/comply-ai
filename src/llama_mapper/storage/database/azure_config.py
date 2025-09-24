"""Azure Database for PostgreSQL configuration and connection management."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import asyncpg
import structlog

try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    AZURE_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    SecretClient = None
    AZURE_AVAILABLE = False

from ...manager.models import (
    DatabaseConnectionError, 
    DatabaseUnavailableError,
    DatabaseOperationError
)

logger = structlog.get_logger(__name__)


@dataclass
class AzureDatabaseConfig:
    """Azure Database for PostgreSQL configuration."""
    
    # Azure-specific settings
    subscription_id: str
    resource_group: str
    server_name: str
    azure_db_host: str  # e.g., myserver.postgres.database.azure.com
    
    # Database settings
    database_name: str = "llama_mapper"
    username: str = "llama_mapper_user"
    password: str = field(repr=False)  # Don't log password
    
    # Connection settings
    ssl_mode: str = "require"
    connection_timeout: int = 30
    command_timeout: int = 60
    
    # Pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    
    # Read replicas
    read_replica_regions: List[str] = field(default_factory=list)
    
    # Azure Monitor settings
    enable_azure_monitor: bool = True
    log_analytics_workspace_id: Optional[str] = None
    
    # Backup settings
    backup_retention_days: int = 7
    geo_redundant_backup: bool = True
    
    def get_connection_string(self) -> str:
        """Get Azure Database connection string."""
        return (
            f"postgresql://{self.username}@{self.server_name}:{self.password}"
            f"@{self.azure_db_host}:5432/{self.database_name}"
            f"?sslmode={self.ssl_mode}"
        )


class CircuitBreaker:
    """Circuit breaker for database connections."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "half-open"
                return False
            return True
        return False
    
    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "Circuit breaker opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold
            )


class AzureDatabaseConnectionManager:
    """Azure Database for PostgreSQL connection management."""
    
    def __init__(self, config: AzureDatabaseConfig):
        self.config = config
        self.write_pool: Optional[asyncpg.Pool] = None
        self.read_pools: Dict[str, asyncpg.Pool] = {}
        self.circuit_breaker = CircuitBreaker()
        self.credential = DefaultAzureCredential()
        self._replica_index = 0
        
    async def initialize_pools(self):
        """Initialize connection pools."""
        try:
            # Azure Database connection string with SSL
            connection_params = {
                'host': self.config.azure_db_host,
                'port': 5432,
                'database': self.config.database_name,
                'user': f"{self.config.username}@{self.config.server_name}",
                'password': self.config.password,
                'ssl': 'require',
                'sslmode': 'require',
                'server_settings': {
                    'application_name': 'llama_mapper_azure',
                    'timezone': 'UTC'
                }
            }
            
            # Primary write pool
            self.write_pool = await asyncpg.create_pool(
                **connection_params,
                min_size=self.config.min_pool_size,
                max_size=self.config.max_pool_size,
                command_timeout=self.config.command_timeout,
                server_settings={
                    **connection_params['server_settings'],
                    'application_name': 'llama_mapper_write'
                }
            )
            
            # Read replica pools (if configured)
            for replica_region in self.config.read_replica_regions:
                replica_host = f"{self.config.server_name}-{replica_region}.postgres.database.azure.com"
                replica_params = {
                    **connection_params,
                    'host': replica_host,
                    'min_size': min(10, self.config.max_pool_size),
                    'max_size': self.config.max_pool_size * 2,
                    'command_timeout': 30
                }
                replica_params['server_settings']['application_name'] = f'llama_mapper_read_{replica_region}'
                
                self.read_pools[replica_region] = await asyncpg.create_pool(**replica_params)
            
            logger.info(
                "Azure Database connection pools initialized",
                write_pool=bool(self.write_pool),
                read_replicas=len(self.read_pools)
            )
            
        except Exception as e:
            logger.error("Failed to initialize Azure Database pools", error=str(e))
                    raise DatabaseConnectionError(f"Failed to initialize connection pools: {e}") from e
    
    async def get_write_connection(self):
        """Get connection for write operations."""
        if not self.write_pool:
            raise DatabaseConnectionError("Write pool not initialized")
        return self.write_pool.acquire()
    
    async def get_read_connection(self, preferred_replica: Optional[str] = None):
        """Get connection for read operations with load balancing."""
        # Try preferred replica first
        if preferred_replica and preferred_replica in self.read_pools:
            return self.read_pools[preferred_replica].acquire()
        
        # Fall back to any available read replica
        if self.read_pools:
            replica_name = self._select_best_replica()
            return self.read_pools[replica_name].acquire()
        
        # Fall back to write pool for reads
        if self.write_pool:
            return self.write_pool.acquire()
        
        raise DatabaseConnectionError("No database connections available")
    
    def _select_best_replica(self) -> str:
        """Select best read replica based on round-robin."""
        replica_names = list(self.read_pools.keys())
        if not replica_names:
            raise DatabaseConnectionError("No read replicas available")
        
        replica_name = replica_names[self._replica_index]
        self._replica_index = (self._replica_index + 1) % len(replica_names)
        
        return replica_name
    
    async def get_azure_connection_info(self):
        """Get Azure-specific connection information."""
        if not self.write_pool:
            raise DatabaseConnectionError("Write pool not initialized")
            
        async with self.write_pool.acquire() as conn:
            info = await conn.fetchrow("""
                SELECT 
                    version() as postgres_version,
                    current_setting('server_version') as server_version,
                    current_setting('ssl', true) as ssl_enabled,
                    pg_size_pretty(pg_database_size(current_database())) as db_size,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_connections
            """)
            return dict(info)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections."""
        health_status = {}
        
        # Check write pool
        try:
            if self.write_pool:
                async with self.write_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                health_status['write_db'] = True
            else:
                health_status['write_db'] = False
        except (asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionError) as e:
            logger.warning("Write database health check failed", error=str(e))
            health_status['write_db'] = False
        
        # Check read replicas
        for replica_name, pool in self.read_pools.items():
            try:
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                health_status[f'read_replica_{replica_name}'] = True
            except (asyncpg.PostgresError, asyncpg.InterfaceError, ConnectionError) as e:
                logger.warning(
                    "Read replica health check failed", 
                    replica=replica_name, 
                    error=str(e)
                )
                health_status[f'read_replica_{replica_name}'] = False
        
        return health_status
    
    async def close_pools(self):
        """Close all connection pools."""
        if self.write_pool:
            await self.write_pool.close()
        
        for pool in self.read_pools.values():
            await pool.close()
        
        logger.info("Azure Database connection pools closed")


class DatabaseErrorHandler:
    """Handles database errors with intelligent retry and fallback."""
    
    def __init__(self, connection_manager: AzureDatabaseConnectionManager):
        self.connection_manager = connection_manager
        self.circuit_breaker = connection_manager.circuit_breaker
        
    async def execute_with_retry(self, operation: Any, max_retries: int = 3):
        """Execute database operation with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.circuit_breaker.is_open():
                    raise DatabaseUnavailableError("Circuit breaker is open")
                
                result = await operation()
                self.circuit_breaker.record_success()
                return result
                
            except (asyncpg.ConnectionDoesNotExistError, asyncpg.InterfaceError) as e:
                logger.warning(
                    "Database connection error", 
                    attempt=attempt + 1, 
                    error=str(e)
                )
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
                self.circuit_breaker.record_failure()
                raise DatabaseConnectionError(f"Failed after {max_retries} attempts: {e}") from e
                
            except asyncpg.PostgresError as e:
                logger.error("Database operation error", error=str(e))
                self.circuit_breaker.record_failure()
                raise DatabaseOperationError(str(e)) from e
    
    async def get_secrets_from_keyvault(self, key_vault_url: str, secret_name: str) -> str:
        """Get database credentials from Azure Key Vault."""
        try:
            client = SecretClient(vault_url=key_vault_url, credential=self.connection_manager.credential)
            secret = client.get_secret(secret_name)
            return secret.value
        except Exception as e:
            logger.error("Failed to get secret from Key Vault", secret=secret_name, error=str(e))
            raise ValueError(f"Failed to get secret {secret_name}") from e
