"""
Azure-Native Storage Manager Implementation

This module provides a comprehensive Azure-native storage manager that integrates
with Azure Database for PostgreSQL, Azure Cache for Redis, Azure Blob Storage,
and Azure Key Vault for the Llama Mapper system.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict

import asyncpg
import redis.asyncio as redis
from azure.storage.blob.aio import BlobServiceClient
from azure.keyvault.secrets.aio import SecretClient
from azure.identity.aio import DefaultAzureCredential
from azure.monitor.opentelemetry import configure_azure_monitor
import structlog

from ..config.settings import AzureStorageConfig, AzureDatabaseConfig, AzureRedisConfig, AzureKeyVaultConfig
from .models import AzureStorageRecord, AzureAuditRecord, AzureTenantConfig

logger = structlog.get_logger(__name__)


@dataclass
class AzureStorageMetrics:
    """Metrics for Azure storage operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    blob_operations: int = 0
    database_operations: int = 0


class AzureStorageManager:
    """
    Azure-native storage manager with integrated services.
    
    Features:
    - Azure Database for PostgreSQL for primary storage
    - Azure Cache for Redis for high-performance caching
    - Azure Blob Storage for immutable storage
    - Azure Key Vault for secrets management
    - Azure Monitor for comprehensive observability
    - Multi-tenant support with row-level security
    - Automatic data lifecycle management
    """
    
    def __init__(
        self,
        storage_config: AzureStorageConfig,
        database_config: AzureDatabaseConfig,
        redis_config: AzureRedisConfig,
        keyvault_config: AzureKeyVaultConfig,
    ):
        """
        Initialize Azure Storage Manager.
        
        Args:
            storage_config: Azure Blob Storage configuration
            database_config: Azure Database for PostgreSQL configuration
            redis_config: Azure Cache for Redis configuration
            keyvault_config: Azure Key Vault configuration
        """
        self.storage_config = storage_config
        self.database_config = database_config
        self.redis_config = redis_config
        self.keyvault_config = keyvault_config
        
        # Azure service clients
        self.blob_client: Optional[BlobServiceClient] = None
        self.keyvault_client: Optional[SecretClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # Metrics tracking
        self.metrics = AzureStorageMetrics()
        
        # Azure Monitor integration
        configure_azure_monitor()
        
        logger.info("Azure Storage Manager initialized", 
                   region=storage_config.region,
                   storage_account=storage_config.storage_account)
    
    async def initialize(self) -> None:
        """Initialize all Azure services."""
        try:
            await self._initialize_blob_storage()
            await self._initialize_key_vault()
            await self._initialize_redis()
            await self._initialize_database()
            await self._create_tables()
            
            logger.info("Azure Storage Manager initialization completed")
        except Exception as e:
            logger.error("Failed to initialize Azure Storage Manager", error=str(e))
            raise
    
    async def _initialize_blob_storage(self) -> None:
        """Initialize Azure Blob Storage client."""
        if self.storage_config.connection_string:
            self.blob_client = BlobServiceClient.from_connection_string(
                self.storage_config.connection_string
            )
        elif self.storage_config.access_key:
            account_url = f"https://{self.storage_config.storage_account}.blob.core.windows.net"
            self.blob_client = BlobServiceClient(
                account_url=account_url,
                credential=self.storage_config.access_key
            )
        else:
            # Use managed identity
            credential = DefaultAzureCredential()
            account_url = f"https://{self.storage_config.storage_account}.blob.core.windows.net"
            self.blob_client = BlobServiceClient(
                account_url=account_url,
                credential=credential
            )
        
        # Ensure container exists
        try:
            await self.blob_client.create_container(
                self.storage_config.container_name,
                public_access=None
            )
            logger.info("Blob storage container created", 
                       container=self.storage_config.container_name)
        except Exception as e:
            if "ContainerAlreadyExists" not in str(e):
                logger.warning("Container creation failed", error=str(e))
    
    async def _initialize_key_vault(self) -> None:
        """Initialize Azure Key Vault client."""
        if self.keyvault_config.use_managed_identity:
            credential = DefaultAzureCredential()
        else:
            # Use client credentials
            from azure.identity.aio import ClientSecretCredential
            credential = ClientSecretCredential(
                tenant_id=self.keyvault_config.tenant_id,
                client_id=self.keyvault_config.client_id,
                client_secret=self.keyvault_config.client_secret
            )
        
        self.keyvault_client = SecretClient(
            vault_url=self.keyvault_config.vault_url,
            credential=credential
        )
        
        logger.info("Key Vault client initialized", 
                   vault_url=self.keyvault_config.vault_url)
    
    async def _initialize_redis(self) -> None:
        """Initialize Azure Cache for Redis client."""
        self.redis_client = redis.Redis(
            host=self.redis_config.host,
            port=self.redis_config.port,
            password=self.redis_config.password,
            ssl=self.redis_config.ssl,
            db=self.redis_config.db,
            max_connections=self.redis_config.max_connections,
            retry_on_timeout=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30
        )
        
        # Test connection
        await self.redis_client.ping()
        logger.info("Redis client initialized", host=self.redis_config.host)
    
    async def _initialize_database(self) -> None:
        """Initialize Azure Database for PostgreSQL connection pool."""
        self.db_pool = await asyncpg.create_pool(
            host=self.database_config.host,
            port=self.database_config.port,
            database=self.database_config.database,
            user=self.database_config.username,
            password=self.database_config.password,
            ssl=self.database_config.ssl_mode,
            min_size=self.database_config.connection_pool_size,
            max_size=self.database_config.max_overflow,
            command_timeout=30,
            server_settings={
                'application_name': 'llama_mapper_storage',
                'timezone': 'UTC'
            }
        )
        
        logger.info("Database connection pool initialized", 
                   host=self.database_config.host,
                   pool_size=self.database_config.connection_pool_size)
    
    async def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        async with self.db_pool.acquire() as conn:
            # Create storage_records table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS storage_records (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_data TEXT NOT NULL,
                    mapped_data JSONB NOT NULL,
                    model_version VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata JSONB,
                    tenant_id VARCHAR(100) NOT NULL,
                    s3_key VARCHAR(500),
                    encrypted BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '90 days',
                    azure_region VARCHAR(50) DEFAULT 'eastus',
                    resource_group VARCHAR(100) DEFAULT 'comply-ai-rg',
                    subscription_id VARCHAR(100)
                );
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_storage_records_tenant_timestamp 
                ON storage_records(tenant_id, timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_storage_records_model_version 
                ON storage_records(model_version);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_storage_records_expires_at 
                ON storage_records(expires_at) WHERE expires_at IS NOT NULL;
            """)
            
            # Enable Row Level Security
            await conn.execute("ALTER TABLE storage_records ENABLE ROW LEVEL SECURITY;")
            
            await conn.execute("""
                CREATE POLICY IF NOT EXISTS tenant_isolation_policy ON storage_records
                    FOR ALL USING (tenant_id = current_setting('app.current_tenant_id', true));
            """)
            
            # Create audit_logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR(255) NOT NULL,
                    user_id VARCHAR(255),
                    action VARCHAR(255) NOT NULL,
                    resource_type VARCHAR(255) NOT NULL,
                    resource_id VARCHAR(255),
                    details JSONB,
                    ip_address INET,
                    user_agent TEXT,
                    azure_region VARCHAR(50) DEFAULT 'eastus',
                    subscription_id VARCHAR(100),
                    resource_group VARCHAR(100),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # Create audit indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_created 
                ON audit_logs(tenant_id, created_at DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_logs_action 
                ON audit_logs(action);
            """)
            
            logger.info("Database tables created successfully")
    
    async def store_record(self, record: AzureStorageRecord) -> str:
        """
        Store record in Azure services.
        
        Args:
            record: Storage record to store
            
        Returns:
            Record ID
        """
        start_time = datetime.utcnow()
        
        try:
            # Store in PostgreSQL (hot storage)
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO storage_records 
                    (id, source_data, mapped_data, model_version, timestamp, metadata, 
                     tenant_id, s3_key, encrypted, azure_region, resource_group, subscription_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    """,
                    record.id, record.source_data, json.dumps(record.mapped_data), 
                    record.model_version, record.timestamp, json.dumps(record.metadata),
                    record.tenant_id, record.s3_key, record.encrypted, 
                    record.azure_region, record.resource_group, record.subscription_id
                )
            
            # Store in Azure Blob Storage (cold storage)
            blob_name = (
                f"mappings/{record.tenant_id}/{record.timestamp.year}/"
                f"{record.timestamp.month:02d}/{record.timestamp.day:02d}/{record.id}.json"
            )
            
            blob_client = self.blob_client.get_blob_client(
                container=self.storage_config.container_name,
                blob=blob_name
            )
            
            await blob_client.upload_blob(
                data=json.dumps(record.mapped_data),
                overwrite=True,
                metadata={
                    "tenant_id": record.tenant_id,
                    "model_version": record.model_version,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                    "azure_region": record.azure_region
                }
            )
            
            # Cache in Redis for fast access
            cache_key = f"mapper:response:{record.tenant_id}:{hash(record.source_data)}"
            await self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(record.mapped_data)
            )
            
            # Update metrics
            self.metrics.successful_requests += 1
            self.metrics.database_operations += 1
            self.metrics.blob_operations += 1
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.metrics.average_latency_ms = (
                (self.metrics.average_latency_ms * (self.metrics.total_requests - 1) + latency) 
                / self.metrics.total_requests
            )
            
            logger.info("Record stored successfully", 
                       record_id=str(record.id),
                       tenant_id=record.tenant_id,
                       latency_ms=latency)
            
            return str(record.id)
            
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error("Failed to store record", 
                        record_id=str(record.id),
                        error=str(e))
            raise
    
    async def retrieve_record(self, record_id: str, tenant_id: str) -> Optional[AzureStorageRecord]:
        """
        Retrieve record from Azure services.
        
        Args:
            record_id: Record ID to retrieve
            tenant_id: Tenant ID for access control
            
        Returns:
            Storage record or None if not found
        """
        try:
            # Try cache first
            cache_key = f"mapper:record:{tenant_id}:{record_id}"
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                self.metrics.cache_hit_rate = (
                    (self.metrics.cache_hit_rate * self.metrics.total_requests + 1) 
                    / (self.metrics.total_requests + 1)
                )
                return json.loads(cached_data)
            
            # Retrieve from database
            async with self.db_pool.acquire() as conn:
                # Set tenant context for RLS
                await conn.execute(f"SET app.current_tenant_id = '{tenant_id}'")
                
                row = await conn.fetchrow(
                    """
                    SELECT id, source_data, mapped_data, model_version, timestamp, 
                           metadata, tenant_id, s3_key, encrypted, created_at, 
                           expires_at, azure_region, resource_group, subscription_id
                    FROM storage_records 
                    WHERE id = $1
                    """,
                    uuid.UUID(record_id)
                )
                
                if not row:
                    return None
                
                record = AzureStorageRecord(
                    id=row['id'],
                    source_data=row['source_data'],
                    mapped_data=json.loads(row['mapped_data']),
                    model_version=row['model_version'],
                    timestamp=row['timestamp'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    tenant_id=row['tenant_id'],
                    s3_key=row['s3_key'],
                    encrypted=row['encrypted'],
                    created_at=row['created_at'],
                    expires_at=row['expires_at'],
                    azure_region=row['azure_region'],
                    resource_group=row['resource_group'],
                    subscription_id=row['subscription_id']
                )
                
                # Cache the result
                await self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour TTL
                    json.dumps(asdict(record), default=str)
                )
                
                self.metrics.database_operations += 1
                return record
                
        except Exception as e:
            logger.error("Failed to retrieve record", 
                        record_id=record_id,
                        tenant_id=tenant_id,
                        error=str(e))
            raise
    
    async def store_audit_log(self, audit_record: AzureAuditRecord) -> None:
        """
        Store audit log in Azure services.
        
        Args:
            audit_record: Audit record to store
        """
        try:
            # Store in PostgreSQL
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_logs 
                    (tenant_id, user_id, action, resource_type, resource_id, details,
                     ip_address, user_agent, azure_region, subscription_id, resource_group)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    audit_record.tenant_id, audit_record.user_id, audit_record.action,
                    audit_record.resource_type, audit_record.resource_id, 
                    json.dumps(audit_record.details),
                    audit_record.ip_address, audit_record.user_agent, 
                    audit_record.azure_region, audit_record.subscription_id, 
                    audit_record.resource_group
                )
            
            # Store in Azure Blob Storage for long-term retention
            blob_name = (
                f"audit/{audit_record.tenant_id}/{audit_record.timestamp.year}/"
                f"{audit_record.timestamp.month:02d}/{audit_record.timestamp.day:02d}/"
                f"{audit_record.event_id}.json"
            )
            
            blob_client = self.blob_client.get_blob_client(
                container=self.storage_config.container_name,
                blob=blob_name
            )
            
            await blob_client.upload_blob(
                data=json.dumps(audit_record.details),
                overwrite=True,
                metadata={
                    "event_id": audit_record.event_id,
                    "tenant_id": audit_record.tenant_id,
                    "action": audit_record.action,
                    "timestamp": audit_record.timestamp.isoformat(),
                    "azure_region": audit_record.azure_region
                }
            )
            
            logger.info("Audit log stored successfully", 
                       event_id=audit_record.event_id,
                       tenant_id=audit_record.tenant_id)
            
        except Exception as e:
            logger.error("Failed to store audit log", 
                        event_id=audit_record.event_id,
                        error=str(e))
            raise
    
    async def get_secret(self, secret_name: str) -> str:
        """
        Get secret from Azure Key Vault.
        
        Args:
            secret_name: Name of the secret
            
        Returns:
            Secret value
        """
        try:
            secret = await self.keyvault_client.get_secret(secret_name)
            logger.debug("Secret retrieved successfully", secret_name=secret_name)
            return secret.value
        except Exception as e:
            logger.error("Failed to retrieve secret", 
                        secret_name=secret_name,
                        error=str(e))
            raise
    
    async def cleanup_expired_records(self) -> int:
        """
        Clean up expired records from hot storage.
        
        Returns:
            Number of records cleaned up
        """
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM storage_records 
                    WHERE expires_at < NOW()
                    """
                )
                
                # Extract number of deleted rows from result
                deleted_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                
                logger.info("Expired records cleaned up", count=deleted_count)
                return deleted_count
                
        except Exception as e:
            logger.error("Failed to cleanup expired records", error=str(e))
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get storage metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (
                self.metrics.successful_requests / self.metrics.total_requests 
                if self.metrics.total_requests > 0 else 0
            ),
            "average_latency_ms": self.metrics.average_latency_ms,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "blob_operations": self.metrics.blob_operations,
            "database_operations": self.metrics.database_operations,
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all Azure services.
        
        Returns:
            Health status of all services
        """
        health_status = {
            "overall": "healthy",
            "services": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check database
        try:
            async with self.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["services"]["database"] = "healthy"
        except Exception as e:
            health_status["services"]["database"] = f"unhealthy: {str(e)}"
            health_status["overall"] = "unhealthy"
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health_status["services"]["redis"] = "healthy"
        except Exception as e:
            health_status["services"]["redis"] = f"unhealthy: {str(e)}"
            health_status["overall"] = "unhealthy"
        
        # Check Blob Storage
        try:
            await self.blob_client.get_account_information()
            health_status["services"]["blob_storage"] = "healthy"
        except Exception as e:
            health_status["services"]["blob_storage"] = f"unhealthy: {str(e)}"
            health_status["overall"] = "unhealthy"
        
        # Check Key Vault
        try:
            await self.keyvault_client.get_properties()
            health_status["services"]["key_vault"] = "healthy"
        except Exception as e:
            health_status["services"]["key_vault"] = f"unhealthy: {str(e)}"
            health_status["overall"] = "unhealthy"
        
        return health_status
    
    async def close(self) -> None:
        """Close all connections and cleanup resources."""
        if self.db_pool:
            await self.db_pool.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.blob_client:
            await self.blob_client.close()
        
        if self.keyvault_client:
            await self.keyvault_client.close()
        
        logger.info("Azure Storage Manager closed")
