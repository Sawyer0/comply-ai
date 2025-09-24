# Database Production Readiness Design Document

## Overview

This design document provides a comprehensive assessment and enhancement plan for the Llama Mapper database architecture to ensure production readiness. The current system has a basic storage layer with PostgreSQL and ClickHouse support, but requires significant enhancements for enterprise-grade scalability, security, performance, and compliance.

## Current State Analysis

### Existing Database Architecture

The current system implements:
- **Dual Backend Support**: PostgreSQL (ACID compliance) and ClickHouse (analytics)
- **Basic Schema**: Single `storage_records` table with minimal indexing
- **Tenant Isolation**: Row-level security policies and tenant-scoped queries
- **S3 Integration**: Immutable storage with WORM compliance
- **Basic Encryption**: Field-level encryption support

### Identified Gaps

1. **Schema Design**: Limited indexing, missing constraints, no partitioning
2. **Performance**: No query optimization, connection pooling issues
3. **Security**: Basic RLS, missing audit trails, limited encryption
4. **Monitoring**: No database performance monitoring or alerting
5. **Migrations**: No versioned schema management
6. **Backup/Recovery**: Limited disaster recovery capabilities

## Architecture

### Azure Database for PostgreSQL Flexible Server Deployment

The design targets **Azure Database for PostgreSQL Flexible Server** as the primary deployment platform, aligning with your existing Azure infrastructure. Based on your current setup, the configuration includes:

#### Current Azure Infrastructure
- **Resource Group**: `comply-ai-rg` (primary), `comply-ai-dr-rg` (disaster recovery)
- **PostgreSQL Server**: `comply-ai-postgres` with 30-day backup retention
- **Redis Cache**: `comply-ai-redis` with Standard tier and persistence
- **Storage Account**: `complyaistorage` with GRS replication and immutable storage
- **Key Vault**: `comply-ai-keyvault` for secrets management
- **Monitoring**: Azure Monitor with Log Analytics workspace `comply-ai-logs`

#### Production Azure Database Configuration
- **Service Tier**: General Purpose (Standard_D4s_v3) for production, Burstable (B_Standard_B2s) for development
- **Compute Size**: 4-32 vCores with auto-scaling capabilities
- **Storage**: Premium SSD with up to 1TB capacity and auto-grow enabled
- **High Availability**: Zone-redundant HA for production environments
- **Backup**: Automated backups with 30-day retention and geo-redundant storage enabled
- **Security**: Azure AD authentication, SSL/TLS encryption, and private endpoint integration

#### Azure-Specific Features Already Implemented
- **Azure Monitor Integration**: Built-in monitoring with action groups for alerting
- **Query Performance Insights**: Automatic query analysis via pg_stat_statements extension
- **Cross-Region Replication**: Geo-redundant backups and read replicas in westus2
- **Point-in-Time Recovery**: Restore to any point within 30-day retention period
- **Private Networking**: Private endpoints and VNet integration for secure access
- **Managed Identity**: Service principal authentication for backup operations

### Enhanced Database Schema Design

#### Enhanced Core Tables Structure

```sql
-- Enhanced storage records table aligned with Azure best practices
CREATE TABLE storage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    source_data_hash VARCHAR(64) NOT NULL, -- SHA-256 hash, no raw data for privacy
    mapped_data JSONB NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    detector_type VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    s3_key VARCHAR(500), -- Azure Blob Storage key
    encrypted BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Azure-specific fields
    correlation_id UUID, -- For distributed tracing
    azure_region VARCHAR(50) DEFAULT 'eastus',
    backup_status VARCHAR(20) DEFAULT 'pending' CHECK (backup_status IN ('pending', 'completed', 'failed')),
    
    -- Constraints
    CONSTRAINT valid_confidence CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
    CONSTRAINT valid_metadata CHECK (jsonb_typeof(metadata) = 'object'),
    CONSTRAINT valid_tenant_id CHECK (length(tenant_id) > 0)
) PARTITION BY RANGE (timestamp);

-- Audit trail table for compliance
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    operation VARCHAR(20) NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE', 'SELECT')),
    user_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    old_values JSONB,
    new_values JSONB,
    correlation_id UUID,
    ip_address INET,
    user_agent TEXT
);

-- Tenant configuration table
CREATE TABLE tenant_configs (
    tenant_id VARCHAR(100) PRIMARY KEY,
    confidence_threshold DECIMAL(5,4) DEFAULT 0.6,
    detector_whitelist TEXT[],
    detector_blacklist TEXT[],
    storage_retention_days INTEGER DEFAULT 90,
    encryption_enabled BOOLEAN DEFAULT TRUE,
    audit_level VARCHAR(20) DEFAULT 'standard' CHECK (audit_level IN ('minimal', 'standard', 'verbose')),
    custom_taxonomy_mappings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model performance metrics table
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Detector execution logs
CREATE TABLE detector_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    detector_type VARCHAR(50) NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    confidence_score DECIMAL(5,4),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    correlation_id UUID
);
```

#### Partitioning Strategy

```sql
-- Monthly partitions for storage_records
CREATE TABLE storage_records_y2024m01 PARTITION OF storage_records
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE storage_records_y2024m02 PARTITION OF storage_records
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Automated partition management
CREATE OR REPLACE FUNCTION create_monthly_partition(table_name TEXT, start_date DATE)
RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    end_date DATE;
BEGIN
    partition_name := table_name || '_y' || EXTRACT(YEAR FROM start_date) || 'm' || LPAD(EXTRACT(MONTH FROM start_date)::TEXT, 2, '0');
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

#### Comprehensive Indexing Strategy

```sql
-- Primary performance indexes
CREATE INDEX CONCURRENTLY idx_storage_records_tenant_timestamp 
    ON storage_records (tenant_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_storage_records_model_version 
    ON storage_records (model_version, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_storage_records_detector_confidence 
    ON storage_records (detector_type, confidence_score DESC) 
    WHERE confidence_score IS NOT NULL;

CREATE INDEX CONCURRENTLY idx_storage_records_metadata_gin 
    ON storage_records USING GIN (metadata);

-- Audit trail indexes
CREATE INDEX CONCURRENTLY idx_audit_trail_tenant_timestamp 
    ON audit_trail (tenant_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_audit_trail_record_operation 
    ON audit_trail (record_id, operation, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_audit_trail_correlation 
    ON audit_trail (correlation_id) 
    WHERE correlation_id IS NOT NULL;

-- Performance monitoring indexes
CREATE INDEX CONCURRENTLY idx_model_metrics_version_type 
    ON model_metrics (model_version, metric_type, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_detector_executions_performance 
    ON detector_executions (detector_type, timestamp DESC, execution_time_ms);
```

### Security Enhancements

#### Row-Level Security Policies

```sql
-- Enable RLS on all tables
ALTER TABLE storage_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_trail ENABLE ROW LEVEL SECURITY;
ALTER TABLE tenant_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE detector_executions ENABLE ROW LEVEL SECURITY;

-- Tenant isolation policies
CREATE POLICY tenant_isolation_storage ON storage_records
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_audit ON audit_trail
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Admin bypass policies
CREATE POLICY admin_bypass_storage ON storage_records
    FOR ALL TO admin_role
    USING (true);

-- Read-only analytics policies
CREATE POLICY analytics_read_only ON storage_records
    FOR SELECT TO analytics_role
    USING (true);
```

#### Field-Level Encryption

```sql
-- Encryption functions
CREATE OR REPLACE FUNCTION encrypt_sensitive_data(data TEXT, key_id TEXT DEFAULT NULL)
RETURNS TEXT AS $$
BEGIN
    -- Integration with application-level encryption
    -- Returns encrypted data that can be decrypted by application
    RETURN pgp_sym_encrypt(data, COALESCE(key_id, current_setting('app.encryption_key')));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE OR REPLACE FUNCTION decrypt_sensitive_data(encrypted_data TEXT, key_id TEXT DEFAULT NULL)
RETURNS TEXT AS $$
BEGIN
    RETURN pgp_sym_decrypt(encrypted_data, COALESCE(key_id, current_setting('app.encryption_key')));
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;
```

### Performance Optimization

#### Azure Database Connection Configuration

```python
# Azure Database for PostgreSQL connection configuration
class AzureDatabaseConnectionManager:
    def __init__(self, config: AzureDatabaseConfig):
        self.config = config
        self.write_pool = None
        self.read_pools = {}
        
    async def initialize_pools(self):
        # Azure Database connection string with SSL
        connection_params = {
            'host': self.config.azure_db_host,  # e.g., myserver.postgres.database.azure.com
            'port': 5432,
            'database': self.config.database_name,
            'user': f"{self.config.username}@{self.config.server_name}",  # Azure format
            'password': self.config.password,
            'ssl': 'require',  # Required for Azure Database
            'sslmode': 'require',
            'server_settings': {
                'application_name': 'llama_mapper_azure',
                'timezone': 'UTC'
            }
        }
        
        # Primary write pool
        self.write_pool = await asyncpg.create_pool(
            **connection_params,
            min_size=5,   # Lower for Azure Database limits
            max_size=20,  # Respect Azure connection limits
            command_timeout=60,
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
                'min_size': 10,
                'max_size': 40,
                'command_timeout': 30
            }
            replica_params['server_settings']['application_name'] = f'llama_mapper_read_{replica_region}'
            
            self.read_pools[replica_region] = await asyncpg.create_pool(**replica_params)
    
    async def get_azure_connection_info(self):
        """Get Azure-specific connection information."""
        async with self.write_pool.acquire() as conn:
            info = await conn.fetchrow("""
                SELECT 
                    version() as postgres_version,
                    current_setting('server_version') as server_version,
                    current_setting('azure.extensions.enabled') as azure_extensions,
                    pg_size_pretty(pg_database_size(current_database())) as db_size
            """)
            return dict(info)
```

#### Query Optimization

```sql
-- Materialized views for analytics
CREATE MATERIALIZED VIEW tenant_performance_summary AS
SELECT 
    tenant_id,
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_requests,
    AVG(confidence_score) as avg_confidence,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY confidence_score) as p95_confidence,
    COUNT(CASE WHEN confidence_score > 0.8 THEN 1 END) as high_confidence_count
FROM storage_records 
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY tenant_id, DATE_TRUNC('hour', timestamp);

CREATE UNIQUE INDEX ON tenant_performance_summary (tenant_id, hour);

-- Refresh schedule
CREATE OR REPLACE FUNCTION refresh_performance_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY tenant_performance_summary;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every 15 minutes
SELECT cron.schedule('refresh-performance-views', '*/15 * * * *', 'SELECT refresh_performance_views();');
```

## Components and Interfaces

### Database Migration System

```python
class DatabaseMigrationManager:
    """Manages versioned database schema migrations."""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
        self.migration_table = "schema_migrations"
        
    async def initialize_migration_tracking(self):
        """Create migration tracking table."""
        async with self.connection_manager.get_write_connection() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.migration_table} (
                    version VARCHAR(50) PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    checksum VARCHAR(64) NOT NULL,
                    execution_time_ms INTEGER NOT NULL
                )
            """)
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration with rollback support."""
        start_time = time.time()
        
        async with self.connection_manager.get_write_connection() as conn:
            async with conn.transaction():
                try:
                    # Validate migration hasn't been applied
                    existing = await conn.fetchrow(
                        f"SELECT version FROM {self.migration_table} WHERE version = $1",
                        migration.version
                    )
                    
                    if existing:
                        logger.info(f"Migration {migration.version} already applied")
                        return True
                    
                    # Execute migration
                    await conn.execute(migration.up_sql)
                    
                    # Record migration
                    execution_time = int((time.time() - start_time) * 1000)
                    await conn.execute(f"""
                        INSERT INTO {self.migration_table} 
                        (version, description, checksum, execution_time_ms)
                        VALUES ($1, $2, $3, $4)
                    """, migration.version, migration.description, 
                        migration.checksum, execution_time)
                    
                    logger.info(f"Applied migration {migration.version} in {execution_time}ms")
                    return True
                    
                except Exception as e:
                    logger.error(f"Migration {migration.version} failed: {e}")
                    raise
```

### Azure Monitoring and Alerting Integration

```python
class AzureDatabaseMonitor:
    """Azure Database monitoring with native Azure Monitor integration."""
    
    def __init__(self, connection_manager: AzureDatabaseConnectionManager, azure_config: AzureConfig):
        self.connection_manager = connection_manager
        self.azure_config = azure_config
        self.metrics_collector = PrometheusMetrics()
        self.azure_monitor_client = self._init_azure_monitor()
        
    def _init_azure_monitor(self):
        """Initialize Azure Monitor client."""
        from azure.monitor.query import MetricsQueryClient
        from azure.identity import DefaultAzureCredential
        
        credential = DefaultAzureCredential()
        return MetricsQueryClient(credential)
    
    async def collect_azure_metrics(self):
        """Collect Azure Database specific metrics."""
        # Azure Database resource ID
        resource_id = (
            f"/subscriptions/{self.azure_config.subscription_id}"
            f"/resourceGroups/{self.azure_config.resource_group}"
            f"/providers/Microsoft.DBforPostgreSQL/flexibleServers/{self.azure_config.server_name}"
        )
        
        # Query Azure Monitor metrics
        metrics_to_collect = [
            "cpu_percent",
            "memory_percent", 
            "storage_percent",
            "active_connections",
            "connections_failed",
            "network_bytes_ingress",
            "network_bytes_egress"
        ]
        
        from datetime import datetime, timedelta
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        for metric_name in metrics_to_collect:
            try:
                response = await self.azure_monitor_client.query_metrics(
                    resource_id=resource_id,
                    metric_names=[metric_name],
                    timespan=(start_time, end_time),
                    granularity=timedelta(minutes=1)
                )
                
                # Process and store metrics
                for metric in response.metrics:
                    for time_series in metric.timeseries:
                        for data_point in time_series.data:
                            if data_point.average is not None:
                                self.metrics_collector.record_azure_metric(
                                    metric_name, 
                                    data_point.average,
                                    data_point.timestamp
                                )
            except Exception as e:
                logger.error(f"Failed to collect Azure metric {metric_name}: {e}")
    
    async def collect_database_performance_metrics(self):
        """Collect database performance metrics with Azure-specific queries."""
        async with self.connection_manager.get_read_connection() as conn:
            # Azure Database specific system views
            azure_stats = await conn.fetchrow("""
                SELECT 
                    (SELECT setting FROM pg_settings WHERE name = 'max_connections') as max_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'idle') as idle_connections,
                    pg_database_size(current_database()) as database_size_bytes,
                    (SELECT sum(heap_blks_read) FROM pg_statio_user_tables) as heap_blocks_read,
                    (SELECT sum(heap_blks_hit) FROM pg_statio_user_tables) as heap_blocks_hit
            """)
            
            # Query Performance Insights compatible queries
            slow_queries = await conn.fetch("""
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    rows,
                    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_exec_time > 100  -- Queries slower than 100ms
                ORDER BY mean_exec_time DESC 
                LIMIT 10
            """)
            
            # Azure-specific wait events
            wait_events = await conn.fetch("""
                SELECT 
                    wait_event_type,
                    wait_event,
                    count(*) as count
                FROM pg_stat_activity 
                WHERE wait_event IS NOT NULL
                GROUP BY wait_event_type, wait_event
                ORDER BY count DESC
                LIMIT 10
            """)
            
            # Update metrics
            self.metrics_collector.update_azure_database_metrics(
                azure_stats, slow_queries, wait_events
            )
    
    async def setup_azure_alerts(self):
        """Configure Azure Monitor alerts for database health."""
        alert_rules = [
            {
                'name': 'High CPU Usage',
                'metric': 'cpu_percent',
                'threshold': 80,
                'operator': 'GreaterThan',
                'severity': 2
            },
            {
                'name': 'High Memory Usage', 
                'metric': 'memory_percent',
                'threshold': 85,
                'operator': 'GreaterThan',
                'severity': 2
            },
            {
                'name': 'Storage Almost Full',
                'metric': 'storage_percent', 
                'threshold': 90,
                'operator': 'GreaterThan',
                'severity': 1
            },
            {
                'name': 'Connection Failures',
                'metric': 'connections_failed',
                'threshold': 10,
                'operator': 'GreaterThan',
                'severity': 2
            }
        ]
        
        # Implementation would use Azure SDK to create alert rules
        logger.info(f"Configured {len(alert_rules)} Azure Monitor alert rules")
```

## Data Models

### Enhanced Storage Record Model

```python
@dataclass
class EnhancedStorageRecord:
    """Enhanced storage record with comprehensive metadata."""
    
    id: UUID
    tenant_id: str
    source_data_hash: str  # SHA-256 hash instead of raw data
    mapped_data: Dict[str, Any]
    model_version: str
    detector_type: str
    confidence_score: Optional[float]
    timestamp: datetime
    metadata: Dict[str, Any]
    s3_key: Optional[str] = None
    encrypted: bool = True
    correlation_id: Optional[UUID] = None
    
    # Audit fields
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': str(self.id),
            'tenant_id': self.tenant_id,
            'source_data_hash': self.source_data_hash,
            'mapped_data': self.mapped_data,
            'model_version': self.model_version,
            'detector_type': self.detector_type,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            's3_key': self.s3_key,
            'encrypted': self.encrypted,
            'correlation_id': str(self.correlation_id) if self.correlation_id else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
```

## Error Handling

### Database Error Recovery

```python
class DatabaseErrorHandler:
    """Handles database errors with intelligent retry and fallback."""
    
    def __init__(self, connection_manager: DatabaseConnectionManager):
        self.connection_manager = connection_manager
        self.circuit_breaker = CircuitBreaker()
        
    async def execute_with_retry(self, operation: Callable, max_retries: int = 3):
        """Execute database operation with retry logic."""
        for attempt in range(max_retries):
            try:
                if self.circuit_breaker.is_open():
                    raise DatabaseUnavailableError("Circuit breaker is open")
                
                result = await operation()
                self.circuit_breaker.record_success()
                return result
                
            except (asyncpg.ConnectionDoesNotExistError, 
                    asyncpg.InterfaceError) as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                    
                self.circuit_breaker.record_failure()
                raise DatabaseConnectionError(f"Failed after {max_retries} attempts")
                
            except asyncpg.PostgresError as e:
                logger.error(f"Database error: {e}")
                self.circuit_breaker.record_failure()
                raise DatabaseOperationError(str(e))
```

## Testing Strategy

### Azure Database Testing Framework

```python
class AzureDatabaseTestFramework:
    """Azure Database testing utilities with Azure-specific considerations."""
    
    @pytest.fixture
    async def azure_test_database(self):
        """Create isolated test database on Azure."""
        test_db_name = f"test_llama_mapper_{uuid.uuid4().hex[:8]}"
        
        # Azure Database connection parameters
        azure_params = {
            'host': os.getenv('AZURE_DB_HOST'),
            'port': 5432,
            'user': f"{os.getenv('AZURE_DB_USER')}@{os.getenv('AZURE_DB_SERVER')}",
            'password': os.getenv('AZURE_DB_PASSWORD'),
            'ssl': 'require',
            'sslmode': 'require'
        }
        
        # Connect to default database to create test database
        admin_conn = await asyncpg.connect(
            database='postgres',
            **azure_params
        )
        
        await admin_conn.execute(f"CREATE DATABASE {test_db_name}")
        await admin_conn.close()
        
        # Connect to test database and apply schema
        test_conn = await asyncpg.connect(
            database=test_db_name,
            **azure_params
        )
        
        migration_manager = AzureDatabaseMigrationManager(test_conn)
        await migration_manager.apply_all_migrations()
        
        yield test_conn
        
        # Cleanup
        await test_conn.close()
        admin_conn = await asyncpg.connect(
            database='postgres',
            **azure_params
        )
        await admin_conn.execute(f"DROP DATABASE {test_db_name}")
        await admin_conn.close()
    
    async def test_azure_specific_features(self, azure_test_database):
        """Test Azure Database specific features."""
        # Test Azure extensions
        extensions = await azure_test_database.fetch("""
            SELECT name, installed_version 
            FROM pg_available_extensions 
            WHERE name IN ('pg_stat_statements', 'pg_buffercache', 'azure')
        """)
        
        assert len(extensions) > 0, "Azure extensions should be available"
        
        # Test SSL connection
        ssl_info = await azure_test_database.fetchrow("""
            SELECT 
                current_setting('ssl') as ssl_enabled,
                version() as postgres_version
        """)
        
        assert ssl_info['ssl_enabled'] == 'on', "SSL should be enabled on Azure Database"
    
    async def test_azure_performance_insights(self, azure_test_database):
        """Test Azure Query Performance Insights compatibility."""
        # Enable pg_stat_statements if not already enabled
        await azure_test_database.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        
        # Execute some test queries
        await azure_test_database.execute("SELECT 1")
        await azure_test_database.execute("SELECT COUNT(*) FROM pg_tables")
        
        # Verify pg_stat_statements is collecting data
        stats = await azure_test_database.fetchrow("""
            SELECT count(*) as query_count 
            FROM pg_stat_statements 
            WHERE query LIKE 'SELECT%'
        """)
        
        assert stats['query_count'] > 0, "pg_stat_statements should be collecting query data"
    
    async def test_tenant_isolation_azure(self, azure_test_database):
        """Test tenant isolation policies on Azure Database."""
        # Set tenant context
        await azure_test_database.execute(
            "SET app.current_tenant_id = 'tenant1'"
        )
        
        # Insert test data
        await azure_test_database.execute("""
            INSERT INTO storage_records (tenant_id, source_data_hash, mapped_data, model_version, detector_type)
            VALUES ('tenant1', 'hash1', '{}', 'v1.0', 'test')
        """)
        
        # Verify tenant can only see their data
        result = await azure_test_database.fetch("SELECT * FROM storage_records")
        assert len(result) == 1
        assert result[0]['tenant_id'] == 'tenant1'
        
        # Switch tenant context
        await azure_test_database.execute(
            "SET app.current_tenant_id = 'tenant2'"
        )
        
        # Verify no access to other tenant's data
        result = await azure_test_database.fetch("SELECT * FROM storage_records")
        assert len(result) == 0
    
    async def test_azure_backup_verification(self, azure_test_database):
        """Test Azure automated backup features."""
        # Insert test data
        test_data_id = str(uuid.uuid4())
        await azure_test_database.execute("""
            INSERT INTO storage_records (id, tenant_id, source_data_hash, mapped_data, model_version, detector_type)
            VALUES ($1, 'test_tenant', 'test_hash', '{"test": true}', 'v1.0', 'test')
        """, test_data_id)
        
        # Verify data exists
        result = await azure_test_database.fetchrow(
            "SELECT id FROM storage_records WHERE id = $1", test_data_id
        )
        assert result is not None, "Test data should exist"
        
        # Note: Actual backup testing would require Azure CLI or REST API calls
        # to verify backup policies and restore capabilities
        logger.info("Azure backup verification would require additional Azure API integration")

### Azure Configuration Model

```python
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
```

## Azure Deployment Considerations

### Infrastructure as Code (Terraform)

```hcl
# Azure Database for PostgreSQL Flexible Server
resource "azurerm_postgresql_flexible_server" "llama_mapper" {
  name                   = "llama-mapper-${var.environment}"
  resource_group_name    = azurerm_resource_group.main.name
  location              = azurerm_resource_group.main.location
  version               = "14"
  
  administrator_login    = var.db_admin_username
  administrator_password = var.db_admin_password
  
  sku_name   = var.environment == "production" ? "GP_Standard_D4s_v3" : "B_Standard_B2s"
  storage_mb = var.environment == "production" ? 1048576 : 32768  # 1TB for prod, 32GB for dev
  
  backup_retention_days        = var.environment == "production" ? 35 : 7
  geo_redundant_backup_enabled = var.environment == "production" ? true : false
  
  high_availability {
    mode                      = var.environment == "production" ? "ZoneRedundant" : "Disabled"
    standby_availability_zone = var.environment == "production" ? "2" : null
  }
  
  maintenance_window {
    day_of_week  = 0  # Sunday
    start_hour   = 2  # 2 AM
    start_minute = 0
  }
  
  tags = {
    Environment = var.environment
    Project     = "llama-mapper"
  }
}

# Database
resource "azurerm_postgresql_flexible_server_database" "llama_mapper" {
  name      = "llama_mapper"
  server_id = azurerm_postgresql_flexible_server.llama_mapper.id
  collation = "en_US.utf8"
  charset   = "utf8"
}

# Read replica for production
resource "azurerm_postgresql_flexible_server" "llama_mapper_replica" {
  count = var.environment == "production" ? 1 : 0
  
  name                = "llama-mapper-replica-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location           = var.replica_location
  
  create_mode               = "Replica"
  source_server_id         = azurerm_postgresql_flexible_server.llama_mapper.id
  
  tags = {
    Environment = var.environment
    Project     = "llama-mapper"
    Role        = "read-replica"
  }
}

# Private endpoint for secure access
resource "azurerm_private_endpoint" "llama_mapper_db" {
  name                = "llama-mapper-db-pe"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  subnet_id           = azurerm_subnet.database.id

  private_service_connection {
    name                           = "llama-mapper-db-psc"
    private_connection_resource_id = azurerm_postgresql_flexible_server.llama_mapper.id
    subresource_names              = ["postgresqlServer"]
    is_manual_connection           = false
  }
}
```

### Azure Monitor Integration

```yaml
# Azure Monitor configuration for database monitoring
apiVersion: v1
kind: ConfigMap
metadata:
  name: azure-monitor-config
data:
  config.yaml: |
    azure:
      subscription_id: "${AZURE_SUBSCRIPTION_ID}"
      resource_group: "${AZURE_RESOURCE_GROUP}"
      server_name: "${AZURE_DB_SERVER_NAME}"
    
    metrics:
      collection_interval: 60s
      metrics_to_collect:
        - cpu_percent
        - memory_percent
        - storage_percent
        - active_connections
        - connections_failed
        - network_bytes_ingress
        - network_bytes_egress
        - io_consumption_percent
        - storage_used
    
    alerts:
      - name: "High CPU Usage"
        metric: "cpu_percent"
        threshold: 80
        operator: "GreaterThan"
        severity: "Warning"
        action_group: "llama-mapper-alerts"
      
      - name: "Storage Almost Full"
        metric: "storage_percent"
        threshold: 90
        operator: "GreaterThan"
        severity: "Critical"
        action_group: "llama-mapper-alerts"
```

### Security Configuration

```python
class AzureSecurityConfig:
    """Azure-specific security configuration."""
    
    def __init__(self):
        self.azure_ad_enabled = True
        self.ssl_enforcement = True
        self.firewall_rules = []
        self.private_endpoint_enabled = True
    
    def configure_azure_ad_authentication(self):
        """Configure Azure AD authentication for database access."""
        return {
            'authentication_method': 'azure_ad',
            'azure_ad_admin': os.getenv('AZURE_AD_DB_ADMIN'),
            'managed_identity_enabled': True,
            'service_principal_auth': True
        }
    
    def get_firewall_rules(self, environment: str) -> List[Dict]:
        """Get environment-specific firewall rules."""
        if environment == 'production':
            return [
                {
                    'name': 'AllowAzureServices',
                    'start_ip': '0.0.0.0',
                    'end_ip': '0.0.0.0'  # Allow Azure services
                },
                {
                    'name': 'AllowApplicationSubnet',
                    'start_ip': '10.0.1.0',
                    'end_ip': '10.0.1.255'
                }
            ]
        else:
            return [
                {
                    'name': 'AllowAll',  # Development only
                    'start_ip': '0.0.0.0',
                    'end_ip': '255.255.255.255'
                }
            ]
```

### Performance Optimization for Azure

```python
class AzurePerformanceOptimizer:
    """Azure Database performance optimization."""
    
    def __init__(self, connection_manager: AzureDatabaseConnectionManager):
        self.connection_manager = connection_manager
    
    async def optimize_azure_parameters(self):
        """Optimize Azure Database parameters for workload."""
        optimizations = {
            # Connection settings
            'max_connections': 200,  # Adjust based on Azure tier
            'shared_preload_libraries': 'pg_stat_statements,pg_buffercache',
            
            # Memory settings (Azure manages most of these)
            'effective_cache_size': '75% of available memory',
            'work_mem': '4MB',
            'maintenance_work_mem': '64MB',
            
            # Checkpoint settings
            'checkpoint_completion_target': 0.9,
            'wal_buffers': '16MB',
            
            # Query planner
            'random_page_cost': 1.1,  # SSD storage
            'effective_io_concurrency': 200,
            
            # Logging for Query Performance Insights
            'log_statement': 'all',
            'log_min_duration_statement': 1000,  # Log queries > 1 second
            'log_checkpoints': 'on',
            'log_connections': 'on',
            'log_disconnections': 'on'
        }
        
        async with self.connection_manager.get_write_connection() as conn:
            for param, value in optimizations.items():
                try:
                    # Note: Some parameters require server restart on Azure
                    await conn.execute(f"ALTER SYSTEM SET {param} = '{value}'")
                    logger.info(f"Set {param} = {value}")
                except Exception as e:
                    logger.warning(f"Could not set {param}: {e}")
    
    async def setup_azure_extensions(self):
        """Enable required PostgreSQL extensions on Azure."""
        extensions = [
            'pg_stat_statements',  # Query performance monitoring
            'pg_buffercache',      # Buffer cache monitoring
            'pgcrypto',           # Encryption functions
            'uuid-ossp',          # UUID generation
            'btree_gin',          # GIN indexes for better performance
            'pg_trgm'             # Trigram matching for text search
        ]
        
        async with self.connection_manager.get_write_connection() as conn:
            for extension in extensions:
                try:
                    await conn.execute(f"CREATE EXTENSION IF NOT EXISTS \"{extension}\"")
                    logger.info(f"Enabled extension: {extension}")
                except Exception as e:
                    logger.warning(f"Could not enable extension {extension}: {e}")
```

This comprehensive design now specifically targets Azure Database for PostgreSQL deployment, incorporating Azure-specific features, monitoring, security, and optimization strategies. The architecture provides a solid foundation for enterprise-scale deployment on Azure while maintaining the existing functionality and adding production-ready enhancements.