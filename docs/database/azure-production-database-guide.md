# Azure Database Production Readiness Guide

## Overview

This guide provides comprehensive documentation for the enhanced Azure Database for PostgreSQL implementation in the Llama Mapper platform. The implementation includes production-ready features for scalability, security, compliance, and monitoring.

## Architecture Overview

### Components

1. **Enhanced Storage Manager** (`src/llama_mapper/storage/manager/enhanced_manager.py`)
   - Combines legacy compatibility with Azure-specific enhancements
   - Supports both PostgreSQL and ClickHouse backends
   - Integrated monitoring and security features

2. **Azure Database Connection Manager** (`src/llama_mapper/storage/database/azure_config.py`)
   - Azure Database for PostgreSQL Flexible Server support
   - Connection pooling with read replicas
   - Circuit breaker pattern for resilience

3. **Database Migration System** (`src/llama_mapper/storage/database/migrations.py`)
   - Versioned schema migrations
   - Rollback capabilities
   - Migration validation and tracking

4. **Enhanced Security** (`src/llama_mapper/storage/security/`)
   - Field-level encryption with Azure Key Vault
   - Enhanced row-level security policies
   - Input sanitization and validation

5. **Azure Monitoring** (`src/llama_mapper/storage/monitoring/`)
   - Azure Monitor integration
   - Performance analysis and alerting
   - Query optimization recommendations

## Database Schema Enhancements

### New Tables

#### Enhanced storage_records
```sql
-- New fields added to existing table
ALTER TABLE storage_records 
ADD COLUMN source_data_hash VARCHAR(64),        -- SHA-256 hash for privacy
ADD COLUMN detector_type VARCHAR(50),
ADD COLUMN confidence_score DECIMAL(5,4),
ADD COLUMN correlation_id UUID,
ADD COLUMN azure_region VARCHAR(50) DEFAULT 'eastus',
ADD COLUMN backup_status VARCHAR(20) DEFAULT 'pending',
ADD COLUMN updated_at TIMESTAMPTZ DEFAULT NOW();
```

#### audit_trail
```sql
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id UUID NOT NULL,
    operation VARCHAR(20) NOT NULL,
    user_id VARCHAR(100),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    old_values JSONB,
    new_values JSONB,
    correlation_id UUID,
    ip_address INET,
    user_agent TEXT
);
```

#### tenant_configs
```sql
CREATE TABLE tenant_configs (
    tenant_id VARCHAR(100) PRIMARY KEY,
    confidence_threshold DECIMAL(5,4) DEFAULT 0.6,
    detector_whitelist TEXT[],
    detector_blacklist TEXT[],
    storage_retention_days INTEGER DEFAULT 90,
    encryption_enabled BOOLEAN DEFAULT TRUE,
    audit_level VARCHAR(20) DEFAULT 'standard',
    custom_taxonomy_mappings JSONB DEFAULT '{}'
);
```

#### model_metrics
```sql
CREATE TABLE model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,6) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
```

#### detector_executions
```sql
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

### Performance Indexes

```sql
-- Composite indexes for common query patterns
CREATE INDEX idx_storage_records_tenant_detector_timestamp 
ON storage_records (tenant_id, detector_type, timestamp DESC);

CREATE INDEX idx_storage_records_confidence_timestamp 
ON storage_records (confidence_score DESC, timestamp DESC) 
WHERE confidence_score IS NOT NULL;

CREATE INDEX idx_audit_trail_tenant_timestamp 
ON audit_trail (tenant_id, timestamp DESC);

CREATE INDEX idx_detector_executions_performance 
ON detector_executions (detector_type, timestamp DESC, execution_time_ms);
```

## Configuration

### Azure Database Configuration

Add to your `config.yaml`:

```yaml
storage:
  azure:
    subscription_id: "your-subscription-id"
    resource_group: "comply-ai-rg"
    server_name: "comply-ai-postgres"
    azure_db_host: "comply-ai-postgres.postgres.database.azure.com"
    key_vault_url: "https://comply-ai-keyvault.vault.azure.net/"
    
    # Connection settings
    ssl_mode: "require"
    connection_timeout: 30
    command_timeout: 60
    
    # Pool settings
    min_pool_size: 5
    max_pool_size: 20
    
    # Read replicas (optional)
    read_replica_regions:
      - "westus2"
    
    # Monitoring
    enable_azure_monitor: true
    log_analytics_workspace_id: "your-workspace-id"
    
    # Backup
    backup_retention_days: 30
    geo_redundant_backup: true

  # Enhanced database settings
  enable_ssl: true
  enable_rls: true
  enable_audit_logging: true
  field_encryption_enabled: true
```

### Environment Variables

```bash
# Azure Database connection
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=comply-ai-rg
AZURE_DB_SERVER=comply-ai-postgres
AZURE_DB_HOST=comply-ai-postgres.postgres.database.azure.com
AZURE_DB_USER=llama_mapper_user
AZURE_DB_PASSWORD=your-secure-password

# Azure Key Vault for secrets
AZURE_KEY_VAULT_URL=https://comply-ai-keyvault.vault.azure.net/

# Monitoring
AZURE_LOG_ANALYTICS_WORKSPACE_ID=your-workspace-id
```

## Usage Examples

### Basic Usage with Enhanced Storage Manager

```python
from llama_mapper.storage.manager.enhanced_manager import EnhancedStorageManager
from llama_mapper.config.settings import StorageConfig

# Initialize enhanced storage manager
config = StorageConfig(
    storage_backend="postgresql",
    db_host="comply-ai-postgres.postgres.database.azure.com",
    db_name="llama_mapper",
    enable_ssl=True,
    field_encryption_enabled=True
)

storage_manager = EnhancedStorageManager(config)
await storage_manager.initialize()

# Store record with enhanced features
from llama_mapper.storage.manager.models import StorageRecord
import uuid

record = StorageRecord(
    id=str(uuid.uuid4()),
    source_data="test data",
    source_data_hash="sha256hash",
    mapped_data='{"category": "test"}',
    model_version="v2.0",
    detector_type="test_detector",
    confidence_score=0.95,
    tenant_id="tenant1",
    correlation_id=str(uuid.uuid4())
)

await storage_manager.store_record_enhanced(record)
```

### Database Migration

```python
from llama_mapper.storage.database.migrations import DatabaseMigrationManager
from llama_mapper.storage.database.azure_config import AzureDatabaseConnectionManager

# Initialize migration manager
connection_manager = AzureDatabaseConnectionManager(azure_config)
await connection_manager.initialize_pools()

migration_manager = DatabaseMigrationManager(connection_manager)
await migration_manager.initialize_migration_tracking()

# Apply migrations
results = await migration_manager.apply_all_migrations()
print(f"Applied {len(results['applied'])} migrations")

# Check migration status
status = await migration_manager.get_migration_status()
print(f"Pending migrations: {status['pending_count']}")
```

### Security Features

```python
from llama_mapper.storage.security.encryption import FieldEncryption
from llama_mapper.storage.security.encryption import EnhancedRowLevelSecurity

# Field-level encryption
encryption = FieldEncryption(key_vault_url="https://your-keyvault.vault.azure.net/")
await encryption.initialize()

# Encrypt sensitive data
sensitive_data = {"name": "John Doe", "email": "john@example.com"}
encrypted_data = encryption.encrypt_dict(sensitive_data, ["name", "email"])

# Enhanced RLS policies
rls_manager = EnhancedRowLevelSecurity(connection_manager)
await rls_manager.create_enhanced_rls_policies()

# Validate tenant isolation
validation_result = await rls_manager.validate_tenant_isolation("tenant1")
```

### Performance Monitoring

```python
from llama_mapper.storage.monitoring.azure_monitor import AzureDatabaseMonitor

# Initialize monitoring
monitor = AzureDatabaseMonitor(connection_manager, azure_config)
await monitor.start_monitoring()

# Get performance metrics
summary = await monitor.get_monitoring_summary()
print(f"Health score: {summary['health_score']}")

# Analyze query performance
from llama_mapper.storage.optimization.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer(connection_manager)
analysis = await optimizer.analyze_query_plans()
recommendations = await optimizer.get_optimization_recommendations()
```

## Deployment Guide

### 1. Prerequisites

- Azure Database for PostgreSQL Flexible Server
- Azure Key Vault for secrets management
- Azure Monitor/Log Analytics workspace
- Required Python packages:
  ```bash
  pip install azure-identity azure-keyvault-secrets azure-monitor-query
  pip install asyncpg prometheus-client structlog
  ```

### 2. Database Setup

1. **Create Azure Database for PostgreSQL Flexible Server**
   ```bash
   az postgres flexible-server create \
     --resource-group comply-ai-rg \
     --name comply-ai-postgres \
     --location eastus \
     --admin-user llama_mapper_admin \
     --admin-password <secure-password> \
     --sku-name Standard_D4s_v3 \
     --storage-size 1024 \
     --tier GeneralPurpose \
     --high-availability ZoneRedundant \
     --backup-retention 30
   ```

2. **Configure firewall rules**
   ```bash
   az postgres flexible-server firewall-rule create \
     --resource-group comply-ai-rg \
     --name comply-ai-postgres \
     --rule-name AllowAzureServices \
     --start-ip-address 0.0.0.0 \
     --end-ip-address 0.0.0.0
   ```

3. **Create database and user**
   ```sql
   CREATE DATABASE llama_mapper;
   CREATE USER llama_mapper_user WITH PASSWORD 'secure-password';
   GRANT ALL PRIVILEGES ON DATABASE llama_mapper TO llama_mapper_user;
   ```

### 3. Application Deployment

1. **Set environment variables**
2. **Run database migrations**
   ```python
   from llama_mapper.storage.database.migrations import create_production_migrations
   
   # This will be done automatically during storage manager initialization
   ```

3. **Initialize monitoring**
   ```python
   # Monitoring starts automatically with EnhancedStorageManager
   ```

### 4. Performance Optimization

```python
from llama_mapper.storage.optimization.query_optimizer import QueryOptimizer

optimizer = QueryOptimizer(connection_manager)

# Create materialized views
await optimizer.create_materialized_views()

# Create performance indexes
await optimizer.create_performance_indexes()

# Setup partitioning
await optimizer.create_partitions()

# Optimize Azure parameters
await optimizer.optimize_azure_parameters()

# Enable extensions
await optimizer.enable_azure_extensions()
```

## Monitoring and Alerting

### Azure Monitor Integration

The system automatically integrates with Azure Monitor to collect:

- **Database Metrics**: CPU, memory, storage, connections
- **Query Performance**: Slow queries, execution times, cache hit ratios
- **Custom Metrics**: Business-specific metrics via Prometheus

### Alert Configuration

Recommended alerts to set up in Azure Monitor:

```yaml
alerts:
  - name: "High CPU Usage"
    metric: "cpu_percent"
    threshold: 80
    severity: "Warning"
  
  - name: "Storage Almost Full"
    metric: "storage_percent"
    threshold: 90
    severity: "Critical"
  
  - name: "Connection Failures"
    metric: "connections_failed"
    threshold: 10
    severity: "Warning"
```

### Performance Dashboards

Access performance insights through:

1. **Azure Portal**: Query Performance Insights
2. **Grafana**: Custom dashboards using Prometheus metrics
3. **Application Monitoring**: Built-in performance analysis

## Security Features

### Field-Level Encryption

- **Automatic**: Sensitive fields encrypted before storage
- **Key Management**: Azure Key Vault integration
- **Rotation**: Automated key rotation support

### Row-Level Security

- **Tenant Isolation**: Automatic tenant data separation
- **Role-Based Access**: Admin, application, and analytics roles
- **Audit Compliance**: All data access logged

### Input Validation

- **SQL Injection Protection**: Parameterized queries and input sanitization
- **XSS Prevention**: HTML escaping and pattern detection
- **Data Length Limits**: Configurable input size restrictions

## Compliance Features

### Audit Trail

All database operations are automatically logged with:
- User identification
- Timestamp and correlation ID
- Before/after values for changes
- IP address and user agent

### Data Retention

- **Configurable**: Per-tenant retention policies
- **Automated Cleanup**: Scheduled removal of expired data
- **Backup Integration**: Coordinated with Azure backup system

### Privacy Compliance

- **Data Minimization**: Hash-based storage for sensitive data
- **Encryption**: Field-level encryption for PII
- **Access Controls**: Strict tenant isolation and role-based access

## Troubleshooting

### Common Issues

1. **Connection Failures**
   ```python
   # Check connection health
   health = await connection_manager.health_check()
   print(health)
   ```

2. **Migration Failures**
   ```python
   # Check migration status
   status = await migration_manager.get_migration_status()
   print(status['failed_migrations'])
   ```

3. **Performance Issues**
   ```python
   # Analyze query performance
   analysis = await optimizer.analyze_query_plans()
   recommendations = await optimizer.get_optimization_recommendations()
   ```

### Monitoring Commands

```python
# Get comprehensive health status
health_status = await storage_manager.get_health_status()

# Analyze database performance
performance = await storage_manager.analyze_performance(hours=24)

# Get tenant-specific metrics
metrics = await storage_manager.get_performance_metrics("tenant1")
```

## Best Practices

### Performance

1. **Use Connection Pooling**: Leverage built-in Azure connection pooling
2. **Monitor Query Performance**: Regular analysis of slow queries
3. **Optimize Indexes**: Use provided performance indexes and monitoring
4. **Partition Large Tables**: Implement time-based partitioning

### Security

1. **Enable SSL**: Always use SSL connections to Azure Database
2. **Use Managed Identity**: Prefer Azure Managed Identity over passwords
3. **Regular Key Rotation**: Implement automated key rotation
4. **Monitor Access**: Review audit logs regularly

### Monitoring

1. **Set Up Alerts**: Configure Azure Monitor alerts for critical metrics
2. **Dashboard Monitoring**: Use Grafana or Azure dashboards
3. **Performance Reviews**: Regular query performance analysis
4. **Capacity Planning**: Monitor growth trends and plan scaling

### Maintenance

1. **Regular Backups**: Verify automated backup functionality
2. **Update Statistics**: Keep database statistics current
3. **Monitor Partitions**: Clean up old partitions as needed
4. **Review Configurations**: Periodic review of Azure Database parameters

## Support and Maintenance

### Automated Tasks

- **Health Monitoring**: Continuous health checks and alerting
- **Performance Analysis**: Regular query performance analysis
- **Backup Verification**: Automated backup status monitoring
- **Capacity Monitoring**: Storage and connection usage tracking

### Manual Tasks

- **Quarterly Reviews**: Performance optimization and capacity planning
- **Security Audits**: Review access patterns and security configurations
- **Documentation Updates**: Keep configuration documentation current
- **Disaster Recovery Testing**: Periodic DR testing procedures

This production-ready implementation provides enterprise-grade database functionality with Azure-specific optimizations, comprehensive monitoring, and robust security features.
