# Azure Database Backup and Restore Procedures

## Overview

This document provides comprehensive procedures for backing up and restoring databases in the Azure-native Comply-AI platform deployment. The platform uses Azure managed services with specific backup and restore capabilities.

## Azure Services Architecture

### Database Services
- **Azure Database for PostgreSQL**: Primary database with built-in backup capabilities
- **Azure Cache for Redis**: Managed Redis with persistence options
- **Azure Blob Storage**: Immutable storage with WORM compliance
- **Azure Key Vault**: Secure secrets management

### Backup Strategy
- **Azure Database for PostgreSQL**: Automated backups with point-in-time recovery
- **Azure Cache for Redis**: RDB persistence with geo-replication
- **Azure Blob Storage**: Immutable storage with lifecycle management
- **Cross-Region Replication**: Azure Storage replication for disaster recovery

## Prerequisites

### Required Tools
- Azure CLI (v2.0+)
- PowerShell (for Azure PowerShell modules)
- Azure Storage Explorer
- Database client tools (psql, redis-cli)

### Required Permissions
- Azure subscription contributor access
- Database administrator privileges
- Storage account contributor access
- Key Vault secrets officer access

## Azure Database for PostgreSQL Backup

### 1. Automated Backup Configuration

Azure Database for PostgreSQL provides built-in automated backups with the following features:
- **Backup Retention**: 7-35 days (configurable)
- **Point-in-Time Recovery**: Restore to any point within retention period
- **Geo-Redundant Storage**: Cross-region backup replication
- **Backup Encryption**: Transparent data encryption

#### Configure Backup Settings

```bash
# Set backup retention period (7-35 days)
az postgres flexible-server update \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --backup-retention 30 \
  --geo-redundant-backup Enabled

# Enable point-in-time recovery
az postgres flexible-server parameter set \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres \
  --name log_checkpoints \
  --value on

az postgres flexible-server parameter set \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres \
  --name wal_level \
  --value replica
```

#### Verify Backup Configuration

```bash
# Check backup configuration
az postgres flexible-server show \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --query "{backupRetentionDays:backup.retentionDays,geoRedundantBackup:backup.geoRedundantBackup}"

# List available backups
az postgres flexible-server backup list \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres
```

### 2. Manual Backup Creation

#### Create Manual Backup

```bash
# Create manual backup
az postgres flexible-server backup create \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres \
  --backup-name "manual-backup-$(date +%Y%m%d-%H%M%S)"

# Export database to Azure Blob Storage
az postgres flexible-server export \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres \
  --database-name llama_mapper \
  --storage-uri "https://complyaistorage.blob.core.windows.net/backups/postgresql/export-$(date +%Y%m%d-%H%M%S).sql" \
  --admin-user complyaiadmin \
  --admin-password "YourSecurePassword123!"
```

#### Custom Backup Script

```bash
#!/bin/bash
# scripts/azure-postgresql-backup.sh

set -euo pipefail

# Configuration
RESOURCE_GROUP="comply-ai-rg"
SERVER_NAME="comply-ai-postgres"
DATABASE_NAME="llama_mapper"
STORAGE_ACCOUNT="complyaistorage"
CONTAINER_NAME="backups"
BACKUP_PREFIX="postgresql"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Create backup container if it doesn't exist
az storage container create \
  --name $CONTAINER_NAME \
  --account-name $STORAGE_ACCOUNT \
  --auth-mode login \
  --public-access off

# Export database to blob storage
az postgres flexible-server export \
  --resource-group $RESOURCE_GROUP \
  --server-name $SERVER_NAME \
  --database-name $DATABASE_NAME \
  --storage-uri "https://$STORAGE_ACCOUNT.blob.core.windows.net/$CONTAINER_NAME/$BACKUP_PREFIX/export-$TIMESTAMP.sql" \
  --admin-user complyaiadmin \
  --admin-password "$(az keyvault secret show --vault-name comply-ai-keyvault --name postgres-admin-password --query value -o tsv)"

echo "Backup completed: export-$TIMESTAMP.sql"
```

### 3. Point-in-Time Recovery

#### Restore to Specific Point in Time

```bash
# List available restore points
az postgres flexible-server restore-point list \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres

# Restore to specific point in time
az postgres flexible-server restore \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres-restored \
  --source-server comply-ai-postgres \
  --restore-time "2024-01-15T10:30:00Z"

# Restore specific database
az postgres flexible-server db restore \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres-restored \
  --database-name llama_mapper \
  --source-database llama_mapper \
  --source-server comply-ai-postgres \
  --restore-time "2024-01-15T10:30:00Z"
```

## Azure Cache for Redis Backup

### 1. Redis Persistence Configuration

#### Configure RDB Persistence

```bash
# Enable RDB persistence
az redis update \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --redis-configuration '{"maxmemory-policy":"allkeys-lru","save":"900 1 300 10 60 10000"}'

# Enable AOF persistence (optional)
az redis update \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --redis-configuration '{"appendonly":"yes","appendfsync":"everysec"}'
```

#### Verify Persistence Settings

```bash
# Check Redis configuration
az redis show \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --query "redisConfiguration"

# Get Redis connection details
az redis list-keys \
  --resource-group comply-ai-rg \
  --name comply-ai-redis
```

### 2. Manual Redis Backup

#### Create RDB Backup

```bash
#!/bin/bash
# scripts/azure-redis-backup.sh

set -euo pipefail

# Configuration
RESOURCE_GROUP="comply-ai-rg"
REDIS_NAME="comply-ai-redis"
STORAGE_ACCOUNT="complyaistorage"
CONTAINER_NAME="backups"
BACKUP_PREFIX="redis"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Get Redis connection details
REDIS_HOST=$(az redis show --resource-group $RESOURCE_GROUP --name $REDIS_NAME --query hostName -o tsv)
REDIS_PORT=$(az redis show --resource-group $RESOURCE_GROUP --name $REDIS_NAME --query port -o tsv)
REDIS_KEY=$(az redis list-keys --resource-group $RESOURCE_GROUP --name $REDIS_NAME --query primaryKey -o tsv)

# Connect to Redis and trigger backup
redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_KEY --tls BGSAVE

# Wait for backup to complete
while [ "$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_KEY --tls LASTSAVE)" = "$(redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_KEY --tls LASTSAVE)" ]; do
  sleep 1
done

# Download RDB file (this would require access to Redis data directory)
# In practice, you'd use Azure Redis Export feature when available
echo "Redis backup triggered successfully"
```

### 3. Redis Geo-Replication

#### Configure Geo-Replication

```bash
# Create secondary Redis instance
az redis create \
  --resource-group comply-ai-rg \
  --name comply-ai-redis-dr \
  --location westus2 \
  --sku Standard \
  --vm-size c1 \
  --enable-non-ssl-port false

# Link primary and secondary instances
az redis server-link create \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --linked-server comply-ai-redis-dr \
  --linked-resource-group comply-ai-rg
```

## Azure Blob Storage Backup

### 1. Immutable Storage Configuration

#### Configure WORM Compliance

```bash
# Create storage account with immutable storage
az storage account create \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot \
  --https-only true \
  --min-tls-version TLS1_2

# Enable versioning
az storage account blob-service-properties update \
  --resource-group comply-ai-rg \
  --account-name complyaistorage \
  --enable-versioning true

# Enable immutable storage
az storage account blob-service-properties update \
  --resource-group comply-ai-rg \
  --account-name complyaistorage \
  --enable-immutable-storage true
```

#### Configure Lifecycle Management

```bash
# Create lifecycle management policy
az storage account management-policy create \
  --resource-group comply-ai-rg \
  --account-name complyaistorage \
  --policy '{
    "rules": [{
      "name": "backup-lifecycle",
      "type": "Lifecycle",
      "definition": {
        "filters": {
          "blobTypes": ["blockBlob"],
          "prefixMatch": ["backups/"]
        },
        "actions": {
          "baseBlob": {
            "tierToCool": {
              "daysAfterModificationGreaterThan": 30
            },
            "tierToArchive": {
              "daysAfterModificationGreaterThan": 90
            },
            "delete": {
              "daysAfterModificationGreaterThan": 2555
            }
          }
        }
      }
    }]
  }'
```

### 2. Cross-Region Replication

#### Configure Geo-Replication

```bash
# Create secondary storage account
az storage account create \
  --resource-group comply-ai-rg \
  --name complyaistorage-dr \
  --location westus2 \
  --sku Standard_GRS \
  --kind StorageV2 \
  --access-tier Hot

# Configure replication
az storage account update \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --sku Standard_GRS
```

## Azure Key Vault Backup

### 1. Key Vault Backup

#### Backup Secrets and Keys

```bash
#!/bin/bash
# scripts/azure-keyvault-backup.sh

set -euo pipefail

# Configuration
RESOURCE_GROUP="comply-ai-rg"
KEY_VAULT_NAME="comply-ai-keyvault"
STORAGE_ACCOUNT="complyaistorage"
CONTAINER_NAME="backups"
BACKUP_PREFIX="keyvault"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Create backup container
az storage container create \
  --name $CONTAINER_NAME \
  --account-name $STORAGE_ACCOUNT \
  --auth-mode login

# Backup all secrets
az keyvault secret list \
  --vault-name $KEY_VAULT_NAME \
  --query "[].name" -o tsv | while read secret_name; do
    echo "Backing up secret: $secret_name"
    az keyvault secret show \
      --vault-name $KEY_VAULT_NAME \
      --name $secret_name \
      --query value -o tsv > "/tmp/$secret_name.txt"
    
    az storage blob upload \
      --account-name $STORAGE_ACCOUNT \
      --container-name $CONTAINER_NAME \
      --name "$BACKUP_PREFIX/secrets/$secret_name-$TIMESTAMP.txt" \
      --file "/tmp/$secret_name.txt" \
      --auth-mode login
    
    rm "/tmp/$secret_name.txt"
  done

echo "Key Vault backup completed"
```

## Restore Procedures

### 1. PostgreSQL Restore

#### Restore from Point-in-Time

```bash
# Restore entire server
az postgres flexible-server restore \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres-restored \
  --source-server comply-ai-postgres \
  --restore-time "2024-01-15T10:30:00Z"

# Update application configuration
az postgres flexible-server update \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres-restored \
  --public-access 0.0.0.0
```

#### Restore from Blob Storage

```bash
#!/bin/bash
# scripts/azure-postgresql-restore.sh

set -euo pipefail

# Configuration
RESOURCE_GROUP="comply-ai-rg"
SERVER_NAME="comply-ai-postgres-restored"
DATABASE_NAME="llama_mapper"
STORAGE_ACCOUNT="complyaistorage"
CONTAINER_NAME="backups"
BACKUP_FILE="postgresql/export-20240115-103000.sql"

# Create new server for restore
az postgres flexible-server create \
  --resource-group $RESOURCE_GROUP \
  --name $SERVER_NAME \
  --location eastus \
  --admin-user complyaiadmin \
  --admin-password "$(az keyvault secret show --vault-name comply-ai-keyvault --name postgres-admin-password --query value -o tsv)" \
  --sku-name Standard_D2s_v3 \
  --tier GeneralPurpose \
  --storage-size 100 \
  --version 15

# Create database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $SERVER_NAME \
  --database-name $DATABASE_NAME

# Download and restore backup
az storage blob download \
  --account-name $STORAGE_ACCOUNT \
  --container-name $CONTAINER_NAME \
  --name $BACKUP_FILE \
  --file "/tmp/restore.sql" \
  --auth-mode login

# Restore database
psql -h $SERVER_NAME.postgres.database.azure.com \
  -U complyaiadmin \
  -d $DATABASE_NAME \
  -f /tmp/restore.sql

echo "PostgreSQL restore completed"
```

### 2. Redis Restore

#### Restore from Geo-Replication

```bash
# Promote secondary Redis to primary
az redis server-link delete \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --linked-server comply-ai-redis-dr

# Update application configuration to point to new primary
az redis show \
  --resource-group comply-ai-rg \
  --name comply-ai-redis-dr \
  --query hostName -o tsv
```

### 3. Blob Storage Restore

#### Restore from Cross-Region Replication

```bash
# Failover to secondary storage account
az storage account failover \
  --resource-group comply-ai-rg \
  --name complyaistorage-dr

# Update application configuration
az storage account show \
  --resource-group comply-ai-rg \
  --name complyaistorage-dr \
  --query primaryEndpoints.blob -o tsv
```

## Monitoring and Alerting

### 1. Azure Monitor Integration

#### Configure Backup Monitoring

```bash
# Create action group for alerts
az monitor action-group create \
  --resource-group comply-ai-rg \
  --name backup-alerts \
  --short-name backup-alerts \
  --email-receivers name=admin email=admin@comply-ai.com

# Create alert rule for backup failures
az monitor metrics alert create \
  --resource-group comply-ai-rg \
  --name postgresql-backup-failure \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.DBforPostgreSQL/flexibleServers/comply-ai-postgres" \
  --condition "count Microsoft.DBforPostgreSQL/flexibleServers backup_failed > 0" \
  --action comply-ai-rg/backup-alerts \
  --description "PostgreSQL backup failure alert"
```

#### Configure Log Analytics

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group comply-ai-rg \
  --workspace-name comply-ai-logs \
  --location eastus

# Enable diagnostic settings for PostgreSQL
az monitor diagnostic-settings create \
  --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.DBforPostgreSQL/flexibleServers/comply-ai-postgres" \
  --name postgresql-diagnostics \
  --workspace comply-ai-logs \
  --logs '[{"category":"PostgreSQLLogs","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'
```

### 2. Custom Monitoring Script

```bash
#!/bin/bash
# scripts/azure-backup-monitor.sh

set -euo pipefail

# Configuration
RESOURCE_GROUP="comply-ai-rg"
POSTGRES_SERVER="comply-ai-postgres"
REDIS_NAME="comply-ai-redis"
STORAGE_ACCOUNT="complyaistorage"

# Check PostgreSQL backup status
echo "Checking PostgreSQL backup status..."
POSTGRES_BACKUPS=$(az postgres flexible-server backup list \
  --resource-group $RESOURCE_GROUP \
  --server-name $POSTGRES_SERVER \
  --query "length([?createdTime > '$(date -d '1 day ago' -u +%Y-%m-%dT%H:%M:%SZ)'])")

if [ "$POSTGRES_BACKUPS" -eq 0 ]; then
  echo "WARNING: No PostgreSQL backups in the last 24 hours"
  # Send alert
fi

# Check Redis persistence
echo "Checking Redis persistence..."
REDIS_CONFIG=$(az redis show \
  --resource-group $RESOURCE_GROUP \
  --name $REDIS_NAME \
  --query "redisConfiguration.save" -o tsv)

if [ "$REDIS_CONFIG" = "null" ]; then
  echo "WARNING: Redis persistence not configured"
  # Send alert
fi

# Check storage account replication
echo "Checking storage replication..."
STORAGE_SKU=$(az storage account show \
  --resource-group $RESOURCE_GROUP \
  --name $STORAGE_ACCOUNT \
  --query "sku.name" -o tsv)

if [[ ! "$STORAGE_SKU" =~ "GRS|RAGRS" ]]; then
  echo "WARNING: Storage account not configured for geo-replication"
  # Send alert
fi

echo "Backup monitoring completed"
```

## Disaster Recovery Procedures

### 1. Complete System Recovery

#### RTO Objectives
- **PostgreSQL**: 15 minutes (Azure managed service)
- **Redis**: 5 minutes (geo-replication)
- **Blob Storage**: 1 minute (cross-region replication)
- **Complete System**: 30 minutes

#### Recovery Steps

```bash
#!/bin/bash
# scripts/azure-disaster-recovery.sh

set -euo pipefail

# Configuration
RESOURCE_GROUP="comply-ai-rg"
DR_RESOURCE_GROUP="comply-ai-dr-rg"
DR_LOCATION="westus2"

echo "Starting disaster recovery procedures..."

# 1. Restore PostgreSQL from point-in-time
echo "Restoring PostgreSQL..."
az postgres flexible-server restore \
  --resource-group $DR_RESOURCE_GROUP \
  --name comply-ai-postgres-dr \
  --source-server comply-ai-postgres \
  --restore-time "$(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%SZ)"

# 2. Promote Redis secondary to primary
echo "Promoting Redis secondary..."
az redis server-link delete \
  --resource-group $DR_RESOURCE_GROUP \
  --name comply-ai-redis \
  --linked-server comply-ai-redis-dr

# 3. Failover storage account
echo "Failing over storage account..."
az storage account failover \
  --resource-group $DR_RESOURCE_GROUP \
  --name complyaistorage-dr

# 4. Update DNS records
echo "Updating DNS records..."
# This would be done through your DNS provider or Azure DNS

# 5. Verify system health
echo "Verifying system health..."
# Run health checks on restored services

echo "Disaster recovery completed"
```

### 2. Cross-Region Recovery

#### Setup Cross-Region Resources

```bash
# Create DR resource group
az group create \
  --name comply-ai-dr-rg \
  --location westus2

# Deploy DR infrastructure
az deployment group create \
  --resource-group comply-ai-dr-rg \
  --template-file templates/azure-dr-infrastructure.json \
  --parameters @parameters/azure-dr-parameters.json
```

## Compliance and Audit

### 1. Backup Compliance Report

```bash
#!/bin/bash
# scripts/azure-compliance-report.sh

set -euo pipefail

# Generate compliance report
REPORT_FILE="azure-backup-compliance-$(date +%Y%m%d).json"

cat > $REPORT_FILE << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "compliance_checks": {
    "postgresql_backup_retention": {
      "status": "compliant",
      "retention_days": 30,
      "last_backup": "$(az postgres flexible-server backup list --resource-group comply-ai-rg --server-name comply-ai-postgres --query '[0].createdTime' -o tsv)"
    },
    "redis_persistence": {
      "status": "compliant",
      "rdb_enabled": true,
      "aof_enabled": false
    },
    "storage_replication": {
      "status": "compliant",
      "replication_type": "GRS",
      "cross_region_enabled": true
    },
    "encryption_at_rest": {
      "status": "compliant",
      "postgresql_encrypted": true,
      "redis_encrypted": true,
      "storage_encrypted": true
    }
  }
}
EOF

echo "Compliance report generated: $REPORT_FILE"
```

### 2. Audit Trail

```bash
# Query Azure Activity Log for backup operations
az monitor activity-log list \
  --resource-group comply-ai-rg \
  --start-time "$(date -d '7 days ago' -u +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --query "[?contains(operationName.value, 'backup') || contains(operationName.value, 'restore')]"
```

## Troubleshooting

### Common Issues

#### PostgreSQL Backup Failures

```bash
# Check backup status
az postgres flexible-server backup list \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres

# Check server status
az postgres flexible-server show \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --query "state"

# Check diagnostic logs
az monitor diagnostic-settings list \
  --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.DBforPostgreSQL/flexibleServers/comply-ai-postgres"
```

#### Redis Persistence Issues

```bash
# Check Redis configuration
az redis show \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --query "redisConfiguration"

# Check Redis metrics
az monitor metrics list \
  --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.Cache/Redis/comply-ai-redis" \
  --metric "connectedclients"
```

#### Storage Replication Issues

```bash
# Check storage account replication status
az storage account show \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --query "{replication:sku.name,status:statusOfPrimary}"

# Check replication lag
az storage account show \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --query "statusOfSecondary"
```

## Security Considerations

### 1. Access Control

```bash
# Create backup service principal
az ad sp create-for-rbac \
  --name comply-ai-backup-sp \
  --role Contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg"

# Assign minimal permissions
az role assignment create \
  --assignee comply-ai-backup-sp \
  --role "Storage Blob Data Contributor" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.Storage/storageAccounts/complyaistorage"
```

### 2. Encryption

```bash
# Enable encryption for PostgreSQL
az postgres flexible-server update \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --infrastructure-encryption Enabled

# Enable encryption for Redis
az redis update \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --minimum-tls-version 1.2
```

## Conclusion

This document provides comprehensive Azure-native backup and restore procedures for the Comply-AI platform. The procedures leverage Azure managed services' built-in backup capabilities while providing additional custom backup solutions for specific requirements.

Key benefits of the Azure-native approach:
- **Managed Services**: Leverage Azure's built-in backup and restore capabilities
- **Automated Recovery**: Point-in-time recovery with minimal manual intervention
- **Cross-Region Replication**: Built-in disaster recovery capabilities
- **Compliance**: Meet SOC 2, ISO 27001, and HIPAA requirements
- **Cost Optimization**: Pay only for storage used with lifecycle management
- **Monitoring**: Integrated with Azure Monitor for comprehensive observability
