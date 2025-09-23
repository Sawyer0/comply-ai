# Azure Database Backup and Restore Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Azure-native database backup and restore system for the Comply-AI platform. The system leverages Azure managed services' built-in backup capabilities while providing additional custom backup solutions for specific requirements.

## Azure Services Architecture

### Database Services
- **Azure Database for PostgreSQL**: Primary database with automated backups and point-in-time recovery
- **Azure Cache for Redis**: Managed Redis with persistence and geo-replication
- **Azure Blob Storage**: Immutable storage with WORM compliance and lifecycle management
- **Azure Key Vault**: Secure secrets management with backup capabilities

### Backup Strategy
- **Azure Database for PostgreSQL**: Built-in automated backups (7-35 days retention)
- **Azure Cache for Redis**: RDB persistence with geo-replication
- **Azure Blob Storage**: Immutable storage with cross-region replication
- **Azure Key Vault**: Secrets backup to blob storage

## Prerequisites

### Required Tools
- Azure CLI (v2.0+)
- Azure PowerShell (optional)
- kubectl (for Kubernetes deployment)
- Helm (v3.0+)

### Required Permissions
- Azure subscription contributor access
- Azure Key Vault secrets officer access
- Kubernetes cluster admin access

## Step 1: Azure Infrastructure Setup

### 1.1 Create Resource Group

```bash
# Create resource group
az group create \
  --name comply-ai-rg \
  --location eastus

# Create DR resource group
az group create \
  --name comply-ai-dr-rg \
  --location westus2
```

### 1.2 Create Azure Database for PostgreSQL

```bash
# Create PostgreSQL server with backup configuration
az postgres flexible-server create \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --location eastus \
  --admin-user complyaiadmin \
  --admin-password "YourSecurePassword123!" \
  --sku-name Standard_D2s_v3 \
  --tier GeneralPurpose \
  --storage-size 100 \
  --version 15 \
  --backup-retention 30 \
  --geo-redundant-backup Enabled

# Create database
az postgres flexible-server db create \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres \
  --database-name llama_mapper

# Configure firewall rules
az postgres flexible-server firewall-rule create \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --rule-name AllowAzureServices \
  --start-ip-address 0.0.0.0 \
  --end-ip-address 0.0.0.0
```

### 1.3 Create Azure Cache for Redis

```bash
# Create Redis cache
az redis create \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --location eastus \
  --sku Standard \
  --vm-size c1 \
  --enable-non-ssl-port false

# Configure Redis persistence
az redis update \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --redis-configuration '{"save":"900 1 300 10 60 10000","maxmemory-policy":"allkeys-lru"}'
```

### 1.4 Create Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --location eastus \
  --sku Standard_GRS \
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

# Create backup container
az storage container create \
  --name backups \
  --account-name complyaistorage \
  --auth-mode login \
  --public-access off
```

### 1.5 Create Azure Key Vault

```bash
# Create Key Vault
az keyvault create \
  --resource-group comply-ai-rg \
  --name comply-ai-keyvault \
  --location eastus \
  --sku standard \
  --enable-rbac-authorization true

# Store database password
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name postgres-admin-password \
  --value "YourSecurePassword123!"

# Store Redis key
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name redis-primary-key \
  --value "$(az redis list-keys --resource-group comply-ai-rg --name comply-ai-redis --query primaryKey -o tsv)"
```

## Step 2: Create Azure Service Principal

### 2.1 Create Service Principal

```bash
# Create service principal for backup operations
az ad sp create-for-rbac \
  --name comply-ai-backup-sp \
  --role Contributor \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg"

# Note the output values for the next step
```

### 2.2 Assign Key Vault Permissions

```bash
# Get service principal object ID
SP_OBJECT_ID=$(az ad sp list --display-name comply-ai-backup-sp --query '[0].id' -o tsv)

# Assign Key Vault Secrets Officer role
az role assignment create \
  --assignee $SP_OBJECT_ID \
  --role "Key Vault Secrets Officer" \
  --scope "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.KeyVault/vaults/comply-ai-keyvault"
```

## Step 3: Deploy Kubernetes Resources

### 3.1 Create Kubernetes Secrets

```bash
# Create namespace
kubectl create namespace llama-mapper

# Create Azure service principal secret
kubectl create secret generic azure-backup-sp \
  --from-literal=client_id="your-client-id" \
  --from-literal=client_secret="your-client-secret" \
  --from-literal=tenant_id="your-tenant-id" \
  --from-literal=subscription_id="your-subscription-id" \
  --namespace llama-mapper
```

### 3.2 Create Custom Values File

```yaml
# values-azure-backup.yaml
azureBackups:
  enabled: true
  azure:
    subscriptionId: "your-subscription-id"
    resourceGroup: "comply-ai-rg"
    keyVaultUrl: "https://comply-ai-keyvault.vault.azure.net/"
    region: "eastus"
    drRegion: "westus2"
    clientIdSecretRef: "azure-backup-sp"
    clientSecretSecretRef: "azure-backup-sp"
    tenantIdSecretRef: "azure-backup-sp"
    subscriptionIdSecretRef: "azure-backup-sp"
  retention:
    days: 30
    complianceDays: 2555
  postgresql:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    serverName: "comply-ai-postgres"
    databaseName: "llama_mapper"
    adminUser: "complyaiadmin"
  redis:
    enabled: true
    schedule: "0 3 * * *"  # Daily at 3 AM
    name: "comply-ai-redis"
  storage:
    accountName: "complyaistorage"
    containerName: "backups"
    sku: "Standard_GRS"
    immutableStorage: true
    versioning: true
    lifecycleManagement: true
  keyvault:
    enabled: true
    schedule: "0 4 * * *"  # Daily at 4 AM
  monitoring:
    enabled: true
    schedule: "0 6 * * *"  # Daily at 6 AM
    logAnalyticsWorkspace: "comply-ai-logs"
    actionGroup: "backup-alerts"

# Service account for Azure backup jobs
serviceAccount:
  create: true
  name: llama-mapper-azure-backup
  annotations:
    azure.workload.identity/client-id: "your-client-id"
```

### 3.3 Deploy with Helm

```bash
# Deploy Azure backup system
helm upgrade --install llama-mapper ./charts/llama-mapper \
  --namespace llama-mapper \
  --values values-azure-backup.yaml \
  --wait
```

## Step 4: Configure Azure Monitor

### 4.1 Create Log Analytics Workspace

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group comply-ai-rg \
  --workspace-name comply-ai-logs \
  --location eastus
```

### 4.2 Create Action Group

```bash
# Create action group for alerts
az monitor action-group create \
  --resource-group comply-ai-rg \
  --name backup-alerts \
  --short-name backup-alerts \
  --email-receivers name=admin email=admin@comply-ai.com
```

### 4.3 Configure Diagnostic Settings

```bash
# Enable diagnostic settings for PostgreSQL
az monitor diagnostic-settings create \
  --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.DBforPostgreSQL/flexibleServers/comply-ai-postgres" \
  --name postgresql-diagnostics \
  --workspace comply-ai-logs \
  --logs '[{"category":"PostgreSQLLogs","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'

# Enable diagnostic settings for Redis
az monitor diagnostic-settings create \
  --resource "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.Cache/Redis/comply-ai-redis" \
  --name redis-diagnostics \
  --workspace comply-ai-logs \
  --logs '[{"category":"RedisLogs","enabled":true}]' \
  --metrics '[{"category":"AllMetrics","enabled":true}]'
```

## Step 5: Verify Backup Deployment

### 5.1 Check CronJobs

```bash
# List Azure backup CronJobs
kubectl get cronjobs -n llama-mapper -l app.kubernetes.io/component=azure-backup

# Check CronJob details
kubectl describe cronjob llama-mapper-azure-pg-backup -n llama-mapper
```

### 5.2 Test Manual Backup

```bash
# Create manual backup job
kubectl create job --from=cronjob/llama-mapper-azure-pg-backup manual-azure-pg-backup-$(date +%Y%m%d-%H%M%S) -n llama-mapper

# Check job status
kubectl get jobs -n llama-mapper

# View backup logs
kubectl logs job/manual-azure-pg-backup-$(date +%Y%m%d-%H%M%S) -n llama-mapper
```

### 5.3 Verify Azure Backups

```bash
# List PostgreSQL backups
az postgres flexible-server backup list \
  --resource-group comply-ai-rg \
  --server-name comply-ai-postgres

# Check Redis persistence
az redis show \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --query "redisConfiguration.save"

# List blob storage backups
az storage blob list \
  --account-name complyaistorage \
  --container-name backups \
  --auth-mode login
```

## Step 6: Test Restore Procedures

### 6.1 Test PostgreSQL Restore

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

# Verify restore
az postgres flexible-server show \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres-restored
```

### 6.2 Test Redis Geo-Replication

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

### 6.3 Test Blob Storage Replication

```bash
# Check storage replication status
az storage account show \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --query "{replication:sku.name,status:statusOfPrimary}"

# Test failover (if needed)
az storage account failover \
  --resource-group comply-ai-rg \
  --name complyaistorage
```

## Step 7: Configure Monitoring and Alerting

### 7.1 Create Alert Rules

```bash
# Create alert rule for backup failures
az monitor metrics alert create \
  --resource-group comply-ai-rg \
  --name postgresql-backup-failure \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.DBforPostgreSQL/flexibleServers/comply-ai-postgres" \
  --condition "count Microsoft.DBforPostgreSQL/flexibleServers backup_failed > 0" \
  --action comply-ai-rg/backup-alerts \
  --description "PostgreSQL backup failure alert"

# Create alert rule for Redis connectivity
az monitor metrics alert create \
  --resource-group comply-ai-rg \
  --name redis-connectivity \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.Cache/Redis/comply-ai-redis" \
  --condition "avg Microsoft.Cache/Redis connectedclients < 1" \
  --action comply-ai-rg/backup-alerts \
  --description "Redis connectivity alert"
```

### 7.2 Test Monitoring

```bash
# Run monitoring job
kubectl create job --from=cronjob/llama-mapper-azure-monitoring monitoring-test-$(date +%Y%m%d-%H%M%S) -n llama-mapper

# Check monitoring logs
kubectl logs job/monitoring-test-$(date +%Y%m%d-%H%M%S) -n llama-mapper
```

## Step 8: Disaster Recovery Testing

### 8.1 Quarterly DR Drill

```bash
#!/bin/bash
# scripts/azure-dr-drill.sh

set -euo pipefail

RESOURCE_GROUP="comply-ai-rg"
DR_RESOURCE_GROUP="comply-ai-dr-rg"

echo "Starting Azure disaster recovery drill..."

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

# 4. Verify system health
echo "Verifying system health..."
# Run health checks on restored services

echo "Disaster recovery drill completed"
```

### 8.2 RTO Measurement

```bash
# Start timing
start_time=$(date +%s)

# Perform restore operations
# ... restore commands ...

# End timing
end_time=$(date +%s)
rto=$((end_time - start_time))
echo "RTO: ${rto} seconds"
```

## Step 9: Compliance and Audit

### 9.1 Generate Compliance Report

```bash
#!/bin/bash
# scripts/azure-compliance-report.sh

set -euo pipefail

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

### 9.2 Audit Trail

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

#### Azure Authentication Failures

```bash
# Check service principal permissions
az role assignment list \
  --assignee comply-ai-backup-sp \
  --all

# Test Azure CLI authentication
az account show
az keyvault secret list --vault-name comply-ai-keyvault
```

#### PostgreSQL Backup Issues

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
# Create backup service principal with minimal permissions
az ad sp create-for-rbac \
  --name comply-ai-backup-sp \
  --role "Storage Blob Data Contributor" \
  --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/comply-ai-rg/providers/Microsoft.Storage/storageAccounts/complyaistorage"
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

### 3. Network Security

```bash
# Configure PostgreSQL firewall rules
az postgres flexible-server firewall-rule create \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --rule-name AllowKubernetes \
  --start-ip-address 10.0.0.0 \
  --end-ip-address 10.0.255.255
```

## Cost Optimization

### 1. Storage Lifecycle Management

```bash
# Configure lifecycle management policy
az storage account management-policy create \
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

### 2. Resource Optimization

```bash
# Monitor storage costs
az consumption usage list \
  --billing-period-name "202401-1" \
  --query "[?contains(instanceName, 'complyaistorage')]"

# Monitor database costs
az consumption usage list \
  --billing-period-name "202401-1" \
  --query "[?contains(instanceName, 'comply-ai-postgres')]"
```

## Conclusion

This deployment guide provides a comprehensive approach to implementing Azure-native database backup and restore procedures for the Comply-AI platform. The system leverages Azure managed services' built-in capabilities while providing additional custom backup solutions for specific requirements.

Key benefits of the Azure-native approach:
- **Managed Services**: Leverage Azure's built-in backup and restore capabilities
- **Automated Recovery**: Point-in-time recovery with minimal manual intervention
- **Cross-Region Replication**: Built-in disaster recovery capabilities
- **Compliance**: Meet SOC 2, ISO 27001, and HIPAA requirements
- **Cost Optimization**: Pay only for storage used with lifecycle management
- **Monitoring**: Integrated with Azure Monitor for comprehensive observability

The system is designed to meet enterprise-grade requirements for data protection, compliance, and disaster recovery while maintaining operational efficiency and cost-effectiveness in the Azure cloud environment.
