# Azure Environment Configuration

This document provides Azure-specific environment configuration templates and examples for the Comply-AI platform.

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Azure-Specific Settings](#azure-specific-settings)
3. [Configuration Templates](#configuration-templates)
4. [Secrets Management](#secrets-management)
5. [Network Configuration](#network-configuration)

## Environment Variables

### Core Application Settings

```bash
# Application Identity
LLAMA_MAPPER_APP_NAME=llama-mapper
LLAMA_MAPPER_VERSION=0.1.0
LLAMA_MAPPER_ENVIRONMENT=production
LLAMA_MAPPER_DEBUG=false

# Service Discovery
SERVICE_DISCOVERY_ENABLED=true
SERVICE_REGISTRY_URL=https://comply-ai-registry.azurewebsites.net
```

### Model Configuration

```bash
# Llama Mapper Model
LLAMA_MAPPER_MODEL__NAME=meta-llama/Llama-2-7b-chat-hf
LLAMA_MAPPER_MODEL__TEMPERATURE=0.1
LLAMA_MAPPER_MODEL__TOP_P=0.9
LLAMA_MAPPER_MODEL__MAX_NEW_TOKENS=200
LLAMA_MAPPER_MODEL__QUANTIZATION=8bit

# Analysis Module Model
ANALYSIS_MODEL__NAME=microsoft/Phi-3-mini-4k-instruct
ANALYSIS_MODEL__TEMPERATURE=0.1
ANALYSIS_MODEL__MAX_TOKENS=1000
ANALYSIS_MODEL__TOP_P=0.9
```

### Serving Configuration

```bash
# Llama Mapper Serving
LLAMA_MAPPER_SERVING__BACKEND=vllm
LLAMA_MAPPER_SERVING__HOST=0.0.0.0
LLAMA_MAPPER_SERVING__PORT=8000
LLAMA_MAPPER_SERVING__WORKERS=1
LLAMA_MAPPER_SERVING__BATCH_SIZE=8
LLAMA_MAPPER_SERVING__GPU_MEMORY_UTILIZATION=0.9
LLAMA_MAPPER_SERVING__DEVICE=auto

# Analysis Module Serving
ANALYSIS_SERVING__HOST=0.0.0.0
ANALYSIS_SERVING__PORT=8001
ANALYSIS_SERVING__WORKERS=1
ANALYSIS_SERVING__BATCH_SIZE=4
ANALYSIS_SERVING__GPU_MEMORY_UTILIZATION=0.8
```

### Detector Orchestration Configuration

```bash
# Orchestration Settings
ORCH_ENVIRONMENT=production
ORCH_LOG_LEVEL=INFO
ORCH_MAX_CONCURRENT_DETECTORS=10
ORCH_DEFAULT_TIMEOUT_MS=5000
ORCH_MAX_RETRIES=2
ORCH_SYNC_REQUEST_SLA_MS=2000
ORCH_ASYNC_REQUEST_SLA_MS=30000
ORCH_MAPPER_TIMEOUT_BUDGET_MS=500

# Health Monitoring
ORCH_HEALTH_CHECK_INTERVAL_SECONDS=30
ORCH_UNHEALTHY_THRESHOLD=3
ORCH_CIRCUIT_BREAKER_ENABLED=true
ORCH_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
ORCH_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS=60
```

## Azure-Specific Settings

### Azure Database for PostgreSQL

```bash
# Database Connection
LLAMA_MAPPER_STORAGE__DB_HOST=comply-ai-postgres.postgres.database.azure.com
LLAMA_MAPPER_STORAGE__DB_PORT=5432
LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper
LLAMA_MAPPER_STORAGE__DB_USER=complyaiadmin
LLAMA_MAPPER_STORAGE__DB_PASSWORD=YourSecurePassword123!
LLAMA_MAPPER_STORAGE__DB_SSL_MODE=require
LLAMA_MAPPER_STORAGE__DB_POOL_SIZE=10
LLAMA_MAPPER_STORAGE__DB_MAX_OVERFLOW=20

# Connection Pool Settings
DB_POOL_RECYCLE_SECONDS=3600
DB_POOL_PRE_PING=true
DB_POOL_TIMEOUT_SECONDS=30
```

### Azure Cache for Redis

```bash
# Redis Connection
REDIS_HOST=comply-ai-redis.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your-redis-primary-key
REDIS_SSL=true
REDIS_DB=0
REDIS_MAX_CONNECTIONS=20
REDIS_RETRY_ON_TIMEOUT=true
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5
```

### Azure Blob Storage (S3-Compatible)

```bash
# Storage Configuration
LLAMA_MAPPER_STORAGE__S3_BUCKET=complyaistorage
LLAMA_MAPPER_STORAGE__S3_PREFIX=mapper-outputs
LLAMA_MAPPER_STORAGE__S3_RETENTION_YEARS=7
LLAMA_MAPPER_STORAGE__AWS_ACCESS_KEY_ID=your-storage-account-key
LLAMA_MAPPER_STORAGE__AWS_SECRET_ACCESS_KEY=your-storage-account-secret
LLAMA_MAPPER_STORAGE__AWS_REGION=eastus
LLAMA_MAPPER_STORAGE__AWS_ENDPOINT_URL=https://complyaistorage.blob.core.windows.net

# Storage Backend
LLAMA_MAPPER_STORAGE__STORAGE_BACKEND=postgresql
LLAMA_MAPPER_STORAGE__RETENTION_DAYS=90
```

### Azure Key Vault Integration

```bash
# Key Vault Configuration
AZURE_KEY_VAULT_URL=https://comply-ai-keyvault.vault.azure.net/
AZURE_KEY_VAULT_TENANT_ID=your-tenant-id
AZURE_KEY_VAULT_CLIENT_ID=your-client-id
AZURE_KEY_VAULT_CLIENT_SECRET=your-client-secret

# Secrets Configuration
LLAMA_MAPPER_SECURITY__SECRETS_BACKEND=azure-keyvault
LLAMA_MAPPER_SECURITY__ENCRYPTION_KEY_ID=your-azure-keyvault-key
LLAMA_MAPPER_SECURITY__KMS_KEY_ID=your-kms-key-id
```

### Azure Monitor and Application Insights

```bash
# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=your-key;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/
APPLICATIONINSIGHTS_ENABLED=true
APPLICATIONINSIGHTS_SAMPLING_PERCENTAGE=100

# Log Analytics
LOG_ANALYTICS_WORKSPACE_ID=your-workspace-id
LOG_ANALYTICS_WORKSPACE_KEY=your-workspace-key

# Monitoring Configuration
LLAMA_MAPPER_MONITORING__ENABLED=true
LLAMA_MAPPER_MONITORING__METRICS_ENABLED=true
LLAMA_MAPPER_MONITORING__TRACING_ENABLED=true
LLAMA_MAPPER_MONITORING__LOGGING_ENABLED=true
```

## Configuration Templates

### Development Environment

```bash
# Development Configuration
cat > .env.development << EOF
# Application
LLAMA_MAPPER_ENVIRONMENT=development
LLAMA_MAPPER_DEBUG=true
LLAMA_MAPPER_LOGGING__LEVEL=DEBUG

# Database (Local/Dev)
LLAMA_MAPPER_STORAGE__DB_HOST=localhost
LLAMA_MAPPER_STORAGE__DB_PORT=5432
LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper_dev
LLAMA_MAPPER_STORAGE__DB_USER=dev_user
LLAMA_MAPPER_STORAGE__DB_PASSWORD=dev_password

# Redis (Local/Dev)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_SSL=false

# Storage (Local/Dev)
LLAMA_MAPPER_STORAGE__S3_BUCKET=dev-mapper-bucket
LLAMA_MAPPER_STORAGE__AWS_ENDPOINT_URL=http://localhost:9000

# Orchestration
ORCH_ENVIRONMENT=development
ORCH_LOG_LEVEL=DEBUG
ORCH_MAX_CONCURRENT_DETECTORS=3
ORCH_DEFAULT_TIMEOUT_MS=10000
EOF
```

### Staging Environment

```bash
# Staging Configuration
cat > .env.staging << EOF
# Application
LLAMA_MAPPER_ENVIRONMENT=staging
LLAMA_MAPPER_DEBUG=false
LLAMA_MAPPER_LOGGING__LEVEL=INFO

# Database (Staging)
LLAMA_MAPPER_STORAGE__DB_HOST=comply-ai-postgres-staging.postgres.database.azure.com
LLAMA_MAPPER_STORAGE__DB_PORT=5432
LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper_staging
LLAMA_MAPPER_STORAGE__DB_USER=staging_admin
LLAMA_MAPPER_STORAGE__DB_PASSWORD=StagingPassword123!

# Redis (Staging)
REDIS_HOST=comply-ai-redis-staging.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=staging-redis-key
REDIS_SSL=true

# Storage (Staging)
LLAMA_MAPPER_STORAGE__S3_BUCKET=complyaistorage-staging
LLAMA_MAPPER_STORAGE__AWS_ENDPOINT_URL=https://complyaistorage-staging.blob.core.windows.net

# Orchestration
ORCH_ENVIRONMENT=staging
ORCH_LOG_LEVEL=INFO
ORCH_MAX_CONCURRENT_DETECTORS=5
ORCH_DEFAULT_TIMEOUT_MS=7000
EOF
```

### Production Environment

```bash
# Production Configuration
cat > .env.production << EOF
# Application
LLAMA_MAPPER_ENVIRONMENT=production
LLAMA_MAPPER_DEBUG=false
LLAMA_MAPPER_LOGGING__LEVEL=INFO
LLAMA_MAPPER_LOGGING__PRIVACY_MODE=true

# Database (Production)
LLAMA_MAPPER_STORAGE__DB_HOST=comply-ai-postgres.postgres.database.azure.com
LLAMA_MAPPER_STORAGE__DB_PORT=5432
LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper
LLAMA_MAPPER_STORAGE__DB_USER=complyaiadmin
LLAMA_MAPPER_STORAGE__DB_PASSWORD=ProductionPassword123!
LLAMA_MAPPER_STORAGE__DB_SSL_MODE=require
LLAMA_MAPPER_STORAGE__DB_POOL_SIZE=20
LLAMA_MAPPER_STORAGE__DB_MAX_OVERFLOW=30

# Redis (Production)
REDIS_HOST=comply-ai-redis.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=production-redis-key
REDIS_SSL=true
REDIS_MAX_CONNECTIONS=50

# Storage (Production)
LLAMA_MAPPER_STORAGE__S3_BUCKET=complyaistorage
LLAMA_MAPPER_STORAGE__S3_PREFIX=mapper-outputs
LLAMA_MAPPER_STORAGE__S3_RETENTION_YEARS=7
LLAMA_MAPPER_STORAGE__AWS_ENDPOINT_URL=https://complyaistorage.blob.core.windows.net

# Orchestration (Production)
ORCH_ENVIRONMENT=production
ORCH_LOG_LEVEL=INFO
ORCH_MAX_CONCURRENT_DETECTORS=10
ORCH_DEFAULT_TIMEOUT_MS=5000
ORCH_SYNC_REQUEST_SLA_MS=2000
ORCH_ASYNC_REQUEST_SLA_MS=30000
ORCH_MAPPER_TIMEOUT_BUDGET_MS=500

# Security (Production)
LLAMA_MAPPER_SECURITY__SECRETS_BACKEND=azure-keyvault
LLAMA_MAPPER_SECURITY__ENCRYPTION_KEY_ID=production-kms-key
LLAMA_MAPPER_SECURITY__KMS_KEY_ID=production-kms-key

# Monitoring (Production)
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=production-key;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/
APPLICATIONINSIGHTS_ENABLED=true
APPLICATIONINSIGHTS_SAMPLING_PERCENTAGE=10
EOF
```

## Secrets Management

### Azure Key Vault Secrets

```bash
# Database Secrets
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name postgres-password \
  --value "YourSecurePassword123!"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name postgres-connection-string \
  --value "postgresql://complyaiadmin:YourSecurePassword123!@comply-ai-postgres.postgres.database.azure.com:5432/llama_mapper?sslmode=require"

# Redis Secrets
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name redis-key \
  --value "$(az redis list-keys --resource-group comply-ai-rg --name comply-ai-redis --query primaryKey -o tsv)"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name redis-connection-string \
  --value "rediss://:$(az redis list-keys --resource-group comply-ai-rg --name comply-ai-redis --query primaryKey -o tsv)@comply-ai-redis.redis.cache.windows.net:6380/0"

# Storage Secrets
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name storage-key \
  --value "$(az storage account keys list --resource-group comply-ai-rg --account-name complyaistorage --query [0].value -o tsv)"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name storage-connection-string \
  --value "$(az storage account show-connection-string --resource-group comply-ai-rg --name complyaistorage --query connectionString -o tsv)"

# Application Secrets
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name jwt-secret \
  --value "$(openssl rand -base64 32)"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name encryption-key \
  --value "$(openssl rand -base64 32)"

# API Keys
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name api-key-admin \
  --value "$(openssl rand -hex 32)"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name api-key-readonly \
  --value "$(openssl rand -hex 32)"
```

### Environment Variable Override with Key Vault

```bash
# Use Key Vault references in environment variables
LLAMA_MAPPER_STORAGE__DB_PASSWORD="@Microsoft.KeyVault(SecretUri=https://comply-ai-keyvault.vault.azure.net/secrets/postgres-password/)"
REDIS_PASSWORD="@Microsoft.KeyVault(SecretUri=https://comply-ai-keyvault.vault.azure.net/secrets/redis-key/)"
LLAMA_MAPPER_STORAGE__AWS_SECRET_ACCESS_KEY="@Microsoft.KeyVault(SecretUri=https://comply-ai-keyvault.vault.azure.net/secrets/storage-key/)"
```

## Network Configuration

### Virtual Network Settings

```bash
# Network Configuration
VNET_NAME=comply-ai-vnet
VNET_ADDRESS_PREFIX=10.0.0.0/16
SUBNET_ORCHESTRATION=10.0.1.0/24
SUBNET_MAPPER=10.0.2.0/24
SUBNET_ANALYSIS=10.0.3.0/24
SUBNET_DATABASE=10.0.4.0/24
SUBNET_GATEWAY=10.0.5.0/24

# DNS Configuration
DNS_SERVERS=168.63.129.16
CUSTOM_DNS_ENABLED=false
```

### Load Balancer Configuration

```bash
# Application Gateway Settings
GATEWAY_NAME=comply-ai-gateway
GATEWAY_SKU=Standard_v2
GATEWAY_CAPACITY=2
GATEWAY_TIER=Standard

# Backend Pool Configuration
BACKEND_POOL_ORCHESTRATION=comply-ai-orchestration.eastus.azurecontainer.io
BACKEND_POOL_MAPPER=comply-ai-mapper.eastus.azurecontainer.io
BACKEND_POOL_ANALYSIS=comply-ai-analysis.eastus.azurecontainer.io

# Health Probe Settings
HEALTH_PROBE_PATH=/health
HEALTH_PROBE_INTERVAL=30
HEALTH_PROBE_TIMEOUT=30
HEALTH_PROBE_UNHEALTHY_THRESHOLD=3
```

### Security Group Rules

```bash
# Network Security Group Rules
NSG_NAME=comply-ai-nsg

# Inbound Rules
ALLOW_HTTPS_INTERNET=443
ALLOW_HTTP_INTERNET=80
ALLOW_SSH_INTERNAL=22
ALLOW_POSTGRES_INTERNAL=5432
ALLOW_REDIS_INTERNAL=6380

# Outbound Rules
ALLOW_HTTPS_OUTBOUND=443
ALLOW_HTTP_OUTBOUND=80
ALLOW_DNS_OUTBOUND=53
ALLOW_NTP_OUTBOUND=123
```

## Configuration Validation

### Environment Validation Script

```bash
#!/bin/bash
# validate-azure-config.sh

echo "Validating Azure environment configuration..."

# Check required environment variables
required_vars=(
    "LLAMA_MAPPER_ENVIRONMENT"
    "LLAMA_MAPPER_STORAGE__DB_HOST"
    "LLAMA_MAPPER_STORAGE__DB_PASSWORD"
    "REDIS_HOST"
    "REDIS_PASSWORD"
    "LLAMA_MAPPER_STORAGE__S3_BUCKET"
)

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Test database connectivity
echo "Testing database connectivity..."
pg_isready -h "$LLAMA_MAPPER_STORAGE__DB_HOST" -p "$LLAMA_MAPPER_STORAGE__DB_PORT" -U "$LLAMA_MAPPER_STORAGE__DB_USER"

# Test Redis connectivity
echo "Testing Redis connectivity..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping

# Test storage connectivity
echo "Testing storage connectivity..."
aws s3 ls s3://"$LLAMA_MAPPER_STORAGE__S3_BUCKET" --endpoint-url "$LLAMA_MAPPER_STORAGE__AWS_ENDPOINT_URL"

echo "Configuration validation completed successfully!"
```

### Configuration Testing

```bash
# Test configuration loading
python -c "
from src.llama_mapper.config.manager import ConfigManager
config = ConfigManager()
print('Configuration loaded successfully')
print(f'Environment: {config.get_config().get(\"environment\")}')
print(f'Database host: {config.get_config().get(\"storage\", {}).get(\"db_host\")}')
"

# Test service connectivity
python -c "
import asyncio
from src.llama_mapper.storage.manager import StorageManager
from src.llama_mapper.config.settings import Settings

async def test_connectivity():
    settings = Settings()
    storage = StorageManager(settings.storage)
    await storage.initialize()
    print('Storage connectivity test passed')

asyncio.run(test_connectivity())
"
```

## Deployment Commands

### Environment-Specific Deployment

```bash
# Deploy to development
az containerapp update \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --set-env-vars @.env.development

# Deploy to staging
az containerapp update \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --set-env-vars @.env.staging

# Deploy to production
az containerapp update \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --set-env-vars @.env.production
```

### Configuration Updates

```bash
# Update configuration without redeployment
az containerapp update \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --set-env-vars \
    ORCH_MAX_CONCURRENT_DETECTORS=15 \
    ORCH_DEFAULT_TIMEOUT_MS=7000

# Rollback configuration
az containerapp update \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --set-env-vars \
    ORCH_MAX_CONCURRENT_DETECTORS=10 \
    ORCH_DEFAULT_TIMEOUT_MS=5000
```

This configuration provides a comprehensive foundation for deploying the Comply-AI platform on Azure with proper environment separation, secrets management, and monitoring integration.
