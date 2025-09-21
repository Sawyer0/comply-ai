# Azure Deployment Scripts

This document provides automated deployment scripts for the Comply-AI platform on Azure.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Scripts](#infrastructure-scripts)
3. [Application Deployment Scripts](#application-deployment-scripts)
4. [Configuration Scripts](#configuration-scripts)
5. [Monitoring Scripts](#monitoring-scripts)
6. [Maintenance Scripts](#maintenance-scripts)

## Prerequisites

### Required Tools

```bash
# Install required tools
sudo apt-get update
sudo apt-get install -y \
    azure-cli \
    jq \
    curl \
    openssl \
    postgresql-client \
    redis-tools

# Verify installations
az --version
jq --version
curl --version
openssl version
psql --version
redis-cli --version
```

### Azure CLI Login

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Verify login
az account show
```

## Infrastructure Scripts

### 1. Complete Infrastructure Setup

```bash
#!/bin/bash
# deploy-infrastructure.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
LOCATION="eastus"
POSTGRES_ADMIN_USER="complyaiadmin"
POSTGRES_ADMIN_PASSWORD="YourSecurePassword123!"
REDIS_SKU="Basic"
REDIS_SIZE="c1"
STORAGE_ACCOUNT="complyaistorage"
KEY_VAULT="comply-ai-keyvault"
ACR_NAME="complyaiacr"

echo "🚀 Starting Azure infrastructure deployment..."

# Create resource group
echo "📦 Creating resource group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --tags Environment=Production Project=ComplyAI

# Create virtual network
echo "🌐 Creating virtual network..."
az network vnet create \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-vnet \
    --address-prefix 10.0.0.0/16 \
    --location $LOCATION

# Create subnets
echo "🔗 Creating subnets..."
az network vnet subnet create \
    --resource-group $RESOURCE_GROUP \
    --vnet-name comply-ai-vnet \
    --name orchestration-subnet \
    --address-prefix 10.0.1.0/24

az network vnet subnet create \
    --resource-group $RESOURCE_GROUP \
    --vnet-name comply-ai-vnet \
    --name mapper-subnet \
    --address-prefix 10.0.2.0/24

az network vnet subnet create \
    --resource-group $RESOURCE_GROUP \
    --vnet-name comply-ai-vnet \
    --name analysis-subnet \
    --address-prefix 10.0.3.0/24

az network vnet subnet create \
    --resource-group $RESOURCE_GROUP \
    --vnet-name comply-ai-vnet \
    --name database-subnet \
    --address-prefix 10.0.4.0/24

az network vnet subnet create \
    --resource-group $RESOURCE_GROUP \
    --vnet-name comply-ai-vnet \
    --name gateway-subnet \
    --address-prefix 10.0.5.0/24

# Create PostgreSQL server
echo "🗄️ Creating PostgreSQL server..."
az postgres flexible-server create \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-postgres \
    --location $LOCATION \
    --admin-user $POSTGRES_ADMIN_USER \
    --admin-password $POSTGRES_ADMIN_PASSWORD \
    --sku-name Standard_D2s_v3 \
    --tier GeneralPurpose \
    --public-access 0.0.0.0 \
    --storage-size 100 \
    --version 15

# Create database
echo "📊 Creating database..."
az postgres flexible-server db create \
    --resource-group $RESOURCE_GROUP \
    --server-name comply-ai-postgres \
    --database-name llama_mapper

# Configure firewall
echo "🔥 Configuring firewall..."
az postgres flexible-server firewall-rule create \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-postgres \
    --rule-name AllowAzureServices \
    --start-ip-address 0.0.0.0 \
    --end-ip-address 0.0.0.0

# Create Redis cache
echo "🔴 Creating Redis cache..."
az redis create \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-redis \
    --location $LOCATION \
    --sku $REDIS_SKU \
    --vm-size $REDIS_SIZE \
    --enable-non-ssl-port false

# Create storage account
echo "💾 Creating storage account..."
az storage account create \
    --resource-group $RESOURCE_GROUP \
    --name $STORAGE_ACCOUNT \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2 \
    --access-tier Hot

# Create storage container
echo "📦 Creating storage container..."
az storage container create \
    --account-name $STORAGE_ACCOUNT \
    --name mapper-outputs \
    --public-access off

# Create Key Vault
echo "🔐 Creating Key Vault..."
az keyvault create \
    --resource-group $RESOURCE_GROUP \
    --name $KEY_VAULT \
    --location $LOCATION \
    --sku standard \
    --enable-rbac-authorization true

# Create Container Registry
echo "🐳 Creating Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Create Log Analytics workspace
echo "📊 Creating Log Analytics workspace..."
az monitor log-analytics workspace create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name comply-ai-logs \
    --location $LOCATION

# Create Application Insights
echo "🔍 Creating Application Insights..."
az monitor app-insights component create \
    --resource-group $RESOURCE_GROUP \
    --app comply-ai-insights \
    --location $LOCATION \
    --kind web \
    --application-type web \
    --workspace comply-ai-logs

echo "✅ Infrastructure deployment completed successfully!"
echo "📋 Next steps:"
echo "   1. Run secrets setup: ./setup-secrets.sh"
echo "   2. Build and push images: ./build-and-push-images.sh"
echo "   3. Deploy applications: ./deploy-applications.sh"
```

### 2. Secrets Setup Script

```bash
#!/bin/bash
# setup-secrets.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
KEY_VAULT="comply-ai-keyvault"
POSTGRES_ADMIN_PASSWORD="YourSecurePassword123!"

echo "🔐 Setting up Azure Key Vault secrets..."

# Get resource information
POSTGRES_SERVER=$(az postgres flexible-server list --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv)
REDIS_NAME=$(az redis list --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv)
STORAGE_ACCOUNT=$(az storage account list --resource-group $RESOURCE_GROUP --query "[0].name" -o tsv)

# Store PostgreSQL password
echo "📝 Storing PostgreSQL password..."
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name postgres-password \
    --value "$POSTGRES_ADMIN_PASSWORD"

# Store PostgreSQL connection string
echo "📝 Storing PostgreSQL connection string..."
POSTGRES_CONNECTION_STRING="postgresql://complyaiadmin:$POSTGRES_ADMIN_PASSWORD@$POSTGRES_SERVER.postgres.database.azure.com:5432/llama_mapper?sslmode=require"
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name postgres-connection-string \
    --value "$POSTGRES_CONNECTION_STRING"

# Store Redis key
echo "📝 Storing Redis key..."
REDIS_KEY=$(az redis list-keys --resource-group $RESOURCE_GROUP --name $REDIS_NAME --query primaryKey -o tsv)
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name redis-key \
    --value "$REDIS_KEY"

# Store Redis connection string
echo "📝 Storing Redis connection string..."
REDIS_CONNECTION_STRING="rediss://:$REDIS_KEY@$REDIS_NAME.redis.cache.windows.net:6380/0"
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name redis-connection-string \
    --value "$REDIS_CONNECTION_STRING"

# Store storage key
echo "📝 Storing storage key..."
STORAGE_KEY=$(az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT --query [0].value -o tsv)
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name storage-key \
    --value "$STORAGE_KEY"

# Store storage connection string
echo "📝 Storing storage connection string..."
STORAGE_CONNECTION_STRING=$(az storage account show-connection-string --resource-group $RESOURCE_GROUP --name $STORAGE_ACCOUNT --query connectionString -o tsv)
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name storage-connection-string \
    --value "$STORAGE_CONNECTION_STRING"

# Generate and store JWT secret
echo "📝 Generating and storing JWT secret..."
JWT_SECRET=$(openssl rand -base64 32)
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name jwt-secret \
    --value "$JWT_SECRET"

# Generate and store encryption key
echo "📝 Generating and storing encryption key..."
ENCRYPTION_KEY=$(openssl rand -base64 32)
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name encryption-key \
    --value "$ENCRYPTION_KEY"

# Generate and store API keys
echo "📝 Generating and storing API keys..."
API_KEY_ADMIN=$(openssl rand -hex 32)
API_KEY_READONLY=$(openssl rand -hex 32)

az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name api-key-admin \
    --value "$API_KEY_ADMIN"

az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name api-key-readonly \
    --value "$API_KEY_READONLY"

echo "✅ Secrets setup completed successfully!"
echo "📋 Generated secrets:"
echo "   - PostgreSQL connection string"
echo "   - Redis connection string"
echo "   - Storage connection string"
echo "   - JWT secret"
echo "   - Encryption key"
echo "   - API keys (admin and readonly)"
```

## Application Deployment Scripts

### 3. Build and Push Images

```bash
#!/bin/bash
# build-and-push-images.sh

set -e

# Configuration
ACR_NAME="complyaiacr"
RESOURCE_GROUP="comply-ai-rg"

echo "🐳 Building and pushing container images..."

# Login to ACR
echo "🔑 Logging into Azure Container Registry..."
az acr login --name $ACR_NAME

# Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
echo "📡 ACR Login Server: $ACR_LOGIN_SERVER"

# Build and push detector-orchestration
echo "🔧 Building detector-orchestration..."
cd detector-orchestration
docker build -t $ACR_LOGIN_SERVER/detector-orchestration:latest .
docker push $ACR_LOGIN_SERVER/detector-orchestration:latest
cd ..

# Build and push llama-mapper
echo "🔧 Building llama-mapper..."
cd src/llama_mapper
docker build -t $ACR_LOGIN_SERVER/llama-mapper:latest .
docker push $ACR_LOGIN_SERVER/llama-mapper:latest
cd ../..

# Build and push analysis-module (when implemented)
if [ -d "src/llama_mapper/analysis" ]; then
    echo "🔧 Building analysis-module..."
    cd src/llama_mapper/analysis
    docker build -t $ACR_LOGIN_SERVER/analysis-module:latest .
    docker push $ACR_LOGIN_SERVER/analysis-module:latest
    cd ../../..
fi

echo "✅ All images built and pushed successfully!"
echo "📋 Available images:"
echo "   - $ACR_LOGIN_SERVER/detector-orchestration:latest"
echo "   - $ACR_LOGIN_SERVER/llama-mapper:latest"
if [ -d "src/llama_mapper/analysis" ]; then
    echo "   - $ACR_LOGIN_SERVER/analysis-module:latest"
fi
```

### 4. Deploy Applications

```bash
#!/bin/bash
# deploy-applications.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
ACR_NAME="complyaiacr"
KEY_VAULT="comply-ai-keyvault"

echo "🚀 Deploying applications to Azure..."

# Get resource information
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value -o tsv)

# Create Container Apps environment
echo "🌍 Creating Container Apps environment..."
az containerapp env create \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-env \
    --location eastus

# Deploy detector-orchestration
echo "🔧 Deploying detector-orchestration..."
az containerapp create \
    --resource-group $RESOURCE_GROUP \
    --name detector-orchestration \
    --environment comply-ai-env \
    --image $ACR_LOGIN_SERVER/detector-orchestration:latest \
    --target-port 8000 \
    --ingress external \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 1 \
    --max-replicas 10 \
    --env-vars \
        DETECTOR_ORCHESTRATION_CONFIG_FILE=/etc/config/config.yaml \
        ORCH_ENVIRONMENT=production \
        ORCH_LOG_LEVEL=INFO \
        ORCH_MAX_CONCURRENT_DETECTORS=10 \
        ORCH_DEFAULT_TIMEOUT_MS=5000 \
        ORCH_MAX_RETRIES=2

# Deploy llama-mapper
echo "🔧 Deploying llama-mapper..."
az container create \
    --resource-group $RESOURCE_GROUP \
    --name llama-mapper \
    --image $ACR_LOGIN_SERVER/llama-mapper:latest \
    --registry-login-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --cpu 4 \
    --memory 16 \
    --gpus 1 \
    --gpu-sku K80 \
    --ports 8000 \
    --ip-address Public \
    --dns-name-label comply-ai-mapper \
    --environment-variables \
        LLAMA_MAPPER_ENVIRONMENT=production \
        LLAMA_MAPPER_SERVING__BACKEND=vllm \
        LLAMA_MAPPER_SERVING__HOST=0.0.0.0 \
        LLAMA_MAPPER_SERVING__PORT=8000 \
        LLAMA_MAPPER_STORAGE__DB_HOST=comply-ai-postgres.postgres.database.azure.com \
        LLAMA_MAPPER_STORAGE__DB_PORT=5432 \
        LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper \
        LLAMA_MAPPER_STORAGE__DB_USER=complyaiadmin \
        LLAMA_MAPPER_STORAGE__S3_BUCKET=complyaistorage \
        LLAMA_MAPPER_STORAGE__AWS_REGION=eastus

# Deploy analysis-module (when implemented)
if [ -d "src/llama_mapper/analysis" ]; then
    echo "🔧 Deploying analysis-module..."
    az container create \
        --resource-group $RESOURCE_GROUP \
        --name analysis-module \
        --image $ACR_LOGIN_SERVER/analysis-module:latest \
        --registry-login-server $ACR_LOGIN_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --cpu 2 \
        --memory 8 \
        --gpus 1 \
        --gpu-sku K80 \
        --ports 8001 \
        --ip-address Public \
        --dns-name-label comply-ai-analysis \
        --environment-variables \
            ANALYSIS_MODEL_PATH=/app/models/phi3-mini \
            ANALYSIS_TEMPERATURE=0.1 \
            FALLBACK_ENABLED=true \
            SERVICE_NAME=analysis
fi

echo "✅ Applications deployed successfully!"
echo "📋 Deployed services:"
echo "   - detector-orchestration (Container Apps)"
echo "   - llama-mapper (Container Instances)"
if [ -d "src/llama_mapper/analysis" ]; then
    echo "   - analysis-module (Container Instances)"
fi
```

## Configuration Scripts

### 5. Environment Configuration

```bash
#!/bin/bash
# configure-environment.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
KEY_VAULT="comply-ai-keyvault"
ENVIRONMENT=${1:-production}

echo "⚙️ Configuring environment: $ENVIRONMENT"

# Create environment-specific configuration
cat > .env.$ENVIRONMENT << EOF
# Application settings
LLAMA_MAPPER_APP_NAME=llama-mapper
LLAMA_MAPPER_VERSION=0.1.0
LLAMA_MAPPER_ENVIRONMENT=$ENVIRONMENT
LLAMA_MAPPER_DEBUG=false

# Model configuration
LLAMA_MAPPER_MODEL__NAME=meta-llama/Llama-2-7b-chat-hf
LLAMA_MAPPER_MODEL__TEMPERATURE=0.1
LLAMA_MAPPER_MODEL__TOP_P=0.9
LLAMA_MAPPER_MODEL__MAX_NEW_TOKENS=200

# Serving configuration
LLAMA_MAPPER_SERVING__BACKEND=vllm
LLAMA_MAPPER_SERVING__HOST=0.0.0.0
LLAMA_MAPPER_SERVING__PORT=8000
LLAMA_MAPPER_SERVING__GPU_MEMORY_UTILIZATION=0.9

# Database configuration
LLAMA_MAPPER_STORAGE__DB_HOST=comply-ai-postgres.postgres.database.azure.com
LLAMA_MAPPER_STORAGE__DB_PORT=5432
LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper
LLAMA_MAPPER_STORAGE__DB_USER=complyaiadmin
LLAMA_MAPPER_STORAGE__DB_SSL_MODE=require

# Redis configuration
REDIS_HOST=comply-ai-redis.redis.cache.windows.net
REDIS_PORT=6380
REDIS_SSL=true

# Storage configuration
LLAMA_MAPPER_STORAGE__S3_BUCKET=complyaistorage
LLAMA_MAPPER_STORAGE__AWS_REGION=eastus
LLAMA_MAPPER_STORAGE__AWS_ENDPOINT_URL=https://complyaistorage.blob.core.windows.net

# Orchestration configuration
ORCH_ENVIRONMENT=$ENVIRONMENT
ORCH_LOG_LEVEL=INFO
ORCH_MAX_CONCURRENT_DETECTORS=10
ORCH_DEFAULT_TIMEOUT_MS=5000
ORCH_MAX_RETRIES=2

# Security configuration
LLAMA_MAPPER_SECURITY__SECRETS_BACKEND=azure-keyvault
LLAMA_MAPPER_SECURITY__ENCRYPTION_KEY_ID=your-azure-keyvault-key

# Monitoring configuration
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=your-key;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/
APPLICATIONINSIGHTS_ENABLED=true
EOF

echo "✅ Environment configuration created: .env.$ENVIRONMENT"
echo "📋 Next steps:"
echo "   1. Update secrets in Azure Key Vault"
echo "   2. Deploy applications with this configuration"
echo "   3. Test connectivity and functionality"
```

### 6. Database Migration Script

```bash
#!/bin/bash
# migrate-database.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
POSTGRES_SERVER="comply-ai-postgres"
DATABASE="llama_mapper"
USER="complyaiadmin"

echo "🗄️ Running database migrations..."

# Get password from Key Vault
PASSWORD=$(az keyvault secret show --vault-name comply-ai-keyvault --name postgres-password --query value -o tsv)

# Set connection string
export PGPASSWORD="$PASSWORD"

# Run migrations
echo "📊 Creating tables..."
psql -h $POSTGRES_SERVER.postgres.database.azure.com -p 5432 -U $USER -d $DATABASE -c "
CREATE TABLE IF NOT EXISTS storage_records (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    detector_outputs JSONB NOT NULL,
    mapping_result JSONB NOT NULL,
    confidence_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '90 days'
);

CREATE INDEX IF NOT EXISTS idx_storage_records_tenant_id ON storage_records(tenant_id);
CREATE INDEX IF NOT EXISTS idx_storage_records_request_id ON storage_records(request_id);
CREATE INDEX IF NOT EXISTS idx_storage_records_created_at ON storage_records(created_at);
CREATE INDEX IF NOT EXISTS idx_storage_records_expires_at ON storage_records(expires_at);
"

echo "📊 Creating audit tables..."
psql -h $POSTGRES_SERVER.postgres.database.azure.com -p 5432 -U $USER -d $DATABASE -c "
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(255) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
"

echo "📊 Creating tenant configuration tables..."
psql -h $POSTGRES_SERVER.postgres.database.azure.com -p 5432 -U $USER -d $DATABASE -c "
CREATE TABLE IF NOT EXISTS tenant_configs (
    id SERIAL PRIMARY KEY,
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    config_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tenant_configs_tenant_id ON tenant_configs(tenant_id);
"

echo "📊 Creating model version tracking..."
psql -h $POSTGRES_SERVER.postgres.database.azure.com -p 5432 -U $USER -d $DATABASE -c "
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(255) NOT NULL,
    model_path TEXT NOT NULL,
    checksum VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    activated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_model_versions_model_name ON model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_version ON model_versions(version);
CREATE INDEX IF NOT EXISTS idx_model_versions_is_active ON model_versions(is_active);
"

echo "✅ Database migrations completed successfully!"
```

## Monitoring Scripts

### 7. Monitoring Setup

```bash
#!/bin/bash
# setup-monitoring.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
KEY_VAULT="comply-ai-keyvault"

echo "📊 Setting up monitoring and alerting..."

# Get Application Insights connection string
APP_INSIGHTS_CONNECTION_STRING=$(az monitor app-insights component show \
    --resource-group $RESOURCE_GROUP \
    --app comply-ai-insights \
    --query connectionString -o tsv)

# Store in Key Vault
az keyvault secret set \
    --vault-name $KEY_VAULT \
    --name app-insights-connection-string \
    --value "$APP_INSIGHTS_CONNECTION_STRING"

# Create action group for alerts
echo "📧 Creating action group..."
az monitor action-group create \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-alerts \
    --short-name complyai \
    --email-receivers name=admin email=admin@yourcompany.com

# Create metric alerts
echo "🚨 Creating metric alerts..."

# High CPU usage alert
az monitor metrics alert create \
    --resource-group $RESOURCE_GROUP \
    --name "High CPU Usage" \
    --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP" \
    --condition "avg Percentage CPU > 80" \
    --description "Alert when CPU usage exceeds 80%" \
    --evaluation-frequency 5m \
    --window-size 15m \
    --severity 2 \
    --action comply-ai-alerts

# High memory usage alert
az monitor metrics alert create \
    --resource-group $RESOURCE_GROUP \
    --name "High Memory Usage" \
    --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP" \
    --condition "avg Percentage Memory > 85" \
    --description "Alert when memory usage exceeds 85%" \
    --evaluation-frequency 5m \
    --window-size 15m \
    --severity 2 \
    --action comply-ai-alerts

# Database connection alert
az monitor metrics alert create \
    --resource-group $RESOURCE_GROUP \
    --name "Database Connection Issues" \
    --scopes "/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.DBforPostgreSQL/flexibleServers/comply-ai-postgres" \
    --condition "avg active_connections > 80" \
    --description "Alert when database connections exceed 80" \
    --evaluation-frequency 5m \
    --window-size 15m \
    --severity 1 \
    --action comply-ai-alerts

echo "✅ Monitoring setup completed successfully!"
echo "📋 Created alerts:"
echo "   - High CPU Usage"
echo "   - High Memory Usage"
echo "   - Database Connection Issues"
```

### 8. Health Check Script

```bash
#!/bin/bash
# health-check.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"

echo "🏥 Running health checks..."

# Check resource group
echo "📦 Checking resource group..."
az group show --name $RESOURCE_GROUP --query "name" -o tsv

# Check PostgreSQL
echo "🗄️ Checking PostgreSQL..."
POSTGRES_STATUS=$(az postgres flexible-server show \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-postgres \
    --query "state" -o tsv)
echo "PostgreSQL Status: $POSTGRES_STATUS"

# Check Redis
echo "🔴 Checking Redis..."
REDIS_STATUS=$(az redis show \
    --resource-group $RESOURCE_GROUP \
    --name comply-ai-redis \
    --query "provisioningState" -o tsv)
echo "Redis Status: $REDIS_STATUS"

# Check Storage Account
echo "💾 Checking Storage Account..."
STORAGE_STATUS=$(az storage account show \
    --resource-group $RESOURCE_GROUP \
    --name complyaistorage \
    --query "provisioningState" -o tsv)
echo "Storage Status: $STORAGE_STATUS"

# Check Container Apps
echo "🌍 Checking Container Apps..."
CONTAINER_APP_STATUS=$(az containerapp show \
    --resource-group $RESOURCE_GROUP \
    --name detector-orchestration \
    --query "properties.provisioningState" -o tsv)
echo "Container App Status: $CONTAINER_APP_STATUS"

# Check Container Instances
echo "🐳 Checking Container Instances..."
CONTAINER_STATUS=$(az container show \
    --resource-group $RESOURCE_GROUP \
    --name llama-mapper \
    --query "instanceView.state" -o tsv)
echo "Container Status: $CONTAINER_STATUS"

# Test connectivity
echo "🔗 Testing connectivity..."

# Test PostgreSQL connectivity
echo "Testing PostgreSQL connectivity..."
PASSWORD=$(az keyvault secret show --vault-name comply-ai-keyvault --name postgres-password --query value -o tsv)
export PGPASSWORD="$PASSWORD"
psql -h comply-ai-postgres.postgres.database.azure.com -p 5432 -U complyaiadmin -d llama_mapper -c "SELECT 1;" > /dev/null
echo "✅ PostgreSQL connectivity: OK"

# Test Redis connectivity
echo "Testing Redis connectivity..."
REDIS_KEY=$(az keyvault secret show --vault-name comply-ai-keyvault --name redis-key --query value -o tsv)
redis-cli -h comply-ai-redis.redis.cache.windows.net -p 6380 -a "$REDIS_KEY" ping
echo "✅ Redis connectivity: OK"

# Test storage connectivity
echo "Testing storage connectivity..."
STORAGE_KEY=$(az keyvault secret show --vault-name comply-ai-keyvault --name storage-key --query value -o tsv)
az storage container list --account-name complyaistorage --account-key "$STORAGE_KEY" > /dev/null
echo "✅ Storage connectivity: OK"

echo "✅ All health checks passed!"
```

## Maintenance Scripts

### 9. Backup Script

```bash
#!/bin/bash
# backup.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
BACKUP_CONTAINER="backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "💾 Starting backup process..."

# Create backup container if it doesn't exist
az storage container create \
    --account-name complyaistorage \
    --name $BACKUP_CONTAINER \
    --public-access off

# Backup PostgreSQL
echo "🗄️ Backing up PostgreSQL..."
PASSWORD=$(az keyvault secret show --vault-name comply-ai-keyvault --name postgres-password --query value -o tsv)
export PGPASSWORD="$PASSWORD"

pg_dump -h comply-ai-postgres.postgres.database.azure.com -p 5432 -U complyaiadmin -d llama_mapper > postgres_backup_$TIMESTAMP.sql

# Upload to storage
az storage blob upload \
    --account-name complyaistorage \
    --container-name $BACKUP_CONTAINER \
    --name "postgres/postgres_backup_$TIMESTAMP.sql" \
    --file "postgres_backup_$TIMESTAMP.sql"

# Clean up local file
rm postgres_backup_$TIMESTAMP.sql

echo "✅ Backup completed successfully!"
echo "📋 Backup location: complyaistorage/$BACKUP_CONTAINER/postgres/postgres_backup_$TIMESTAMP.sql"
```

### 10. Scaling Script

```bash
#!/bin/bash
# scale-services.sh

set -e

# Configuration
RESOURCE_GROUP="comply-ai-rg"
SCALE_TYPE=${1:-up}  # up or down
SCALE_FACTOR=${2:-2}

echo "📈 Scaling services: $SCALE_TYPE by factor $SCALE_FACTOR"

if [ "$SCALE_TYPE" = "up" ]; then
    # Scale up Container Apps
    echo "🔧 Scaling up detector-orchestration..."
    CURRENT_REPLICAS=$(az containerapp show \
        --resource-group $RESOURCE_GROUP \
        --name detector-orchestration \
        --query "properties.template.scale.minReplicas" -o tsv)
    
    NEW_REPLICAS=$((CURRENT_REPLICAS * SCALE_FACTOR))
    
    az containerapp update \
        --resource-group $RESOURCE_GROUP \
        --name detector-orchestration \
        --min-replicas $NEW_REPLICAS \
        --max-replicas $((NEW_REPLICAS * 2))
    
    echo "✅ Scaled detector-orchestration to $NEW_REPLICAS replicas"
    
elif [ "$SCALE_TYPE" = "down" ]; then
    # Scale down Container Apps
    echo "🔧 Scaling down detector-orchestration..."
    CURRENT_REPLICAS=$(az containerapp show \
        --resource-group $RESOURCE_GROUP \
        --name detector-orchestration \
        --query "properties.template.scale.minReplicas" -o tsv)
    
    NEW_REPLICAS=$((CURRENT_REPLICAS / SCALE_FACTOR))
    NEW_REPLICAS=$((NEW_REPLICAS > 1 ? NEW_REPLICAS : 1))
    
    az containerapp update \
        --resource-group $RESOURCE_GROUP \
        --name detector-orchestration \
        --min-replicas $NEW_REPLICAS \
        --max-replicas $((NEW_REPLICAS * 2))
    
    echo "✅ Scaled detector-orchestration to $NEW_REPLICAS replicas"
fi

echo "✅ Scaling completed successfully!"
```

## Usage Instructions

### Complete Deployment

```bash
# 1. Deploy infrastructure
./deploy-infrastructure.sh

# 2. Setup secrets
./setup-secrets.sh

# 3. Build and push images
./build-and-push-images.sh

# 4. Deploy applications
./deploy-applications.sh

# 5. Configure environment
./configure-environment.sh production

# 6. Run database migrations
./migrate-database.sh

# 7. Setup monitoring
./setup-monitoring.sh

# 8. Run health checks
./health-check.sh
```

### Maintenance Operations

```bash
# Backup data
./backup.sh

# Scale services up
./scale-services.sh up 2

# Scale services down
./scale-services.sh down 2

# Health check
./health-check.sh
```

### Environment Management

```bash
# Create development environment
./configure-environment.sh development

# Create staging environment
./configure-environment.sh staging

# Create production environment
./configure-environment.sh production
```

These scripts provide a complete automation solution for deploying and managing the Comply-AI platform on Azure.
