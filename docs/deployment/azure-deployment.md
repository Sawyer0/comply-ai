# Azure Deployment Guide

This document provides comprehensive instructions for deploying the Comply-AI platform to Microsoft Azure.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Azure Services Architecture](#azure-services-architecture)
4. [Infrastructure Setup](#infrastructure-setup)
5. [Service Deployment](#service-deployment)
6. [Configuration](#configuration)
7. [Monitoring and Security](#monitoring-and-security)
8. [Cost Management](#cost-management)
9. [Troubleshooting](#troubleshooting)

## Overview

The Comply-AI platform consists of three main services:

- **Detector Orchestration**: Lightweight orchestration service
- **Llama Mapper**: AI/ML model serving (8B model)
- **Analysis Module**: AI/ML analysis service (Phi-3 Mini)

### Azure Services Used

- **Azure Container Instances (ACI)**: GPU-enabled containers for AI models
- **Azure Container Apps**: Lightweight orchestration service
- **Azure Database for PostgreSQL**: Primary database
- **Azure Cache for Redis**: Caching layer
- **Azure Blob Storage**: Immutable storage (S3-compatible)
- **Azure Key Vault**: Secrets management
- **Azure Monitor**: Monitoring and alerting
- **Azure Application Gateway**: Load balancing and SSL termination

## Prerequisites

### Azure Account Setup

1. **Azure Subscription**: Active Azure subscription
2. **Azure CLI**: Install and configure Azure CLI
3. **Docker**: For building container images
4. **Azure Container Registry**: For storing container images

### Required Permissions

```bash
# Required Azure roles
- Contributor (for resource creation)
- Key Vault Administrator (for secrets management)
- Application Administrator (for service principals)
```

### Azure CLI Setup

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Create resource group
az group create --name comply-ai-rg --location eastus
```

## Azure Services Architecture

### Infrastructure Overview

```yaml
# Resource Group: comply-ai-rg
# Location: East US

Services:
  detector-orchestration:
    service: Azure Container Apps
    cpu: 1-2 vCPUs
    memory: 2-4 GB
    cost: ~$40/month

  llama-mapper:
    service: Azure Container Instances (GPU)
    gpu: 1x T4 (16GB VRAM)
    cpu: 4 vCPUs
    memory: 16 GB
    cost: ~$180/month

  analysis-module:
    service: Azure Container Instances (GPU)
    gpu: 1x T4 (16GB VRAM)
    cpu: 2 vCPUs
    memory: 8 GB
    cost: ~$90/month

  postgresql:
    service: Azure Database for PostgreSQL
    tier: General Purpose
    vCores: 2
    memory: 10 GB
    storage: 100 GB
    cost: ~$40/month

  redis:
    service: Azure Cache for Redis
    tier: Basic
    size: C1 (1 GB)
    cost: ~$15/month

  storage:
    service: Azure Blob Storage
    tier: Standard
    redundancy: LRS
    cost: ~$5/month

  networking:
    service: Azure Application Gateway
    tier: Standard_v2
    size: Medium
    cost: ~$25/month

  monitoring:
    service: Azure Monitor
    features: Application Insights, Log Analytics
    cost: ~$20/month

  security:
    service: Azure Key Vault
    tier: Standard
    cost: ~$5/month

# Total Monthly Cost: ~$435
```

### Network Architecture

```yaml
# Virtual Network: comply-ai-vnet
# Address Space: 10.0.0.0/16

Subnets:
  orchestration-subnet: 10.0.1.0/24
  mapper-subnet: 10.0.2.0/24
  analysis-subnet: 10.0.3.0/24
  database-subnet: 10.0.4.0/24
  gateway-subnet: 10.0.5.0/24
```

## Infrastructure Setup

### 1. Create Resource Group

```bash
# Create resource group
az group create \
  --name comply-ai-rg \
  --location eastus \
  --tags Environment=Production Project=ComplyAI
```

### 2. Create Virtual Network

```bash
# Create virtual network
az network vnet create \
  --resource-group comply-ai-rg \
  --name comply-ai-vnet \
  --address-prefix 10.0.0.0/16 \
  --location eastus

# Create subnets
az network vnet subnet create \
  --resource-group comply-ai-rg \
  --vnet-name comply-ai-vnet \
  --name orchestration-subnet \
  --address-prefix 10.0.1.0/24

az network vnet subnet create \
  --resource-group comply-ai-rg \
  --vnet-name comply-ai-vnet \
  --name mapper-subnet \
  --address-prefix 10.0.2.0/24

az network vnet subnet create \
  --resource-group comply-ai-rg \
  --vnet-name comply-ai-vnet \
  --name analysis-subnet \
  --address-prefix 10.0.3.0/24

az network vnet subnet create \
  --resource-group comply-ai-rg \
  --vnet-name comply-ai-vnet \
  --name database-subnet \
  --address-prefix 10.0.4.0/24

az network vnet subnet create \
  --resource-group comply-ai-rg \
  --vnet-name comply-ai-vnet \
  --name gateway-subnet \
  --address-prefix 10.0.5.0/24
```

### 3. Create Azure Database for PostgreSQL

```bash
# Create PostgreSQL server
az postgres flexible-server create \
  --resource-group comply-ai-rg \
  --name comply-ai-postgres \
  --location eastus \
  --admin-user complyaiadmin \
  --admin-password "YourSecurePassword123!" \
  --sku-name Standard_D2s_v3 \
  --tier GeneralPurpose \
  --public-access 0.0.0.0 \
  --storage-size 100 \
  --version 15

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

### 4. Create Azure Cache for Redis

```bash
# Create Redis cache
az redis create \
  --resource-group comply-ai-rg \
  --name comply-ai-redis \
  --location eastus \
  --sku Basic \
  --vm-size c1 \
  --enable-non-ssl-port false
```

### 5. Create Azure Blob Storage

```bash
# Create storage account
az storage account create \
  --resource-group comply-ai-rg \
  --name complyaistorage \
  --location eastus \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot

# Create container
az storage container create \
  --account-name complyaistorage \
  --name mapper-outputs \
  --public-access off
```

### 6. Create Azure Key Vault

```bash
# Create Key Vault
az keyvault create \
  --resource-group comply-ai-rg \
  --name comply-ai-keyvault \
  --location eastus \
  --sku standard \
  --enable-rbac-authorization true

# Create secrets
az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name postgres-password \
  --value "YourSecurePassword123!"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name redis-key \
  --value "$(az redis list-keys --resource-group comply-ai-rg --name comply-ai-redis --query primaryKey -o tsv)"

az keyvault secret set \
  --vault-name comply-ai-keyvault \
  --name storage-key \
  --value "$(az storage account keys list --resource-group comply-ai-rg --account-name complyaistorage --query [0].value -o tsv)"
```

### 7. Create Azure Container Registry

```bash
# Create container registry
az acr create \
  --resource-group comply-ai-rg \
  --name complyaiacr \
  --sku Basic \
  --admin-enabled true

# Get login server
az acr show --name complyaiacr --query loginServer --output tsv
```

## Service Deployment

### 1. Build and Push Container Images

```bash
# Login to ACR
az acr login --name complyaiacr

# Build and push detector-orchestration
cd detector-orchestration
docker build -t complyaiacr.azurecr.io/detector-orchestration:latest .
docker push complyaiacr.azurecr.io/detector-orchestration:latest

# Build and push llama-mapper
cd ../src/llama_mapper
docker build -t complyaiacr.azurecr.io/llama-mapper:latest .
docker push complyaiacr.azurecr.io/llama-mapper:latest

# Build and push analysis-module (when implemented)
cd ../analysis-module
docker build -t complyaiacr.azurecr.io/analysis-module:latest .
docker push complyaiacr.azurecr.io/analysis-module:latest
```

### 2. Deploy Detector Orchestration (Azure Container Apps)

```bash
# Create Container Apps environment
az containerapp env create \
  --resource-group comply-ai-rg \
  --name comply-ai-env \
  --location eastus

# Deploy detector-orchestration
az containerapp create \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --environment comply-ai-env \
  --image complyaiacr.azurecr.io/detector-orchestration:latest \
  --target-port 8000 \
  --ingress external \
  --registry-server complyaiacr.azurecr.io \
  --registry-username complyaiacr \
  --registry-password "$(az acr credential show --name complyaiacr --query passwords[0].value -o tsv)" \
  --cpu 1.0 \
  --memory 2.0Gi \
  --min-replicas 1 \
  --max-replicas 10 \
  --env-vars \
    DETECTOR_ORCHESTRATION_CONFIG_FILE=/etc/config/config.yaml \
    ORCH_ENVIRONMENT=production \
    ORCH_LOG_LEVEL=INFO
```

### 3. Deploy Llama Mapper (Azure Container Instances)

```bash
# Create container group for llama-mapper
az container create \
  --resource-group comply-ai-rg \
  --name llama-mapper \
  --image complyaiacr.azurecr.io/llama-mapper:latest \
  --registry-login-server complyaiacr.azurecr.io \
  --registry-username complyaiacr \
  --registry-password "$(az acr credential show --name complyaiacr --query passwords[0].value -o tsv)" \
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
```

### 4. Deploy Analysis Module (Azure Container Instances)

```bash
# Create container group for analysis-module
az container create \
  --resource-group comply-ai-rg \
  --name analysis-module \
  --image complyaiacr.azurecr.io/analysis-module:latest \
  --registry-login-server complyaiacr.azurecr.io \
  --registry-username complyaiacr \
  --registry-password "$(az acr credential show --name complyaiacr --query passwords[0].value -o tsv)" \
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
```

### 5. Create Application Gateway

```bash
# Create public IP
az network public-ip create \
  --resource-group comply-ai-rg \
  --name comply-ai-gateway-ip \
  --allocation-method Static \
  --sku Standard

# Create application gateway
az network application-gateway create \
  --resource-group comply-ai-rg \
  --name comply-ai-gateway \
  --location eastus \
  --sku Standard_v2 \
  --capacity 2 \
  --vnet-name comply-ai-vnet \
  --subnet gateway-subnet \
  --public-ip-address comply-ai-gateway-ip \
  --servers comply-ai-mapper.eastus.azurecontainer.io \
  --http-settings-port 8000 \
  --http-settings-protocol Http \
  --frontend-port 80 \
  --routing-rule-type Basic
```

## Configuration

### Environment Variables

Create a comprehensive environment configuration file:

```bash
# Create azure.env file
cat > azure.env << EOF
# Application settings
LLAMA_MAPPER_APP_NAME=llama-mapper
LLAMA_MAPPER_VERSION=0.1.0
LLAMA_MAPPER_ENVIRONMENT=production
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

# Azure Database for PostgreSQL
LLAMA_MAPPER_STORAGE__DB_HOST=comply-ai-postgres.postgres.database.azure.com
LLAMA_MAPPER_STORAGE__DB_PORT=5432
LLAMA_MAPPER_STORAGE__DB_NAME=llama_mapper
LLAMA_MAPPER_STORAGE__DB_USER=complyaiadmin
LLAMA_MAPPER_STORAGE__DB_PASSWORD=YourSecurePassword123!

# Azure Blob Storage (S3-compatible)
LLAMA_MAPPER_STORAGE__S3_BUCKET=complyaistorage
LLAMA_MAPPER_STORAGE__AWS_ACCESS_KEY_ID=your-storage-account-key
LLAMA_MAPPER_STORAGE__AWS_SECRET_ACCESS_KEY=your-storage-account-secret
LLAMA_MAPPER_STORAGE__AWS_REGION=eastus

# Azure Cache for Redis
REDIS_HOST=comply-ai-redis.redis.cache.windows.net
REDIS_PORT=6380
REDIS_PASSWORD=your-redis-key
REDIS_SSL=true

# Logging configuration
LLAMA_MAPPER_LOGGING__LEVEL=INFO
LLAMA_MAPPER_LOGGING__FORMAT=json
LLAMA_MAPPER_LOGGING__PRIVACY_MODE=true

# Security configuration
LLAMA_MAPPER_SECURITY__SECRETS_BACKEND=azure-keyvault
LLAMA_MAPPER_SECURITY__ENCRYPTION_KEY_ID=your-azure-keyvault-key

# Detector Orchestration
ORCH_ENVIRONMENT=production
ORCH_LOG_LEVEL=INFO
ORCH_MAX_CONCURRENT_DETECTORS=10
ORCH_DEFAULT_TIMEOUT_MS=5000
ORCH_MAX_RETRIES=2
EOF
```

### Key Vault Integration

```bash
# Create managed identity for services
az identity create \
  --resource-group comply-ai-rg \
  --name comply-ai-identity

# Grant Key Vault access
az keyvault set-policy \
  --name comply-ai-keyvault \
  --object-id "$(az identity show --resource-group comply-ai-rg --name comply-ai-identity --query principalId -o tsv)" \
  --secret-permissions get list
```

## Monitoring and Security

### 1. Azure Monitor Setup

```bash
# Create Log Analytics workspace
az monitor log-analytics workspace create \
  --resource-group comply-ai-rg \
  --workspace-name comply-ai-logs \
  --location eastus

# Create Application Insights
az monitor app-insights component create \
  --resource-group comply-ai-rg \
  --app comply-ai-insights \
  --location eastus \
  --kind web \
  --application-type web \
  --workspace comply-ai-logs
```

### 2. Security Configuration

```bash
# Enable Azure Security Center
az security pricing create \
  --name VirtualMachines \
  --tier standard

# Create Network Security Groups
az network nsg create \
  --resource-group comply-ai-rg \
  --name comply-ai-nsg \
  --location eastus

# Add security rules
az network nsg rule create \
  --resource-group comply-ai-rg \
  --nsg-name comply-ai-nsg \
  --name AllowHTTPS \
  --priority 100 \
  --source-address-prefixes Internet \
  --destination-port-ranges 443 \
  --access Allow \
  --protocol Tcp \
  --direction Inbound
```

### 3. Backup Configuration

```bash
# Create backup vault
az backup vault create \
  --resource-group comply-ai-rg \
  --name comply-ai-backup-vault \
  --location eastus

# Configure PostgreSQL backup
az backup protection enable-for-azurewl \
  --resource-group comply-ai-rg \
  --vault-name comply-ai-backup-vault \
  --policy-name DefaultPolicy \
  --protectable-item-type AzureVMWorkload \
  --protectable-item-name comply-ai-postgres
```

## Cost Management

### 1. Cost Optimization

```yaml
# Use Spot Instances for non-critical workloads
# Enable auto-scaling
# Use reserved instances for predictable workloads
# Monitor and alert on cost thresholds
```

### 2. Cost Monitoring

```bash
# Create budget alert
az consumption budget create \
  --resource-group comply-ai-rg \
  --budget-name comply-ai-budget \
  --amount 500 \
  --time-grain Monthly \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### 3. Resource Tagging

```bash
# Tag resources for cost tracking
az resource tag \
  --resource-group comply-ai-rg \
  --tags Environment=Production Project=ComplyAI CostCenter=Engineering
```

## Troubleshooting

### Common Issues

1. **Container Startup Failures**
   ```bash
   # Check container logs
   az container logs --resource-group comply-ai-rg --name llama-mapper
   
   # Check container status
   az container show --resource-group comply-ai-rg --name llama-mapper --query instanceView
   ```

2. **Database Connection Issues**
   ```bash
   # Test database connectivity
   az postgres flexible-server show \
     --resource-group comply-ai-rg \
     --name comply-ai-postgres
   
   # Check firewall rules
   az postgres flexible-server firewall-rule list \
     --resource-group comply-ai-rg \
     --name comply-ai-postgres
   ```

3. **GPU Resource Issues**
   ```bash
   # Check GPU availability
   az vm list-skus --location eastus --query "[?contains(name, 'Standard_NC')]"
   
   # Monitor GPU usage
   az monitor metrics list \
     --resource comply-ai-rg \
     --metric "GPUUtilization"
   ```

### Performance Monitoring

```bash
# Monitor application performance
az monitor app-insights query \
  --app comply-ai-insights \
  --analytics-query "requests | summarize count() by bin(timestamp, 1h)"

# Check resource utilization
az monitor metrics list \
  --resource comply-ai-rg \
  --metric "CpuPercentage" "MemoryPercentage"
```

### Scaling Operations

```bash
# Scale Container Apps
az containerapp update \
  --resource-group comply-ai-rg \
  --name detector-orchestration \
  --min-replicas 2 \
  --max-replicas 20

# Scale Container Instances (requires recreation)
az container delete --resource-group comply-ai-rg --name llama-mapper
# Then recreate with new resource specifications
```

## Next Steps

1. **Implement Analysis Module**: Complete the analysis module implementation
2. **Add CI/CD Pipeline**: Set up Azure DevOps or GitHub Actions
3. **Implement Monitoring**: Add custom metrics and alerting
4. **Security Hardening**: Implement additional security measures
5. **Performance Optimization**: Fine-tune resource allocation
6. **Disaster Recovery**: Implement backup and recovery procedures

## Support

For issues and questions:
- Check Azure Service Health: https://status.azure.com/
- Azure Support: https://azure.microsoft.com/support/
- Documentation: https://docs.microsoft.com/azure/
