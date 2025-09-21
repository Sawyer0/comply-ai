"""
Azure-Native Data Models for Llama Mapper System

This module defines all data models used in the Azure-native implementation
of the Llama Mapper system, including storage records, audit logs, and
configuration models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum
import uuid

from pydantic import BaseModel, Field, field_validator


class TenantAccessLevel(str, Enum):
    """Tenant access levels for data isolation."""
    STRICT = "strict"  # Complete isolation, no cross-tenant access
    SHARED = "shared"  # Limited shared resources with tenant filtering
    ADMIN = "admin"    # Administrative access across tenants


class StorageBackend(str, Enum):
    """Supported storage backends."""
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    BLOB_STORAGE = "blob_storage"


class EncryptionType(str, Enum):
    """Encryption types for data protection."""
    AES256 = "AES256"
    AES256_KMS = "AES256-KMS"
    AZURE_KEY_VAULT = "azure_key_vault"


# Core Data Models

@dataclass
class AzureStorageRecord:
    """Azure-optimized storage record with region and subscription tracking."""
    id: uuid.UUID
    source_data: str
    mapped_data: Dict[str, Any]
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tenant_id: str
    s3_key: Optional[str] = None
    encrypted: bool = False
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Azure-specific fields
    azure_region: str = "eastus"
    resource_group: str = "comply-ai-rg"
    subscription_id: Optional[str] = None
    blob_url: Optional[str] = None
    container_name: str = "mapper-outputs"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(days=90)


@dataclass
class AzureAuditRecord:
    """Azure-compliant audit record with resource tracking."""
    event_id: str
    tenant_id: str
    user_id: Optional[str] = None
    action: str = ""
    resource_type: str = ""
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Azure-specific fields
    azure_region: str = "eastus"
    subscription_id: Optional[str] = None
    resource_group: str = "comply-ai-rg"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary."""
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "azure_region": self.azure_region,
            "subscription_id": self.subscription_id,
            "resource_group": self.resource_group,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AzureTenantConfig:
    """Tenant configuration with Azure resource mapping."""
    tenant_id: str
    config_data: Dict[str, Any]
    azure_subscription_id: Optional[str] = None
    azure_resource_group: str = "comply-ai-rg"
    azure_region: str = "eastus"
    azure_key_vault_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update tenant configuration."""
        self.config_data.update(new_config)
        self.updated_at = datetime.utcnow()


@dataclass
class AzureModelVersion:
    """Model version tracking with Azure Blob Storage integration."""
    model_name: str
    version: str
    model_path: str
    azure_blob_url: Optional[str] = None
    azure_container_name: str = "model-artifacts"
    checksum: Optional[str] = None
    is_active: bool = False
    azure_region: str = "eastus"
    created_at: datetime = field(default_factory=datetime.utcnow)
    activated_at: Optional[datetime] = None
    
    def activate(self) -> None:
        """Activate this model version."""
        self.is_active = True
        self.activated_at = datetime.utcnow()


@dataclass
class AzureKeyVaultSecret:
    """Azure Key Vault secret management."""
    secret_name: str
    secret_value: str
    vault_url: str
    tenant_id: str
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_expired(self) -> bool:
        """Check if secret is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class AzureMonitorMetrics:
    """Azure Monitor metrics and alerting."""
    metric_name: str
    metric_value: float
    dimensions: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resource_id: str = ""
    azure_region: str = "eastus"
    
    def to_azure_format(self) -> Dict[str, Any]:
        """Convert to Azure Monitor format."""
        return {
            "name": self.metric_name,
            "value": self.metric_value,
            "dimensions": self.dimensions,
            "timestamp": self.timestamp.isoformat(),
            "resourceId": self.resource_id,
            "region": self.azure_region
        }


# Configuration Models

class AzureStorageConfig(BaseModel):
    """Azure Blob Storage configuration."""
    storage_account: str = Field(..., description="Azure Storage Account name")
    container_name: str = Field(default="mapper-outputs", description="Container name")
    access_key: Optional[str] = Field(None, description="Storage account access key")
    connection_string: Optional[str] = Field(None, description="Storage connection string")
    sas_token: Optional[str] = Field(None, description="SAS token for access")
    endpoint_url: Optional[str] = Field(None, description="Custom endpoint URL")
    region: str = Field(default="eastus", description="Azure region")
    
    @field_validator("storage_account")
    @classmethod
    def validate_storage_account(cls, v: str) -> str:
        """Validate storage account name."""
        if not v or len(v) < 3 or len(v) > 24:
            raise ValueError("Storage account name must be 3-24 characters")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Storage account name must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()


class AzureDatabaseConfig(BaseModel):
    """Azure Database for PostgreSQL configuration."""
    host: str = Field(..., description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(..., description="Database name")
    username: str = Field(..., description="Database username")
    password: Optional[str] = Field(None, description="Database password")
    ssl_mode: str = Field(default="require", description="SSL mode")
    connection_pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=1, le=200, description="Max overflow connections")
    region: str = Field(default="eastus", description="Azure region")
    
    @field_validator("ssl_mode")
    @classmethod
    def validate_ssl_mode(cls, v: str) -> str:
        """Validate SSL mode."""
        valid_modes = ["disable", "allow", "prefer", "require", "verify-ca", "verify-full"]
        if v not in valid_modes:
            raise ValueError(f"SSL mode must be one of: {valid_modes}")
        return v


class AzureRedisConfig(BaseModel):
    """Azure Cache for Redis configuration."""
    host: str = Field(..., description="Redis host")
    port: int = Field(default=6380, ge=1, le=65535, description="Redis port")
    password: Optional[str] = Field(None, description="Redis password")
    ssl: bool = Field(default=True, description="Enable SSL")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    max_connections: int = Field(default=50, ge=1, le=1000, description="Max connections")
    region: str = Field(default="eastus", description="Azure region")
    
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate Redis port."""
        if v not in [6379, 6380]:
            raise ValueError("Redis port must be 6379 (non-SSL) or 6380 (SSL)")
        return v


class AzureKeyVaultConfig(BaseModel):
    """Azure Key Vault configuration."""
    vault_url: str = Field(..., description="Key Vault URL")
    tenant_id: str = Field(..., description="Azure tenant ID")
    client_id: Optional[str] = Field(None, description="Client ID for authentication")
    client_secret: Optional[str] = Field(None, description="Client secret")
    use_managed_identity: bool = Field(default=True, description="Use managed identity")
    
    @field_validator("vault_url")
    @classmethod
    def validate_vault_url(cls, v: str) -> str:
        """Validate Key Vault URL."""
        if not v.startswith("https://") or not v.endswith(".vault.azure.net/"):
            raise ValueError("Key Vault URL must be in format: https://{vault-name}.vault.azure.net/")
        return v


class AzureMonitorConfig(BaseModel):
    """Azure Monitor configuration."""
    workspace_id: str = Field(..., description="Log Analytics workspace ID")
    workspace_key: Optional[str] = Field(None, description="Workspace key")
    application_insights_key: Optional[str] = Field(None, description="App Insights key")
    region: str = Field(default="eastus", description="Azure region")
    
    @field_validator("workspace_id")
    @classmethod
    def validate_workspace_id(cls, v: str) -> str:
        """Validate workspace ID format."""
        if not v or len(v) != 36:
            raise ValueError("Workspace ID must be a 36-character GUID")
        return v


# API Models

class MapperPayload(BaseModel):
    """Locked handoff schema from Orchestrator → Mapper."""
    detector: str = Field(..., description="Originating orchestrated detector label")
    output: str = Field(..., description="Aggregated raw indication")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Handoff metadata")
    tenant_id: str = Field(..., min_length=1, max_length=64, description="Tenant identifier")
    
    @field_validator("detector")
    @classmethod
    def validate_detector(cls, v: str) -> str:
        """Validate detector name."""
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Detector name must contain only alphanumeric characters, hyphens, and underscores")
        return v


class MappingResponse(BaseModel):
    """Response model following pillars-detectors/schema.json."""
    taxonomy: List[str] = Field(..., min_length=1, description="Canonical taxonomy labels")
    scores: Dict[str, float] = Field(..., description="Map of taxonomy labels to normalized scores [0,1]")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model-calibrated confidence")
    notes: Optional[str] = Field(None, max_length=500, description="Optional debugging notes")
    provenance: Optional[Dict[str, Any]] = None
    policy_context: Optional[Dict[str, Any]] = None
    version_info: Optional[Dict[str, str]] = Field(None, description="Component version tags")
    
    @field_validator("taxonomy")
    @classmethod
    def validate_taxonomy_format(cls, v: List[str]) -> List[str]:
        """Validate taxonomy label format."""
        import re
        pattern = r"^[A-Z][A-Z0-9_]*(\.[A-Za-z0-9_]+)*$"
        for item in v:
            if not re.match(pattern, item):
                raise ValueError(f'Taxonomy label "{item}" does not match required pattern')
        return v
    
    @field_validator("scores")
    @classmethod
    def validate_scores_range(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all scores are in [0,1] range."""
        for label, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f'Score for "{label}" must be between 0.0 and 1.0')
        return v


class Provenance(BaseModel):
    """Provenance information for tracking detector outputs."""
    vendor: Optional[str] = None
    detector: Optional[str] = None
    detector_version: Optional[str] = None
    raw_ref: Optional[str] = Field(None, description="Pointer/ID to raw event")
    route: Optional[str] = None
    model: Optional[str] = None
    tenant_id: Optional[str] = None
    ts: Optional[datetime] = None


class PolicyContext(BaseModel):
    """Policy context for detector expectations."""
    expected_detectors: Optional[List[str]] = None
    environment: Optional[str] = Field(None, pattern="^(dev|stage|prod)$")


class VersionInfo(BaseModel):
    """Version information for components."""
    taxonomy: str = Field(..., description="Taxonomy version")
    frameworks: str = Field(..., description="Frameworks version")
    model: str = Field(..., description="Model version")


# Compliance Models

@dataclass
class ComplianceControlMapping:
    """Mapping between taxonomy labels and compliance controls."""
    taxonomy_label: str
    framework_name: str
    control_id: str
    control_description: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "taxonomy_label": self.taxonomy_label,
            "framework_name": self.framework_name,
            "control_id": self.control_id,
            "control_description": self.control_description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ComplianceReport:
    """Compliance report for audit purposes."""
    report_id: str
    tenant_id: str
    framework_name: str
    report_period_start: datetime
    report_period_end: datetime
    total_events: int
    compliance_score: float
    violations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "tenant_id": self.tenant_id,
            "framework_name": self.framework_name,
            "report_period_start": self.report_period_start.isoformat(),
            "report_period_end": self.report_period_end.isoformat(),
            "total_events": self.total_events,
            "compliance_score": self.compliance_score,
            "violations": self.violations,
            "recommendations": self.recommendations,
            "generated_at": self.generated_at.isoformat(),
        }


# Metrics Models

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
    key_vault_operations: int = 0
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.get_success_rate(),
            "average_latency_ms": self.average_latency_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "blob_operations": self.blob_operations,
            "database_operations": self.database_operations,
            "key_vault_operations": self.key_vault_operations,
        }


@dataclass
class HealthStatus:
    """Health status for Azure services."""
    service_name: str
    status: str  # healthy, unhealthy, degraded
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "status": self.status,
            "last_check": self.last_check.isoformat(),
            "error_message": self.error_message,
            "response_time_ms": self.response_time_ms,
        }
