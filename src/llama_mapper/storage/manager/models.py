"""Data structures shared by storage manager components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, List
# UUID imports available if needed
# from uuid import UUID, uuid4


class StorageBackend(Enum):
    """Supported storage backends."""

    CLICKHOUSE = "clickhouse"
    POSTGRESQL = "postgresql"


@dataclass
class StorageRecord:
    """Record structure for storing mapping results."""

    # pylint: disable=too-many-instance-attributes

    id: str
    source_data: str  # This will be deprecated in favor of source_data_hash for privacy
    mapped_data: str
    model_version: str
    timestamp: datetime
    metadata: Dict[str, Any]
    tenant_id: str = "unknown"
    s3_key: Optional[str] = None
    encrypted: bool = False
    
    # Enhanced fields for production readiness
    source_data_hash: Optional[str] = None  # SHA-256 hash for privacy compliance
    detector_type: Optional[str] = None
    confidence_score: Optional[float] = None
    correlation_id: Optional[str] = None
    azure_region: str = "eastus"
    backup_status: str = "pending"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'source_data': self.source_data,
            'source_data_hash': self.source_data_hash,
            'mapped_data': self.mapped_data,
            'model_version': self.model_version,
            'detector_type': self.detector_type,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
            'tenant_id': self.tenant_id,
            's3_key': self.s3_key,
            'encrypted': self.encrypted,
            'correlation_id': self.correlation_id,
            'azure_region': self.azure_region,
            'backup_status': self.backup_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


@dataclass
class AuditRecord:
    """Audit trail record for compliance tracking."""
    
    id: str
    tenant_id: str
    table_name: str
    record_id: str
    operation: str  # INSERT, UPDATE, DELETE, SELECT
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'table_name': self.table_name,
            'record_id': self.record_id,
            'operation': self.operation,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'old_values': self.old_values,
            'new_values': self.new_values,
            'correlation_id': self.correlation_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent
        }


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""
    
    tenant_id: str
    confidence_threshold: float = 0.6
    detector_whitelist: Optional[List[str]] = None
    detector_blacklist: Optional[List[str]] = None
    storage_retention_days: int = 90
    encryption_enabled: bool = True
    audit_level: str = "standard"  # minimal, standard, verbose
    custom_taxonomy_mappings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'tenant_id': self.tenant_id,
            'confidence_threshold': self.confidence_threshold,
            'detector_whitelist': self.detector_whitelist,
            'detector_blacklist': self.detector_blacklist,
            'storage_retention_days': self.storage_retention_days,
            'encryption_enabled': self.encryption_enabled,
            'audit_level': self.audit_level,
            'custom_taxonomy_mappings': self.custom_taxonomy_mappings,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ModelMetric:
    """Model performance metrics."""
    
    id: str
    model_version: str
    tenant_id: str
    metric_type: str
    metric_value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'model_version': self.model_version,
            'tenant_id': self.tenant_id,
            'metric_type': self.metric_type,
            'metric_value': self.metric_value,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class DetectorExecution:
    """Detector execution logging."""
    
    id: str
    tenant_id: str
    detector_type: str
    execution_time_ms: int
    confidence_score: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'detector_type': self.detector_type,
            'execution_time_ms': self.execution_time_ms,
            'confidence_score': self.confidence_score,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id
        }


class StorageAccessError(PermissionError):
    """Raised when tenant isolation rules forbid an operation."""


class DatabaseMigrationError(Exception):
    """Raised when database migration operations fail."""


class DatabaseConnectionError(Exception):
    """Raised when database connection operations fail."""


class DatabaseOperationError(Exception):
    """Raised when database operations fail."""


class DatabaseUnavailableError(Exception):
    """Raised when database is unavailable (circuit breaker open)."""


__all__ = [
    "StorageBackend", 
    "StorageRecord", 
    "AuditRecord",
    "TenantConfig",
    "ModelMetric",
    "DetectorExecution",
    "StorageAccessError",
    "DatabaseMigrationError",
    "DatabaseConnectionError",
    "DatabaseOperationError",
    "DatabaseUnavailableError"
]
