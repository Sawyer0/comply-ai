"""
Shared Tenant Isolation Interfaces

This module provides shared interfaces and models for tenant isolation
across all microservices, based on the existing llama_mapper implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from abc import ABC, abstractmethod


class TenantAccessLevel(Enum):
    """Tenant access levels for data isolation."""

    STRICT = "strict"  # Complete isolation, no cross-tenant access
    SHARED = "shared"  # Limited shared resources with tenant filtering
    ADMIN = "admin"  # Administrative access across tenants


@dataclass
class TenantContext:
    """Context information for tenant operations."""

    tenant_id: str
    access_level: TenantAccessLevel = TenantAccessLevel.STRICT
    allowed_tenants: Optional[Set[str]] = None  # For shared access
    configuration_overrides: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TenantConfig:
    """Tenant-specific configuration."""

    tenant_id: str
    confidence_threshold: Optional[float] = None
    detector_whitelist: Optional[List[str]] = None
    detector_blacklist: Optional[List[str]] = None
    storage_retention_days: Optional[int] = None
    encryption_enabled: bool = True
    audit_level: str = "standard"  # minimal, standard, verbose
    custom_taxonomy_mappings: Optional[Dict[str, str]] = None


class TenantIsolationError(Exception):
    """Exception raised when tenant isolation is violated."""

    pass


class ITenantIsolationManager(ABC):
    """Interface for tenant isolation management across services."""

    @abstractmethod
    def create_tenant_context(
        self,
        tenant_id: str,
        access_level: TenantAccessLevel = TenantAccessLevel.STRICT,
        allowed_tenants: Optional[Set[str]] = None,
        configuration_overrides: Optional[Dict[str, Any]] = None,
    ) -> TenantContext:
        """Create a tenant context for scoped operations."""
        pass

    @abstractmethod
    def get_tenant_context(self, tenant_id: str) -> Optional[TenantContext]:
        """Get cached tenant context."""
        pass

    @abstractmethod
    def validate_tenant_access(
        self, requesting_tenant: str, target_tenant: str, operation: str = "read"
    ) -> bool:
        """Validate if a tenant can access another tenant's data."""
        pass

    @abstractmethod
    def apply_tenant_filter(
        self, query: str, tenant_context: TenantContext, table_alias: str = ""
    ) -> str:
        """Apply tenant filtering to database queries."""
        pass

    @abstractmethod
    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Get tenant-specific configuration with fallback to defaults."""
        pass

    @abstractmethod
    def update_tenant_config(
        self, tenant_id: str, config_or_overrides: Union[TenantConfig, Dict[str, Any]]
    ) -> None:
        """Update tenant-specific configuration."""
        pass

    @abstractmethod
    def validate_detector_access(self, tenant_id: str, detector_name: str) -> bool:
        """Validate if a tenant can use a specific detector."""
        pass

    @abstractmethod
    def get_effective_confidence_threshold(self, tenant_id: str) -> float:
        """Get the effective confidence threshold for a tenant."""
        pass

    @abstractmethod
    def create_tenant_scoped_record_id(self, tenant_id: str, base_id: str) -> str:
        """Create a tenant-scoped record ID to prevent ID collisions."""
        pass

    @abstractmethod
    def extract_tenant_from_record_id(self, scoped_id: str) -> tuple[str, str]:
        """Extract tenant ID and base ID from a scoped record ID."""
        pass

    @abstractmethod
    def clear_tenant_context(self, tenant_id: str) -> None:
        """Clear cached tenant context and configuration."""
        pass

    @abstractmethod
    def get_tenant_statistics(self) -> Dict[str, Any]:
        """Get statistics about tenant usage and isolation."""
        pass
