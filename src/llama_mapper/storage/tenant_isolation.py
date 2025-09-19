"""
Tenant Isolation Manager for multi-tenant data access control.

This module provides the TenantIsolationManager class that handles:
- Tenant-scoped data access and queries
- Prevention of cross-tenant data leakage
- Support for tenant-specific configuration overrides
- Row-level security enforcement
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import structlog

from ..config.settings import Settings

logger = structlog.get_logger(__name__)


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


class TenantIsolationManager:
    """
    Manages tenant isolation and data access control.

    Features:
    - Tenant-scoped database queries with row-level security
    - Cross-tenant data leakage prevention
    - Tenant-specific configuration management
    - Audit trail for tenant access patterns
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings
        self.logger = logger.bind(component="tenant_isolation")

        # Tenant configurations cache
        self._tenant_configs: Dict[str, TenantConfig] = {}

        # Active tenant contexts
        self._active_contexts: Dict[str, TenantContext] = {}

        # Tenant access patterns for audit
        self._access_patterns: Dict[str, List[str]] = {}

    def create_tenant_context(
        self,
        tenant_id: str,
        access_level: TenantAccessLevel = TenantAccessLevel.STRICT,
        allowed_tenants: Optional[Set[str]] = None,
        configuration_overrides: Optional[Dict[str, Any]] = None,
    ) -> TenantContext:
        """
        Create a tenant context for scoped operations.

        Args:
            tenant_id: The tenant identifier
            access_level: Level of access permissions
            allowed_tenants: Set of tenant IDs for shared access
            configuration_overrides: Tenant-specific config overrides

        Returns:
            TenantContext object for use in operations
        """
        if not tenant_id or not isinstance(tenant_id, str):
            raise TenantIsolationError("Invalid tenant_id: must be non-empty string")

        # Validate allowed_tenants for shared access
        if access_level == TenantAccessLevel.SHARED and not allowed_tenants:
            raise TenantIsolationError("Shared access requires allowed_tenants list")

        context = TenantContext(
            tenant_id=tenant_id,
            access_level=access_level,
            allowed_tenants=allowed_tenants,
            configuration_overrides=configuration_overrides or {},
        )

        # Cache the context
        self._active_contexts[tenant_id] = context

        self.logger.info(
            "Created tenant context",
            tenant_id=tenant_id,
            access_level=access_level.value,
        )

        return context

    def get_tenant_context(self, tenant_id: str) -> Optional[TenantContext]:
        """Get cached tenant context."""
        return self._active_contexts.get(tenant_id)

    def validate_tenant_access(
        self, requesting_tenant: str, target_tenant: str, operation: str = "read"
    ) -> bool:
        """
        Validate if a tenant can access another tenant's data.

        Args:
            requesting_tenant: The tenant making the request
            target_tenant: The tenant whose data is being accessed
            operation: Type of operation (read, write, delete)

        Returns:
            True if access is allowed, False otherwise
        """
        # Same tenant always has access
        if requesting_tenant == target_tenant:
            return True

        context = self._active_contexts.get(requesting_tenant)
        if not context:
            self.logger.warning(
                "No context found for requesting tenant",
                requesting_tenant=requesting_tenant,
            )
            return False

        # Admin access allows cross-tenant operations
        if context.access_level == TenantAccessLevel.ADMIN:
            self._log_access_pattern(
                requesting_tenant, target_tenant, operation, "admin"
            )
            return True

        # Shared access with explicit allowlist
        if (
            context.access_level == TenantAccessLevel.SHARED
            and context.allowed_tenants
            and target_tenant in context.allowed_tenants
        ):
            self._log_access_pattern(
                requesting_tenant, target_tenant, operation, "shared"
            )
            return True

        # Strict isolation - no cross-tenant access
        if context.access_level == TenantAccessLevel.STRICT:
            self.logger.warning(
                "Cross-tenant access denied",
                requesting_tenant=requesting_tenant,
                target_tenant=target_tenant,
                operation=operation,
            )
            return False

        return False

    def apply_tenant_filter(
        self, query: str, tenant_context: TenantContext, table_alias: str = ""
    ) -> str:
        """
        Apply tenant filtering to database queries.

        Args:
            query: Original SQL query
            tenant_context: Tenant context for filtering
            table_alias: Table alias for column references

        Returns:
            Modified query with tenant filtering
        """
        table_prefix = f"{table_alias}." if table_alias else ""
        tenant_filter = f"{table_prefix}tenant_id = '{tenant_context.tenant_id}'"

        # For shared access, include allowed tenants
        if (
            tenant_context.access_level == TenantAccessLevel.SHARED
            and tenant_context.allowed_tenants
        ):
            allowed_list = "', '".join(tenant_context.allowed_tenants)
            tenant_filter = f"{table_prefix}tenant_id IN ('{tenant_context.tenant_id}', '{allowed_list}')"

        # For admin access, no filtering needed
        if tenant_context.access_level == TenantAccessLevel.ADMIN:
            return query

        # Add WHERE clause or extend existing one
        if "WHERE" in query.upper():
            filtered_query = query.replace("WHERE", f"WHERE {tenant_filter} AND", 1)
        else:
            # Find the position to insert WHERE clause
            # Look for ORDER BY, GROUP BY, HAVING, LIMIT
            insert_keywords = ["ORDER BY", "GROUP BY", "HAVING", "LIMIT"]
            insert_pos = len(query)

            for keyword in insert_keywords:
                pos = query.upper().find(keyword)
                if pos != -1 and pos < insert_pos:
                    insert_pos = pos

            filtered_query = (
                query[:insert_pos].rstrip()
                + f" WHERE {tenant_filter} "
                + query[insert_pos:]
            )

        self.logger.debug(
            "Applied tenant filter to query",
            tenant_id=tenant_context.tenant_id,
            original_query=query[:100] + "..." if len(query) > 100 else query,
            filtered_query=filtered_query[:100] + "..."
            if len(filtered_query) > 100
            else filtered_query,
        )

        return filtered_query

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """
        Get tenant-specific configuration with fallback to defaults.

        Args:
            tenant_id: The tenant identifier

        Returns:
            TenantConfig with tenant-specific overrides applied
        """
        # Check cache first
        if tenant_id in self._tenant_configs:
            return self._tenant_configs[tenant_id]

        # Create default config
        config = TenantConfig(tenant_id=tenant_id)

        # Apply any context-specific overrides
        context = self._active_contexts.get(tenant_id)
        if context and context.configuration_overrides:
            overrides = context.configuration_overrides

            if "confidence_threshold" in overrides:
                config.confidence_threshold = overrides["confidence_threshold"]
            if "detector_whitelist" in overrides:
                config.detector_whitelist = overrides["detector_whitelist"]
            if "detector_blacklist" in overrides:
                config.detector_blacklist = overrides["detector_blacklist"]
            if "storage_retention_days" in overrides:
                config.storage_retention_days = overrides["storage_retention_days"]
            if "encryption_enabled" in overrides:
                config.encryption_enabled = overrides["encryption_enabled"]
            if "audit_level" in overrides:
                config.audit_level = overrides["audit_level"]
            if "custom_taxonomy_mappings" in overrides:
                config.custom_taxonomy_mappings = overrides["custom_taxonomy_mappings"]

        # Cache the config
        self._tenant_configs[tenant_id] = config

        self.logger.info(
            "Retrieved tenant configuration",
            tenant_id=tenant_id,
            has_overrides=bool(context and context.configuration_overrides),
        )

        return config

    def update_tenant_config(
        self, tenant_id: str, config_or_overrides: Union[TenantConfig, Dict[str, Any]]
    ) -> None:
        """
        Update tenant-specific configuration.

        Args:
            tenant_id: The tenant identifier
            config_or_overrides: Updated tenant configuration or dictionary of overrides
        """
        if isinstance(config_or_overrides, dict):
            # Handle dictionary overrides
            existing_config = self.get_tenant_config(tenant_id)

            # Apply overrides
            if "confidence_threshold" in config_or_overrides:
                existing_config.confidence_threshold = config_or_overrides[
                    "confidence_threshold"
                ]
            if "encryption_enabled" in config_or_overrides:
                existing_config.encryption_enabled = config_or_overrides[
                    "encryption_enabled"
                ]
            if "storage_retention_days" in config_or_overrides:
                existing_config.storage_retention_days = config_or_overrides[
                    "storage_retention_days"
                ]
            if "detector_whitelist" in config_or_overrides:
                existing_config.detector_whitelist = config_or_overrides[
                    "detector_whitelist"
                ]
            if "detector_blacklist" in config_or_overrides:
                existing_config.detector_blacklist = config_or_overrides[
                    "detector_blacklist"
                ]
            if "audit_level" in config_or_overrides:
                existing_config.audit_level = config_or_overrides["audit_level"]
            if "custom_taxonomy_mappings" in config_or_overrides:
                existing_config.custom_taxonomy_mappings = config_or_overrides[
                    "custom_taxonomy_mappings"
                ]

            config = existing_config
        else:
            # Handle TenantConfig object
            config = config_or_overrides
            if config.tenant_id != tenant_id:
                raise TenantIsolationError(
                    "Config tenant_id must match provided tenant_id"
                )

        self._tenant_configs[tenant_id] = config

        self.logger.info(
            "Updated tenant configuration",
            tenant_id=tenant_id,
            confidence_threshold=config.confidence_threshold,
            encryption_enabled=config.encryption_enabled,
            audit_level=config.audit_level,
        )

    def validate_detector_access(self, tenant_id: str, detector_name: str) -> bool:
        """
        Validate if a tenant can use a specific detector.

        Args:
            tenant_id: The tenant identifier
            detector_name: Name of the detector

        Returns:
            True if detector access is allowed
        """
        config = self.get_tenant_config(tenant_id)

        # Check blacklist first
        if config.detector_blacklist and detector_name in config.detector_blacklist:
            self.logger.warning(
                "Detector access denied by blacklist",
                tenant_id=tenant_id,
                detector_name=detector_name,
            )
            return False

        # Check whitelist if configured
        if config.detector_whitelist and detector_name not in config.detector_whitelist:
            self.logger.warning(
                "Detector access denied by whitelist",
                tenant_id=tenant_id,
                detector_name=detector_name,
            )
            return False

        return True

    def get_effective_confidence_threshold(self, tenant_id: str) -> float:
        """
        Get the effective confidence threshold for a tenant.

        Args:
            tenant_id: The tenant identifier

        Returns:
            Confidence threshold (tenant-specific or default)
        """
        config = self.get_tenant_config(tenant_id)

        if config.confidence_threshold is not None:
            return config.confidence_threshold

        # Fallback to global default
        assert self.settings is not None
        return self.settings.confidence.default_threshold

    def create_tenant_scoped_record_id(self, tenant_id: str, base_id: str) -> str:
        """
        Create a tenant-scoped record ID to prevent ID collisions.

        Args:
            tenant_id: The tenant identifier
            base_id: Base record identifier

        Returns:
            Tenant-scoped record ID
        """
        return f"{tenant_id}:{base_id}"

    def extract_tenant_from_record_id(self, scoped_id: str) -> tuple[str, str]:
        """
        Extract tenant ID and base ID from a scoped record ID.

        Args:
            scoped_id: Tenant-scoped record ID

        Returns:
            Tuple of (tenant_id, base_id)
        """
        if ":" not in scoped_id:
            raise TenantIsolationError(f"Invalid scoped record ID format: {scoped_id}")

        tenant_id, base_id = scoped_id.split(":", 1)
        return tenant_id, base_id

    def get_tenant_access_audit(self, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Get audit trail for tenant access patterns.

        Args:
            tenant_id: The tenant identifier

        Returns:
            List of access audit entries
        """
        patterns = self._access_patterns.get(tenant_id, [])
        return [
            {
                "timestamp": pattern.split("|")[0],
                "target_tenant": pattern.split("|")[1],
                "operation": pattern.split("|")[2],
                "access_type": pattern.split("|")[3],
            }
            for pattern in patterns
        ]

    def _log_access_pattern(
        self,
        requesting_tenant: str,
        target_tenant: str,
        operation: str,
        access_type: str,
    ) -> None:
        """Log tenant access patterns for audit purposes."""
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()
        pattern = f"{timestamp}|{target_tenant}|{operation}|{access_type}"

        if requesting_tenant not in self._access_patterns:
            self._access_patterns[requesting_tenant] = []

        self._access_patterns[requesting_tenant].append(pattern)

        # Keep only last 1000 entries per tenant
        if len(self._access_patterns[requesting_tenant]) > 1000:
            self._access_patterns[requesting_tenant] = self._access_patterns[
                requesting_tenant
            ][-1000:]

    def clear_tenant_context(self, tenant_id: str) -> None:
        """
        Clear cached tenant context and configuration.

        Args:
            tenant_id: The tenant identifier
        """
        self._active_contexts.pop(tenant_id, None)
        self._tenant_configs.pop(tenant_id, None)

        self.logger.info("Cleared tenant context", tenant_id=tenant_id)

    def get_tenant_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tenant usage and isolation.

        Returns:
            Dictionary with tenant statistics
        """
        return {
            "active_contexts": len(self._active_contexts),
            "cached_configs": len(self._tenant_configs),
            "tenants_with_access_patterns": len(self._access_patterns),
            "total_access_events": sum(
                len(patterns) for patterns in self._access_patterns.values()
            ),
            "tenant_access_levels": {
                level.value: sum(
                    1
                    for ctx in self._active_contexts.values()
                    if ctx.access_level == level
                )
                for level in TenantAccessLevel
            },
        }
