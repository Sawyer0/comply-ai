"""
Shared tenant isolation manager for analysis service.

This module implements the shared tenant isolation interfaces
for consistent tenant management across all microservices.
"""

from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

from ..shared_integration import (
    get_shared_logger,
    ITenantIsolationManager,
    TenantContext,
    TenantAccessLevel,
    TenantConfig,
    TenantIsolationError,
)
import time

logger = get_shared_logger(__name__)


class AnalysisTenantManager(ITenantIsolationManager):
    """Tenant isolation manager for analysis service."""

    def __init__(self):
        """Initialize the tenant manager."""
        self._tenant_contexts: Dict[str, TenantContext] = {}
        self._tenant_configs: Dict[str, TenantConfig] = {}
        self._default_config = TenantConfig()
        
        logger.info("Analysis tenant manager initialized")

    def create_tenant_context(
        self,
        tenant_id: str,
        access_level: TenantAccessLevel = TenantAccessLevel.STRICT,
        allowed_tenants: Optional[Set[str]] = None,
        configuration_overrides: Optional[Dict[str, Any]] = None,
    ) -> TenantContext:
        """Create a tenant context for scoped operations."""
        try:
            # Validate tenant ID
            if not tenant_id or not isinstance(tenant_id, str):
                raise TenantIsolationError("Invalid tenant ID")

            # Create tenant context
            context = TenantContext(
                tenant_id=tenant_id,
                access_level=access_level,
                allowed_tenants=allowed_tenants or {tenant_id},
                configuration_overrides=configuration_overrides or {},
            )

            # Cache the context
            self._tenant_contexts[tenant_id] = context
            
            logger.info(
                "Created tenant context",
                tenant_id=tenant_id,
                access_level=access_level.value,
                allowed_tenants=list(allowed_tenants) if allowed_tenants else [tenant_id]
            )

            return context

        except Exception as e:
            logger.error("Failed to create tenant context", tenant_id=tenant_id, error=str(e))
            raise TenantIsolationError(f"Failed to create tenant context: {str(e)}")

    def get_tenant_context(self, tenant_id: str) -> Optional[TenantContext]:
        """Get cached tenant context."""
        try:
            context = self._tenant_contexts.get(tenant_id)
            if context:
                return context
            
            return None

        except Exception as e:
            logger.error("Failed to get tenant context", tenant_id=tenant_id, error=str(e))
            return None

    def validate_tenant_access(
        self, requesting_tenant: str, target_tenant: str, operation: str = "read"
    ) -> bool:
        """Validate if a tenant can access another tenant's data."""
        try:
            # Get requesting tenant context
            context = self.get_tenant_context(requesting_tenant)
            if not context:
                logger.warning("No tenant context found", tenant_id=requesting_tenant)
                return False

            # Check if requesting tenant can access target tenant
            if context.allowed_tenants and target_tenant in context.allowed_tenants:
                logger.debug(
                    "Tenant access validated",
                    requesting_tenant=requesting_tenant,
                    target_tenant=target_tenant,
                    operation=operation
                )
                return True

            logger.warning(
                "Tenant access denied",
                requesting_tenant=requesting_tenant,
                target_tenant=target_tenant,
                operation=operation
            )
            return False

        except Exception as e:
            logger.error("Failed to validate tenant access", error=str(e))
            return False

    def apply_tenant_filter(
        self, query: str, tenant_context: TenantContext, table_alias: str = ""
    ) -> str:
        """Apply tenant filtering to database queries."""
        try:
            # Add tenant filter to query
            if table_alias:
                filter_clause = f"AND {table_alias}.tenant_id = '{tenant_context.tenant_id}'"
            else:
                filter_clause = f"AND tenant_id = '{tenant_context.tenant_id}'"

            # Insert filter before any ORDER BY or LIMIT clauses
            query_lower = query.lower()
            if "order by" in query_lower:
                insert_pos = query_lower.find("order by")
            elif "limit" in query_lower:
                insert_pos = query_lower.find("limit")
            else:
                insert_pos = len(query)
            
            filtered_query = query[:insert_pos] + filter_clause + " " + query[insert_pos:]
            
            logger.debug("Applied tenant filter to query", tenant_id=tenant_context.tenant_id)
            return filtered_query

        except Exception as e:
            logger.error("Failed to apply tenant filter", error=str(e))
            return query

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Get tenant-specific configuration with fallback to defaults."""
        try:
            config = self._tenant_configs.get(tenant_id, self._default_config)
            logger.debug("Retrieved tenant config", tenant_id=tenant_id)
            return config

        except Exception as e:
            logger.error("Failed to get tenant config", tenant_id=tenant_id, error=str(e))
            return self._default_config

    def update_tenant_config(
        self, tenant_id: str, config_or_overrides: Union[TenantConfig, Dict[str, Any]]
    ) -> None:
        """Update tenant-specific configuration."""
        try:
            if isinstance(config_or_overrides, dict):
                # Merge with existing config
                existing_config = self._tenant_configs.get(tenant_id, self._default_config)
                updated_config = TenantConfig(
                    **{**existing_config.__dict__, **config_or_overrides}
                )
            else:
                updated_config = config_or_overrides

            self._tenant_configs[tenant_id] = updated_config
            
            logger.info("Updated tenant config", tenant_id=tenant_id)

        except Exception as e:
            logger.error("Failed to update tenant config", tenant_id=tenant_id, error=str(e))

    def validate_detector_access(self, tenant_id: str, detector_name: str) -> bool:
        """Validate if a tenant can use a specific detector."""
        try:
            config = self.get_tenant_config(tenant_id)
            allowed_detectors = getattr(config, 'allowed_detectors', None)
            
            if allowed_detectors is None:
                # No restrictions
                return True
            
            has_access = detector_name in allowed_detectors
            logger.debug(
                "Detector access validation",
                tenant_id=tenant_id,
                detector_name=detector_name,
                has_access=has_access
            )
            return has_access

        except Exception as e:
            logger.error("Failed to validate detector access", error=str(e))
            return False

    def get_effective_confidence_threshold(self, tenant_id: str) -> float:
        """Get the effective confidence threshold for a tenant."""
        try:
            config = self.get_tenant_config(tenant_id)
            threshold = getattr(config, 'confidence_threshold', 0.7)
            
            logger.debug(
                "Retrieved confidence threshold",
                tenant_id=tenant_id,
                threshold=threshold
            )
            return threshold

        except Exception as e:
            logger.error("Failed to get confidence threshold", error=str(e))
            return 0.7

    def create_tenant_scoped_record_id(self, tenant_id: str, base_id: str) -> str:
        """Create a tenant-scoped record ID."""
        try:
            scoped_id = f"{tenant_id}:{base_id}"
            logger.debug("Created tenant-scoped record ID", tenant_id=tenant_id, base_id=base_id)
            return scoped_id

        except Exception as e:
            logger.error("Failed to create tenant-scoped record ID", error=str(e))
            return base_id

    def extract_tenant_from_record_id(self, scoped_id: str) -> tuple[str, str]:
        """Extract tenant ID and base ID from a scoped record ID."""
        try:
            if ":" in scoped_id:
                tenant_id, base_id = scoped_id.split(":", 1)
                logger.debug("Extracted tenant and base ID from record ID", tenant_id=tenant_id, base_id=base_id)
                return tenant_id, base_id
            return "", scoped_id

        except Exception as e:
            logger.error("Failed to extract tenant from record ID", error=str(e))
            return "", scoped_id

    def clear_tenant_context(self, tenant_id: str) -> None:
        """Clear cached tenant context."""
        try:
            if tenant_id in self._tenant_contexts:
                del self._tenant_contexts[tenant_id]
                logger.info("Cleared tenant context", tenant_id=tenant_id)

        except Exception as e:
            logger.error("Failed to clear tenant context", error=str(e))

    def get_tenant_statistics(self) -> Dict[str, Any]:
        """Get statistics about tenant usage and isolation."""
        try:
            return {
                "total_tenants": len(self._tenant_contexts),
                "active_tenants": len(self._tenant_contexts),
                "tenant_configs": len(self._tenant_configs),
                "contexts": list(self._tenant_contexts.keys()),
            }

        except Exception as e:
            logger.error("Failed to get tenant statistics", error=str(e))
            return {}


# Global tenant manager instance
_tenant_manager: Optional[AnalysisTenantManager] = None


def get_shared_tenant_manager() -> AnalysisTenantManager:
    """Get the global tenant manager instance."""
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = AnalysisTenantManager()
    return _tenant_manager


__all__ = [
    "AnalysisTenantManager",
    "get_shared_tenant_manager",
]
