"""
Shared tenant isolation manager for mapper service.

This module implements the shared tenant isolation interfaces
for consistent tenant management across all microservices.
"""

from typing import Any, Dict, List, Optional, Set, Union, Callable, TypeVar
from enum import Enum
import time
import threading
from functools import wraps

from ..shared_integration import (
    get_shared_logger,
    ITenantIsolationManager,
    TenantContext,
    TenantAccessLevel,
    TenantConfig,
    TenantIsolationError,
)

logger = get_shared_logger(__name__)

T = TypeVar("T")


def handle_tenant_errors(default_return: Any = None, raise_on_error: bool = False):
    """Decorator to handle common tenant operation errors following DRY principle."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except TenantIsolationError:
                # Re-raise tenant isolation errors as-is
                raise
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.error(f"Error in {func.__name__}", error=str(e))
                if raise_on_error:
                    raise TenantIsolationError(
                        f"Operation failed in {func.__name__}: {str(e)}"
                    ) from e
                return default_return
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Broad exception handling is acceptable in error handling decorators
                logger.error(f"Unexpected error in {func.__name__}", error=str(e))
                if raise_on_error:
                    raise TenantIsolationError(
                        f"Unexpected error in {func.__name__}: {str(e)}"
                    ) from e
                return default_return

        return wrapper

    return decorator


class TenantOperationStatus(Enum):
    """Status enum for tenant operations."""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class MapperTenantManager(ITenantIsolationManager):
    """Tenant isolation manager for mapper service."""

    def __init__(self):
        """Initialize the tenant manager."""
        self._tenant_contexts: Dict[str, TenantContext] = {}
        self._tenant_configs: Dict[str, TenantConfig] = {}
        self._tenant_operations: Dict[str, List[Dict[str, Any]]] = {}
        self._tenant_timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._default_config = TenantConfig(tenant_id="default")

        logger.info("Mapper tenant manager initialized")

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

            # Cache the context with timestamp
            with self._lock:
                self._tenant_contexts[tenant_id] = context
                self._tenant_timestamps[tenant_id] = time.time()
                self._track_tenant_operation(
                    tenant_id,
                    "context_created",
                    {
                        "access_level": access_level.value,
                        "allowed_tenants": (
                            list(allowed_tenants) if allowed_tenants else [tenant_id]
                        ),
                    },
                )

            logger.info(
                "Created tenant context",
                tenant_id=tenant_id,
                access_level=access_level.value,
                allowed_tenants=(list(allowed_tenants) if allowed_tenants else [tenant_id]),
            )

            return context

        except (ValueError, TypeError) as e:
            logger.error("Failed to create tenant context", tenant_id=tenant_id, error=str(e))
            raise TenantIsolationError(f"Failed to create tenant context: {str(e)}") from e
        except Exception as e:
            logger.error(
                "Unexpected error creating tenant context",
                tenant_id=tenant_id,
                error=str(e),
            )
            raise TenantIsolationError(f"Failed to create tenant context: {str(e)}") from e

    @handle_tenant_errors(default_return=None)
    def get_tenant_context(self, tenant_id: str) -> Optional[TenantContext]:
        """Get cached tenant context."""
        with self._lock:
            context = self._tenant_contexts.get(tenant_id)
            if context:
                # Update access timestamp
                self._tenant_timestamps[tenant_id] = time.time()
                self._track_tenant_operation(tenant_id, "context_accessed", {})
                return context

            return None

    @handle_tenant_errors(default_return=False)
    def validate_tenant_access(
        self, requesting_tenant: str, target_tenant: str, operation: str = "read"
    ) -> bool:
        """Validate if a tenant can access another tenant's data."""
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
                operation=operation,
            )
            return True

        logger.warning(
            "Tenant access denied",
            requesting_tenant=requesting_tenant,
            target_tenant=target_tenant,
            operation=operation,
        )
        return False

    def apply_tenant_filter(
        self, query: str, tenant_context: TenantContext, table_alias: str = ""
    ) -> str:
        """Apply tenant filtering to database queries."""
        if not query:
            return query

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

        except (AttributeError, ValueError) as e:
            logger.error("Failed to apply tenant filter", error=str(e))
            return query

    def get_tenant_config(self, tenant_id: str) -> TenantConfig:
        """Get tenant-specific configuration with fallback to defaults."""
        try:
            with self._lock:
                config = self._tenant_configs.get(tenant_id, self._default_config)
                logger.debug("Retrieved tenant config", tenant_id=tenant_id)
                return config

        except (KeyError, AttributeError) as e:
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
                updated_config = TenantConfig(**{**existing_config.__dict__, **config_or_overrides})
            else:
                updated_config = config_or_overrides

            self._tenant_configs[tenant_id] = updated_config

            logger.info("Updated tenant config", tenant_id=tenant_id)

        except (TypeError, AttributeError, ValueError) as e:
            logger.error("Failed to update tenant config", tenant_id=tenant_id, error=str(e))
            raise TenantIsolationError(f"Failed to update tenant config: {str(e)}") from e

    @handle_tenant_errors(default_return=False)
    def validate_detector_access(self, tenant_id: str, detector_name: str) -> bool:
        """Validate if a tenant can use a specific detector."""
        config = self.get_tenant_config(tenant_id)
        allowed_detectors = getattr(config, "detector_whitelist", None)

        if allowed_detectors is None:
            # No restrictions
            return True

        has_access = detector_name in allowed_detectors
        logger.debug(
            "Detector access validation",
            tenant_id=tenant_id,
            detector_name=detector_name,
            has_access=has_access,
        )
        return has_access

    @handle_tenant_errors(default_return=0.7)
    def get_effective_confidence_threshold(self, tenant_id: str) -> float:
        """Get the effective confidence threshold for a tenant."""
        config = self.get_tenant_config(tenant_id)
        threshold = getattr(config, "confidence_threshold", 0.7) or 0.7

        logger.debug("Retrieved confidence threshold", tenant_id=tenant_id, threshold=threshold)
        return threshold

    def create_tenant_scoped_record_id(self, tenant_id: str, base_id: str) -> str:
        """Create a tenant-scoped record ID."""
        try:
            if not tenant_id or not base_id:
                return base_id

            scoped_id = f"{tenant_id}:{base_id}"
            logger.debug("Created tenant-scoped record ID", tenant_id=tenant_id, base_id=base_id)
            return scoped_id

        except (TypeError, ValueError) as e:
            logger.error("Failed to create tenant-scoped record ID", error=str(e))
            return base_id

    @handle_tenant_errors(default_return=("", ""))
    def extract_tenant_from_record_id(self, scoped_id: str) -> tuple[str, str]:
        """Extract tenant ID and base ID from a scoped record ID."""
        if not scoped_id or ":" not in scoped_id:
            return "", scoped_id

        tenant_id, base_id = scoped_id.split(":", 1)
        logger.debug(
            "Extracted tenant and base ID from record ID",
            tenant_id=tenant_id,
            base_id=base_id,
        )
        return tenant_id, base_id

    @handle_tenant_errors()
    def clear_tenant_context(self, tenant_id: str) -> None:
        """Clear cached tenant context."""
        with self._lock:
            if tenant_id in self._tenant_contexts:
                self._cleanup_tenant_data(tenant_id)
                self._track_tenant_operation(tenant_id, "context_cleared", {})
                logger.info("Cleared tenant context", tenant_id=tenant_id)

    @handle_tenant_errors(default_return={})
    def get_tenant_statistics(self) -> Dict[str, Any]:
        """Get statistics about tenant usage and isolation."""
        with self._lock:
            return {
                "total_tenants": len(self._tenant_contexts),
                "active_tenants": len(self._tenant_contexts),
                "tenant_configs": len(self._tenant_configs),
                "contexts": list(self._tenant_contexts.keys()),
                "operations_tracked": sum(len(ops) for ops in self._tenant_operations.values()),
            }

    def _track_tenant_operation(
        self, tenant_id: str, operation: str, metadata: Dict[str, Any]
    ) -> None:
        """Track tenant operations for audit and monitoring purposes."""
        # This method is called within locked context, minimal error handling
        try:
            if tenant_id not in self._tenant_operations:
                self._tenant_operations[tenant_id] = []

            operation_record = {
                "operation": operation,
                "timestamp": time.time(),
                "metadata": metadata,
                "status": TenantOperationStatus.ACTIVE.value,
            }

            self._tenant_operations[tenant_id].append(operation_record)

            # Keep only last 100 operations per tenant
            if len(self._tenant_operations[tenant_id]) > 100:
                self._tenant_operations[tenant_id] = self._tenant_operations[tenant_id][-100:]

        except (AttributeError, KeyError) as e:
            logger.warning("Failed to track tenant operation", tenant_id=tenant_id, error=str(e))

    @handle_tenant_errors(default_return=[])
    def get_tenant_operations_history(
        self, tenant_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent operations history for a tenant."""
        with self._lock:
            operations = self._tenant_operations.get(tenant_id, [])
            # Return most recent operations first
            return sorted(operations, key=lambda x: x["timestamp"], reverse=True)[:limit]

    @handle_tenant_errors(default_return=[])
    def cleanup_expired_contexts(self, max_age_seconds: int = 3600) -> List[str]:
        """Clean up tenant contexts that haven't been accessed recently."""
        current_time = time.time()
        expired_tenants = []

        with self._lock:
            for tenant_id, timestamp in list(self._tenant_timestamps.items()):
                if current_time - timestamp > max_age_seconds:
                    expired_tenants.append(tenant_id)
                    self._cleanup_tenant_data(tenant_id)

        if expired_tenants:
            logger.info("Cleaned up expired tenant contexts", expired_tenants=expired_tenants)

        return expired_tenants

    def _cleanup_tenant_data(self, tenant_id: str) -> None:
        """Internal method to clean up all data for a tenant."""
        # This method is called within locked context, no additional error handling needed
        self._tenant_contexts.pop(tenant_id, None)
        self._tenant_configs.pop(tenant_id, None)
        self._tenant_operations.pop(tenant_id, None)
        self._tenant_timestamps.pop(tenant_id, None)


# Global tenant manager instance
_tenant_manager: Optional[MapperTenantManager] = None


def get_shared_tenant_manager() -> MapperTenantManager:
    """Get the global tenant manager instance."""
    global _tenant_manager  # pylint: disable=global-statement
    if _tenant_manager is None:
        _tenant_manager = MapperTenantManager()
    return _tenant_manager


__all__ = [
    "MapperTenantManager",
    "TenantOperationStatus",
    "get_shared_tenant_manager",
]
