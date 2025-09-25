"""Tenant data isolation functionality following SRP.

This module provides ONLY tenant data isolation - ensuring tenant data separation.
Single Responsibility: Enforce tenant data isolation and access controls.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from contextlib import asynccontextmanager

from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import AuthenticationError, ValidationError

logger = logging.getLogger(__name__)


class TenantContext:
    """Tenant context for request processing."""

    def __init__(
        self,
        tenant_id: str,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        permissions: Optional[Set[str]] = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.request_id = request_id or get_correlation_id()
        self.permissions = permissions or set()
        self.created_at = datetime.utcnow()


class TenantIsolationManager:
    """Manages tenant data isolation and access controls.

    Single Responsibility: Enforce tenant data separation and access controls.
    Does NOT handle: tenant management, routing, configuration.
    """

    def __init__(self):
        """Initialize tenant isolation manager."""
        self._tenant_data: Dict[str, Dict[str, Any]] = {}  # tenant_id -> data
        self._tenant_sessions: Dict[str, Set[str]] = {}  # tenant_id -> session_ids
        self._active_contexts: Dict[str, TenantContext] = {}  # request_id -> context
        self._cross_tenant_access_log: List[Dict[str, Any]] = []

    @asynccontextmanager
    async def tenant_context(self, tenant_id: str, user_id: Optional[str] = None):
        """Create a tenant context for request processing.

        Args:
            tenant_id: Tenant identifier
            user_id: User identifier (optional)

        Yields:
            TenantContext object
        """
        correlation_id = get_correlation_id()

        # Validate tenant
        if not self._is_valid_tenant(tenant_id):
            raise AuthenticationError(
                f"Invalid tenant: {tenant_id}", correlation_id=correlation_id
            )

        # Create context
        context = TenantContext(
            tenant_id=tenant_id, user_id=user_id, request_id=correlation_id
        )

        # Store active context
        self._active_contexts[correlation_id] = context

        # Add to tenant sessions
        if tenant_id not in self._tenant_sessions:
            self._tenant_sessions[tenant_id] = set()
        self._tenant_sessions[tenant_id].add(correlation_id)

        logger.debug(
            "Created tenant context for %s",
            tenant_id,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
            },
        )

        try:
            yield context
        finally:
            # Clean up context
            self._active_contexts.pop(correlation_id, None)
            if tenant_id in self._tenant_sessions:
                self._tenant_sessions[tenant_id].discard(correlation_id)

                # Clean up empty session sets
                if not self._tenant_sessions[tenant_id]:
                    del self._tenant_sessions[tenant_id]

    def store_tenant_data(
        self, tenant_id: str, key: str, data: Any, request_id: Optional[str] = None
    ) -> bool:
        """Store data for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            key: Data key
            data: Data to store
            request_id: Request identifier for context validation

        Returns:
            True if data stored successfully
        """
        correlation_id = request_id or get_correlation_id()

        try:
            # Validate tenant context
            if not self._validate_tenant_access(tenant_id, correlation_id):
                logger.warning(
                    "Unauthorized attempt to store data for tenant %s",
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "key": key,
                    },
                )
                return False

            # Initialize tenant data if needed
            if tenant_id not in self._tenant_data:
                self._tenant_data[tenant_id] = {}

            # Store data
            self._tenant_data[tenant_id][key] = {
                "data": data,
                "stored_at": datetime.utcnow(),
                "stored_by": correlation_id,
            }

            logger.debug(
                "Stored data for tenant %s, key %s",
                tenant_id,
                key,
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "key": key,
                },
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to store data for tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "key": key,
                    "error": str(e),
                },
            )
            return False

    def get_tenant_data(
        self, tenant_id: str, key: str, request_id: Optional[str] = None
    ) -> Optional[Any]:
        """Get data for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            key: Data key
            request_id: Request identifier for context validation

        Returns:
            Data if found and authorized, None otherwise
        """
        correlation_id = request_id or get_correlation_id()

        try:
            # Validate tenant access
            if not self._validate_tenant_access(tenant_id, correlation_id):
                logger.warning(
                    "Unauthorized attempt to access data for tenant %s",
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "key": key,
                    },
                )
                return None

            # Get data
            tenant_data = self._tenant_data.get(tenant_id, {})
            data_entry = tenant_data.get(key)

            if data_entry:
                logger.debug(
                    "Retrieved data for tenant %s, key %s",
                    tenant_id,
                    key,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "key": key,
                    },
                )
                return data_entry["data"]

            return None

        except Exception as e:
            logger.error(
                "Failed to get data for tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "key": key,
                    "error": str(e),
                },
            )
            return None

    def delete_tenant_data(
        self, tenant_id: str, key: str, request_id: Optional[str] = None
    ) -> bool:
        """Delete data for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            key: Data key
            request_id: Request identifier for context validation

        Returns:
            True if data deleted successfully
        """
        correlation_id = request_id or get_correlation_id()

        try:
            # Validate tenant access
            if not self._validate_tenant_access(tenant_id, correlation_id):
                logger.warning(
                    "Unauthorized attempt to delete data for tenant %s",
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "key": key,
                    },
                )
                return False

            # Delete data
            tenant_data = self._tenant_data.get(tenant_id, {})
            if key in tenant_data:
                del tenant_data[key]

                logger.debug(
                    "Deleted data for tenant %s, key %s",
                    tenant_id,
                    key,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "key": key,
                    },
                )

                return True

            return False

        except Exception as e:
            logger.error(
                "Failed to delete data for tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "key": key,
                    "error": str(e),
                },
            )
            return False

    def list_tenant_data_keys(
        self, tenant_id: str, request_id: Optional[str] = None
    ) -> List[str]:
        """List data keys for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            request_id: Request identifier for context validation

        Returns:
            List of data keys for the tenant
        """
        correlation_id = request_id or get_correlation_id()

        try:
            # Validate tenant access
            if not self._validate_tenant_access(tenant_id, correlation_id):
                logger.warning(
                    "Unauthorized attempt to list data for tenant %s",
                    tenant_id,
                    extra={"correlation_id": correlation_id, "tenant_id": tenant_id},
                )
                return []

            # Get keys
            tenant_data = self._tenant_data.get(tenant_id, {})
            return list(tenant_data.keys())

        except Exception as e:
            logger.error(
                "Failed to list data keys for tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            return []

    def purge_tenant_data(self, tenant_id: str) -> bool:
        """Purge all data for a specific tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if data purged successfully
        """
        correlation_id = get_correlation_id()

        try:
            # Remove all tenant data
            if tenant_id in self._tenant_data:
                data_count = len(self._tenant_data[tenant_id])
                del self._tenant_data[tenant_id]

                logger.info(
                    "Purged %d data entries for tenant %s",
                    data_count,
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "data_count": data_count,
                    },
                )

            # Remove tenant sessions
            if tenant_id in self._tenant_sessions:
                session_count = len(self._tenant_sessions[tenant_id])
                del self._tenant_sessions[tenant_id]

                logger.info(
                    "Purged %d sessions for tenant %s",
                    session_count,
                    tenant_id,
                    extra={
                        "correlation_id": correlation_id,
                        "tenant_id": tenant_id,
                        "session_count": session_count,
                    },
                )

            return True

        except Exception as e:
            logger.error(
                "Failed to purge data for tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            return False

    def get_tenant_isolation_stats(self) -> Dict[str, Any]:
        """Get tenant isolation statistics.

        Returns:
            Dictionary with isolation statistics
        """
        total_tenants = len(self._tenant_data)
        total_sessions = sum(
            len(sessions) for sessions in self._tenant_sessions.values()
        )
        total_data_entries = sum(len(data) for data in self._tenant_data.values())

        return {
            "total_tenants_with_data": total_tenants,
            "total_active_sessions": total_sessions,
            "total_data_entries": total_data_entries,
            "active_contexts": len(self._active_contexts),
            "cross_tenant_access_attempts": len(self._cross_tenant_access_log),
            "tenants_with_sessions": list(self._tenant_sessions.keys()),
            "data_per_tenant": {
                tenant_id: len(data) for tenant_id, data in self._tenant_data.items()
            },
        }

    def _validate_tenant_access(self, tenant_id: str, request_id: str) -> bool:
        """Validate that a request has access to tenant data.

        Args:
            tenant_id: Tenant identifier
            request_id: Request identifier

        Returns:
            True if access is authorized
        """
        # Check if there's an active context for this request
        context = self._active_contexts.get(request_id)
        if not context:
            # Log potential cross-tenant access attempt
            self._log_cross_tenant_access(tenant_id, request_id, "no_context")
            return False

        # Check if context tenant matches requested tenant
        if context.tenant_id != tenant_id:
            # Log cross-tenant access attempt
            self._log_cross_tenant_access(
                tenant_id,
                request_id,
                "tenant_mismatch",
                context_tenant=context.tenant_id,
            )
            return False

        return True

    def _is_valid_tenant(self, tenant_id: str) -> bool:
        """Check if a tenant ID is valid.

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if tenant is valid
        """
        # Basic validation - in a real implementation, this would check
        # against a tenant registry or database
        return (
            tenant_id
            and isinstance(tenant_id, str)
            and len(tenant_id) >= 3
            and len(tenant_id) <= 50
        )

    def _log_cross_tenant_access(
        self,
        requested_tenant: str,
        request_id: str,
        reason: str,
        context_tenant: Optional[str] = None,
    ):
        """Log cross-tenant access attempts for security monitoring.

        Args:
            requested_tenant: Tenant that was requested
            request_id: Request identifier
            reason: Reason for access denial
            context_tenant: Tenant from context (if available)
        """
        access_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "requested_tenant": requested_tenant,
            "request_id": request_id,
            "reason": reason,
            "context_tenant": context_tenant,
        }

        self._cross_tenant_access_log.append(access_log)

        # Keep only last 1000 entries
        if len(self._cross_tenant_access_log) > 1000:
            self._cross_tenant_access_log = self._cross_tenant_access_log[-1000:]

        logger.warning(
            "Cross-tenant access attempt: %s requested %s (reason: %s)",
            request_id,
            requested_tenant,
            reason,
            extra={
                "correlation_id": request_id,
                "requested_tenant": requested_tenant,
                "context_tenant": context_tenant,
                "reason": reason,
            },
        )

    def get_cross_tenant_access_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent cross-tenant access attempts.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recent access attempts
        """
        return self._cross_tenant_access_log[-limit:]


# Export only the tenant isolation functionality
__all__ = [
    "TenantIsolationManager",
    "TenantContext",
]
