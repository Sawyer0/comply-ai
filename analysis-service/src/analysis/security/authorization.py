"""
Authorization module for the Analysis Service.

Handles permission checking and access control.
"""

from typing import Any, Dict

import structlog

from .exceptions import AuthorizationError

logger = structlog.get_logger(__name__)


class AuthorizationManager:
    """Manages authorization and permission checking."""

    def __init__(self):
        self.logger = logger.bind(component="authorization_manager")

    async def authorize_action(
        self, user_info: Dict[str, Any], action: str, resource: str
    ) -> bool:
        """
        Check if user is authorized to perform action on resource.

        Args:
            user_info: User information from authentication
            action: Action being performed (read, write, admin)
            resource: Resource being accessed

        Returns:
            True if authorized, False otherwise
        """
        permissions = user_info.get("permissions", [])

        # Check for specific permission
        required_permission = f"{action}:{resource}"
        if required_permission in permissions:
            return True

        # Check for wildcard permissions
        wildcard_permission = f"{action}:*"
        if wildcard_permission in permissions:
            return True

        admin_permission = "admin:*"
        if admin_permission in permissions:
            return True

        self.logger.warning(
            "Authorization failed",
            user_id=user_info.get("user_id"),
            action=action,
            resource=resource,
            permissions=permissions,
        )

        return False

    def require_permission(self, permission: str):
        """
        Decorator to require specific permission for a function.

        Args:
            permission: Required permission string

        Returns:
            Decorator function
        """

        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Extract user_info from kwargs or context
                user_info = kwargs.get("user_info")
                if not user_info:
                    raise AuthorizationError("No user information available")

                # Parse permission into action and resource
                if ":" in permission:
                    action, resource = permission.split(":", 1)
                else:
                    action, resource = permission, "*"

                if not await self.authorize_action(user_info, action, resource):
                    raise AuthorizationError(
                        f"Insufficient permissions. Required: {permission}"
                    )

                return await func(*args, **kwargs)

            return wrapper

        return decorator

    async def check_tenant_access(
        self, user_info: Dict[str, Any], tenant_id: str
    ) -> bool:
        """
        Check if user has access to specific tenant.

        Args:
            user_info: User information from authentication
            tenant_id: Tenant ID to check access for

        Returns:
            True if user has access to tenant
        """
        user_tenant_id = user_info.get("tenant_id")

        # Check if user belongs to the tenant
        if user_tenant_id == tenant_id:
            return True

        # Check for cross-tenant permissions
        permissions = user_info.get("permissions", [])
        if "admin:*" in permissions or f"tenant:{tenant_id}" in permissions:
            return True

        self.logger.warning(
            "Tenant access denied",
            user_id=user_info.get("user_id"),
            user_tenant_id=user_tenant_id,
            requested_tenant_id=tenant_id,
        )

        return False
