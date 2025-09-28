"""Role-based access control (RBAC) management following SRP.

This module provides ONLY RBAC functionality - roles, permissions, and access control.
Single Responsibility: Manage roles and permissions for authorization.
"""

import logging
from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """System permissions enumeration."""

    # Orchestration permissions
    ORCHESTRATE_DETECTORS = "orchestrate:detectors"
    REGISTER_DETECTORS = "register:detectors"
    UNREGISTER_DETECTORS = "unregister:detectors"
    VIEW_DETECTOR_STATUS = "view:detector_status"

    # Policy permissions
    CREATE_POLICIES = "create:policies"
    UPDATE_POLICIES = "update:policies"
    DELETE_POLICIES = "delete:policies"
    VIEW_POLICIES = "view:policies"

    # Health and monitoring permissions
    VIEW_HEALTH = "view:health"
    VIEW_METRICS = "view:metrics"
    VIEW_LOGS = "view:logs"

    # Administrative permissions
    MANAGE_TENANTS = "manage:tenants"
    MANAGE_USERS = "manage:users"
    MANAGE_API_KEYS = "manage:api_keys"

    # System permissions
    ADMIN_ACCESS = "admin:access"
    SYSTEM_CONFIG = "system:config"


class Role(str, Enum):
    """System roles enumeration."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"


@dataclass
class RoleDefinition:
    """Role definition with permissions."""

    role: Role
    permissions: Set[Permission]
    description: str
    is_tenant_scoped: bool = True


class RBACManager:
    """Manages role-based access control.

    Single Responsibility: Manage roles, permissions, and access control decisions.
    Does NOT handle: authentication, API key management, tenant management.
    """

    def __init__(self):
        """Initialize RBAC manager with default roles and permissions."""
        self._role_definitions = self._initialize_default_roles()
        self._user_roles: Dict[str, Dict[str, Role]] = (
            {}
        )  # user_id -> {tenant_id: role}
        self._tenant_permissions: Dict[str, Set[Permission]] = (
            {}
        )  # tenant_id -> custom permissions

    def _initialize_default_roles(self) -> Dict[Role, RoleDefinition]:
        """Initialize default system roles and their permissions."""
        return {
            Role.VIEWER: RoleDefinition(
                role=Role.VIEWER,
                permissions={
                    Permission.VIEW_DETECTOR_STATUS,
                    Permission.VIEW_POLICIES,
                    Permission.VIEW_HEALTH,
                },
                description="Read-only access to system status and policies",
                is_tenant_scoped=True,
            ),
            Role.OPERATOR: RoleDefinition(
                role=Role.OPERATOR,
                permissions={
                    Permission.ORCHESTRATE_DETECTORS,
                    Permission.VIEW_DETECTOR_STATUS,
                    Permission.VIEW_POLICIES,
                    Permission.VIEW_HEALTH,
                    Permission.VIEW_METRICS,
                },
                description="Can orchestrate detectors and view system status",
                is_tenant_scoped=True,
            ),
            Role.ADMIN: RoleDefinition(
                role=Role.ADMIN,
                permissions={
                    Permission.ORCHESTRATE_DETECTORS,
                    Permission.REGISTER_DETECTORS,
                    Permission.UNREGISTER_DETECTORS,
                    Permission.VIEW_DETECTOR_STATUS,
                    Permission.CREATE_POLICIES,
                    Permission.UPDATE_POLICIES,
                    Permission.DELETE_POLICIES,
                    Permission.VIEW_POLICIES,
                    Permission.VIEW_HEALTH,
                    Permission.VIEW_METRICS,
                    Permission.VIEW_LOGS,
                    Permission.MANAGE_API_KEYS,
                },
                description="Full access to tenant resources",
                is_tenant_scoped=True,
            ),
            Role.TENANT_ADMIN: RoleDefinition(
                role=Role.TENANT_ADMIN,
                permissions={
                    Permission.ORCHESTRATE_DETECTORS,
                    Permission.REGISTER_DETECTORS,
                    Permission.UNREGISTER_DETECTORS,
                    Permission.VIEW_DETECTOR_STATUS,
                    Permission.CREATE_POLICIES,
                    Permission.UPDATE_POLICIES,
                    Permission.DELETE_POLICIES,
                    Permission.VIEW_POLICIES,
                    Permission.VIEW_HEALTH,
                    Permission.VIEW_METRICS,
                    Permission.VIEW_LOGS,
                    Permission.MANAGE_USERS,
                    Permission.MANAGE_API_KEYS,
                },
                description="Administrative access within tenant scope",
                is_tenant_scoped=True,
            ),
            Role.SUPER_ADMIN: RoleDefinition(
                role=Role.SUPER_ADMIN,
                permissions=set(Permission),  # All permissions
                description="Full system administrative access",
                is_tenant_scoped=False,
            ),
        }

    def assign_role(self, user_id: str, tenant_id: str, role: Role) -> bool:
        """Assign a role to a user for a specific tenant.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            role: Role to assign

        Returns:
            True if assignment successful, False otherwise
        """
        correlation_id = get_correlation_id()

        if role not in self._role_definitions:
            logger.error(
                "Invalid role assignment attempted: %s",
                role.value,
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                },
            )
            return False

        if user_id not in self._user_roles:
            self._user_roles[user_id] = {}

        self._user_roles[user_id][tenant_id] = role

        logger.info(
            "Assigned role %s to user %s for tenant %s",
            role.value,
            user_id,
            tenant_id,
            extra={
                "correlation_id": correlation_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": role.value,
            },
        )

        return True


    def revoke_role(self, user_id: str, tenant_id: str) -> bool:
        """Revoke a user's role for a specific tenant.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier

        Returns:
            True if revocation successful, False otherwise
        """
        correlation_id = get_correlation_id()

        if user_id not in self._user_roles or tenant_id not in self._user_roles[user_id]:
            logger.warning(
                "Cannot revoke role: user %s has no role for tenant %s",
                user_id,
                tenant_id,
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                },
            )
            return False

        role = self._user_roles[user_id].pop(tenant_id)
        if not self._user_roles[user_id]:
            del self._user_roles[user_id]

        logger.info(
            "Revoked role %s from user %s for tenant %s",
            role.value,
            user_id,
            tenant_id,
            extra={
                "correlation_id": correlation_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": role.value,
            },
        )

        return True


    def check_permission(
        self, user_id: str, tenant_id: str, required_permission: Permission
    ) -> bool:
        """Check if a user has a specific permission for a tenant.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            required_permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        correlation_id = get_correlation_id()

        user_role = self.get_user_role(user_id, tenant_id)
        if not user_role:
            logger.debug(
                "Permission denied: user %s has no role for tenant %s",
                user_id,
                tenant_id,
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "required_permission": required_permission.value,
                },
            )
            return False

        role_definition = self._role_definitions.get(user_role)
        if not role_definition:
            logger.warning(
                "Permission denied: invalid role %s for user %s",
                user_role.value,
                user_id,
                extra={
                    "correlation_id": correlation_id,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "role": user_role.value,
                    "required_permission": required_permission.value,
                },
            )
            return False

        has_permission = required_permission in role_definition.permissions
        if not has_permission:
            tenant_permissions = self._tenant_permissions.get(tenant_id, set())
            has_permission = required_permission in tenant_permissions

        logger.debug(
            "Permission check: user %s %s permission %s for tenant %s",
            user_id,
            "has" if has_permission else "lacks",
            required_permission.value,
            tenant_id,
            extra={
                "correlation_id": correlation_id,
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": user_role.value,
                "required_permission": required_permission.value,
                "has_permission": has_permission,
            },
        )

        return has_permission


    def get_user_role(self, user_id: str, tenant_id: str) -> Optional[Role]:
        """Get a user's role for a specific tenant.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier

        Returns:
            User's role for the tenant, or None if no role assigned
        """
        return self._user_roles.get(user_id, {}).get(tenant_id)

    def get_user_permissions(self, user_id: str, tenant_id: str) -> Set[Permission]:
        """Get all permissions for a user in a specific tenant.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier

        Returns:
            Set of permissions for the user
        """
        user_role = self.get_user_role(user_id, tenant_id)
        if not user_role:
            return set()

        role_definition = self._role_definitions.get(user_role)
        if not role_definition:
            return set()

        permissions = role_definition.permissions.copy()

        # Add tenant-specific permissions
        tenant_permissions = self._tenant_permissions.get(tenant_id, set())
        permissions.update(tenant_permissions)

        return permissions

    def list_user_tenants(self, user_id: str) -> List[str]:
        """List all tenants where a user has roles.

        Args:
            user_id: User identifier

        Returns:
            List of tenant IDs where user has roles
        """
        return list(self._user_roles.get(user_id, {}).keys())

    def list_tenant_users(self, tenant_id: str) -> Dict[str, Role]:
        """List all users with roles in a specific tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary of user_id -> role for the tenant
        """
        tenant_users = {}
        for user_id, user_tenants in self._user_roles.items():
            if tenant_id in user_tenants:
                tenant_users[user_id] = user_tenants[tenant_id]
        return tenant_users

    def add_tenant_permission(self, tenant_id: str, permission: Permission) -> bool:
        """Add a custom permission for a specific tenant.

        Args:
            tenant_id: Tenant identifier
            permission: Permission to add

        Returns:
            True if permission added successfully
        """
        correlation_id = get_correlation_id()

        permissions = self._tenant_permissions.setdefault(tenant_id, set())
        if permission in permissions:
            logger.debug(
                "Permission %s already granted to tenant %s",
                permission.value,
                tenant_id,
                extra={"correlation_id": correlation_id},
            )
            return True

        permissions.add(permission)

        logger.info(
            "Added custom permission %s for tenant %s",
            permission.value,
            tenant_id,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "permission": permission.value,
            },
        )

        return True


    def get_role_definition(self, role: Role) -> Optional[RoleDefinition]:
        """Get the definition for a specific role.

        Args:
            role: Role to get definition for

        Returns:
            Role definition if found, None otherwise
        """
        return self._role_definitions.get(role)

    def list_available_roles(self) -> List[RoleDefinition]:
        """List all available role definitions.

        Returns:
            List of all role definitions
        """
        return list(self._role_definitions.values())


# Export only the RBAC functionality
__all__ = [
    "RBACManager",
    "Permission",
    "Role",
    "RoleDefinition",
]
