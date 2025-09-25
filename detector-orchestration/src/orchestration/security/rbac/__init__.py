"""Role-based access control functionality following SRP.

This module provides RBAC capabilities:
- RBAC Manager: Role and permission management
- Permissions: System permission definitions
- Roles: System role definitions
"""

from .rbac_manager import (
    RBACManager,
    Permission,
    Role,
    RoleDefinition,
)

__all__ = [
    "RBACManager",
    "Permission",
    "Role",
    "RoleDefinition",
]
