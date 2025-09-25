"""
Authorization Service for Mapper Service

Handles authorization logic and permission checking
following Single Responsibility Principle.
"""

from enum import Enum
from typing import List, Set

import structlog
from fastapi import HTTPException
from pydantic import BaseModel

from .api_key_manager import APIKeyInfo

logger = structlog.get_logger(__name__)


class Permission(Enum):
    """Available permissions in the mapper service."""

    # Mapping permissions
    MAP_CANONICAL = "map:canonical"
    MAP_FRAMEWORK = "map:framework"
    MAP_BATCH = "map:batch"

    # Model permissions
    MODEL_INFERENCE = "model:inference"
    MODEL_TRAINING = "model:training"
    MODEL_DEPLOYMENT = "model:deployment"

    # Validation permissions
    VALIDATE_INPUT = "validate:input"
    VALIDATE_OUTPUT = "validate:output"

    # Cost monitoring permissions
    COST_VIEW = "cost:view"
    COST_MANAGE = "cost:manage"

    # Admin permissions
    ADMIN_KEYS = "admin:keys"
    ADMIN_TENANTS = "admin:tenants"
    ADMIN_SYSTEM = "admin:system"


class Scope(Enum):
    """Available scopes in the mapper service."""

    # Data scopes
    READ = "read"
    WRITE = "write"

    # Resource scopes
    MAPPING = "mapping"
    TRAINING = "training"
    MONITORING = "monitoring"
    ADMIN = "admin"


class AuthorizationResult(BaseModel):
    """Authorization result model."""

    authorized: bool
    reason: str = ""


class AuthorizationService:
    """
    Authorization service responsible for permission and scope checking.

    Follows SRP by focusing solely on authorization logic:
    - Checking permissions against API key
    - Validating scopes
    - Managing authorization policies
    """

    def __init__(self):
        self.logger = logger.bind(component="authorization_service")

        # Define permission hierarchies
        self.permission_hierarchies = {
            Permission.ADMIN_SYSTEM: {
                Permission.ADMIN_KEYS,
                Permission.ADMIN_TENANTS,
                Permission.COST_MANAGE,
                Permission.MODEL_DEPLOYMENT,
                Permission.MODEL_TRAINING,
                Permission.MODEL_INFERENCE,
                Permission.MAP_BATCH,
                Permission.MAP_FRAMEWORK,
                Permission.MAP_CANONICAL,
                Permission.VALIDATE_INPUT,
                Permission.VALIDATE_OUTPUT,
                Permission.COST_VIEW,
            },
            Permission.ADMIN_KEYS: {Permission.COST_VIEW},
            Permission.MODEL_DEPLOYMENT: {
                Permission.MODEL_TRAINING,
                Permission.MODEL_INFERENCE,
            },
            Permission.MODEL_TRAINING: {Permission.MODEL_INFERENCE},
            Permission.MAP_BATCH: {Permission.MAP_FRAMEWORK, Permission.MAP_CANONICAL},
            Permission.MAP_FRAMEWORK: {Permission.MAP_CANONICAL},
            Permission.COST_MANAGE: {Permission.COST_VIEW},
        }

    def check_permission(
        self, api_key_info: APIKeyInfo, required_permission: Permission
    ) -> AuthorizationResult:
        """
        Check if API key has required permission.

        Args:
            api_key_info: API key information
            required_permission: Required permission

        Returns:
            Authorization result
        """
        try:
            # Convert string permissions to enum
            user_permissions = set()
            for perm_str in api_key_info.permissions:
                try:
                    user_permissions.add(Permission(perm_str))
                except ValueError:
                    self.logger.warning(
                        "Unknown permission in API key", permission=perm_str
                    )

            # Check direct permission
            if required_permission in user_permissions:
                return AuthorizationResult(authorized=True)

            # Check hierarchical permissions
            for user_perm in user_permissions:
                if required_permission in self.permission_hierarchies.get(
                    user_perm, set()
                ):
                    return AuthorizationResult(authorized=True)

            self.logger.warning(
                "Permission denied",
                tenant_id=api_key_info.tenant_id,
                key_id=api_key_info.key_id,
                required_permission=required_permission.value,
                user_permissions=[p.value for p in user_permissions],
            )

            return AuthorizationResult(
                authorized=False,
                reason=f"Missing required permission: {required_permission.value}",
            )

        except Exception as e:
            self.logger.error("Authorization check failed", error=str(e))
            return AuthorizationResult(authorized=False, reason="Authorization error")

    def check_scope(
        self, api_key_info: APIKeyInfo, required_scopes: List[Scope]
    ) -> AuthorizationResult:
        """
        Check if API key has required scopes.

        Args:
            api_key_info: API key information
            required_scopes: Required scopes

        Returns:
            Authorization result
        """
        try:
            # Convert string scopes to enum
            user_scopes = set()
            for scope_str in api_key_info.scopes:
                try:
                    user_scopes.add(Scope(scope_str))
                except ValueError:
                    self.logger.warning("Unknown scope in API key", scope=scope_str)

            # Check if all required scopes are present
            required_scope_set = set(required_scopes)

            if required_scope_set.issubset(user_scopes):
                return AuthorizationResult(authorized=True)

            missing_scopes = required_scope_set - user_scopes

            self.logger.warning(
                "Scope denied",
                tenant_id=api_key_info.tenant_id,
                key_id=api_key_info.key_id,
                required_scopes=[s.value for s in required_scopes],
                user_scopes=[s.value for s in user_scopes],
                missing_scopes=[s.value for s in missing_scopes],
            )

            return AuthorizationResult(
                authorized=False,
                reason=f"Missing required scopes: {[s.value for s in missing_scopes]}",
            )

        except Exception as e:
            self.logger.error("Scope check failed", error=str(e))
            return AuthorizationResult(authorized=False, reason="Scope check error")

    def check_tenant_access(
        self, api_key_info: APIKeyInfo, resource_tenant_id: str
    ) -> AuthorizationResult:
        """
        Check if API key can access resources for a specific tenant.

        Args:
            api_key_info: API key information
            resource_tenant_id: Tenant ID of the resource being accessed

        Returns:
            Authorization result
        """
        # API key can only access resources from its own tenant
        # unless it has admin permissions
        if api_key_info.tenant_id == resource_tenant_id:
            return AuthorizationResult(authorized=True)

        # Check for admin permissions that allow cross-tenant access
        admin_result = self.check_permission(api_key_info, Permission.ADMIN_SYSTEM)
        if admin_result.authorized:
            return AuthorizationResult(authorized=True)

        self.logger.warning(
            "Cross-tenant access denied",
            api_key_tenant=api_key_info.tenant_id,
            resource_tenant=resource_tenant_id,
            key_id=api_key_info.key_id,
        )

        return AuthorizationResult(
            authorized=False, reason="Cross-tenant access not allowed"
        )

    def create_permission_dependency(self, required_permission: Permission):
        """
        Create FastAPI dependency for permission checking.

        Args:
            required_permission: Required permission

        Returns:
            FastAPI dependency function
        """

        def check_permission_dependency(api_key_info: APIKeyInfo) -> APIKeyInfo:
            result = self.check_permission(api_key_info, required_permission)

            if not result.authorized:
                raise HTTPException(status_code=403, detail=result.reason)

            return api_key_info

        return check_permission_dependency

    def create_scope_dependency(self, required_scopes: List[Scope]):
        """
        Create FastAPI dependency for scope checking.

        Args:
            required_scopes: Required scopes

        Returns:
            FastAPI dependency function
        """

        def check_scope_dependency(api_key_info: APIKeyInfo) -> APIKeyInfo:
            result = self.check_scope(api_key_info, required_scopes)

            if not result.authorized:
                raise HTTPException(status_code=403, detail=result.reason)

            return api_key_info

        return check_scope_dependency

    def create_tenant_dependency(self, resource_tenant_id: str):
        """
        Create FastAPI dependency for tenant access checking.

        Args:
            resource_tenant_id: Tenant ID of the resource

        Returns:
            FastAPI dependency function
        """

        def check_tenant_dependency(api_key_info: APIKeyInfo) -> APIKeyInfo:
            result = self.check_tenant_access(api_key_info, resource_tenant_id)

            if not result.authorized:
                raise HTTPException(status_code=403, detail=result.reason)

            return api_key_info

        return check_tenant_dependency
