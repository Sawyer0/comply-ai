"""Authentication and authorization services for all services."""

from __future__ import annotations

import jwt
import hashlib
import secrets
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

from fastapi import Depends

from ..config.settings import SecurityConfig
from ..exceptions.base import AuthenticationError, AuthorizationError
from .rate_limiting import RateLimitingService, RateLimitResult

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions."""
    
    # Orchestration permissions
    ORCHESTRATE_DETECTOR = "orchestrate:detector"
    MANAGE_DETECTOR = "manage:detector"
    VIEW_HEALTH = "view:health"
    
    # Analysis permissions
    ANALYZE_CONTENT = "analyze:content"
    MANAGE_POLICIES = "manage:policies"
    VIEW_REPORTS = "view:reports"
    
    # Mapper permissions
    MAP_CANONICAL = "map:canonical"
    MANAGE_TAXONOMY = "manage:taxonomy"
    VIEW_METRICS = "view:metrics"
    
    # Admin permissions
    SYSTEM_ADMIN = "system:admin"
    TENANT_ADMIN = "tenant:admin"


@dataclass
class APIKeyInfo:
    """API key information."""
    
    key_id: str
    key_hash: str
    tenant_id: str
    permissions: List[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if API key has a specific permission."""
        return permission in self.permissions or Permission.SYSTEM_ADMIN in self.permissions
    
    def is_expired(self) -> bool:
        """Check if API key is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class AuthenticationService:
    """Handles authentication for API keys and JWT tokens."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._api_keys: Dict[str, APIKeyInfo] = {}
        self._initialize_api_keys()
    
    def _initialize_api_keys(self):
        """Initialize API keys from configuration."""
        for i, key in enumerate(self.config.api_keys):
            key_hash = self._hash_api_key(key)
            key_info = APIKeyInfo(
                key_id=f"key-{i}",
                key_hash=key_hash,
                tenant_id="default",  # In production, this would come from a database
                permissions=list(Permission),  # Default to all permissions for demo
                created_at=datetime.utcnow(),
            )
            self._api_keys[key_hash] = key_info
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def validate_api_key(self, api_key: str) -> APIKeyInfo:
        """Validate an API key and return key information."""
        if not api_key:
            raise AuthenticationError("API key required")
        
        key_hash = self._hash_api_key(api_key)
        key_info = self._api_keys.get(key_hash)
        
        if not key_info:
            raise AuthenticationError("Invalid API key")
        
        if not key_info.is_active:
            raise AuthenticationError("API key is inactive")
        
        if key_info.is_expired():
            raise AuthenticationError("API key is expired")
        
        return key_info
    
    def generate_jwt_token(self, key_info: APIKeyInfo, expires_in: timedelta = None) -> str:
        """Generate a JWT token for an authenticated API key."""
        if not self.config.jwt_secret:
            raise AuthenticationError("JWT not configured")
        
        expires_in = expires_in or timedelta(hours=1)
        payload = {
            "key_id": key_info.key_id,
            "tenant_id": key_info.tenant_id,
            "permissions": [p.value for p in key_info.permissions],
            "exp": datetime.utcnow() + expires_in,
            "iat": datetime.utcnow(),
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate a JWT token and return payload."""
        if not self.config.jwt_secret:
            raise AuthenticationError("JWT not configured")
        
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")


class AuthorizationService:
    """Handles authorization checks."""
    
    def __init__(self):
        self._role_permissions = {
            "admin": list(Permission),
            "orchestration": [Permission.ORCHESTRATE_DETECTOR, Permission.VIEW_HEALTH],
            "analysis": [Permission.ANALYZE_CONTENT, Permission.VIEW_REPORTS],
            "mapper": [Permission.MAP_CANONICAL, Permission.VIEW_METRICS],
            "readonly": [Permission.VIEW_HEALTH, Permission.VIEW_REPORTS, Permission.VIEW_METRICS],
        }
    
    def check_permission(self, key_info: APIKeyInfo, permission: Permission, resource_tenant_id: str = None) -> bool:
        """Check if an API key has permission for a resource."""
        # Check direct permission
        if key_info.has_permission(permission):
            pass
        else:
            raise AuthorizationError(f"Insufficient permissions: {permission.value}")
        
        # Check tenant access if tenant isolation is enabled
        if resource_tenant_id and key_info.tenant_id != resource_tenant_id:
            # Only system admins can cross tenant boundaries
            if Permission.SYSTEM_ADMIN not in key_info.permissions:
                raise AuthorizationError("Cross-tenant access denied")
        
        return True
    
    def get_permissions_for_role(self, role: str) -> List[Permission]:
        """Get permissions for a role."""
        return self._role_permissions.get(role, [])


class SecurityManager:
    """Unified security manager combining authentication, authorization, and rate limiting."""
    
    def __init__(
        self,
        auth_service: AuthenticationService,
        authz_service: AuthorizationService,
        *,
        rate_limiting_service: Optional[RateLimitingService] = None,
    ):
        self.auth_service = auth_service
        self.authz_service = authz_service
        self.rate_limiting_service = rate_limiting_service
    
    def authenticate_request(self, headers: Dict[str, str]) -> APIKeyInfo:
        """Authenticate a request from headers."""
        # Try API key authentication first
        api_key = headers.get("X-API-Key")
        if api_key:
            return self.auth_service.validate_api_key(api_key)
        
        # Try JWT authentication
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = self.auth_service.validate_jwt_token(token)
            # Convert JWT payload back to APIKeyInfo (simplified)
            key_info = APIKeyInfo(
                key_id=payload["key_id"],
                key_hash="jwt",  # Placeholder
                tenant_id=payload["tenant_id"],
                permissions=[Permission(p) for p in payload["permissions"]],
                created_at=datetime.utcnow(),
            )
            return key_info
        
        # If no authentication required, return default
        if not self.auth_service.config.require_api_key:
            return APIKeyInfo(
                key_id="anonymous",
                key_hash="anonymous",
                tenant_id="default",
                permissions=list(Permission),  # Full permissions for demo
                created_at=datetime.utcnow(),
            )
        
        raise AuthenticationError("Authentication required")
    
    def authorize_request(self, key_info: APIKeyInfo, permission: Permission, resource_tenant_id: str = None) -> bool:
        """Authorize a request."""
        return self.authz_service.check_permission(key_info, permission, resource_tenant_id)

    async def check_rate_limit(
        self,
        request: Any,
        subject: Optional[APIKeyInfo] = None,
    ) -> RateLimitResult:
        """Apply rate limiting if a service is configured."""

        if not self.rate_limiting_service:
            return RateLimitResult(
                allowed=True,
                remaining=9999,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
            )

        return await self.rate_limiting_service.check_rate_limits(request, subject)


# FastAPI dependency functions
def get_security_manager() -> SecurityManager:
    """Get the security manager (to be implemented by each service)."""
    # This should be overridden by each service to provide their instance
    raise NotImplementedError("Each service must implement get_security_manager")


def require_permission(permission: Permission):
    """FastAPI dependency to require a specific permission."""
    def dependency(
        request_headers: Dict[str, str] = None,
        security_manager: SecurityManager = Depends(get_security_manager)
    ) -> APIKeyInfo:
        """Check if the request has the required permission."""
        # Authenticate the request
        key_info = security_manager.authenticate_request(request_headers)
        
        # Authorize the request
        security_manager.authorize_request(key_info, permission)
        
        return key_info
    
    return dependency


def authenticate_request(
    request_headers: Dict[str, str] = None,
    security_manager: SecurityManager = Depends(get_security_manager)
) -> APIKeyInfo:
    """Authenticate a request (FastAPI dependency)."""
    return security_manager.authenticate_request(request_headers or {})
