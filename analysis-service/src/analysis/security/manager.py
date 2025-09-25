"""
Main Security Manager for the Analysis Service.

Orchestrates all security components following SRP.
"""

from typing import Any, Dict

import structlog

from .audit_logger import AuditLogger
from .authentication import AuthenticationManager
from .authorization import AuthorizationManager
from .config import SecurityConfig
from .content_scanner import ContentScanner
from .exceptions import AuthenticationError, AuthorizationError, RateLimitError
from .rate_limiter import RateLimiter
from ..shared_integration import get_shared_database

logger = structlog.get_logger(__name__)


class SecurityManager:
    """
    Main security manager that orchestrates all security components.

    This class follows the Single Responsibility Principle by delegating
    specific security concerns to specialized components.
    """

    def __init__(self, config: SecurityConfig, db_pool=None):
        self.config = config
        self.logger = logger.bind(component="security_manager")

        # Use shared database if not provided
        db_pool = db_pool or get_shared_database()

        # Initialize security components
        self.auth = AuthenticationManager(config, db_pool)
        self.authz = AuthorizationManager()
        self.rate_limiter = RateLimiter(config)
        self.content_scanner = ContentScanner(config)
        self.audit_logger = AuditLogger(config, db_pool)

    async def authenticate_request(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Authenticate incoming request.

        Args:
            headers: Request headers

        Returns:
            Authentication result with user info

        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            result = await self.auth.authenticate_request(headers)

            await self.audit_logger.log_authentication_event(
                success=True,
                user_id=result.get("user_id"),
                method=result.get("auth_method", "unknown"),
            )

            return result

        except AuthenticationError as e:
            await self.audit_logger.log_authentication_event(
                success=False, details={"error": str(e)}
            )
            raise

    async def authorize_action(
        self, user_info: Dict[str, Any], action: str, resource: str
    ) -> bool:
        """
        Check if user is authorized to perform action on resource.

        Args:
            user_info: User information from authentication
            action: Action being performed
            resource: Resource being accessed

        Returns:
            True if authorized

        Raises:
            AuthorizationError: If authorization fails
        """
        try:
            authorized = await self.authz.authorize_action(user_info, action, resource)

            await self.audit_logger.log_authorization_event(
                success=authorized,
                user_id=user_info.get("user_id"),
                action=action,
                resource=resource,
            )

            if not authorized:
                raise AuthorizationError(
                    f"Insufficient permissions for {action} on {resource}"
                )

            return True

        except AuthorizationError:
            raise

    async def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """
        Check if request is within rate limits.

        Args:
            client_id: Client identifier
            endpoint: API endpoint being accessed

        Returns:
            True if within limits

        Raises:
            RateLimitError: If rate limit is exceeded
        """
        try:
            return await self.rate_limiter.check_rate_limit(client_id, endpoint)
        except RateLimitError as e:
            await self.audit_logger.log_rate_limit_event(
                client_id=client_id, endpoint=endpoint, details={"error": str(e)}
            )
            raise

    async def validate_input(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Validate and sanitize input content.

        Args:
            content: Input content to validate
            content_type: Content type header

        Returns:
            Validation result with sanitized content

        Raises:
            ContentSecurityError: If validation fails
        """
        return await self.content_scanner.validate_input(content, content_type)

    async def scan_content(self, content: str) -> Dict[str, Any]:
        """
        Scan content for security threats.

        Args:
            content: Content to scan

        Returns:
            Scan results with threat information
        """
        return await self.content_scanner.scan_content(content)

    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        try:
            audit_metrics = await self.audit_logger.get_security_metrics()

            return {
                "audit_metrics": audit_metrics,
                "rate_limiter_buckets": len(self.rate_limiter._rate_limit_buckets),
                "config": {
                    "audit_enabled": self.config.enable_audit_logging,
                    "content_scanning_enabled": self.config.enable_content_scanning,
                    "rate_limit_per_minute": self.config.rate_limit_requests_per_minute,
                    "max_input_size_mb": self.config.max_input_size_bytes
                    / (1024 * 1024),
                },
            }

        except Exception as e:
            self.logger.error("Failed to get security metrics", error=str(e))
            return {"error": str(e)}

    # Convenience methods for common operations
    async def create_api_key(self, tenant_id: str, permissions: list) -> str:
        """Create a new API key."""
        return await self.auth.create_api_key(tenant_id, permissions)

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        return await self.auth.revoke_api_key(api_key)

    async def create_jwt_token(
        self, user_id: str, tenant_id: str, permissions: list
    ) -> str:
        """Create a JWT token."""
        return await self.auth.create_jwt_token(user_id, tenant_id, permissions)

    def require_permission(self, permission: str):
        """Decorator to require specific permission."""
        return self.authz.require_permission(permission)

    async def cleanup_audit_logs(self) -> int:
        """Clean up old audit logs."""
        return await self.audit_logger.cleanup_old_logs()
