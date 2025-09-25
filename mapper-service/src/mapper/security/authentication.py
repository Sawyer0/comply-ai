"""
Authentication Service for Mapper Service

Handles authentication logic and request validation
following Single Responsibility Principle.
"""

from datetime import datetime
from typing import Dict, Optional

import structlog
from fastapi import HTTPException, Request
from pydantic import BaseModel

from .api_key_manager import APIKeyManager, APIKeyInfo

logger = structlog.get_logger(__name__)


class AuthenticationResult(BaseModel):
    """Authentication result model."""

    authenticated: bool
    tenant_id: Optional[str] = None
    api_key_info: Optional[APIKeyInfo] = None
    error: Optional[str] = None


class AuthenticationService:
    """
    Authentication service responsible for request authentication.

    Follows SRP by focusing solely on authentication logic:
    - Extracting authentication credentials from requests
    - Validating API keys
    - Managing authentication state
    """

    def __init__(self, api_key_manager: APIKeyManager):
        self.api_key_manager = api_key_manager
        self.logger = logger.bind(component="authentication_service")

    async def authenticate_request(self, request: Request) -> AuthenticationResult:
        """
        Authenticate an incoming request.

        Args:
            request: FastAPI request object

        Returns:
            Authentication result
        """
        try:
            # Extract API key from headers
            api_key = self._extract_api_key(request)

            if not api_key:
                return AuthenticationResult(
                    authenticated=False, error="Missing API key"
                )

            # Validate API key
            api_key_info = await self.api_key_manager.validate_api_key(api_key)

            if not api_key_info:
                self.logger.warning(
                    "Invalid API key used",
                    key_prefix=api_key[:8] if len(api_key) > 8 else "short_key",
                    ip_address=self._get_client_ip(request),
                )
                return AuthenticationResult(
                    authenticated=False, error="Invalid API key"
                )

            self.logger.info(
                "Request authenticated successfully",
                tenant_id=api_key_info.tenant_id,
                key_id=api_key_info.key_id,
                ip_address=self._get_client_ip(request),
            )

            return AuthenticationResult(
                authenticated=True,
                tenant_id=api_key_info.tenant_id,
                api_key_info=api_key_info,
            )

        except Exception as e:
            self.logger.error("Authentication failed", error=str(e))
            return AuthenticationResult(
                authenticated=False, error="Authentication error"
            )

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """
        Extract API key from request headers.

        Supports multiple header formats:
        - X-API-Key: <key>
        - Authorization: Bearer <key>
        - Authorization: ApiKey <key>
        """
        # Try X-API-Key header first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Try Authorization header
        auth_header = request.headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        if auth_header.startswith("ApiKey "):
            return auth_header[7:]  # Remove "ApiKey " prefix

        return None

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first (for load balancers/proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host

        return "unknown"

    def create_authentication_dependency(self):
        """
        Create FastAPI dependency for authentication.

        Returns:
            FastAPI dependency function
        """

        async def authenticate(request: Request) -> APIKeyInfo:
            result = await self.authenticate_request(request)

            if not result.authenticated:
                raise HTTPException(
                    status_code=401, detail=result.error or "Authentication required"
                )

            return result.api_key_info

        return authenticate

    def create_optional_authentication_dependency(self):
        """
        Create FastAPI dependency for optional authentication.

        Returns:
            FastAPI dependency function that returns None if not authenticated
        """

        async def optional_authenticate(request: Request) -> Optional[APIKeyInfo]:
            result = await self.authenticate_request(request)

            if result.authenticated:
                return result.api_key_info

            return None

        return optional_authenticate
