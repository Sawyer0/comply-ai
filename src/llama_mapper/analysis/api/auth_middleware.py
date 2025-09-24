"""
Authentication middleware for the Analysis Module API.

This module provides FastAPI middleware for API key authentication,
authorization, and rate limiting.
"""

import logging
import time
from typing import Callable, Optional, Set

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from ..infrastructure.auth import APIKeyManager, APIKeyScope

logger = logging.getLogger(__name__)


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    API key authentication middleware.

    Validates API keys and enforces authorization for API endpoints.
    """

    def __init__(
        self,
        app,
        api_key_manager: APIKeyManager,
        required_scopes: Optional[Set[APIKeyScope]] = None,
        rate_limit_requests_per_minute: int = 60,
    ):
        """
        Initialize the authentication middleware.

        Args:
            app: FastAPI application
            api_key_manager: API key manager instance
            required_scopes: Required scopes for all endpoints
            rate_limit_requests_per_minute: Rate limit per minute
        """
        super().__init__(app)
        self.api_key_manager = api_key_manager
        self.required_scopes = required_scopes or {APIKeyScope.ANALYZE}
        self.rate_limit_requests_per_minute = rate_limit_requests_per_minute

        # Rate limiting storage (in production, use Redis)
        self._rate_limit_storage = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through authentication middleware.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Skip authentication for health checks and docs
        if self._should_skip_auth(request):
            return await call_next(request)

        # Extract API key from headers
        api_key = self._extract_api_key(request)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate API key
        validated_key = self.api_key_manager.validate_api_key(
            api_key, list(self.required_scopes)
        )

        if not validated_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check rate limiting
        if not self._check_rate_limit(validated_key, request):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        # Add authentication info to request state
        request.state.api_key = validated_key
        request.state.tenant_id = validated_key.tenant_id

        # Process request
        response = await call_next(request)

        # Add security headers
        response.headers["X-API-Key-ID"] = validated_key.key_id
        response.headers["X-Tenant-ID"] = validated_key.tenant_id

        return response

    def _should_skip_auth(self, request: Request) -> bool:
        """
        Check if authentication should be skipped for this request.

        Args:
            request: HTTP request

        Returns:
            True if auth should be skipped
        """
        skip_paths = {
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/health/live",
            "/health/ready",
            "/api/v1/analysis/health",
            "/api/v1/analysis/metrics",
        }

        return request.url.path in skip_paths

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """
        Extract API key from request headers.

        Args:
            request: HTTP request

        Returns:
            API key if found, None otherwise
        """
        # Check X-API-Key header first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        return None

    def _check_rate_limit(self, api_key, request: Request) -> bool:
        """
        Check if request is within rate limits.

        Args:
            api_key: Validated API key
            request: HTTP request

        Returns:
            True if within limits, False otherwise
        """
        # Use API key's rate limit if specified, otherwise use default
        rate_limit = api_key.rate_limit or self.rate_limit_requests_per_minute

        # Create rate limit key
        rate_key = f"{api_key.key_id}:{int(time.time() // 60)}"

        # Check current count
        current_count = self._rate_limit_storage.get(rate_key, 0)

        if current_count >= rate_limit:
            logger.warning("Rate limit exceeded for API key %s", api_key.key_id)
            return False

        # Increment count
        self._rate_limit_storage[rate_key] = current_count + 1

        # Clean up old entries (simple cleanup)
        current_minute = int(time.time() // 60)
        self._rate_limit_storage = {
            k: v
            for k, v in self._rate_limit_storage.items()
            if int(k.split(":")[1]) >= current_minute - 1
        }

        return True


class APIKeyAuthDependency:
    """
    FastAPI dependency for API key authentication.

    Can be used as a dependency in endpoint functions.
    """

    def __init__(
        self,
        api_key_manager: APIKeyManager,
        required_scopes: Optional[Set[APIKeyScope]] = None,
    ):
        """
        Initialize the authentication dependency.

        Args:
            api_key_manager: API key manager instance
            required_scopes: Required scopes for the endpoint
        """
        self.api_key_manager = api_key_manager
        self.required_scopes = required_scopes or {APIKeyScope.ANALYZE}

    async def __call__(self, request: Request) -> dict:
        """
        Validate API key and return authentication info.

        Args:
            request: HTTP request

        Returns:
            Dictionary with authentication info

        Raises:
            HTTPException: If authentication fails
        """
        # Extract API key
        api_key = self._extract_api_key(request)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="API key required"
            )

        # Validate API key
        validated_key = self.api_key_manager.validate_api_key(
            api_key, list(self.required_scopes)
        )

        if not validated_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key",
            )

        return {
            "api_key": validated_key,
            "tenant_id": validated_key.tenant_id,
            "scopes": [scope.value for scope in validated_key.scopes],
        }

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Check X-API-Key header first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return api_key

        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        return None


def create_auth_middleware(
    api_key_manager: APIKeyManager,
    required_scopes: Optional[Set[APIKeyScope]] = None,
    rate_limit_requests_per_minute: int = 60,
) -> Callable:
    """
    Create authentication middleware factory.

    Args:
        api_key_manager: API key manager instance
        required_scopes: Required scopes for all endpoints
        rate_limit_requests_per_minute: Rate limit per minute

    Returns:
        Middleware factory function
    """

    def middleware_factory(app):
        return APIKeyAuthMiddleware(
            app=app,
            api_key_manager=api_key_manager,
            required_scopes=required_scopes,
            rate_limit_requests_per_minute=rate_limit_requests_per_minute,
        )

    return middleware_factory


def create_auth_dependency(
    api_key_manager: APIKeyManager, required_scopes: Optional[Set[APIKeyScope]] = None
) -> APIKeyAuthDependency:
    """
    Create authentication dependency.

    Args:
        api_key_manager: API key manager instance
        required_scopes: Required scopes for the endpoint

    Returns:
        Authentication dependency instance
    """
    return APIKeyAuthDependency(
        api_key_manager=api_key_manager, required_scopes=required_scopes
    )
