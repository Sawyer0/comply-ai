"""
Authentication module for the Analysis Service.

Handles API key and JWT token authentication.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from .config import SecurityConfig
from .exceptions import AuthenticationError, SecurityError
from ..shared_integration import get_shared_database

try:
    import jwt
except ImportError:
    jwt = None

logger = structlog.get_logger(__name__)


class AuthenticationManager:
    """Manages authentication for API keys and JWT tokens."""

    def __init__(self, config: SecurityConfig, db_pool=None):
        self.config = config
        self.db = db_pool or get_shared_database()
        self.logger = logger.bind(component="authentication_manager")

    async def authenticate_request(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Authenticate incoming request using API key or JWT token.

        Args:
            headers: Request headers

        Returns:
            Authentication result with user info

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.config.require_api_key:
            return {"authenticated": True, "user": "anonymous"}

        # Try JWT token first (production-grade authentication)
        jwt_token = headers.get("Authorization", "").replace("Bearer ", "")
        if jwt_token:
            return await self._authenticate_jwt_token(jwt_token)

        # Fallback to API key authentication
        api_key = headers.get(self.config.api_key_header)
        if not api_key:
            raise AuthenticationError(
                "Missing authentication credentials (API key or JWT token)"
            )

        # Validate API key using secure database storage
        key_info = await self._get_api_key_info(api_key)
        if not key_info:
            self.logger.warning(
                "Invalid API key used",
                api_key_prefix=(
                    api_key[:8] + "..." if len(api_key) > 8 else "short_key"
                ),
            )
            raise AuthenticationError("Invalid API key")

        # Check if key is expired
        if key_info.get("expires_at") and datetime.now() > key_info["expires_at"]:
            self.logger.warning("Expired API key used", user_id=key_info.get("user_id"))
            raise AuthenticationError("API key expired")

        # Check if key is active
        if not key_info.get("active", True):
            self.logger.warning(
                "Inactive API key used", user_id=key_info.get("user_id")
            )
            raise AuthenticationError("API key inactive")

        # Update last used timestamp
        await self._update_api_key_usage(api_key)

        self.logger.info(
            "API key authentication successful",
            user_id=key_info.get("user_id"),
            api_key_id=key_info.get("key_id"),
        )

        return {
            "authenticated": True,
            "user_id": key_info.get("user_id"),
            "tenant_id": key_info.get("tenant_id"),
            "permissions": key_info.get("permissions", []),
            "api_key_id": key_info.get("key_id"),
            "auth_method": "api_key",
        }

    async def _authenticate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Authenticate JWT token for production-grade authentication."""
        if jwt is None:
            raise AuthenticationError(
                "JWT authentication not available: PyJWT not installed"
            )

        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )

            # Check token expiration
            if payload.get("exp") and datetime.utcnow().timestamp() > payload["exp"]:
                raise AuthenticationError("JWT token expired")

            # Validate required claims
            if not payload.get("sub"):
                raise AuthenticationError("Invalid JWT token: missing subject")

            # Check token permissions
            permissions = payload.get("permissions", [])
            tenant_id = payload.get("tenant_id")

            self.logger.info(
                "JWT authentication successful",
                user_id=payload.get("sub"),
                tenant_id=tenant_id,
            )

            return {
                "authenticated": True,
                "user_id": payload.get("sub"),
                "tenant_id": tenant_id,
                "permissions": permissions,
                "auth_method": "jwt",
                "expires_at": (
                    datetime.fromtimestamp(payload.get("exp", 0))
                    if payload.get("exp")
                    else None
                ),
            }

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("JWT token expired") from None
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid JWT token: {str(e)}") from e
        except Exception as e:
            self.logger.error("JWT authentication failed", error=str(e))
            raise AuthenticationError(f"Authentication failed: {str(e)}") from e

    async def create_jwt_token(
        self,
        user_id: str,
        tenant_id: str,
        permissions: List[str],
        expires_hours: int = 24,
    ) -> str:
        """Create JWT token for production authentication."""
        if jwt is None:
            raise SecurityError("JWT token creation not available: PyJWT not installed")

        try:
            # Create token payload
            payload = {
                "sub": user_id,
                "tenant_id": tenant_id,
                "permissions": permissions,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=expires_hours),
                "iss": "analysis-service",
                "aud": "compliance-platform",
            }

            # Generate JWT token
            token = jwt.encode(
                payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm
            )

            self.logger.info(
                "JWT token created successfully",
                user_id=user_id,
                tenant_id=tenant_id,
                expires_hours=expires_hours,
            )

            return token

        except Exception as e:
            self.logger.error("Failed to create JWT token", error=str(e))
            raise SecurityError(f"Failed to create JWT token: {str(e)}") from e

    async def _get_api_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get API key information from secure database storage."""
        try:
            # Hash the API key for secure lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Query database for API key
            query = """
                SELECT key_id, tenant_id, permissions, expires_at, is_active, created_at
                FROM api_keys
                WHERE key_hash = $1 AND is_active = true
            """

            result = await self.db.fetchrow(query, key_hash)
            if result:
                return {
                    "key_id": result["key_id"],
                    "tenant_id": result["tenant_id"],
                    "permissions": result["permissions"],
                    "expires_at": result["expires_at"],
                    "created_at": result["created_at"],
                }
            return None

        except Exception as e:
            self.logger.error("Failed to retrieve API key info", error=str(e))
            return None

    async def _update_api_key_usage(self, api_key: str) -> None:
        """Update API key last used timestamp."""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            query = """
                UPDATE api_keys 
                SET last_used = $1 
                WHERE key_hash = $2
            """
            await self.db.execute(query, datetime.utcnow(), key_hash)
        except Exception as e:
            self.logger.error("Failed to update API key usage", error=str(e))

    async def create_api_key(
        self, tenant_id: str, permissions: List[str], expires_days: int = 365
    ) -> str:
        """Create a new API key with secure storage."""
        try:
            # Generate secure API key
            api_key = f"ak_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Calculate expiration
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

            # Store in database
            query = """
                INSERT INTO api_keys (key_hash, tenant_id, permissions, expires_at, is_active)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING key_id
            """

            key_id = await self.db.fetchval(
                query, key_hash, tenant_id, permissions, expires_at, True
            )

            self.logger.info(
                "API key created successfully",
                key_id=key_id,
                tenant_id=tenant_id,
                expires_at=expires_at,
            )

            return api_key

        except Exception as e:
            self.logger.error("Failed to create API key", error=str(e))
            raise SecurityError(f"Failed to create API key: {str(e)}") from e

    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            query = "UPDATE api_keys SET is_active = false WHERE key_hash = $1"
            result = await self.db.execute(query, key_hash)

            return result == "UPDATE 1"

        except Exception as e:
            self.logger.error("Failed to revoke API key", error=str(e))
            return False
