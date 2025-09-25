"""API key management functionality following SRP.

This module provides ONLY API key management - creation, validation, and rotation.
Single Responsibility: Manage API keys for authentication.
"""

import hashlib
import secrets
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from enum import Enum

from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import AuthenticationError, ValidationError

logger = logging.getLogger(__name__)


class ApiKeyStatus(str, Enum):
    """API key status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ApiKey:
    """API key data structure."""

    def __init__(
        self,
        key_id: str,
        tenant_id: str,
        key_hash: str,
        status: ApiKeyStatus = ApiKeyStatus.ACTIVE,
        created_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        last_used: Optional[datetime] = None,
        permissions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.key_id = key_id
        self.tenant_id = tenant_id
        self.key_hash = key_hash
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.expires_at = expires_at
        self.last_used = last_used
        self.permissions = permissions or []
        self.metadata = metadata or {}


class ApiKeyManager:
    """Manages API keys for authentication.

    Single Responsibility: Create, validate, and manage API keys.
    Does NOT handle: authorization, rate limiting, tenant management.
    """

    def __init__(self, default_expiry_days: int = 365):
        """Initialize API key manager.

        Args:
            default_expiry_days: Default expiry period for API keys in days
        """
        self.default_expiry_days = default_expiry_days
        self._api_keys: Dict[str, ApiKey] = {}  # key_id -> ApiKey
        self._key_hash_index: Dict[str, str] = {}  # key_hash -> key_id
        self._tenant_keys: Dict[str, List[str]] = {}  # tenant_id -> [key_ids]

    def generate_api_key(
        self,
        tenant_id: str,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Generate a new API key for a tenant.

        Args:
            tenant_id: Tenant identifier
            permissions: List of permissions for the key
            expires_in_days: Expiry period in days (optional)
            metadata: Additional metadata for the key

        Returns:
            Tuple of (key_id, raw_api_key)
        """
        correlation_id = get_correlation_id()

        try:
            # Generate secure API key
            raw_key = self._generate_secure_key()
            key_hash = self._hash_key(raw_key)
            key_id = f"ak_{secrets.token_urlsafe(16)}"

            # Calculate expiry
            expires_in = expires_in_days or self.default_expiry_days
            expires_at = datetime.utcnow() + timedelta(days=expires_in)

            # Create API key object
            api_key = ApiKey(
                key_id=key_id,
                tenant_id=tenant_id,
                key_hash=key_hash,
                expires_at=expires_at,
                permissions=permissions or [],
                metadata=metadata or {},
            )

            # Store API key
            self._api_keys[key_id] = api_key
            self._key_hash_index[key_hash] = key_id

            # Update tenant index
            if tenant_id not in self._tenant_keys:
                self._tenant_keys[tenant_id] = []
            self._tenant_keys[tenant_id].append(key_id)

            logger.info(
                "Generated API key for tenant %s",
                tenant_id,
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "key_id": key_id,
                    "expires_at": expires_at.isoformat(),
                    "permissions": permissions or [],
                },
            )

            return key_id, raw_key

        except Exception as e:
            logger.error(
                "Failed to generate API key for tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            raise ValidationError(
                f"Failed to generate API key: {str(e)}", correlation_id=correlation_id
            ) from e

    def validate_api_key(self, raw_key: str) -> Optional[ApiKey]:
        """Validate an API key and return key information.

        Args:
            raw_key: Raw API key to validate

        Returns:
            ApiKey object if valid, None otherwise
        """
        correlation_id = get_correlation_id()

        try:
            # Hash the provided key
            key_hash = self._hash_key(raw_key)

            # Look up key ID
            key_id = self._key_hash_index.get(key_hash)
            if not key_id:
                logger.warning(
                    "API key validation failed: key not found",
                    extra={"correlation_id": correlation_id},
                )
                return None

            # Get API key object
            api_key = self._api_keys.get(key_id)
            if not api_key:
                logger.warning(
                    "API key validation failed: key object not found",
                    extra={"correlation_id": correlation_id, "key_id": key_id},
                )
                return None

            # Check key status
            if api_key.status != ApiKeyStatus.ACTIVE:
                logger.warning(
                    "API key validation failed: key not active",
                    extra={
                        "correlation_id": correlation_id,
                        "key_id": key_id,
                        "status": api_key.status.value,
                    },
                )
                return None

            # Check expiry
            if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                logger.warning(
                    "API key validation failed: key expired",
                    extra={
                        "correlation_id": correlation_id,
                        "key_id": key_id,
                        "expires_at": api_key.expires_at.isoformat(),
                    },
                )
                # Update status to expired
                api_key.status = ApiKeyStatus.EXPIRED
                return None

            # Update last used timestamp
            api_key.last_used = datetime.utcnow()

            logger.debug(
                "API key validated successfully",
                extra={
                    "correlation_id": correlation_id,
                    "key_id": key_id,
                    "tenant_id": api_key.tenant_id,
                },
            )

            return api_key

        except Exception as e:
            logger.error(
                "API key validation error: %s",
                str(e),
                extra={"correlation_id": correlation_id, "error": str(e)},
            )
            return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: Key identifier to revoke

        Returns:
            True if revocation successful, False otherwise
        """
        correlation_id = get_correlation_id()

        try:
            api_key = self._api_keys.get(key_id)
            if not api_key:
                logger.warning(
                    "Cannot revoke API key: key not found",
                    extra={"correlation_id": correlation_id, "key_id": key_id},
                )
                return False

            # Update status
            api_key.status = ApiKeyStatus.REVOKED

            # Remove from hash index
            self._key_hash_index.pop(api_key.key_hash, None)

            logger.info(
                "API key revoked",
                extra={
                    "correlation_id": correlation_id,
                    "key_id": key_id,
                    "tenant_id": api_key.tenant_id,
                },
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to revoke API key %s: %s",
                key_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "key_id": key_id,
                    "error": str(e),
                },
            )
            return False

    def list_tenant_keys(self, tenant_id: str) -> List[ApiKey]:
        """List all API keys for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of API keys for the tenant
        """
        key_ids = self._tenant_keys.get(tenant_id, [])
        return [
            self._api_keys[key_id] for key_id in key_ids if key_id in self._api_keys
        ]

    def cleanup_expired_keys(self) -> int:
        """Clean up expired API keys.

        Returns:
            Number of keys cleaned up
        """
        correlation_id = get_correlation_id()
        current_time = datetime.utcnow()
        cleaned_count = 0

        expired_keys = []
        for key_id, api_key in self._api_keys.items():
            if (
                api_key.expires_at
                and current_time > api_key.expires_at
                and api_key.status == ApiKeyStatus.ACTIVE
            ):
                expired_keys.append(key_id)

        for key_id in expired_keys:
            api_key = self._api_keys[key_id]
            api_key.status = ApiKeyStatus.EXPIRED
            self._key_hash_index.pop(api_key.key_hash, None)
            cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                "Cleaned up %d expired API keys",
                cleaned_count,
                extra={"correlation_id": correlation_id},
            )

        return cleaned_count

    def _generate_secure_key(self) -> str:
        """Generate a secure API key."""
        # Generate 32 bytes of random data and encode as URL-safe base64
        return f"llm_{secrets.token_urlsafe(32)}"

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def get_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get information about an API key (without sensitive data).

        Args:
            key_id: Key identifier

        Returns:
            Dictionary with key information
        """
        api_key = self._api_keys.get(key_id)
        if not api_key:
            return None

        return {
            "key_id": api_key.key_id,
            "tenant_id": api_key.tenant_id,
            "status": api_key.status.value,
            "created_at": api_key.created_at.isoformat(),
            "expires_at": (
                api_key.expires_at.isoformat() if api_key.expires_at else None
            ),
            "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            "permissions": api_key.permissions,
            "metadata": api_key.metadata,
        }


# Export only the API key management functionality
__all__ = [
    "ApiKeyManager",
    "ApiKey",
    "ApiKeyStatus",
]
