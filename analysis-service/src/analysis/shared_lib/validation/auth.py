"""Authentication and authorization validation functions.

This module provides validation functions for API keys, tokens, and tenant access
following the microservice security guidelines.
"""

import logging
from typing import Optional
import hashlib
import hmac
import time

logger = logging.getLogger(__name__)

# In a real implementation, these would be stored in a secure database
# For now, we'll use a simple in-memory store for demonstration
API_KEYS = {
    "test-api-key-123": {
        "tenant_id": "test-tenant",
        "permissions": ["read", "write"],
        "created_at": time.time(),
        "last_used": time.time(),
        "active": True,
    },
    "demo-api-key-456": {
        "tenant_id": "demo-tenant",
        "permissions": ["read"],
        "created_at": time.time(),
        "last_used": time.time(),
        "active": True,
    },
}


async def validate_api_key(api_key: str) -> bool:
    """Validate an API key.

    Args:
        api_key: The API key to validate

    Returns:
        True if the API key is valid and active, False otherwise
    """
    try:
        if not api_key or not isinstance(api_key, str):
            return False

        # In a real implementation, this would query a database
        # and check for proper hashing, expiration, etc.
        key_info = API_KEYS.get(api_key)
        if not key_info:
            logger.warning(
                "Invalid API key attempted", extra={"api_key_prefix": api_key[:8]}
            )
            return False

        if not key_info.get("active", False):
            logger.warning(
                "Inactive API key attempted", extra={"api_key_prefix": api_key[:8]}
            )
            return False

        # Update last used timestamp
        key_info["last_used"] = time.time()

        logger.debug(
            "API key validated successfully", extra={"api_key_prefix": api_key[:8]}
        )
        return True

    except Exception as e:
        logger.error("API key validation failed", extra={"error": str(e)})
        return False


async def get_tenant_from_api_key(api_key: str) -> Optional[str]:
    """Extract tenant ID from an API key.

    Args:
        api_key: The API key to extract tenant ID from

    Returns:
        The tenant ID if the API key is valid, None otherwise
    """
    try:
        if not api_key or not isinstance(api_key, str):
            return None

        # Validate the API key first
        is_valid = await validate_api_key(api_key)
        if not is_valid:
            return None

        # Get tenant ID from key info
        key_info = API_KEYS.get(api_key)
        if key_info:
            return key_info.get("tenant_id")

        return None

    except Exception as e:
        logger.error("Failed to extract tenant from API key", extra={"error": str(e)})
        return None


async def check_api_key_permissions(api_key: str, required_permission: str) -> bool:
    """Check if an API key has the required permission.

    Args:
        api_key: The API key to check
        required_permission: The permission to check for

    Returns:
        True if the API key has the required permission, False otherwise
    """
    try:
        if not api_key or not isinstance(api_key, str):
            return False

        # Validate the API key first
        is_valid = await validate_api_key(api_key)
        if not is_valid:
            return False

        # Check permissions
        key_info = API_KEYS.get(api_key)
        if key_info:
            permissions = key_info.get("permissions", [])
            return required_permission in permissions

        return False

    except Exception as e:
        logger.error("Failed to check API key permissions", extra={"error": str(e)})
        return False


def generate_api_key(tenant_id: str, permissions: list = None) -> str:
    """Generate a new API key for a tenant.

    Args:
        tenant_id: The tenant ID to generate the key for
        permissions: List of permissions for the key

    Returns:
        The generated API key
    """
    if permissions is None:
        permissions = ["read"]

    # In a real implementation, this would use proper cryptographic key generation
    # and store the key securely in a database
    import uuid

    api_key = f"{tenant_id}-{uuid.uuid4().hex[:16]}"

    # Store the key info
    API_KEYS[api_key] = {
        "tenant_id": tenant_id,
        "permissions": permissions,
        "created_at": time.time(),
        "last_used": time.time(),
        "active": True,
    }

    logger.info(
        "Generated new API key",
        extra={"tenant_id": tenant_id, "api_key_prefix": api_key[:8]},
    )
    return api_key


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key.

    Args:
        api_key: The API key to revoke

    Returns:
        True if the key was revoked, False if not found
    """
    try:
        if api_key in API_KEYS:
            API_KEYS[api_key]["active"] = False
            logger.info("API key revoked", extra={"api_key_prefix": api_key[:8]})
            return True

        return False

    except Exception as e:
        logger.error("Failed to revoke API key", extra={"error": str(e)})
        return False
