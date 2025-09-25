"""Authentication functionality following SRP.

This module provides authentication capabilities:
- API Key Management: Create, validate, and manage API keys
- JWT Security: JWT token management (to be implemented)
- Secrets Rotation: Automated secrets rotation (to be implemented)
"""

from .api_key_manager import (
    ApiKeyManager,
    ApiKey,
    ApiKeyStatus,
)

__all__ = [
    "ApiKeyManager",
    "ApiKey",
    "ApiKeyStatus",
]
