"""
OIDC integration stub for future use.

This module defines a minimal interface for an OIDC provider to enable
future migration from API key auth to token-based auth.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class OIDCProvider:
    """Stub OIDC provider interface (not implemented)."""

    def __init__(
        self, issuer_url: str, client_id: str, audience: Optional[str] = None
    ) -> None:
        self.issuer_url = issuer_url
        self.client_id = client_id
        self.audience = audience

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify an OIDC bearer token and return claims."""
        import jwt
        import httpx
        from datetime import datetime, timedelta
        
        try:
            # For demo purposes, return mock claims for valid-looking tokens
            if token.startswith("eyJ") and len(token) > 100:
                # Mock successful verification
                return {
                    "sub": "user-123",
                    "email": "user@example.com",
                    "iss": self.issuer_url,
                    "aud": self.audience or self.client_id,
                    "exp": int((datetime.now() + timedelta(hours=1)).timestamp()),
                    "iat": int(datetime.now().timestamp()),
                    "tenant_id": "demo-tenant",
                    "scopes": ["map:write", "map:read"]
                }
            else:
                raise ValueError("Invalid token format")
                
        except Exception as e:
            raise ValueError(f"Token verification failed: {e}")
