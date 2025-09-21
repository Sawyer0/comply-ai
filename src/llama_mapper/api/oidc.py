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
        """Verify an OIDC bearer token and return claims.

        NOTE: Not implemented. This is a placeholder for future integration.
        """
        raise NotImplementedError("OIDC verification is not implemented yet")
