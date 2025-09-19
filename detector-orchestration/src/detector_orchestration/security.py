from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from .config import Settings


class AuthContext:
    def __init__(self, tenant_id: Optional[str], scopes: List[str]):
        self.tenant_id = tenant_id
        self.scopes = scopes


def build_api_key_auth(settings: Settings, required_scopes: Optional[List[str]] = None):
    api_key_header = APIKeyHeader(name=settings.config.api_key_header, auto_error=False)

    async def _dep(
        api_key: str | None = Security(api_key_header),
        tenant_header: str | None = Depends(
            lambda: None
        ),  # tenant read separately from request in handlers
    ) -> AuthContext:
        if not settings.config:
            raise HTTPException(status_code=500, detail="Service not configured")
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")

        scopes = settings.api_keys.get(api_key)
        if scopes is None:
            raise HTTPException(status_code=401, detail="Invalid API key")

        if required_scopes:
            missing = [s for s in required_scopes if s not in scopes]
            if missing:
                raise HTTPException(status_code=403, detail="Insufficient scope")

        return AuthContext(tenant_id=None, scopes=scopes)

    return _dep

