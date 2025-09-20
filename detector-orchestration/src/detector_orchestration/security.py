from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader

from .config import Settings
from .metrics import OrchestrationMetricsCollector


class AuthContext:
    def __init__(self, tenant_id: Optional[str], scopes: List[str]):
        self.tenant_id = tenant_id
        self.scopes = scopes


def build_api_key_auth(settings: Settings, required_scopes: Optional[List[str]] = None, metrics: OrchestrationMetricsCollector | None = None):
    """
    Build an API key dependency. If no API keys are configured, authentication is
    treated as disabled (useful for tests and dev environments).
    """
    api_key_header = APIKeyHeader(name=settings.config.api_key_header, auto_error=False)

    async def _dep(
        request: Request,
        api_key: str | None = Security(api_key_header),
    ) -> AuthContext:
        if not settings.config:
            raise HTTPException(status_code=500, detail="Service not configured")

        tenant = request.headers.get(settings.config.tenant_header) or None

        # If no API keys configured, allow unauthenticated access (test/dev mode)
        try:
            if not settings.api_keys:
                if metrics:
                    try:
                        metrics.record_rbac(request.url.path, tenant, decision="allowed", scope="none")
                    except Exception:
                        pass
                return AuthContext(tenant_id=None, scopes=[])
        except Exception:
            # Be permissive if settings.api_keys is not well-formed
            if metrics:
                try:
                    metrics.record_rbac(request.url.path, tenant, decision="allowed", scope="none")
                except Exception:
                    pass
            return AuthContext(tenant_id=None, scopes=[])

        # Support multiple auth types: API key header or Bearer token
        credential: Optional[str] = api_key
        if not credential:
            try:
                authz = request.headers.get("Authorization")
                if isinstance(authz, str) and authz.lower().startswith("bearer "):
                    credential = authz.split(" ", 1)[1].strip()
            except Exception:
                credential = None

        # Enforce credential when keys are configured
        if not credential:
            if metrics:
                try:
                    metrics.record_rbac(request.url.path, tenant, decision="missing_api_key")
                except Exception:
                    pass
            raise HTTPException(status_code=401, detail="Missing API key")

        scopes = settings.api_keys.get(credential)
        if scopes is None:
            if metrics:
                try:
                    metrics.record_rbac(request.url.path, tenant, decision="invalid_api_key")
                except Exception:
                    pass
            raise HTTPException(status_code=401, detail="Invalid API key")

        if required_scopes:
            missing = [s for s in required_scopes if s not in scopes]
            if missing:
                if metrics:
                    try:
                        metrics.record_rbac(request.url.path, tenant, decision="insufficient_scope", scope=missing[0] if missing else None)
                    except Exception:
                        pass
                raise HTTPException(status_code=403, detail="Insufficient scope")

        if metrics:
            try:
                metrics.record_rbac(request.url.path, tenant, decision="allowed")
            except Exception:
                pass
        return AuthContext(tenant_id=None, scopes=scopes)

    return _dep

