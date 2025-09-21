"""
API authentication and idempotency utilities for FastAPI.

- API key authentication with per-tenant scopes
- Optional tenant header requirement
- Simple in-memory idempotency cache for POST endpoints
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Header, HTTPException, Request

from ..config.manager import APIKeyInfo, ConfigManager


@dataclass
class AuthContext:
    """Resolved authentication context for a request."""

    tenant_id: Optional[str]
    api_key_id: Optional[str]
    scopes: List[str]
    authenticated: bool


class IdempotencyCache:
    """A simple in-memory TTL cache for idempotency results."""

    def __init__(self, ttl_seconds: int = 600, max_items: int = 1000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        # key -> (expires_at, value)
        self._store: Dict[str, Tuple[float, Any]] = {}

    def _purge(self) -> None:
        now = time.time()
        # Remove expired
        to_delete = [k for k, (exp, _) in self._store.items() if exp <= now]
        for k in to_delete:
            self._store.pop(k, None)
        # Bound size
        if len(self._store) > self.max_items:
            # Drop oldest by expiration time
            for k, _ in sorted(self._store.items(), key=lambda kv: kv[1][0])[
                : len(self._store) - self.max_items
            ]:
                self._store.pop(k, None)

    def get(self, key: str) -> Optional[Any]:
        if self.ttl_seconds <= 0:
            return None
        self._purge()
        item = self._store.get(key)
        if not item:
            return None
        exp, value = item
        if exp <= time.time():
            # expired
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if self.ttl_seconds <= 0:
            return
        self._purge()
        exp = time.time() + self.ttl_seconds
        self._store[key] = (exp, value)


def build_idempotency_key(
    tenant_id: Optional[str], path: str, idempotency_key: Optional[str]
) -> Optional[str]:
    """Compose a stable cache key from tenant, path, and idempotency key value."""
    if not idempotency_key:
        return None
    tenant_part = tenant_id or "anonymous"
    return f"idem::{tenant_part}::{path}::{idempotency_key}"


def build_api_key_auth(
    config_manager: ConfigManager, required_scopes: Optional[List[str]] = None
) -> Callable:
    """
    Build a FastAPI dependency that enforces API key auth and per-tenant scopes.

    If auth is disabled in configuration, the dependency is a no-op but still
    attaches an AuthContext to request.state for downstream use.
    """
    required_scopes = required_scopes or []

    # Resolve header names safely (handle mocks)
    sec_cfg = getattr(config_manager, "security", None)
    api_key_header_name = getattr(sec_cfg, "api_key_header", "X-API-Key")
    if not isinstance(api_key_header_name, str):
        api_key_header_name = "X-API-Key"
    tenant_header_name = getattr(sec_cfg, "tenant_header", "X-Tenant-ID")
    if not isinstance(tenant_header_name, str):
        tenant_header_name = "X-Tenant-ID"

    # Resolve auth enablement safely (handle mocks)
    auth_cfg = getattr(config_manager, "auth", None)
    _require_tenant_header_val = getattr(auth_cfg, "require_tenant_header", True)
    require_tenant_header = (
        bool(_require_tenant_header_val)
        if isinstance(_require_tenant_header_val, bool)
        else True
    )

    async def _dependency(
        request: Request,
        api_key: Optional[str] = Header(default=None, alias=api_key_header_name),
        tenant_id_header: Optional[str] = Header(
            default=None, alias=tenant_header_name
        ),
    ) -> AuthContext:
        # If disabled, attach context and continue (safe check)
        _enabled_val = getattr(auth_cfg, "enabled", False)
        auth_enabled = bool(_enabled_val) if isinstance(_enabled_val, bool) else False
        if not auth_enabled:
            ctx = AuthContext(
                tenant_id=tenant_id_header,
                api_key_id=None,
                scopes=[],
                authenticated=False,
            )
            request.state.auth = ctx
            return ctx

        # Enforce presence of API key
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key")

        # Lookup API key
        # Access api_keys safely
        api_keys = getattr(auth_cfg, "api_keys", {}) or {}
        # If api_keys is a dict of plain dicts, handle both types
        key_info: Optional[APIKeyInfo]
        try:
            key_info = api_keys.get(api_key)  # type: ignore[arg-type]
        except (AttributeError, TypeError, KeyError) as _e:
            # API key lookup failed - this can happen if api_keys structure is unexpected
            # Treat as invalid key to be safe
            key_info = None
        if not key_info or not key_info.active:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")

        # Resolve tenant
        resolved_tenant = tenant_id_header or key_info.tenant_id
        if require_tenant_header and not tenant_id_header:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required tenant header: {tenant_header_name}",
            )

        # Check scope requirements
        if required_scopes and not set(required_scopes).issubset(set(key_info.scopes)):
            raise HTTPException(
                status_code=403, detail="Insufficient scopes for this operation"
            )

        ctx = AuthContext(
            tenant_id=resolved_tenant,
            api_key_id="***masked***",
            scopes=key_info.scopes,
            authenticated=True,
        )
        request.state.auth = ctx
        return ctx

    return _dependency
