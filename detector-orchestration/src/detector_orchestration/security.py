"""Security and authentication utilities for detector orchestration."""


from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .config import Settings
from .metrics import OrchestrationMetricsCollector

logger = logging.getLogger(__name__)


class AuthContext:
    """Authentication context containing tenant and scope information."""

    def __init__(self, tenant_id: Optional[str], scopes: List[str]):
        """Initialize authentication context.

        Args:
            tenant_id: The tenant identifier
            scopes: List of authorized scopes for this context
        """
        self.tenant_id = tenant_id
        self.scopes = scopes

    def is_authenticated(self) -> bool:
        """Check if the user is authenticated (has a tenant_id)."""
        return self.tenant_id is not None

    def has_scope(self, scope: str) -> bool:
        """Check if the user has a specific scope."""
        return scope in self.scopes

    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if the user has any of the specified scopes."""
        return any(scope in self.scopes for scope in scopes)


def _record_rbac_allowance(
    metrics: OrchestrationMetricsCollector | None,
    path: str,
    tenant: Optional[str],
) -> None:
    """Record RBAC allowance in metrics.

    Args:
        metrics: Optional metrics collector
        path: Request path
        tenant: Tenant identifier
    """
    if metrics:
        try:
            metrics.record_rbac(path, tenant, decision="allowed", scope="none")
        except (ConnectionError, TimeoutError, AttributeError) as e:
            # Metrics recording failed - log but don't break authentication
            # These specific exceptions indicate metrics system issues, not auth issues
            logger.warning(
                "Failed to record RBAC allowance metrics",
                extra={
                    "error": str(e),
                    "path": path,
                    "tenant": tenant,
                    "decision": "allowed"
                }
            )


def _record_rbac_metrics(
    metrics: OrchestrationMetricsCollector | None,
    path: str,
    tenant: Optional[str],
    decision: str,
    scope: Optional[str] = None,
) -> None:
    """Record RBAC decision in metrics.

    Args:
        metrics: Optional metrics collector
        path: Request path
        tenant: Tenant identifier
        decision: RBAC decision type
        scope: Optional scope information
    """
    if metrics:
        try:
            metrics.record_rbac(path, tenant, decision=decision, scope=scope)
        except (ConnectionError, TimeoutError, AttributeError) as e:
            # Metrics recording failed - log but don't break authentication
            # These specific exceptions indicate metrics system issues, not auth issues
            logger.warning(
                "Failed to record RBAC metrics",
                extra={
                    "error": str(e),
                    "path": path,
                    "tenant": tenant,
                    "decision": decision,
                    "scope": scope
                }
            )


def build_api_key_auth(
    settings: Settings,
    required_scopes: Optional[List[str]] = None,
    metrics: OrchestrationMetricsCollector | None = None,
) -> callable:  # type: ignore
    """Build an API key dependency.

    If no API keys are configured, authentication is treated as disabled
    (useful for tests and dev environments).

    Args:
        settings: Configuration settings containing API key information
        required_scopes: List of required scopes for the endpoint
        metrics: Optional metrics collector for RBAC recording

    Returns:
        A FastAPI dependency function that validates API keys and scopes
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
        if not settings.api_keys:
            _record_rbac_allowance(metrics, request.url.path, tenant)
            return AuthContext(tenant_id=None, scopes=[])

        # Support multiple auth types: API key header or Bearer token
        credential: Optional[str] = api_key
        if not credential:
            authz = request.headers.get("Authorization")
            if isinstance(authz, str) and authz.lower().startswith("bearer "):
                credential = authz.split(" ", 1)[1].strip()

        # Enforce credential when keys are configured
        if not credential:
            _record_rbac_metrics(metrics, request.url.path, tenant, decision="missing_api_key")
            raise HTTPException(status_code=401, detail="Missing API key")

        scopes = settings.api_keys.get(credential)
        if scopes is None:
            _record_rbac_metrics(metrics, request.url.path, tenant, decision="invalid_api_key")
            raise HTTPException(status_code=401, detail="Invalid API key")

        if required_scopes:
            missing = [s for s in required_scopes if s not in scopes]
            if missing:
                _record_rbac_metrics(
                    metrics, request.url.path, tenant,
                    decision="insufficient_scope",
                    scope=missing[0] if missing else None
                )
                raise HTTPException(status_code=403, detail="Insufficient scope")

        _record_rbac_metrics(metrics, request.url.path, tenant, decision="allowed")
        return AuthContext(tenant_id=tenant, scopes=scopes)

    return _dep
