from __future__ import annotations

import hashlib
import math
import time
from typing import Callable, Dict, Tuple, Awaitable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from .config import Settings


class _Bucket:
    def __init__(self) -> None:
        self.tokens: float = 0.0
        self.last_refill: float = time.monotonic()


class OrchestratorRateLimitMiddleware(BaseHTTPMiddleware):
    """Minimal token-bucket rate limiting per tenant for orchestrator API.

    Returns 403 with error_code=RATE_LIMITED per service contract.
    """

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings
        self._buckets: Dict[str, _Bucket] = {}

    def _key(self, tenant: str) -> str:
        return hashlib.sha256(tenant.encode("utf-8")).hexdigest()

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        cfg = self.settings.config
        if not cfg.rate_limit_enabled:
            return await call_next(request)
        if request.url.path not in ("/orchestrate", "/orchestrate/batch") or request.method != "POST":
            return await call_next(request)

        tenant = request.headers.get(cfg.tenant_header) or "anonymous"
        key = self._key(tenant)
        limit = max(1, cfg.rate_limit_tenant_limit)
        window = max(1, cfg.rate_limit_window_seconds)
        rate_per_sec = limit / float(window)

        b = self._buckets.get(key)
        now = time.monotonic()
        if b is None:
            b = _Bucket()
            b.tokens = float(limit)
            b.last_refill = now
            self._buckets[key] = b
        # Refill
        elapsed = max(0.0, now - b.last_refill)
        b.tokens = min(float(limit), b.tokens + elapsed * rate_per_sec)
        b.last_refill = now

        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return await call_next(request)

        # Block with 403 RATE_LIMITED as per contract
        deficit = max(0.0, 1.0 - b.tokens)
        reset_seconds = math.ceil(deficit / rate_per_sec) if rate_per_sec > 0 else window
        return JSONResponse(
            status_code=403,
            content={
                "error": "RATE_LIMITED",
                "error_code": "RATE_LIMITED",
                "detail": "Too many requests for tenant",
                "tenant": tenant,
                "limit": limit,
                "reset_seconds": reset_seconds,
            },
        )
