"""
Rate limiting middleware and in-memory token bucket backend.

- Identity precedence: API key -> tenant -> client IP
- Endpoint keys: "map" for /map, "map_batch" for /map/batch (others bypass)
- Emits standard RateLimit-* headers and optional legacy X-RateLimit-* headers
- Records Prometheus + legacy counters via MetricsCollector
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Literal, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector

IdentityKind = Literal["api_key", "tenant", "ip"]


@dataclass
class AllowResult:
    allowed: bool
    remaining: int
    limit: int
    reset_seconds: float
    identity_kind: IdentityKind


@dataclass
class _BucketState:
    tokens: float
    last_refill: float
    limit: int
    window: int


class RateLimiterBackend:
    async def allow(
        self,
        endpoint: str,
        identity: str,
        limit: int,
        window: int,
        cost: int = 1,
    ) -> AllowResult:  # pragma: no cover - interface
        raise NotImplementedError


class MemoryRateLimiterBackend(RateLimiterBackend):
    def __init__(self) -> None:
        # (endpoint, identity) -> bucket state
        self._buckets: Dict[Tuple[str, str], _BucketState] = {}
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    def _get_lock(self, key: Tuple[str, str]) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def allow(
        self, endpoint: str, identity: str, limit: int, window: int, cost: int = 1
    ) -> AllowResult:
        now = time.monotonic()
        key = (endpoint, identity)
        lock = self._get_lock(key)
        async with lock:
            rate_per_sec = limit / float(window)
            state = self._buckets.get(key)
            if state is None:
                state = _BucketState(
                    tokens=float(limit), last_refill=now, limit=limit, window=window
                )
                self._buckets[key] = state
            # Refill tokens
            elapsed = max(0.0, now - state.last_refill)
            refill = elapsed * rate_per_sec
            state.tokens = min(state.limit, state.tokens + refill)
            state.last_refill = now

            if state.tokens >= cost:
                state.tokens -= cost
                remaining = int(math.floor(state.tokens))
                return AllowResult(
                    True, remaining, state.limit, 0.0, "api_key"
                )  # identity_kind set by caller
            else:
                # Time until enough tokens for 1 request
                deficit = max(0.0, cost - state.tokens)
                reset_seconds = (
                    deficit / rate_per_sec if rate_per_sec > 0 else float("inf")
                )
                return AllowResult(False, 0, state.limit, reset_seconds, "api_key")


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_untrusted_hop(xff: str, trusted_proxies: int) -> Optional[str]:
    # X-Forwarded-For is a comma-separated list client, proxy1, proxy2, ...
    parts = [p.strip() for p in xff.split(",") if p.strip()]
    if not parts:
        return None
    # Trusted proxies are at the end; take the element before the trusted tail
    index = max(0, len(parts) - 1 - trusted_proxies)
    return parts[index] if index < len(parts) else parts[0]


def _extract_identity(
    request: Request, config: ConfigManager
) -> Tuple[IdentityKind, str]:
    # 1) API Key
    api_key_header = getattr(config.security, "api_key_header", "X-API-Key")
    api_key_val = request.headers.get(api_key_header)
    if api_key_val:
        return ("api_key", _hash_value(api_key_val))

    # 2) Tenant
    tenant_header = getattr(config.security, "tenant_header", "X-Tenant-ID")
    tenant_id = request.headers.get(tenant_header)
    if not tenant_id:
        # Try auth context if present
        auth = getattr(request.state, "auth", None)
        tenant_id = getattr(auth, "tenant_id", None)
    if tenant_id:
        return ("tenant", tenant_id)

    # 3) Client IP
    trusted = (
        int(getattr(config.rate_limit, "trusted_proxies", 0))
        if hasattr(config, "rate_limit")
        else 0
    )
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        ip = _first_untrusted_hop(xff, trusted)
        if ip:
            return ("ip", ip)
    client_host = request.client.host if request.client else "unknown"
    return ("ip", client_host)


def _endpoint_key(path: str) -> Optional[str]:
    if path == "/map":
        return "map"
    if path == "/map/batch":
        return "map_batch"
    return None


def _limits_for(
    endpoint: str, identity_kind: IdentityKind, config: ConfigManager
) -> Tuple[int, int]:
    rl = getattr(config, "rate_limit", None)
    if rl is None:
        return (600, 60)
    window = int(getattr(rl, "window_seconds", 60))
    ep_cfg = rl.endpoints.get(endpoint) if hasattr(rl, "endpoints") else None
    if ep_cfg is None:
        # default
        limit = 600 if identity_kind in ("api_key", "tenant") else 120
        return (limit, window)
    if identity_kind == "api_key":
        return (int(ep_cfg.api_key_limit), window)
    if identity_kind == "tenant":
        return (int(ep_cfg.tenant_limit), window)
    return (int(ep_cfg.ip_limit), window)


def _emit_headers(
    response: Response,
    limit: int,
    remaining: int,
    reset_seconds: float,
    window: int,
    emit_standard: bool,
    emit_legacy: bool,
) -> None:
    reset_int = int(math.ceil(max(0.0, reset_seconds)))
    if emit_standard:
        response.headers["RateLimit-Limit"] = f"{limit};w={window}"
        response.headers["RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["RateLimit-Reset"] = str(reset_int)
        if remaining <= 0:
            response.headers["Retry-After"] = str(reset_int)
    if emit_legacy:
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        config_manager: ConfigManager,
        metrics_collector: MetricsCollector,
    ) -> None:
        super().__init__(app)
        self.config = config_manager
        self.metrics = metrics_collector
        # Choose backend based on config
        backend_choice = getattr(
            getattr(self.config, "rate_limit", object()), "backend", "memory"
        )
        # Explicit backend type to satisfy type-checker when switching implementations
        self.backend: RateLimiterBackend
        if backend_choice == "redis":
            try:
                from .rate_limit_redis import RedisRateLimiterBackend

                # Pull optional URL from environment/config if present
                # Reuse mapper auth redis URL if shared, else default localhost
                redis_url = (
                    getattr(
                        getattr(self.config, "auth", object()),
                        "idempotency_redis_url",
                        None,
                    )
                    or "redis://localhost:6379/0"
                )
                rb = RedisRateLimiterBackend(redis_url=redis_url)
                # Health-check; fallback if down
                if rb.is_healthy():
                    self.backend = rb
                    try:
                        self.metrics.redis_backend_up.labels(component="rate_limit").set(1)  # type: ignore[attr-defined]
                    except Exception:
                        self.metrics.set_gauge("redis_rate_limit_up", 1)
                else:
                    self.backend = MemoryRateLimiterBackend()
                    try:
                        self.metrics.redis_backend_up.labels(component="rate_limit").set(0)  # type: ignore[attr-defined]
                        self.metrics.redis_backend_fallback_total.labels(component="rate_limit").inc()  # type: ignore[attr-defined]
                    except Exception:
                        self.metrics.set_gauge("redis_rate_limit_up", 0)
                        self.metrics.increment_counter(
                            "redis_rate_limit_fallback_total"
                        )
            except Exception:
                self.backend = MemoryRateLimiterBackend()
                try:
                    self.metrics.redis_backend_up.labels(component="rate_limit").set(0)  # type: ignore[attr-defined]
                    self.metrics.redis_backend_fallback_total.labels(component="rate_limit").inc()  # type: ignore[attr-defined]
                except Exception:
                    self.metrics.set_gauge("redis_rate_limit_up", 0)
                    self.metrics.increment_counter("redis_rate_limit_fallback_total")
        else:
            self.backend = MemoryRateLimiterBackend()
        self._skip_paths = {"/health", "/metrics", "/openapi.json", "/openapi.yaml"}
        self.logger = logging.getLogger(__name__)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rl_cfg = getattr(self.config, "rate_limit", None)
        if not rl_cfg or not rl_cfg.enabled:
            return await call_next(request)

        path = request.url.path
        if path in self._skip_paths or request.method == "OPTIONS":
            return await call_next(request)

        ep = _endpoint_key(path)
        if ep is None:
            return await call_next(request)

        identity_kind, identity = _extract_identity(request, self.config)
        limit, window = _limits_for(ep, identity_kind, self.config)

        try:
            result = await self.backend.allow(ep, identity, limit, window, cost=1)
        except Exception as ex:
            # Backend error: fail closed (block) and count error
            try:
                self.metrics.rate_limit_backend_errors_total.inc()  # type: ignore[attr-defined]
            except Exception:
                # Fallback legacy counter
                self.metrics.increment_counter("rate_limit_backend_errors_total")
            reset_seconds = float(window)
            request_id = getattr(
                getattr(request, "state", object()), "request_id", None
            )
            self.logger.error(
                "Rate limit backend error",
                extra={
                    "endpoint": ep,
                    "identity_kind": identity_kind,
                    "reset_seconds": reset_seconds,
                    "request_id": request_id,
                    "error": str(ex),
                },
            )
            body = {
                "detail": "Rate limit backend error",
                "limit": limit,
                "remaining": 0,
                "reset_seconds": reset_seconds,
                "identity_kind": identity_kind,
                "endpoint": ep,
            }
            err_response = JSONResponse(status_code=429, content=body)
            _emit_headers(
                err_response,
                limit,
                0,
                reset_seconds,
                window,
                rl_cfg.headers.emit_standard,
                rl_cfg.headers.emit_legacy,
            )
            return err_response

        # Align identity_kind on result for upstream metrics usage
        result.identity_kind = identity_kind  # type: ignore[attr-defined]

        if result.allowed:
            # Proceed
            response = await call_next(request)
            try:
                # Prometheus counter
                self.metrics.rate_limit_requests_total.labels(endpoint=ep, identity_kind=identity_kind, action="allow").inc()  # type: ignore[attr-defined]
            except Exception:
                # Legacy counter
                self.metrics.increment_counter(
                    "rate_limit_allowed_total",
                    {"endpoint": ep, "identity": identity_kind},
                )
            self.logger.debug(
                "Rate limit allow",
                extra={
                    "endpoint": ep,
                    "identity_kind": identity_kind,
                    "remaining": result.remaining,
                    "limit": result.limit,
                    "request_id": getattr(
                        getattr(request, "state", object()), "request_id", None
                    ),
                },
            )
            _emit_headers(
                response,
                result.limit,
                result.remaining,
                result.reset_seconds,
                window,
                rl_cfg.headers.emit_standard,
                rl_cfg.headers.emit_legacy,
            )
            return response
        else:
            # Block with 429
            body = {
                "detail": "Rate limit exceeded",
                "limit": result.limit,
                "remaining": 0,
                "reset_seconds": result.reset_seconds,
                "identity_kind": identity_kind,
                "endpoint": ep,
            }
            block_response = JSONResponse(status_code=429, content=body)
            try:
                self.metrics.rate_limit_requests_total.labels(endpoint=ep, identity_kind=identity_kind, action="block").inc()  # type: ignore[attr-defined]
                self.metrics.rate_limit_reset_seconds.labels(endpoint=ep, identity_kind=identity_kind).observe(result.reset_seconds)  # type: ignore[attr-defined]
            except Exception:
                self.metrics.increment_counter(
                    "rate_limit_blocked_total",
                    {"endpoint": ep, "identity": identity_kind},
                )
            self.logger.warning(
                "Rate limit block",
                extra={
                    "endpoint": ep,
                    "identity_kind": identity_kind,
                    "remaining": 0,
                    "limit": result.limit,
                    "reset_seconds": result.reset_seconds,
                    "request_id": getattr(
                        getattr(request, "state", object()), "request_id", None
                    ),
                },
            )
            _emit_headers(
                block_response,
                result.limit,
                0,
                result.reset_seconds,
                window,
                rl_cfg.headers.emit_standard,
                rl_cfg.headers.emit_legacy,
            )
            return block_response
