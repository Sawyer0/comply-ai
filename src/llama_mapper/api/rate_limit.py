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
from typing import Awaitable, Callable, Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector
from .rate_limit_base import (
    AllowResult,
    IdentityKind,
    RateLimiterBackend,
    RateLimitRequest,
)

# Optional Redis import
try:
    from .rate_limit_redis import RedisRateLimiterBackend
except ImportError:
    RedisRateLimiterBackend = None  # type: ignore[assignment]


@dataclass
class _BucketState:
    tokens: float
    last_refill: float
    limit: int
    window: int


@dataclass
class _RateLimitHeadersConfig:
    emit_standard: bool
    emit_legacy: bool


@dataclass
class _RateLimitResponseData:
    """Data object for rate limit response information."""

    limit: int
    remaining: int
    reset_seconds: float
    window: int


class MemoryRateLimiterBackend(RateLimiterBackend):
    """In-memory token bucket rate limiter backend.

    This implementation uses concurrent data structures to provide
    thread-safe rate limiting without external dependencies.

    Note: Only has one public method as it's an interface implementation.
    """

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

    async def allow(self, request: RateLimitRequest) -> AllowResult:
        now = time.monotonic()
        key = (request.endpoint, request.identity)
        lock = self._get_lock(key)
        async with lock:
            rate_per_sec = request.limit / float(request.window)
            state = self._buckets.get(key)
            if state is None:
                state = _BucketState(
                    tokens=float(request.limit),
                    last_refill=now,
                    limit=request.limit,
                    window=request.window,
                )
                self._buckets[key] = state

            # Refill tokens
            elapsed = max(0.0, now - state.last_refill)
            refill = elapsed * rate_per_sec
            state.tokens = min(state.limit, state.tokens + refill)
            state.last_refill = now

            if state.tokens >= request.cost:
                state.tokens -= request.cost
                remaining = int(math.floor(state.tokens))
                return AllowResult(
                    True, remaining, state.limit, 0.0, "api_key"
                )  # identity_kind set by caller

            # Calculate reset time for insufficient tokens
            deficit = max(0.0, request.cost - state.tokens)
            reset_seconds = deficit / rate_per_sec if rate_per_sec > 0 else float("inf")
            return AllowResult(False, 0, state.limit, reset_seconds, "api_key")

    async def is_healthy(self) -> bool:
        """Check if the memory rate limiter backend is healthy.

        Returns:
            True as the in-memory implementation is always available.
        """
        return True


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
    response_data: _RateLimitResponseData,
    headers_config: _RateLimitHeadersConfig,
) -> None:
    reset_int = int(math.ceil(max(0.0, response_data.reset_seconds)))
    if headers_config.emit_standard:
        response.headers["RateLimit-Limit"] = (
            f"{response_data.limit};w={response_data.window}"
        )
        response.headers["RateLimit-Remaining"] = str(max(0, response_data.remaining))
        response.headers["RateLimit-Reset"] = str(reset_int)
        if response_data.remaining <= 0:
            response.headers["Retry-After"] = str(reset_int)
    if headers_config.emit_legacy:
        response.headers["X-RateLimit-Limit"] = str(response_data.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, response_data.remaining))


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for API endpoints.

    Implements token bucket rate limiting with support for multiple
    identity types (API key, tenant, IP) and configurable backends.
    Supports both Redis and in-memory backends with automatic fallback.

    Note: Only has one public method as it's a middleware implementation.
    """

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
            if RedisRateLimiterBackend is None:
                # Redis backend not available, fallback to memory
                self.backend = MemoryRateLimiterBackend()
            else:
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
                        self.metrics.redis_backend_up.labels(  # type: ignore[attr-defined]
                            component="rate_limit"
                        ).set(
                            1
                        )
                    except AttributeError:
                        self.metrics.set_gauge("redis_rate_limit_up", 1)
                else:
                    self.backend = MemoryRateLimiterBackend()
                    try:
                        self.metrics.redis_backend_up.labels(  # type: ignore[attr-defined]
                            component="rate_limit"
                        ).set(
                            0
                        )
                        self.metrics.redis_backend_fallback_total.labels(  # type: ignore[attr-defined]
                            component="rate_limit"
                        ).inc()
                    except AttributeError:
                        self.metrics.set_gauge("redis_rate_limit_up", 0)
                        self.metrics.increment_counter(
                            "redis_rate_limit_fallback_total"
                        )
        else:
            self.backend = MemoryRateLimiterBackend()
        self._skip_paths = {"/health", "/metrics", "/openapi.json", "/openapi.yaml"}
        self.logger = logging.getLogger(__name__)

    def _create_error_response(
        self,
        ep: str,
        identity_kind: str,
        limit: int,
        window: int,
        headers_config: _RateLimitHeadersConfig,
    ) -> JSONResponse:
        """Create error response for backend failures."""
        reset_seconds = float(window)
        body = {
            "detail": "Rate limit backend error",
            "limit": limit,
            "remaining": 0,
            "reset_seconds": reset_seconds,
            "identity_kind": identity_kind,
            "endpoint": ep,
        }
        err_response = JSONResponse(status_code=429, content=body)
        response_data = _RateLimitResponseData(
            limit=limit, remaining=0, reset_seconds=reset_seconds, window=window
        )
        _emit_headers(err_response, response_data, headers_config)
        return err_response

    def _handle_rate_limit_allow(
        self,
        request: Request,
        response: Response,
        result: AllowResult,
        ep: str,
        identity_kind: str,
        window: int,
        headers_config: _RateLimitHeadersConfig,
    ) -> Response:
        """Handle successful rate limit allowance."""
        try:
            # Prometheus counter
            self.metrics.rate_limit_requests_total.labels(  # type: ignore[attr-defined]
                endpoint=ep, identity_kind=identity_kind, action="allow"
            ).inc()
        except AttributeError:
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
        response_data = _RateLimitResponseData(
            limit=result.limit,
            remaining=result.remaining,
            reset_seconds=result.reset_seconds,
            window=window,
        )
        _emit_headers(response, response_data, headers_config)
        return response

    def _handle_rate_limit_block(
        self,
        request: Request,
        result: AllowResult,
        ep: str,
        identity_kind: str,
        window: int,
        headers_config: _RateLimitHeadersConfig,
    ) -> JSONResponse:
        """Handle rate limit blocking."""
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
            self.metrics.rate_limit_requests_total.labels(  # type: ignore[attr-defined]
                endpoint=ep, identity_kind=identity_kind, action="block"
            ).inc()
            self.metrics.rate_limit_reset_seconds.labels(  # type: ignore[attr-defined]
                endpoint=ep, identity_kind=identity_kind
            ).observe(result.reset_seconds)
        except AttributeError:
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
        response_data = _RateLimitResponseData(
            limit=result.limit,
            remaining=0,
            reset_seconds=result.reset_seconds,
            window=window,
        )
        _emit_headers(block_response, response_data, headers_config)
        return block_response

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

        # Pre-calculate headers config to reduce local variables
        headers_config = _RateLimitHeadersConfig(
            emit_standard=rl_cfg.headers.emit_standard if rl_cfg else False,
            emit_legacy=rl_cfg.headers.emit_legacy if rl_cfg else False,
        )

        try:
            request_obj = RateLimitRequest(
                endpoint=ep, identity=identity, limit=limit, window=window, cost=1
            )
            result = await self.backend.allow(request_obj)
        except (ConnectionError, TimeoutError, RuntimeError) as ex:
            # Backend error: fail closed (block) and count error
            try:
                self.metrics.rate_limit_backend_errors_total.inc()  # type: ignore[attr-defined]
            except AttributeError:
                # Fallback legacy counter
                self.metrics.increment_counter("rate_limit_backend_errors_total")
            self.logger.error(
                "Rate limit backend error",
                extra={
                    "endpoint": ep,
                    "identity_kind": identity_kind,
                    "reset_seconds": float(window),
                    "request_id": getattr(
                        getattr(request, "state", object()), "request_id", None
                    ),
                    "error": str(ex),
                },
            )
            return self._create_error_response(
                ep, identity_kind, limit, int(window), headers_config
            )

        # Align identity_kind on result for upstream metrics usage
        result.identity_kind = identity_kind  # type: ignore[attr-defined]

        if result.allowed:
            # Proceed
            response = await call_next(request)
            return self._handle_rate_limit_allow(
                request, response, result, ep, identity_kind, window, headers_config
            )

        return self._handle_rate_limit_block(
            request, result, ep, identity_kind, window, headers_config
        )

    def is_healthy(self) -> bool:
        """Check if the rate limit middleware is healthy.

        Returns:
            True as the middleware is always available.
        """
        return True
