"""
Redis backend for rate limiting (optional, enabled via config).

Implements a simple fixed-window counter using INCR with key expiry, which is
good enough for RATE_LIMITED support and cross-instance consistency. For more
granular smoothing, migrate to a token bucket with Lua/EVAL in a later change.
"""

from __future__ import annotations

import time
from typing import Any

from .rate_limit_base import AllowResult, RateLimitRequest, RateLimiterBackend

try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore[assignment]


class RedisRateLimiterBackend(RateLimiterBackend):
    """Redis-based rate limiter backend.

    Implements rate limiting using Redis for cross-instance consistency.
    Uses a fixed-window counter with INCR operations and key expiry.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "rl:mapper:",
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis: Any | None = None
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        if redis is None:
            self._available = False
            return False

        try:
            self._redis = redis.Redis.from_url(
                self.redis_url, socket_timeout=0.2
            )  # type: ignore[arg-type]
            try:
                client = self._redis
                if client is not None:
                    client.ping()
            except (ConnectionError, TimeoutError):
                pass
            return True
        except (ConnectionError, TimeoutError, ValueError):
            self._available = False
            return False

    async def allow(self, request: RateLimitRequest) -> AllowResult:
        # If Redis unavailable, signal backend error by returning a blocked result
        if not self._ensure_client():
            return AllowResult(False, 0, request.limit, float(request.window), "api_key")
        if self._redis is None:
            return AllowResult(False, 0, request.limit, float(request.window), "api_key")

        now = time.time()
        # Fixed window key: floor(now/window) bucket
        bucket = int(now // request.window)
        key = f"{self.key_prefix}{request.endpoint}:{request.identity}:{bucket}"
        try:
            # Atomic increment with expiry set on first write
            count = self._redis.incrby(key, request.cost)
            if count == request.cost:
                self._redis.expire(key, request.window)
            remaining = max(0, request.limit - int(count))
            if count <= request.limit:
                return AllowResult(True, remaining, request.limit, 0.0, "api_key")
            # Compute seconds until next window
            reset_seconds = float(request.window - int(now % request.window))
            return AllowResult(False, 0, request.limit, reset_seconds, "api_key")
        except (ConnectionError, TimeoutError, RuntimeError):
            # Return blocked; middleware will record backend error path
            return AllowResult(False, 0, request.limit, float(request.window), "api_key")

    async def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except (ConnectionError, TimeoutError, RuntimeError):
            return False
