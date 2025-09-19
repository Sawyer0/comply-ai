"""
Redis backend for rate limiting (optional, enabled via config).

Implements a simple fixed-window counter using INCR with key expiry, which is
good enough for RATE_LIMITED support and cross-instance consistency. For more
granular smoothing, migrate to a token bucket with Lua/EVAL in a later change.
"""

from __future__ import annotations

import time

from .rate_limit import AllowResult, RateLimiterBackend


class RedisRateLimiterBackend(RateLimiterBackend):
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "rl:mapper:",
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        from typing import Any as _Any
        self._redis: _Any | None = None
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis  # type: ignore

            self._redis = redis.Redis.from_url(self.redis_url, socket_timeout=0.2)  # type: ignore[arg-type]
            try:
                client = self._redis
                if client is not None:
                    client.ping()
            except Exception:
                pass
            return True
        except Exception:
            self._available = False
            return False

    async def allow(
        self,
        endpoint: str,
        identity: str,
        limit: int,
        window: int,
        cost: int = 1,
    ) -> AllowResult:
        # If Redis unavailable, signal backend error by returning a blocked result
        if not self._ensure_client():
            return AllowResult(False, 0, limit, float(window), "api_key")
        if self._redis is None:
            return AllowResult(False, 0, limit, float(window), "api_key")

        now = time.time()
        # Fixed window key: floor(now/window) bucket
        bucket = int(now // window)
        key = f"{self.key_prefix}{endpoint}:{identity}:{bucket}"
        try:
            # Atomic increment with expiry set on first write
            count = self._redis.incrby(key, cost)
            if count == cost:
                self._redis.expire(key, window)
            remaining = max(0, limit - int(count))
            if count <= limit:
                return AllowResult(True, remaining, limit, 0.0, "api_key")
            # Compute seconds until next window
            reset_seconds = float(window - int(now % window))
            return AllowResult(False, 0, limit, reset_seconds, "api_key")
        except Exception:
            # Return blocked; middleware will record backend error path
            return AllowResult(False, 0, limit, float(window), "api_key")

    def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
