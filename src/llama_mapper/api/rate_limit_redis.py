"""
Redis backend stub for rate limiting (optional, not enabled by default).

This module defines a RedisRateLimiterBackend that follows the same interface
as MemoryRateLimiterBackend but is not implemented here. In production, use a
Lua script (EVALSHA) or Redis modules (e.g., RedisCell) for atomic token bucket
operations across instances.
"""
from __future__ import annotations


from .rate_limit import AllowResult, RateLimiterBackend


class RedisRateLimiterBackend(RateLimiterBackend):
    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url
        # In a real implementation, initialize an async Redis client here

    async def allow(
        self,
        endpoint: str,
        identity: str,
        limit: int,
        window: int,
        cost: int = 1,
    ) -> AllowResult:
        raise NotImplementedError(
            "RedisRateLimiterBackend is a stub. Implement atomic token bucket via Lua/EVALSHA."
        )
