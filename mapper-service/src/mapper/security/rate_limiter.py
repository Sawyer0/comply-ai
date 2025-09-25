"""
Rate Limiter for Mapper Service

Implements rate limiting functionality following Single Responsibility Principle.
Supports multiple rate limiting strategies and storage backends.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: int = 100
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    window_size_seconds: int = 60

    def __post_init__(self):
        if self.burst_size > self.requests_per_minute:
            self.burst_size = self.requests_per_minute


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.remaining + (0 if self.allowed else 1)),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp())),
        }

        if self.retry_after:
            headers["Retry-After"] = str(self.retry_after)

        return headers


class RateLimitStorage(ABC):
    """Abstract base class for rate limit storage backends."""

    @abstractmethod
    async def get_bucket_state(self, key: str) -> Optional[Dict[str, any]]:
        """Get current bucket state."""
        pass

    @abstractmethod
    async def update_bucket_state(
        self, key: str, state: Dict[str, any], ttl: int
    ) -> None:
        """Update bucket state with TTL."""
        pass

    @abstractmethod
    async def increment_counter(self, key: str, window_start: int, ttl: int) -> int:
        """Increment counter for sliding window."""
        pass

    @abstractmethod
    async def get_window_count(
        self, key: str, window_start: int, window_size: int
    ) -> int:
        """Get count for time window."""
        pass


class MemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage (for development/testing)."""

    def __init__(self):
        self._buckets: Dict[str, Dict[str, any]] = {}
        self._counters: Dict[str, List[Tuple[int, int]]] = (
            {}
        )  # key -> [(timestamp, count)]
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup()

    def _start_cleanup(self):
        """Start background cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                current_time = int(time.time())

                # Clean up expired buckets
                expired_buckets = [
                    key
                    for key, state in self._buckets.items()
                    if state.get("expires_at", 0) < current_time
                ]
                for key in expired_buckets:
                    del self._buckets[key]

                # Clean up old counter entries
                for key in list(self._counters.keys()):
                    self._counters[key] = [
                        (ts, count)
                        for ts, count in self._counters[key]
                        if ts > current_time - 3600  # Keep last hour
                    ]
                    if not self._counters[key]:
                        del self._counters[key]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Rate limit cleanup error", error=str(e))

    async def get_bucket_state(self, key: str) -> Optional[Dict[str, any]]:
        """Get current bucket state."""
        state = self._buckets.get(key)
        if state and state.get("expires_at", 0) > time.time():
            return state
        return None

    async def update_bucket_state(
        self, key: str, state: Dict[str, any], ttl: int
    ) -> None:
        """Update bucket state with TTL."""
        state["expires_at"] = time.time() + ttl
        self._buckets[key] = state

    async def increment_counter(self, key: str, window_start: int, ttl: int) -> int:
        """Increment counter for sliding window."""
        if key not in self._counters:
            self._counters[key] = []

        # Add new entry
        self._counters[key].append((int(time.time()), 1))

        # Return total count in current window
        return sum(count for ts, count in self._counters[key] if ts >= window_start)

    async def get_window_count(
        self, key: str, window_start: int, window_size: int
    ) -> int:
        """Get count for time window."""
        if key not in self._counters:
            return 0

        return sum(
            count
            for ts, count in self._counters[key]
            if window_start <= ts < window_start + window_size
        )


class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based rate limit storage."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.logger = logger.bind(component="redis_rate_limit_storage")

    async def get_bucket_state(self, key: str) -> Optional[Dict[str, any]]:
        """Get current bucket state."""
        try:
            data = await self.redis.hgetall(f"rate_limit:bucket:{key}")
            if data:
                return {
                    "tokens": float(data.get("tokens", 0)),
                    "last_refill": float(data.get("last_refill", 0)),
                }
            return None
        except Exception as e:
            self.logger.error("Failed to get bucket state", error=str(e), key=key)
            return None

    async def update_bucket_state(
        self, key: str, state: Dict[str, any], ttl: int
    ) -> None:
        """Update bucket state with TTL."""
        try:
            pipe = self.redis.pipeline()
            pipe.hset(
                f"rate_limit:bucket:{key}",
                mapping={
                    "tokens": str(state["tokens"]),
                    "last_refill": str(state["last_refill"]),
                },
            )
            pipe.expire(f"rate_limit:bucket:{key}", ttl)
            await pipe.execute()
        except Exception as e:
            self.logger.error("Failed to update bucket state", error=str(e), key=key)

    async def increment_counter(self, key: str, window_start: int, ttl: int) -> int:
        """Increment counter for sliding window."""
        try:
            pipe = self.redis.pipeline()
            counter_key = f"rate_limit:counter:{key}:{window_start}"
            pipe.incr(counter_key)
            pipe.expire(counter_key, ttl)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            self.logger.error("Failed to increment counter", error=str(e), key=key)
            return 0

    async def get_window_count(
        self, key: str, window_start: int, window_size: int
    ) -> int:
        """Get count for time window."""
        try:
            # Get all counter keys in the window
            pattern = f"rate_limit:counter:{key}:*"
            keys = await self.redis.keys(pattern)

            total = 0
            for redis_key in keys:
                # Extract timestamp from key
                try:
                    timestamp = int(redis_key.split(":")[-1])
                    if window_start <= timestamp < window_start + window_size:
                        count = await self.redis.get(redis_key)
                        total += int(count) if count else 0
                except (ValueError, IndexError):
                    continue

            return total
        except Exception as e:
            self.logger.error("Failed to get window count", error=str(e), key=key)
            return 0


class RateLimiter:
    """
    Rate limiter implementation following SRP.

    Responsible for:
    - Checking rate limits using various strategies
    - Managing rate limit state
    - Providing rate limit information
    """

    def __init__(self, storage: RateLimitStorage):
        self.storage = storage
        self.logger = logger.bind(component="rate_limiter")

    async def check_rate_limit(
        self, key: str, config: RateLimitConfig
    ) -> RateLimitResult:
        """
        Check if request is within rate limits.

        Args:
            key: Unique identifier for the rate limit (e.g., API key, IP)
            config: Rate limit configuration

        Returns:
            RateLimitResult with decision and metadata
        """
        try:
            if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key, config)
            elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key, config)
            elif config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key, config)
            else:
                raise ValueError(f"Unsupported rate limit strategy: {config.strategy}")

        except Exception as e:
            self.logger.error("Rate limit check failed", error=str(e), key=key)
            # Fail open - allow request if rate limiting fails
            return RateLimitResult(
                allowed=True,
                remaining=config.requests_per_minute,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
            )

    async def _check_token_bucket(
        self, key: str, config: RateLimitConfig
    ) -> RateLimitResult:
        """Check rate limit using token bucket algorithm."""
        now = time.time()

        # Get current bucket state
        state = await self.storage.get_bucket_state(key)

        if state is None:
            # Initialize new bucket
            state = {"tokens": float(config.burst_size), "last_refill": now}
        else:
            # Refill tokens based on time elapsed
            time_elapsed = now - state["last_refill"]
            tokens_to_add = time_elapsed * (config.requests_per_minute / 60.0)
            state["tokens"] = min(config.burst_size, state["tokens"] + tokens_to_add)
            state["last_refill"] = now

        # Check if request can be processed
        if state["tokens"] >= 1:
            state["tokens"] -= 1
            allowed = True
            remaining = int(state["tokens"])
        else:
            allowed = False
            remaining = 0

        # Update state
        await self.storage.update_bucket_state(
            key, state, config.window_size_seconds * 2
        )

        # Calculate reset time
        if remaining == 0:
            seconds_to_refill = (1 - state["tokens"]) / (
                config.requests_per_minute / 60.0
            )
            reset_time = datetime.utcnow() + timedelta(seconds=seconds_to_refill)
            retry_after = int(seconds_to_refill) + 1
        else:
            reset_time = datetime.utcnow() + timedelta(minutes=1)
            retry_after = None

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )

    async def _check_sliding_window(
        self, key: str, config: RateLimitConfig
    ) -> RateLimitResult:
        """Check rate limit using sliding window algorithm."""
        now = int(time.time())
        window_start = now - config.window_size_seconds

        # Get current count in window
        current_count = await self.storage.get_window_count(
            key, window_start, config.window_size_seconds
        )

        if current_count < config.requests_per_minute:
            # Increment counter
            await self.storage.increment_counter(
                key, now, config.window_size_seconds * 2
            )
            allowed = True
            remaining = config.requests_per_minute - current_count - 1
        else:
            allowed = False
            remaining = 0

        reset_time = datetime.utcnow() + timedelta(seconds=config.window_size_seconds)
        retry_after = config.window_size_seconds if not allowed else None

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )

    async def _check_fixed_window(
        self, key: str, config: RateLimitConfig
    ) -> RateLimitResult:
        """Check rate limit using fixed window algorithm."""
        now = int(time.time())
        window_start = (now // config.window_size_seconds) * config.window_size_seconds

        # Get current count in window
        current_count = await self.storage.increment_counter(
            f"{key}:{window_start}", window_start, config.window_size_seconds
        )

        if current_count <= config.requests_per_minute:
            allowed = True
            remaining = config.requests_per_minute - current_count
        else:
            allowed = False
            remaining = 0

        # Calculate reset time (start of next window)
        next_window = window_start + config.window_size_seconds
        reset_time = datetime.fromtimestamp(next_window)
        retry_after = next_window - now if not allowed else None

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
        )


class RateLimitManager:
    """
    Rate limit manager that coordinates different rate limiters.

    Responsible for:
    - Managing multiple rate limit configurations
    - Coordinating different rate limit keys (IP, API key, tenant)
    - Providing unified rate limit checking
    """

    def __init__(self, storage: RateLimitStorage):
        self.rate_limiter = RateLimiter(storage)
        self.logger = logger.bind(component="rate_limit_manager")

        # Default configurations
        self.default_configs = {
            "api_key": RateLimitConfig(requests_per_minute=1000, burst_size=100),
            "ip": RateLimitConfig(requests_per_minute=100, burst_size=20),
            "tenant": RateLimitConfig(requests_per_minute=5000, burst_size=500),
            "endpoint": RateLimitConfig(requests_per_minute=10000, burst_size=1000),
        }

    async def check_multiple_limits(
        self,
        api_key_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        custom_configs: Optional[Dict[str, RateLimitConfig]] = None,
    ) -> RateLimitResult:
        """
        Check multiple rate limits and return the most restrictive result.

        Args:
            api_key_id: API key identifier
            tenant_id: Tenant identifier
            ip_address: Client IP address
            endpoint: API endpoint
            custom_configs: Custom rate limit configurations

        Returns:
            Most restrictive rate limit result
        """
        configs = custom_configs or {}
        results = []

        # Check API key rate limit
        if api_key_id:
            config = configs.get("api_key", self.default_configs["api_key"])
            result = await self.rate_limiter.check_rate_limit(
                f"api_key:{api_key_id}", config
            )
            results.append(("api_key", result))

        # Check tenant rate limit
        if tenant_id:
            config = configs.get("tenant", self.default_configs["tenant"])
            result = await self.rate_limiter.check_rate_limit(
                f"tenant:{tenant_id}", config
            )
            results.append(("tenant", result))

        # Check IP rate limit
        if ip_address:
            config = configs.get("ip", self.default_configs["ip"])
            result = await self.rate_limiter.check_rate_limit(
                f"ip:{ip_address}", config
            )
            results.append(("ip", result))

        # Check endpoint rate limit
        if endpoint:
            config = configs.get("endpoint", self.default_configs["endpoint"])
            result = await self.rate_limiter.check_rate_limit(
                f"endpoint:{endpoint}", config
            )
            results.append(("endpoint", result))

        if not results:
            # No rate limits to check
            return RateLimitResult(
                allowed=True,
                remaining=1000,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
            )

        # Find most restrictive result
        most_restrictive = None
        for limit_type, result in results:
            if not result.allowed:
                self.logger.warning(
                    "Rate limit exceeded",
                    limit_type=limit_type,
                    remaining=result.remaining,
                    reset_time=result.reset_time,
                )
                return result

            if (
                most_restrictive is None
                or result.remaining < most_restrictive.remaining
            ):
                most_restrictive = result

        return most_restrictive or results[0][1]

    def update_default_config(self, limit_type: str, config: RateLimitConfig) -> None:
        """Update default configuration for a limit type."""
        self.default_configs[limit_type] = config
        self.logger.info(
            "Updated default rate limit config", limit_type=limit_type, config=config
        )
