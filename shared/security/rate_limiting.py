from __future__ import annotations

"""Shared rate limiting primitives and service helpers."""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol

logger = logging.getLogger(__name__)


class SupportsPipeline(Protocol):  # pragma: no cover - runtime protocol
    """Subset of aioredis pipeline operations used by the Redis storage."""

    async def hset(self, key: str, mapping: Mapping[str, str]) -> Any: ...

    async def expire(self, key: str, ttl: int) -> Any: ...

    async def execute(self) -> Any: ...


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

    def __post_init__(self) -> None:
        if self.burst_size > self.requests_per_minute:
            self.burst_size = self.requests_per_minute


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert the result to HTTP-compatible headers."""

        headers = {
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_time.timestamp())),
        }

        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)

        return headers


class RateLimitStorage:
    """Abstract base class for rate limit storage backends."""

    async def get_bucket_state(self, key: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    async def update_bucket_state(self, key: str, state: Mapping[str, Any], ttl: int) -> None:
        raise NotImplementedError

    async def increment_counter(self, key: str, window_start: int, ttl: int) -> int:
        raise NotImplementedError

    async def get_window_count(self, key: str, window_start: int, window_size: int) -> int:
        raise NotImplementedError


class MemoryRateLimitStorage(RateLimitStorage):
    """In-memory rate limit storage (primarily for development/testing)."""

    def __init__(self) -> None:
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._counters: Dict[str, list[tuple[int, int]]] = {}
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._start_cleanup()

    def _start_cleanup(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._cleanup_task is None and loop.is_running():  # pragma: no branch
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)
                current_time = int(time.time())

                expired_buckets = [
                    key for key, state in self._buckets.items() if state.get("expires_at", 0) < current_time
                ]
                for key in expired_buckets:
                    self._buckets.pop(key, None)

                for key in list(self._counters.keys()):
                    self._counters[key] = [
                        (timestamp, count)
                        for timestamp, count in self._counters[key]
                        if timestamp > current_time - 3600
                    ]
                    if not self._counters[key]:
                        self._counters.pop(key, None)

            except asyncio.CancelledError:  # pragma: no cover - shutdown behaviour
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Rate limit cleanup error", extra={"error": str(exc)})

    async def get_bucket_state(self, key: str) -> Optional[Dict[str, Any]]:
        state = self._buckets.get(key)
        if state and state.get("expires_at", 0) > time.time():
            return state
        return None

    async def update_bucket_state(self, key: str, state: Mapping[str, Any], ttl: int) -> None:
        new_state = dict(state)
        new_state["expires_at"] = time.time() + ttl
        self._buckets[key] = new_state

    async def increment_counter(self, key: str, window_start: int, ttl: int) -> int:
        bucket = self._counters.setdefault(key, [])
        bucket.append((int(time.time()), 1))
        return sum(count for timestamp, count in bucket if timestamp >= window_start)

    async def get_window_count(self, key: str, window_start: int, window_size: int) -> int:
        bucket = self._counters.get(key)
        if not bucket:
            return 0
        window_end = window_start + window_size
        return sum(
            count
            for timestamp, count in bucket
            if window_start <= timestamp <= window_end
        )


class RedisRateLimitStorage(RateLimitStorage):
    """Redis-based rate limit storage."""

    def __init__(self, redis_client: Any) -> None:
        self.redis = redis_client
        self.logger = logger.getChild("redis_rate_limit_storage")

    async def get_bucket_state(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            data = await self.redis.hgetall(f"rate_limit:bucket:{key}")
            if data:
                return {
                    "tokens": float(data.get("tokens", 0)),
                    "last_refill": float(data.get("last_refill", 0)),
                }
            return None
        except Exception as exc:  # pragma: no cover - redis failures
            self.logger.error("Failed to get bucket state", extra={"error": str(exc), "key": key})
            return None

    async def update_bucket_state(self, key: str, state: Mapping[str, Any], ttl: int) -> None:
        try:
            pipeline = self.redis.pipeline()
            pipeline.hset(
                f"rate_limit:bucket:{key}",
                mapping={
                    "tokens": str(state["tokens"]),
                    "last_refill": str(state["last_refill"]),
                },
            )
            pipeline.expire(f"rate_limit:bucket:{key}", ttl)
            await pipeline.execute()
        except Exception as exc:  # pragma: no cover - redis failures
            self.logger.error("Failed to update bucket state", extra={"error": str(exc), "key": key})

    async def increment_counter(self, key: str, window_start: int, ttl: int) -> int:
        try:
            counter_key = f"rate_limit:counter:{key}:{window_start}"
            new_value = await self.redis.incr(counter_key)
            await self.redis.expire(counter_key, ttl)
            return int(new_value)
        except Exception as exc:  # pragma: no cover - redis failures
            self.logger.error("Failed to increment counter", extra={"error": str(exc), "key": key})
            return 0

    async def get_window_count(self, key: str, window_start: int, window_size: int) -> int:
        try:
            pattern = f"rate_limit:counter:{key}:*"
            keys = await self.redis.keys(pattern)

            totals = []
            window_end = window_start + window_size
            for redis_key in keys:
                try:
                    key_str = redis_key.decode() if isinstance(redis_key, (bytes, bytearray)) else str(redis_key)
                    timestamp = int(key_str.rsplit(":", maxsplit=1)[-1])
                    if window_start <= timestamp <= window_end:
                        totals.append(redis_key)
                except (ValueError, TypeError):
                    continue

            if not totals:
                return 0

            values = await self.redis.mget(*totals)
            return sum(
                int(value.decode() if isinstance(value, (bytes, bytearray)) else value)
                for value in values
                if value is not None
            )
        except Exception as exc:  # pragma: no cover - redis failures
            self.logger.error("Failed to get window count", extra={"error": str(exc), "key": key})
            return 0


class RateLimiter:
    """Core rate limiter implementation."""

    def __init__(self, storage: RateLimitStorage) -> None:
        self.storage = storage
        self.logger = logger.getChild("rate_limiter")

    async def check_rate_limit(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        try:
            if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                return await self._check_token_bucket(key, config)
            if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                return await self._check_sliding_window(key, config)
            if config.strategy == RateLimitStrategy.FIXED_WINDOW:
                return await self._check_fixed_window(key, config)
            raise ValueError(f"Unsupported rate limit strategy: {config.strategy}")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Rate limit check failed", extra={"error": str(exc), "key": key})
            return RateLimitResult(
                allowed=True,
                remaining=config.requests_per_minute,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
            )

    async def _check_token_bucket(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        now = time.time()
        state = await self.storage.get_bucket_state(key)

        if state is None:
            state = {"tokens": float(config.burst_size), "last_refill": now}
        else:
            elapsed = now - state["last_refill"]
            refill = elapsed * (config.requests_per_minute / 60.0)
            state["tokens"] = min(config.burst_size, state["tokens"] + refill)
            state["last_refill"] = now

        if state["tokens"] >= 1:
            state["tokens"] -= 1
            allowed = True
            remaining = int(state["tokens"])
        else:
            allowed = False
            remaining = 0

        await self.storage.update_bucket_state(key, state, config.window_size_seconds * 2)

        if remaining == 0:
            seconds_to_refill = (1 - state["tokens"]) / (config.requests_per_minute / 60.0)
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

    async def _check_sliding_window(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        now = int(time.time())
        window_start = now - config.window_size_seconds
        current_count = await self.storage.get_window_count(key, window_start, config.window_size_seconds)

        if current_count < config.requests_per_minute:
            await self.storage.increment_counter(key, now, config.window_size_seconds * 2)
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

    async def _check_fixed_window(self, key: str, config: RateLimitConfig) -> RateLimitResult:
        now = int(time.time())
        window_start = (now // config.window_size_seconds) * config.window_size_seconds
        current_count = await self.storage.increment_counter(
            f"{key}:{window_start}", window_start, config.window_size_seconds
        )

        if current_count <= config.requests_per_minute:
            allowed = True
            remaining = config.requests_per_minute - current_count
        else:
            allowed = False
            remaining = 0

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
    """Coordinator that evaluates multiple rate limit dimensions."""

    def __init__(self, storage: RateLimitStorage) -> None:
        self.rate_limiter = RateLimiter(storage)
        self.logger = logger.getChild("rate_limit_manager")
        self.default_configs: Dict[str, RateLimitConfig] = {
            "api_key": RateLimitConfig(requests_per_minute=1000, burst_size=100),
            "ip": RateLimitConfig(requests_per_minute=100, burst_size=20),
            "tenant": RateLimitConfig(requests_per_minute=5000, burst_size=500),
            "endpoint": RateLimitConfig(requests_per_minute=10000, burst_size=1000),
        }

    async def check_multiple_limits(
        self,
        *,
        api_key_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        custom_configs: Optional[Dict[str, RateLimitConfig]] = None,
    ) -> RateLimitResult:
        configs = custom_configs or {}
        results: list[RateLimitResult] = []

        async def _maybe_check(key: str, identifier: Optional[str]) -> None:
            if not identifier:
                return
            config = configs.get(key, self.default_configs.get(key))
            if not config:
                return
            result = await self.rate_limiter.check_rate_limit(f"{key}:{identifier}", config)
            results.append(result)

        await asyncio.gather(
            _maybe_check("api_key", api_key_id),
            _maybe_check("tenant", tenant_id),
            _maybe_check("ip", ip_address),
            _maybe_check("endpoint", endpoint),
        )

        if not results:
            return RateLimitResult(
                allowed=True,
                remaining=1000,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
            )

        for result in results:
            if not result.allowed:
                self.logger.warning(
                    "Rate limit exceeded",
                    extra={
                        "remaining": result.remaining,
                        "reset_time": result.reset_time.isoformat(),
                    },
                )
                return result

        return min(results, key=lambda res: res.remaining)

    def update_default_config(self, key: str, config: RateLimitConfig) -> None:
        self.default_configs[key] = config
        self.logger.info("Updated rate limit defaults", extra={"key": key, "config": config})


class RateLimitingService:
    """High-level helper that extracts identifiers and applies rate limits."""

    def __init__(
        self,
        storage: Optional[RateLimitStorage] = None,
        *,
        redis_client: Optional[Any] = None,
        default_configs: Optional[Dict[str, RateLimitConfig]] = None,
    ) -> None:
        if storage is None:
            storage = RedisRateLimitStorage(redis_client) if redis_client else MemoryRateLimitStorage()
        self.manager = RateLimitManager(storage)
        if default_configs:
            for key, config in default_configs.items():
                self.manager.update_default_config(key, config)

    async def check_rate_limits(self, request: Any, subject: Optional[Any] = None) -> RateLimitResult:
        ip_address = self._extract_client_ip(request)
        endpoint = self._normalize_endpoint(getattr(request, "url", None))

        custom_configs: Dict[str, RateLimitConfig] = {}
        if subject is not None:
            custom_configs = self._subject_overrides(subject)

        return await self.manager.check_multiple_limits(
            api_key_id=getattr(subject, "key_id", None),
            tenant_id=getattr(subject, "tenant_id", None),
            ip_address=ip_address,
            endpoint=endpoint,
            custom_configs=custom_configs,
        )

    async def get_rate_limit_status(
        self,
        *,
        api_key_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return current counters for debugging or observability."""

        status: Dict[str, Any] = {}

        async def _check(label: str, identifier: Optional[str]) -> None:
            if not identifier:
                return
            config = self.manager.default_configs.get(label)
            if not config:
                return
            result = await self.manager.rate_limiter.check_rate_limit(f"{label}:{identifier}", config)
            status[label] = {
                "remaining": result.remaining,
                "reset_time": result.reset_time.isoformat(),
                "allowed": result.allowed,
            }

        await _check("api_key", api_key_id)
        await _check("tenant", tenant_id)
        await _check("ip", ip_address)
        await _check("endpoint", endpoint)
        return status

    def _subject_overrides(self, subject: Any) -> Dict[str, RateLimitConfig]:
        overrides: Dict[str, RateLimitConfig] = {}
        per_minute = getattr(subject, "rate_limit_per_minute", None)
        if isinstance(per_minute, int) and per_minute > 0:
            overrides["api_key"] = RateLimitConfig(
                requests_per_minute=per_minute,
                burst_size=min(max(1, per_minute // 10), per_minute),
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )

        permissions = {str(perm) for perm in getattr(subject, "permissions", [])}
        if "admin:system" in permissions:
            overrides["api_key"] = RateLimitConfig(
                requests_per_minute=10000,
                burst_size=1000,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )
        if "map:batch" in permissions:
            overrides["endpoint:/api/v1/map"] = RateLimitConfig(
                requests_per_minute=5000,
                burst_size=500,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )
            overrides["endpoint:/api/v1/map/batch"] = RateLimitConfig(
                requests_per_minute=500,
                burst_size=50,
                strategy=RateLimitStrategy.TOKEN_BUCKET,
            )

        return overrides

    @staticmethod
    def _extract_client_ip(request: Any) -> Optional[str]:
        headers = getattr(request, "headers", {}) or {}
        forwarded_for = headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        client = getattr(request, "client", None)
        if client and getattr(client, "host", None):
            return str(client.host)
        return None

    @staticmethod
    def _normalize_endpoint(url: Any) -> Optional[str]:
        if url is None:
            return None
        path = getattr(url, "path", str(url))
        if "?" in path:
            path = path.split("?")[0]
        parts = [part for part in path.split("/") if part]
        normalised: list[str] = []
        for part in parts:
            if (len(part) == 36 and part.count("-") == 4) or part.isdigit():
                normalised.append("{id}")
            else:
                normalised.append(part)
        return "/" + "/".join(normalised) if normalised else "/"
