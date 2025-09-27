"""Caching utilities for idempotency and response caching.

This module provides caching functionality that can be shared across services
following the Single Responsibility Principle.
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union
import logging

import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    fingerprint: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: int = 3600


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    def __init__(self, ttl_seconds: int = 3600, namespace: str = "shared"):
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, fingerprint: Optional[str] = None) -> None:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass

    def _build_key(self, key: str) -> str:
        """Build cache key with namespace."""
        return f"{self.namespace}:{key}"


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 3600,
        namespace: str = "shared",
        **kwargs
    ):
        super().__init__(ttl_seconds, namespace)
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None

    async def initialize(self):
        """Initialize Redis connection."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            if not self.redis_client:
                await self.initialize()

            cache_key = self._build_key(key)
            data = await self.redis_client.get(cache_key)

            if data:
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("Failed to decode cached data for key: %s", key)
                    return None
            return None
        except Exception as e:
            logger.error("Redis cache get failed: %s", str(e))
            return None

    async def set(self, key: str, value: Any, fingerprint: Optional[str] = None) -> None:
        """Set value in Redis cache."""
        try:
            if not self.redis_client:
                await self.initialize()

            cache_key = self._build_key(key)
            entry = CacheEntry(data=value, fingerprint=fingerprint, ttl_seconds=self.ttl_seconds)

            await self.redis_client.setex(
                cache_key,
                self.ttl_seconds,
                json.dumps(entry.__dict__, default=str)
            )
        except Exception as e:
            logger.error("Redis cache set failed: %s", str(e))

    async def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            if not self.redis_client:
                await self.initialize()

            cache_key = self._build_key(key)
            result = await self.redis_client.delete(cache_key)
            return result > 0
        except Exception as e:
            logger.error("Redis cache delete failed: %s", str(e))
            return False


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000, namespace: str = "shared"):
        super().__init__(ttl_seconds, namespace)
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list = []  # For LRU eviction

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        entry = self._cache.get(key)
        if entry:
            # Check if expired
            if time.time() - entry.created_at.timestamp() > entry.ttl_seconds:
                await self.delete(key)
                return None

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return entry.data
        return None

    async def set(self, key: str, value: Any, fingerprint: Optional[str] = None) -> None:
        """Set value in memory cache."""
        # Evict if at max size
        if len(self._cache) >= self.max_size and key not in self._cache:
            await self._evict_lru()

        self._cache[key] = CacheEntry(
            data=value,
            fingerprint=fingerprint,
            ttl_seconds=self.ttl_seconds
        )

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    async def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            return True
        return False

    async def _evict_lru(self):
        """Evict least recently used item."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            await self.delete(lru_key)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self._cache)
        expired_entries = sum(
            1 for entry in self._cache.values()
            if time.time() - entry.created_at.timestamp() > entry.ttl_seconds
        )
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "max_size": self.max_size,
            "utilization": total_entries / self.max_size if self.max_size > 0 else 0,
        }


class IdempotencyCache:
    """Idempotency cache implementation for safe request deduplication."""

    def __init__(self, cache_backend: Optional[CacheBackend] = None, ttl_seconds: int = 3600):
        self.cache = cache_backend or MemoryCache(ttl_seconds=ttl_seconds, namespace="idempotency")

    async def get(self, idempotency_key: str) -> Optional[Any]:
        """Get cached response for idempotency key."""
        if not idempotency_key:
            return None

        return await self.cache.get(idempotency_key)

    async def set(self, idempotency_key: str, response: Any, fingerprint: Optional[str] = None) -> None:
        """Cache response for idempotency key."""
        if not idempotency_key:
            return

        await self.cache.set(idempotency_key, response, fingerprint)

    async def generate_fingerprint(self, request_data: Dict[str, Any]) -> str:
        """Generate fingerprint for request to detect duplicates."""
        # Create a hash of the request content, detectors, and parameters
        fingerprint_data = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]

    async def is_healthy(self) -> bool:
        """Check if cache is healthy."""
        try:
            test_key = f"health_check_{time.time()}"
            await self.cache.set(test_key, "healthy")
            result = await self.cache.get(test_key)
            await self.cache.delete(test_key)
            return result == "healthy"
        except Exception as e:
            logger.error("Idempotency cache health check failed: %s", str(e))
            return False


class ResponseCache:
    """Response cache for caching expensive operation results."""

    def __init__(self, cache_backend: Optional[CacheBackend] = None, ttl_seconds: int = 1800):
        self.cache = cache_backend or MemoryCache(ttl_seconds=ttl_seconds, namespace="response")

    @staticmethod
    def build_key(*args, **kwargs) -> str:
        """Build deterministic cache key from request parameters."""
        # Create a string from all arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, (list, tuple)):
                key_parts.append(",".join(str(x) for x in arg))
            else:
                key_parts.append(str(arg))

        for k, v in sorted(kwargs.items()):
            if isinstance(v, (list, tuple)):
                key_parts.append(f"{k}={','.join(str(x) for x in v)}")
            else:
                key_parts.append(f"{k}={v}")

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Optional[Any]:
        """Get cached response."""
        return await self.cache.get(key)

    async def set(self, key: str, response: Any) -> None:
        """Cache response."""
        await self.cache.set(key, response)

    async def is_healthy(self) -> bool:
        """Check if cache is healthy."""
        return await self.cache.is_healthy()


def create_cache_backend(
    backend_type: str = "memory",
    ttl_seconds: int = 3600,
    namespace: str = "shared",
    **kwargs
) -> CacheBackend:
    """Factory function to create cache backends."""
    if backend_type.lower() == "redis":
        return RedisCache(ttl_seconds=ttl_seconds, namespace=namespace, **kwargs)
    elif backend_type.lower() == "memory":
        return MemoryCache(ttl_seconds=ttl_seconds, namespace=namespace, **kwargs)
    else:
        raise ValueError(f"Unknown cache backend type: {backend_type}")


def create_idempotency_cache(
    backend_type: str = "memory",
    ttl_seconds: int = 3600,
    **kwargs
) -> IdempotencyCache:
    """Factory function to create idempotency caches."""
    backend = create_cache_backend(backend_type, ttl_seconds, "idempotency", **kwargs)
    return IdempotencyCache(backend, ttl_seconds)


def create_response_cache(
    backend_type: str = "memory",
    ttl_seconds: int = 1800,
    **kwargs
) -> ResponseCache:
    """Factory function to create response caches."""
    backend = create_cache_backend(backend_type, ttl_seconds, "response", **kwargs)
    return ResponseCache(backend, ttl_seconds)
