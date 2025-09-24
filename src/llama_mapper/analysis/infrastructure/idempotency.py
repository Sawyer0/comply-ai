"""
Infrastructure implementation of the idempotency manager for the Analysis Module.

This module contains the concrete implementation of the IIdempotencyManager interface
for managing idempotency and caching.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from ..domain.entities import BatchAnalysisResponse
from ..domain.interfaces import IIdempotencyManager

logger = logging.getLogger(__name__)


class IdempotencyManager(IIdempotencyManager):
    """
    Idempotency manager implementation.

    Provides concrete implementation of the IIdempotencyManager interface
    for managing idempotency and caching of analysis responses.
    """

    def __init__(self, ttl_hours: int = 24, max_items: int = 1000):
        """
        Initialize the idempotency manager.

        Args:
            ttl_hours: Time-to-live for cache entries in hours
            max_items: Maximum number of cache items
        """
        self.ttl_hours = ttl_hours
        self.max_items = max_items
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

        logger.info(
            f"Initialized Idempotency Manager with TTL: {ttl_hours}h, Max items: {max_items}"
        )

    async def get_cached_response(
        self, idempotency_key: str, request_data: Dict[str, Any]
    ) -> Optional[BatchAnalysisResponse]:
        """
        Get cached response for idempotency key.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data

        Returns:
            Cached response if found and not expired, None otherwise
        """
        async with self._lock:
            try:
                # Generate cache key from idempotency key and request data
                cache_key = self._generate_cache_key(idempotency_key, request_data)

                if cache_key not in self._cache:
                    return None

                cache_entry = self._cache[cache_key]

                # Check if entry has expired
                if self._is_expired(cache_entry):
                    del self._cache[cache_key]
                    logger.debug("Cache entry expired for key: %s", cache_key)
                    return None

                # Return cached response
                logger.debug("Cache hit for key: %s", cache_key)
                return BatchAnalysisResponse(**cache_entry["response"])

            except Exception as e:
                logger.error("Error retrieving cached response: %s", e)
                return None

    async def cache_response(
        self,
        idempotency_key: str,
        request_data: Dict[str, Any],
        response: BatchAnalysisResponse,
    ) -> None:
        """
        Cache response for idempotency key.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data
            response: Response to cache
        """
        async with self._lock:
            try:
                # Generate cache key
                cache_key = self._generate_cache_key(idempotency_key, request_data)

                # Check cache size limit
                if len(self._cache) >= self.max_items:
                    await self._evict_oldest_entries()

                # Store response in cache
                cache_entry = {
                    "response": response.dict(),
                    "created_at": datetime.now(timezone.utc),
                    "expires_at": datetime.now(timezone.utc)
                    + timedelta(hours=self.ttl_hours),
                    "idempotency_key": idempotency_key,
                }

                self._cache[cache_key] = cache_entry
                logger.debug("Cached response for key: %s", cache_key)

            except Exception as e:
                logger.error("Error caching response: %s", e)

    async def cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries.

        Returns:
            Number of entries cleaned up
        """
        async with self._lock:
            try:
                expired_keys = []

                for cache_key, cache_entry in self._cache.items():
                    if self._is_expired(cache_entry):
                        expired_keys.append(cache_key)

                # Remove expired entries
                for key in expired_keys:
                    del self._cache[key]

                if expired_keys:
                    logger.info(
                        "Cleaned up %s expired cache entries", len(expired_keys)
                    )

                return len(expired_keys)

            except Exception as e:
                logger.error("Error cleaning up expired entries: %s", e)
                return 0

    def _generate_cache_key(
        self, idempotency_key: str, request_data: Dict[str, Any]
    ) -> str:
        """
        Generate cache key from idempotency key and request data.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data

        Returns:
            Cache key
        """
        # Create a hash of the request data for consistency
        request_hash = hashlib.md5(
            json.dumps(request_data, sort_keys=True).encode()
        ).hexdigest()

        return f"{idempotency_key}:{request_hash}"

    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """
        Check if cache entry has expired.

        Args:
            cache_entry: Cache entry

        Returns:
            True if expired, False otherwise
        """
        expires_at = cache_entry.get("expires_at")
        if not expires_at:
            return True

        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))

        return datetime.now(timezone.utc) > expires_at

    async def _evict_oldest_entries(self) -> None:
        """Evict oldest cache entries to make room for new ones."""
        try:
            # Sort entries by creation time
            sorted_entries = sorted(
                self._cache.items(), key=lambda x: x[1].get("created_at", datetime.min)
            )

            # Remove oldest 10% of entries
            evict_count = max(1, len(sorted_entries) // 10)

            for i in range(evict_count):
                cache_key = sorted_entries[i][0]
                del self._cache[cache_key]

            logger.info("Evicted %s oldest cache entries", evict_count)

        except Exception as e:
            logger.error("Error evicting oldest entries: %s", e)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Cache statistics
        """
        try:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values() if self._is_expired(entry)
            )

            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "active_entries": total_entries - expired_entries,
                "max_items": self.max_items,
                "ttl_hours": self.ttl_hours,
                "cache_hit_rate": 0.0,  # This would be calculated over time
            }

        except Exception as e:
            logger.error("Error getting cache stats: %s", e)
            return {}


class MemoryIdempotencyManager(IdempotencyManager):
    """
    Memory-based idempotency manager.

    Simple in-memory implementation for development and testing.
    """

    def __init__(self, ttl_hours: int = 24, max_items: int = 1000):
        """
        Initialize the memory-based idempotency manager.

        Args:
            ttl_hours: Time-to-live for cache entries in hours
            max_items: Maximum number of cache items
        """
        super().__init__(ttl_hours, max_items)
        logger.info("Using memory-based idempotency manager")


class RedisIdempotencyManager(IdempotencyManager):
    """
    Redis-based idempotency manager.

    Redis implementation for production use with distributed caching.
    """

    def __init__(self, redis_client, ttl_hours: int = 24, max_items: int = 1000):
        """
        Initialize the Redis-based idempotency manager.

        Args:
            redis_client: Redis client instance
            ttl_hours: Time-to-live for cache entries in hours
            max_items: Maximum number of cache items
        """
        super().__init__(ttl_hours, max_items)
        self.redis_client = redis_client
        logger.info("Using Redis-based idempotency manager")

    async def get_cached_response(
        self, idempotency_key: str, request_data: Dict[str, Any]
    ) -> Optional[BatchAnalysisResponse]:
        """
        Get cached response from Redis.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data

        Returns:
            Cached response if found and not expired, None otherwise
        """
        try:
            cache_key = self._generate_cache_key(idempotency_key, request_data)
            redis_key = f"analysis:idempotency:{cache_key}"

            cached_data = await self.redis_client.get(redis_key)
            if not cached_data:
                return None

            response_data = json.loads(cached_data)
            return BatchAnalysisResponse(**response_data)

        except Exception as e:
            logger.error("Error retrieving cached response from Redis: %s", e)
            return None

    async def cache_response(
        self,
        idempotency_key: str,
        request_data: Dict[str, Any],
        response: BatchAnalysisResponse,
    ) -> None:
        """
        Cache response in Redis.

        Args:
            idempotency_key: Idempotency key
            request_data: Request data
            response: Response to cache
        """
        try:
            cache_key = self._generate_cache_key(idempotency_key, request_data)
            redis_key = f"analysis:idempotency:{cache_key}"

            # Set with TTL
            ttl_seconds = self.ttl_hours * 3600
            await self.redis_client.setex(
                redis_key, ttl_seconds, json.dumps(response.dict())
            )

            logger.debug("Cached response in Redis for key: %s", cache_key)

        except Exception as e:
            logger.error("Error caching response in Redis: %s", e)

    async def cleanup_expired_entries(self) -> int:
        """
        Clean up expired entries in Redis.

        Returns:
            Number of entries cleaned up
        """
        try:
            # Redis handles TTL automatically, so we just return 0
            # In a real implementation, you might want to scan for keys
            # and check their TTL
            return 0

        except Exception as e:
            logger.error("Error cleaning up expired entries in Redis: %s", e)
            return 0
