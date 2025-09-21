"""
Redis-backed idempotency cache for MappingResponse and BatchMappingResponse.

Falls back to a no-op (in-memory) approach if redis-py is not available or
configuration is incomplete. This cache stores serialized Pydantic payloads
under a configured key prefix with TTL semantics.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Union

from pydantic import BaseModel

from ..api.models import BatchMappingResponse, MappingResponse

logger = logging.getLogger(__name__)


class RedisIdempotencyCache:
    """Redis-backed idempotency cache with fallback to in-memory storage.

    This class provides idempotency for mapping operations by caching responses
    in Redis with TTL support. If Redis is unavailable, it gracefully falls back
    to a no-op implementation that doesn't persist data.

    Uses lazy initialization to avoid Redis import unless needed.

    Args:
        redis_url: Redis server URL for connection
        key_prefix: Namespace prefix for cache keys
        ttl_seconds: Time-to-live for cached entries in seconds
    """

    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        ttl_seconds: int,
    ) -> None:
        """Initialize the Redis idempotency cache.

        Args:
            redis_url: Redis server URL for connection
            key_prefix: Namespace prefix for cache keys
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

        self._redis: Any | None = None  # Lazy init to avoid import unless needed
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis  # type: ignore

            # Use blocking client with short timeouts; optional switch to asyncio later
            self._redis = redis.Redis.from_url(  # type: ignore[arg-type]
                self.redis_url, socket_timeout=0.2
            )
            # Basic ping check
            try:
                client = self._redis
                if client is not None:
                    client.ping()
            except (ConnectionError, TimeoutError, OSError):
                pass
            return True
        except (ImportError, ConnectionError, TimeoutError, OSError) as e:  # noqa: BLE001
            self._available = False
            logger.warning(
                "RedisIdempotencyCache unavailable, falling back to in-memory: %s", e
            )
            return False

    def _serialize(self, value: Union[MappingResponse, BatchMappingResponse]) -> str:
        """Serialize a mapping response to JSON string.

        Args:
            value: The mapping response to serialize

        Returns:
            JSON string representation of the value
        """
        if isinstance(value, MappingResponse):
            payload = {"kind": "single", "payload": value.model_dump()}
        elif isinstance(value, BatchMappingResponse):
            payload = {"kind": "batch", "payload": value.model_dump()}
        else:
            # Generic pydantic model or dict
            if isinstance(value, BaseModel):
                payload = {"kind": "unknown", "payload": value.model_dump()}
            else:
                payload = {"kind": "raw", "payload": value}
        return json.dumps(payload, separators=(",", ":"))

    def _deserialize(
        self, raw: str
    ) -> Optional[Union[MappingResponse, BatchMappingResponse]]:
        """Deserialize a JSON string back to a mapping response.

        Args:
            raw: JSON string to deserialize

        Returns:
            The deserialized mapping response or None if deserialization fails
        """
        try:
            data = json.loads(raw)
            kind = data.get("kind")
            payload = data.get("payload")
            if kind == "single" and isinstance(payload, dict):
                return MappingResponse(**payload)
            if kind == "batch" and isinstance(payload, dict):
                return BatchMappingResponse(**payload)
            return None
        except (ValueError, TypeError):
            return None

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a cached mapping response by key.

        Args:
            key: Cache key to retrieve

        Returns:
            The cached mapping response or None if not found or expired
        """
        if self.ttl_seconds <= 0:
            return None
        if not self._ensure_client():
            return None
        if self._redis is None:
            return None
        try:
            namespaced = f"{self.key_prefix}{key}"
            raw = self._redis.get(namespaced)
            if raw is None:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return self._deserialize(raw)
        except (ConnectionError, TimeoutError, OSError) as e:  # noqa: BLE001
            logger.debug("Redis get error (idempotency): %s", e)
            return None

    def set(
        self, key: str, value: Union[MappingResponse, BatchMappingResponse]
    ) -> None:
        """Store a mapping response in the cache.

        Args:
            key: Cache key to store under
            value: Mapping response to cache
        """
        if self.ttl_seconds <= 0:
            return
        if not self._ensure_client():
            return
        if self._redis is None:
            return
        try:
            namespaced = f"{self.key_prefix}{key}"
            payload = self._serialize(value)
            self._redis.set(namespaced, payload, ex=self.ttl_seconds)
        except (ConnectionError, TimeoutError, OSError) as e:  # noqa: BLE001
            logger.debug("Redis set error (idempotency): %s", e)

    def is_healthy(self) -> bool:
        """Attempt a lightweight health check on the Redis client."""
        if not self._ensure_client():
            return False
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except (ConnectionError, TimeoutError, OSError):
            return False
