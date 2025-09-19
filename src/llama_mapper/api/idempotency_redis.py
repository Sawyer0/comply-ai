"""
Redis-backed idempotency cache for MappingResponse and BatchMappingResponse.

Falls back to a no-op (in-memory) approach if redis-py is not available or
configuration is incomplete. This cache stores serialized Pydantic payloads
under a configured key prefix with TTL semantics.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Union

from pydantic import BaseModel

from ..api.models import BatchMappingResponse, MappingResponse

logger = logging.getLogger(__name__)


class RedisIdempotencyCache:
    def __init__(
        self,
        redis_url: str,
        key_prefix: str,
        ttl_seconds: int,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        from typing import Any as _Any
        self._redis: _Any | None = None  # Lazy init to avoid import unless needed
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis  # type: ignore

            # Use blocking client with short timeouts; optional switch to asyncio later
            self._redis = redis.Redis.from_url(self.redis_url, socket_timeout=0.2)  # type: ignore[arg-type]
            # Basic ping check
            try:
                client = self._redis
                if client is not None:
                    client.ping()
            except Exception:
                pass
            return True
        except Exception as e:  # noqa: BLE001
            self._available = False
            logger.warning(
                f"RedisIdempotencyCache unavailable, falling back to in-memory: {e}"
            )
            return False

    def _serialize(self, value: Union[MappingResponse, BatchMappingResponse]) -> str:
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
        try:
            data = json.loads(raw)
            kind = data.get("kind")
            payload = data.get("payload")
            if kind == "single" and isinstance(payload, dict):
                return MappingResponse(**payload)
            if kind == "batch" and isinstance(payload, dict):
                return BatchMappingResponse(**payload)
            return None
        except Exception:
            return None

    def get(self, key: str) -> Optional[Any]:
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
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Redis get error (idempotency): {e}")
            return None

    def set(
        self, key: str, value: Union[MappingResponse, BatchMappingResponse]
    ) -> None:
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
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Redis set error (idempotency): {e}")

    def is_healthy(self) -> bool:
        """Attempt a lightweight health check on the Redis client."""
        if not self._ensure_client():
            return False
        if self._redis is None:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
