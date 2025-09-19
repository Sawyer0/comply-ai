from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import logging

from .models import OrchestrationResponse

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class IdempotencyCache:
    def __init__(self, ttl_seconds: int = 60 * 60 * 24):
        self._store: Dict[str, _CacheEntry] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.value

    def set(self, key: str, value: OrchestrationResponse) -> None:
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self._ttl)


class RedisIdempotencyCache:
    """Redis-backed idempotency cache for orchestrator responses."""

    def __init__(self, redis_url: str, ttl_seconds: int = 60 * 60 * 24, key_prefix: str = "idem:orch:") -> None:
        self._url = redis_url
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        self._redis = None
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis  # type: ignore

            self._redis = redis.Redis.from_url(self._url, socket_timeout=0.2)  # type: ignore[arg-type]
            try:
                self._redis.ping()
            except Exception:
                pass
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RedisIdempotencyCache unavailable, falling back to memory: {e}")
            self._available = False
            return False

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        if self._ttl <= 0:
            return None
        if not self._ensure_client():
            return None
        try:
            raw = self._redis.get(f"{self._prefix}{key}")
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            return OrchestrationResponse(**data)
        except Exception:
            return None

    def set(self, key: str, value: OrchestrationResponse) -> None:
        if self._ttl <= 0:
            return
        if not self._ensure_client():
            return
        try:
            payload = value.model_dump_json()
            self._redis.set(f"{self._prefix}{key}", payload, ex=self._ttl)
        except Exception:
            return

    def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False


class ResponseCache:
    def __init__(self, ttl_seconds: int = 300):
        self._store: Dict[str, _CacheEntry] = {}
        self._ttl = ttl_seconds

    @staticmethod
    def build_key(content: str, detector_set: Tuple[str, ...], policy_bundle: str) -> str:
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        det = ",".join(sorted(detector_set))
        return hashlib.sha256(f"{h}|{det}|{policy_bundle}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.value

    def set(self, key: str, value: OrchestrationResponse) -> None:
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self._ttl)


class RedisResponseCache:
    """Redis-backed response cache for orchestrator responses."""

    def __init__(self, redis_url: str, ttl_seconds: int = 300, key_prefix: str = "resp:orch:") -> None:
        self._url = redis_url
        self._ttl = ttl_seconds
        self._prefix = key_prefix
        self._redis = None
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        try:
            import redis  # type: ignore

            self._redis = redis.Redis.from_url(self._url, socket_timeout=0.2)  # type: ignore[arg-type]
            try:
                self._redis.ping()
            except Exception:
                pass
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"RedisResponseCache unavailable, falling back to memory: {e}")
            self._available = False
            return False

    @staticmethod
    def build_key(content: str, detector_set: Tuple[str, ...], policy_bundle: str) -> str:
        # Keep deterministic key derivation same as in-memory
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        det = ",".join(sorted(detector_set))
        return hashlib.sha256(f"{h}|{det}|{policy_bundle}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        if self._ttl <= 0:
            return None
        if not self._ensure_client():
            return None
        try:
            raw = self._redis.get(f"{self._prefix}{key}")
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            return OrchestrationResponse(**data)
        except Exception:
            return None

    def set(self, key: str, value: OrchestrationResponse) -> None:
        if self._ttl <= 0:
            return
        if not self._ensure_client():
            return
        try:
            payload = value.model_dump_json()
            self._redis.set(f"{self._prefix}{key}", payload, ex=self._ttl)
        except Exception:
            return

    def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
