"""Caching module for orchestrator responses and intermediate results.

This module provides caching functionality for detector orchestration results,
including Redis-based caching for production and in-memory caching for testing.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Set
import json
import logging

from .models import OrchestrationResponse

try:  # pragma: no cover - optional dependency
    import redis  # type: ignore[import-not-found]
    from redis.exceptions import RedisError  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - redis is optional
    redis = None  # type: ignore[assignment]

    class RedisError(Exception):
        """Fallback Redis error used when the dependency is unavailable."""

# Redis client type alias (Optional[Any] keeps typing lenient without redis installed)
RedisClient = Any


logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry with expiration."""
    value: OrchestrationResponse
    expires_at: float


@dataclass
class _IdemEntry:
    """Internal idempotency cache entry with fingerprint."""
    value: OrchestrationResponse
    fingerprint: Optional[str]
    expires_at: float


class IdempotencyCache:
    """Cache for idempotent orchestrator responses."""
    def __init__(self, ttl_seconds: int = 60 * 60 * 24):
        self._store: Dict[str, _IdemEntry] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        """Retrieve the cached response for the idempotency key, if present."""

        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.value

    def get_entry(self, key: str) -> Optional[_IdemEntry]:
        """Return the full idempotency entry (value + fingerprint)."""

        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e

    def set(
        self, key: str, value: OrchestrationResponse, fingerprint: Optional[str] = None
    ) -> None:
        """Store an idempotent response, overwriting existing values."""

        self._store[key] = _IdemEntry(
            value=value, fingerprint=fingerprint, expires_at=time.time() + self._ttl
        )

    def is_healthy(self) -> bool:
        """In-memory cache is always healthy."""
        return True


class RedisIdempotencyCache(IdempotencyCache):
    """Redis-backed idempotency cache for orchestrator responses."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 60 * 60 * 24,
        key_prefix: str = "idem:orch:",
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._url = redis_url
        self._prefix = key_prefix
        self._redis: RedisClient | None = None
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        if redis is None:
            logger.warning(
                "Redis dependency not installed; idempotency cache fallback active"
            )
            self._available = False
            return False
        try:
            self._redis = redis.Redis.from_url(  # type: ignore[attr-defined]
                self._url,
                socket_timeout=0.2,
            )
            self._redis.ping()
            return True
        except RedisError as exc:
            logger.warning(
                "RedisIdempotencyCache unavailable, falling back to memory: %s", exc
            )
            self._available = False
            self._redis = None
            return False

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        entry = self.get_entry(key)
        return entry.value if entry else None

    def get_entry(self, key: str) -> Optional[_IdemEntry]:
        if self._ttl <= 0:
            return None
        if not self._ensure_client():
            return None
        try:
            client = self._redis
            if client is None:
                return None
            raw = client.get(f"{self._prefix}{key}")
            entry: Optional[_IdemEntry] = None
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
                expiry = time.time() + self._ttl
                if isinstance(data, dict) and "resp" in data:
                    resp = OrchestrationResponse(**data.get("resp", {}))
                    entry = _IdemEntry(
                        value=resp,
                        fingerprint=data.get("fp"),
                        expires_at=expiry,
                    )
                else:
                    resp = OrchestrationResponse(**data)
                    entry = _IdemEntry(value=resp, fingerprint=None, expires_at=expiry)
            return entry
        except (RedisError, json.JSONDecodeError, TypeError):
            return None

    def set(
        self, key: str, value: OrchestrationResponse, fingerprint: Optional[str] = None
    ) -> None:
        if self._ttl <= 0:
            return
        if not self._ensure_client():
            return
        try:
            payload_obj = {"resp": value.model_dump(), "fp": fingerprint}
            payload = json.dumps(payload_obj, separators=(",", ":"))
            client = self._redis
            if client is None:
                return
            client.set(f"{self._prefix}{key}", payload, ex=self._ttl)
        except RedisError as exc:
            logger.warning("Failed to persist Redis idempotency entry: %s", exc)

    def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        try:
            client = self._redis
            if client is None:
                return False
            client.ping()
            return True
        except RedisError as exc:
            logger.warning("Redis health check failed: %s", exc)
            return False


class ResponseCache:
    """Cache of orchestration responses keyed by request fingerprint."""

    def __init__(self, ttl_seconds: int = 300):
        self._store: Dict[str, _CacheEntry] = {}
        self._ttl = ttl_seconds
        # Indexes for invalidation
        self._by_policy: Dict[str, Set[str]] = {}
        self._by_detector: Dict[str, Set[str]] = {}

    @staticmethod
    def build_key(
        content: str, detector_set: Tuple[str, ...], policy_bundle: str
    ) -> str:
        """Build a deterministic cache key from content, detectors, and policy."""

        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        det = ",".join(sorted(detector_set))
        return hashlib.sha256(f"{h}|{det}|{policy_bundle}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        """Fetch a cached orchestration response if available and not expired."""

        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.value

    def _index_key(self, key: str, value: OrchestrationResponse) -> None:
        """Index cache keys by policy and detector for fast invalidation."""

        try:
            policy = value.routing_decision.policy_applied
            if policy:
                s = self._by_policy.setdefault(policy, set())
                s.add(key)
            for det in value.routing_decision.selected_detectors:
                ds = self._by_detector.setdefault(det, set())
                ds.add(key)
        except Exception:  # pylint: disable=broad-exception-caught
            # Be permissive; indexing is best-effort
            pass

    def set(self, key: str, value: OrchestrationResponse) -> None:
        """Insert or update a cached orchestration response."""

        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self._ttl)
        self._index_key(key, value)

    def delete_key(self, key: str) -> None:
        """Remove a specific cache entry without touching indexes."""

        self._store.pop(key, None)
        # Lazy cleanup of indexes: full cleanup occurs on invalidate

    def invalidate_for_policy(self, policy_bundle: str) -> int:
        """Invalidate every cached response associated with the policy bundle."""

        keys = list(self._by_policy.get(policy_bundle, set()))
        count = 0
        for k in keys:
            if k in self._store:
                self._store.pop(k, None)
                count += 1
        # Remove from index
        self._by_policy.pop(policy_bundle, None)
        # Also remove keys from detector indexes (best-effort)
        for det, det_keys in list(self._by_detector.items()):
            det_keys.difference_update(keys)
            if not det_keys:
                self._by_detector.pop(det, None)
        return count

    def invalidate_for_detector(self, detector: str) -> int:
        """Invalidate responses using a particular detector."""

        keys = list(self._by_detector.get(detector, set()))
        count = 0
        for k in keys:
            if k in self._store:
                self._store.pop(k, None)
                count += 1
        # Remove from index
        self._by_detector.pop(detector, None)
        # Also prune from policy indexes
        for pol, pol_keys in list(self._by_policy.items()):
            pol_keys.difference_update(keys)
            if not pol_keys:
                self._by_policy.pop(pol, None)
        return count

    def is_healthy(self) -> bool:
        """In-memory cache is always healthy."""
        return True


class RedisResponseCache(ResponseCache):
    """Redis-backed response cache for orchestrator responses."""

    def __init__(
        self, redis_url: str, ttl_seconds: int = 300, key_prefix: str = "resp:orch:"
    ) -> None:
        super().__init__(ttl_seconds=ttl_seconds)
        self._url = redis_url
        self._prefix = key_prefix
        self._redis: RedisClient | None = None
        self._available = True

    def _ensure_client(self) -> bool:
        if self._redis is not None:
            return True
        if redis is None:
            logger.warning(
                "Redis dependency not installed; response cache fallback active"
            )
            self._available = False
            return False
        try:
            self._redis = redis.Redis.from_url(  # type: ignore[attr-defined]
                self._url,
                socket_timeout=0.2,
            )
            self._redis.ping()
            return True
        except RedisError as exc:
            logger.warning(
                "RedisResponseCache unavailable, falling back to memory: %s", exc
            )
            self._available = False
            self._redis = None
            return False

    @staticmethod
    def build_key(
        content: str, detector_set: Tuple[str, ...], policy_bundle: str
    ) -> str:
        # Keep deterministic key derivation same as in-memory
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        det = ",".join(sorted(detector_set))
        return hashlib.sha256(f"{h}|{det}|{policy_bundle}".encode("utf-8")).hexdigest()

    def _index_key(self, key: str, value: OrchestrationResponse) -> None:
        if not self._ensure_client():
            return
        try:
            client = self._redis
            if client is None:
                return
            pipe = client.pipeline()
            policy = value.routing_decision.policy_applied
            if policy:
                pol_idx = f"{self._prefix}idx:policy:{policy}"
                pipe.sadd(pol_idx, key)
                pipe.expire(pol_idx, self._ttl)
            for det in value.routing_decision.selected_detectors:
                det_idx = f"{self._prefix}idx:detector:{det}"
                pipe.sadd(det_idx, key)
                pipe.expire(det_idx, self._ttl)
            pipe.execute()
        except RedisError as exc:
            logger.warning("Failed to index Redis cache entry: %s", exc)

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        if self._ttl <= 0:
            return None
        if not self._ensure_client():
            return None
        try:
            client = self._redis
            if client is None:
                return None
            raw = client.get(f"{self._prefix}{key}")
            if not raw:
                return None
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            return OrchestrationResponse(**data)
        except (RedisError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to read Redis cache entry: %s", exc)
            return None

    def set(self, key: str, value: OrchestrationResponse) -> None:
        if self._ttl <= 0:
            return
        if not self._ensure_client():
            return
        try:
            payload = value.model_dump_json()
            client = self._redis
            if client is None:
                return
            pipe = client.pipeline()
            pipe.set(f"{self._prefix}{key}", payload, ex=self._ttl)
            # Also index for invalidation
            policy = value.routing_decision.policy_applied
            if policy:
                pol_idx = f"{self._prefix}idx:policy:{policy}"
                pipe.sadd(pol_idx, key)
                pipe.expire(pol_idx, self._ttl)
            for det in value.routing_decision.selected_detectors:
                det_idx = f"{self._prefix}idx:detector:{det}"
                pipe.sadd(det_idx, key)
                pipe.expire(det_idx, self._ttl)
            pipe.execute()
        except RedisError as exc:
            logger.warning("Failed to store Redis cache entry: %s", exc)

    def invalidate_for_policy(self, policy_bundle: str) -> int:
        if not self._ensure_client():
            return 0
        try:
            pol_idx = f"{self._prefix}idx:policy:{policy_bundle}"
            client = self._redis
            if client is None:
                return 0
            keys = client.smembers(pol_idx)
            if not keys:
                return 0
            # Redis returns bytes items
            key_list = [
                k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in keys
            ]
            pipe = client.pipeline()
            for k in key_list:
                pipe.delete(f"{self._prefix}{k}")
            pipe.delete(pol_idx)
            pipe.execute()
            return len(key_list)
        except RedisError as exc:
            logger.warning("Failed policy invalidation in Redis cache: %s", exc)
            return 0

    def invalidate_for_detector(self, detector: str) -> int:
        if not self._ensure_client():
            return 0
        try:
            det_idx = f"{self._prefix}idx:detector:{detector}"
            client = self._redis
            if client is None:
                return 0
            keys = client.smembers(det_idx)
            if not keys:
                return 0
            key_list = [
                k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in keys
            ]
            pipe = client.pipeline()
            for k in key_list:
                pipe.delete(f"{self._prefix}{k}")
            pipe.delete(det_idx)
            pipe.execute()
            return len(key_list)
        except RedisError as exc:
            logger.warning("Failed detector invalidation in Redis cache: %s", exc)
            return 0

    def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        try:
            client = self._redis
            if client is None:
                return False
            client.ping()
            return True
        except RedisError:
            return False
