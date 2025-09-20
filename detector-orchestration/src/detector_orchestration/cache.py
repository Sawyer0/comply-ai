from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Set
import json
import logging

from .models import OrchestrationResponse

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


@dataclass
class _IdemEntry:
    value: OrchestrationResponse
    fingerprint: Optional[str]
    expires_at: float


class IdempotencyCache:
    def __init__(self, ttl_seconds: int = 60 * 60 * 24):
        self._store: Dict[str, _IdemEntry] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[OrchestrationResponse]:
        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e.value

    def get_entry(self, key: str) -> Optional[_IdemEntry]:
        e = self._store.get(key)
        if not e:
            return None
        if e.expires_at < time.time():
            self._store.pop(key, None)
            return None
        return e

    def set(self, key: str, value: OrchestrationResponse, fingerprint: Optional[str] = None) -> None:
        self._store[key] = _IdemEntry(value=value, fingerprint=fingerprint, expires_at=time.time() + self._ttl)


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
        entry = self.get_entry(key)
        return entry.value if entry else None

    def get_entry(self, key: str) -> Optional[_IdemEntry]:
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
            # Backward/compat: handle both {..response fields..} and {"resp": {...}, "fp": "..."}
            if isinstance(data, dict) and "resp" in data:
                resp = OrchestrationResponse(**data.get("resp", {}))
                fp = data.get("fp")
                return _IdemEntry(value=resp, fingerprint=fp, expires_at=time.time() + self._ttl)
            # Older format: whole object is the response JSON
            resp = OrchestrationResponse(**data)
            return _IdemEntry(value=resp, fingerprint=None, expires_at=time.time() + self._ttl)
        except Exception:
            return None

    def set(self, key: str, value: OrchestrationResponse, fingerprint: Optional[str] = None) -> None:
        if self._ttl <= 0:
            return
        if not self._ensure_client():
            return
        try:
            payload_obj = {"resp": value.model_dump(), "fp": fingerprint}
            payload = json.dumps(payload_obj, separators=(",", ":"))
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
        # Indexes for invalidation
        self._by_policy: Dict[str, Set[str]] = {}
        self._by_detector: Dict[str, Set[str]] = {}

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

    def _index_key(self, key: str, value: OrchestrationResponse) -> None:
        try:
            policy = value.routing_decision.policy_applied
            if policy:
                s = self._by_policy.setdefault(policy, set())
                s.add(key)
            for det in value.routing_decision.selected_detectors:
                ds = self._by_detector.setdefault(det, set())
                ds.add(key)
        except Exception:
            # Be permissive; indexing is best-effort
            pass

    def set(self, key: str, value: OrchestrationResponse) -> None:
        self._store[key] = _CacheEntry(value=value, expires_at=time.time() + self._ttl)
        self._index_key(key, value)

    def delete_key(self, key: str) -> None:
        self._store.pop(key, None)
        # Lazy cleanup of indexes: full cleanup occurs on invalidate

    def invalidate_for_policy(self, policy_bundle: str) -> int:
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

    def _index_key(self, key: str, value: OrchestrationResponse) -> None:
        try:
            if not self._ensure_client():
                return
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
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
        except Exception:
            # Best-effort indexing
            pass

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
            namespaced = f"{self._prefix}{key}"
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            pipe.set(namespaced, payload, ex=self._ttl)
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
        except Exception:
            return

    def invalidate_for_policy(self, policy_bundle: str) -> int:
        if not self._ensure_client():
            return 0
        try:
            pol_idx = f"{self._prefix}idx:policy:{policy_bundle}"
            keys = self._redis.smembers(pol_idx)  # type: ignore[union-attr]
            if not keys:
                return 0
            # Redis returns bytes items
            key_list = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in keys]
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            for k in key_list:
                pipe.delete(f"{self._prefix}{k}")
            pipe.delete(pol_idx)
            pipe.execute()
            return len(key_list)
        except Exception:
            return 0

    def invalidate_for_detector(self, detector: str) -> int:
        if not self._ensure_client():
            return 0
        try:
            det_idx = f"{self._prefix}idx:detector:{detector}"
            keys = self._redis.smembers(det_idx)  # type: ignore[union-attr]
            if not keys:
                return 0
            key_list = [k.decode("utf-8") if isinstance(k, bytes) else str(k) for k in keys]
            pipe = self._redis.pipeline()  # type: ignore[union-attr]
            for k in key_list:
                pipe.delete(f"{self._prefix}{k}")
            pipe.delete(det_idx)
            pipe.execute()
            return len(key_list)
        except Exception:
            return 0

    def is_healthy(self) -> bool:
        if not self._ensure_client():
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
