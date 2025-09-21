"""Unit tests for cache utilities."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from detector_orchestration.cache import IdempotencyCache, ResponseCache
from detector_orchestration.models import (
    DetectorResult,
    DetectorStatus,
    OrchestrationResponse,
    ProcessingMode,
    RoutingDecision,
)


def build_response(detector: str = "det-1") -> OrchestrationResponse:
    """Create a minimal orchestration response for cache tests."""

    result = DetectorResult(
        detector=detector,
        status=DetectorStatus.SUCCESS,
        output="ok",
        processing_time_ms=42,
    )
    decision = RoutingDecision(
        selected_detectors=[detector],
        routing_reason="test",
        policy_applied="default",
        coverage_requirements={"min_success_fraction": 1.0},
        health_status={detector: True},
    )
    return OrchestrationResponse(
        request_id="req-123",
        processing_mode=ProcessingMode.SYNC,
        detector_results=[result],
        aggregated_payload=None,
        mapping_result=None,
        total_processing_time_ms=100,
        detectors_attempted=1,
        detectors_succeeded=1,
        detectors_failed=0,
        coverage_achieved=1.0,
        routing_decision=decision,
        fallback_used=False,
        timestamp=datetime.now(timezone.utc),
    )


class TestIdempotencyCache:
    def test_set_and_get_round_trip(self) -> None:
        cache = IdempotencyCache(ttl_seconds=60)
        response = build_response()

        cache.set("key", response, fingerprint="fp")

        assert cache.get("key") == response
        entry = cache.get_entry("key")
        assert entry is not None
        assert entry.fingerprint == "fp"

    def test_expired_entry_evicted(self) -> None:
        cache = IdempotencyCache(ttl_seconds=1)
        cache.set("key", build_response(), fingerprint=None)

        time.sleep(1.1)

        assert cache.get("key") is None
        assert cache.get_entry("key") is None

    def test_missing_entry_returns_none(self) -> None:
        cache = IdempotencyCache()
        assert cache.get("missing") is None
        assert cache.get_entry("missing") is None

    def test_is_healthy_always_true(self) -> None:
        cache = IdempotencyCache()
        assert cache.is_healthy() is True


class TestResponseCache:
    def test_build_key_is_deterministic(self) -> None:
        key1 = ResponseCache.build_key("content", ("a", "b"), "policy")
        key2 = ResponseCache.build_key("content", ("b", "a"), "policy")
        assert key1 == key2

    def test_set_and_get_round_trip(self) -> None:
        cache = ResponseCache(ttl_seconds=60)
        response = build_response()
        key = ResponseCache.build_key("content", ("a",), "policy")

        cache.set(key, response)

        assert cache.get(key) == response

    def test_eviction_by_policy(self) -> None:
        cache = ResponseCache(ttl_seconds=60)
        resp_a = build_response("a")
        resp_b = build_response("b")

        key_a = ResponseCache.build_key("content", ("a",), "policy-a")
        key_b = ResponseCache.build_key("content", ("b",), "policy-b")

        cache.set(key_a, resp_a)
        cache.set(key_b, resp_b)

        removed = cache.invalidate_for_policy("policy-a")
        assert removed == 1
        assert cache.get(key_a) is None
        assert cache.get(key_b) == resp_b

    def test_eviction_by_detector(self) -> None:
        cache = ResponseCache(ttl_seconds=60)
        resp_a = build_response("a")
        resp_b = build_response("b")

        key_a = ResponseCache.build_key("content", ("a",), "policy")
        key_b = ResponseCache.build_key("other", ("b",), "policy")

        cache.set(key_a, resp_a)
        cache.set(key_b, resp_b)

        removed = cache.invalidate_for_detector("a")
        assert removed == 1
        assert cache.get(key_a) is None
        assert cache.get(key_b) == resp_b

    def test_expired_response_returns_none(self) -> None:
        cache = ResponseCache(ttl_seconds=1)
        key = ResponseCache.build_key("content", ("a",), "policy")
        cache.set(key, build_response())

        time.sleep(1.1)

        assert cache.get(key) is None
