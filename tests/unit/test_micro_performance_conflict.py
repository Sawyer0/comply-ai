from __future__ import annotations

import sys
from pathlib import Path
import time
from typing import List

import pytest


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.conflict import (  # type: ignore  # noqa: E402
    ConflictResolutionRequest,
    ConflictResolver,
)
from detector_orchestration.aggregator import ResponseAggregator  # type: ignore  # noqa: E402
from detector_orchestration.models import DetectorResult, DetectorStatus, ContentType, RoutingPlan  # type: ignore  # noqa: E402


def _res(det: str, out: str, conf: float) -> DetectorResult:
    return DetectorResult(detector=det, status=DetectorStatus.SUCCESS, output=out, confidence=conf, processing_time_ms=5)


@pytest.mark.performance
@pytest.mark.asyncio
async def test_conflict_resolver_micro_performance():
    # Build 200 detectors with a slight bias toward 'safe' but some 'toxic'
    results: List[DetectorResult] = []
    for i in range(100):
        results.append(_res(f"d-safe-{i}", "safe", 0.7))
    for i in range(100):
        results.append(_res(f"d-toxic-{i}", "toxic", 0.65))

    resolver = ConflictResolver()
    start = time.perf_counter()
    request = ConflictResolutionRequest(
        tenant_id="t",
        policy_bundle="b",
        content_type=ContentType.IMAGE,
        detector_results=results,
    )
    # Run 200 resolves (should be well under a few seconds even on CI)
    for _ in range(200):
        out = await resolver.resolve(request)
    duration = time.perf_counter() - start
    # Generous threshold to avoid flakes; adjust as needed
    assert duration < 2.5


@pytest.mark.performance
def test_aggregator_micro_performance():
    # Use the same results; plan with required_set to avoid taxonomy cost
    results: List[DetectorResult] = []
    for i in range(100):
        results.append(_res(f"d-safe-{i}", "safe", 0.7))
    for i in range(100):
        results.append(_res(f"d-toxic-{i}", "toxic", 0.65))

    plan = RoutingPlan(
        primary_detectors=[r.detector for r in results],
        parallel_groups=[[]],
        timeout_config={},
        retry_config={},
        coverage_method="required_set",
        weights={},
        required_taxonomy_categories=[],
    )
    agg = ResponseAggregator()
    start = time.perf_counter()
    for _ in range(200):
        payload, cov = agg.aggregate(results, plan, tenant_id="t")
        assert payload is not None
        assert 0.0 <= cov <= 1.0
    duration = time.perf_counter() - start
    assert duration < 2.5
