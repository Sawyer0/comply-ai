from __future__ import annotations

import sys
from pathlib import Path
import random
from typing import Optional

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
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    DetectorResult,
    DetectorStatus,
    ContentType,
)


def _res(detector: str, output: str, conf: float) -> DetectorResult:
    return DetectorResult(
        detector=detector,
        status=DetectorStatus.SUCCESS,
        output=output,
        confidence=conf,
        processing_time_ms=5,
    )


def _request(
    *,
    content_type: ContentType,
    results: list[DetectorResult],
    weights: Optional[dict[str, float]] = None,
) -> ConflictResolutionRequest:
    return ConflictResolutionRequest(
        tenant_id="t1",
        policy_bundle="default",
        content_type=content_type,
        detector_results=list(results),
        weights=weights,
    )


@pytest.mark.asyncio
async def test_order_invariance_highest_confidence():
    # Highest-confidence strategy should be invariant to permutation of results order
    resolver = ConflictResolver()
    base = [
        _res("d1", "toxic", 0.80),
        _res("d2", "safe", 0.85),
        _res("d3", "toxic", 0.75),
        _res("d4", "safe", 0.65),
        _res("d5", "safe", 0.55),
    ]
    # Baseline winner
    out0 = await resolver.resolve(
        _request(content_type=ContentType.IMAGE, results=list(base))
    )
    expected_output = out0.winning_output
    expected_detector = out0.winning_detector

    rnd = random.Random(1234)
    # Test multiple random permutations
    for _ in range(20):
        perm = list(base)
        rnd.shuffle(perm)
        out = await resolver.resolve(
            _request(content_type=ContentType.IMAGE, results=perm)
        )
        assert out.winning_output == expected_output
        assert out.winning_detector == expected_detector


@pytest.mark.asyncio
async def test_weight_scaling_invariance_weighted_average():
    # Scaling all weights by a constant should not change the winner
    resolver = ConflictResolver()
    results = [
        _res("d1", "toxic", 0.80),
        _res("d2", "safe", 0.70),
        _res("d3", "toxic", 0.60),
    ]
    base_weights = {"d1": 3.0, "d2": 1.0, "d3": 2.0}

    out_base = await resolver.resolve(
        _request(
            content_type=ContentType.TEXT,
            results=results,
            weights=base_weights,
        )
    )

    for k in [0.1, 2.5, 10.0, 100.0]:
        scaled = {d: w * k for d, w in base_weights.items()}
        out_scaled = await resolver.resolve(
            _request(
                content_type=ContentType.TEXT,
                results=results,
                weights=scaled,
            )
        )
        assert out_scaled.winning_output == out_base.winning_output
