from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
from hypothesis import given, settings, assume
import hypothesis.strategies as st


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.conflict import ConflictResolver  # type: ignore  # noqa: E402
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
        processing_time_ms=1,
    )


# Strategy: IMAGE -> highest_confidence; should be invariant to input order
@given(
    st.lists(
        st.builds(
            _res,
            detector=st.text(min_size=1, max_size=6),
            output=st.sampled_from(["safe", "toxic"]),
            conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=2,
        max_size=8,
    )
)
@settings(max_examples=40, deadline=None)
@pytest.mark.asyncio
async def test_order_invariance_hypothesis(results: List[DetectorResult]):
    outs = {r.output for r in results}
    assume("safe" in outs and "toxic" in outs)
    resolver = ConflictResolver()
    base = await resolver.resolve(
        tenant_id="t", policy_bundle="b", content_type=ContentType.IMAGE, detector_results=list(results)
    )
    # Simple permutation (reverse) suffices; Hypothesis varies input lists
    rev = list(reversed(results))
    out = await resolver.resolve(
        tenant_id="t", policy_bundle="b", content_type=ContentType.IMAGE, detector_results=rev
    )
    assert out.winning_output == base.winning_output


# Strategy: TEXT -> weighted_average; scaling all weights by a constant should not change winner
@given(
    st.lists(
        st.builds(
            _res,
            detector=st.text(min_size=1, max_size=6),
            output=st.sampled_from(["safe", "toxic"]),
            conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=2,
        max_size=8,
    ),
    st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
@pytest.mark.asyncio
async def test_weight_scaling_invariance_hypothesis(
    results: List[DetectorResult], scale: float
):
    outs = {r.output for r in results}
    assume("safe" in outs and "toxic" in outs)

    # Deterministically derive a positive weight per detector
    dets = sorted({r.detector for r in results})
    base_weights = {det: float(i + 1) for i, det in enumerate(dets)}

    resolver = ConflictResolver()
    base = await resolver.resolve(
        tenant_id="t",
        policy_bundle="b",
        content_type=ContentType.TEXT,
        detector_results=results,
        weights=base_weights,
    )
    scaled = {k: v * scale for k, v in base_weights.items()}
    out = await resolver.resolve(
        tenant_id="t",
        policy_bundle="b",
        content_type=ContentType.TEXT,
        detector_results=results,
        weights=scaled,
    )
    assert out.winning_output == base.winning_output


# Normalized score bounds: always in [0,1]
@given(
    st.lists(
        st.builds(
            _res,
            detector=st.text(min_size=1, max_size=6),
            output=st.sampled_from(["safe", "toxic", "none"]),
            conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    )
)
@settings(max_examples=40, deadline=None)
@pytest.mark.asyncio
async def test_normalized_scores_bounds_hypothesis(results: List[DetectorResult]):
    resolver = ConflictResolver()
    out = await resolver.resolve(
        tenant_id="t", policy_bundle="b", content_type=ContentType.IMAGE, detector_results=results
    )
    for v in out.normalized_scores.values():
        assert 0.0 <= v <= 1.0


# Tie determinism: When highest confidences tie across outputs, fallback is alphabetical by output
@given(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=30, deadline=None)
@pytest.mark.asyncio
async def test_tie_determinism_alphabetical_highest_confidence(c: float):
    # Build a synthetic tie: one 'safe' and one 'toxic' with identical top confidence c
    # plus some lower-confidence noise that doesn't exceed c
    results = [
        _res("det-safe", "safe", c),
        _res("det-toxic", "toxic", c),
        _res("det-safe-2", "safe", max(0.0, c - 0.3)),
        _res("det-toxic-2", "toxic", max(0.0, c - 0.2)),
    ]
    resolver = ConflictResolver()
    out = await resolver.resolve(
        tenant_id="t",
        policy_bundle="b",
        content_type=ContentType.IMAGE,  # highest_confidence strategy
        detector_results=results,
    )
    # Alphabetical fallback between 'safe' and 'toxic' should pick 'safe'
    assert out.winning_output in ("safe", "toxic")
    assert out.winning_output == "safe"


# Monotonicity (majority_vote): adding a vote for the current winner should not flip the outcome
@given(
    st.lists(
        st.builds(
            _res,
            detector=st.text(min_size=1, max_size=6),
            output=st.sampled_from(["safe", "toxic"]),
            conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    ),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=None)
@pytest.mark.asyncio
async def test_monotonicity_majority_vote(results: List[DetectorResult], extra_conf: float):
    resolver = ConflictResolver()
    base = await resolver.resolve(
        tenant_id="t",
        policy_bundle="b",
        content_type=ContentType.CODE,  # majority_vote
        detector_results=results,
    )
    # If no winner (tie or single unique output), skip
    if base.winning_output not in ("safe", "toxic"):
        pytest.skip("no clear winner; skipping monotonicity assertion")
    # Add one more vote for the current winner
    extra = _res("extra", base.winning_output or "safe", extra_conf)
    augmented = list(results) + [extra]
    out = await resolver.resolve(
        tenant_id="t",
        policy_bundle="b",
        content_type=ContentType.CODE,
        detector_results=augmented,
    )
    assert out.winning_output == base.winning_output


# Robustness: empty list should not crash and produce bounded normalized scores
@given(
    st.lists(
        st.builds(
            _res,
            detector=st.text(min_size=1, max_size=6),
            output=st.sampled_from(["safe", "toxic"]),
            conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=0,
        max_size=5,
    )
)
@settings(max_examples=30, deadline=None)
@pytest.mark.asyncio
async def test_empty_and_small_lists_do_not_crash(results: List[DetectorResult]):
    resolver = ConflictResolver()
    out = await resolver.resolve(
        tenant_id="t", policy_bundle="b", content_type=ContentType.IMAGE, detector_results=results
    )
    for v in out.normalized_scores.values():
        assert 0.0 <= v <= 1.0
