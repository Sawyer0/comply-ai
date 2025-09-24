from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.conflict import (  # type: ignore  # noqa: E402
    ConflictResolutionRequest,
    ConflictResolutionStrategy,
    ConflictResolver,
)
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    ContentType,
    DetectorResult,
    DetectorStatus,
)


def _res(detector: str, output: str, conf: float) -> DetectorResult:
    return DetectorResult(
        detector=detector,
        status=DetectorStatus.SUCCESS,
        output=output,
        confidence=conf,
        processing_time_ms=5,
    )


@pytest.mark.asyncio
async def test_highest_confidence_default_for_image(conflict_scenarios):
    # For images, default strategy is HIGHEST_CONFIDENCE
    resolver = ConflictResolver()
    # Use 'agreement' scenario (both safe) to ensure clear winner
    scen = conflict_scenarios["agreement"]
    results = [
        _res(
            scen["detectors"][0]["detector"],
            scen["detectors"][0]["output"],
            scen["detectors"][0]["confidence"],
        ),
        _res(
            scen["detectors"][1]["detector"],
            scen["detectors"][1]["output"],
            scen["detectors"][1]["confidence"],
        ),
    ]
    request = ConflictResolutionRequest(
        tenant_id="t1",
        policy_bundle="default",
        content_type=ContentType.IMAGE,
        detector_results=results,
    )
    out = await resolver.resolve(request)
    assert out.strategy_used == ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    assert out.winning_output == "safe"
    # Normalized scores within [0,1]
    assert 0.0 <= out.normalized_scores["safe"] <= 1.0


@pytest.mark.asyncio
async def test_weighted_average_default_for_text_with_weights(conflict_scenarios):
    # For text, default strategy is WEIGHTED_AVERAGE; weights can flip result
    resolver = ConflictResolver()
    scen = conflict_scenarios["outlier"]
    results = [
        _res(d["detector"], d["output"], d["confidence"]) for d in scen["detectors"]
    ]
    weights = scen.get("weights", {})
    request = ConflictResolutionRequest(
        tenant_id="t1",
        policy_bundle="default",
        content_type=ContentType.TEXT,
        detector_results=results,
        weights=weights,
    )
    out = await resolver.resolve(request)
    assert out.strategy_used == ConflictResolutionStrategy.WEIGHTED_AVERAGE
    # Weighted should pick the outlier toxic
    assert out.winning_output == "toxic"


@pytest.mark.asyncio
async def test_majority_vote_default_for_code_with_tie_breaker(conflict_scenarios):
    # For code, default strategy is MAJORITY_VOTE
    resolver = ConflictResolver()
    tie = conflict_scenarios["tie"]
    results = [
        _res("det-1", tie["detectors"][0]["output"], tie["detectors"][0]["confidence"]),
        _res("det-2", tie["detectors"][1]["output"], tie["detectors"][1]["confidence"]),
    ]
    request = ConflictResolutionRequest(
        tenant_id="t1",
        policy_bundle="default",
        content_type=ContentType.CODE,
        detector_results=results,
    )
    out = await resolver.resolve(request)
    assert out.strategy_used == ConflictResolutionStrategy.MAJORITY_VOTE
    # With a 1-1 tie, tie-breaker path is used
    assert out.tie_breaker_applied is not None
    # With equal confidences, alphabetical fallback selects 'safe'
    assert out.winning_output == "safe"


@pytest.mark.asyncio
async def test_most_restrictive_default_for_document_approximates_weighted(
    conflict_scenarios,
):
    # For documents, default strategy is MOST_RESTRICTIVE (approximated via weighted)
    resolver = ConflictResolver()
    scen = conflict_scenarios["outlier"]
    results = [
        _res(d["detector"], d["output"], d["confidence"]) for d in scen["detectors"]
    ]
    request = ConflictResolutionRequest(
        tenant_id="t1",
        policy_bundle="default",
        content_type=ContentType.DOCUMENT,
        detector_results=results,
        weights=scen.get("weights", {}),
    )
    out = await resolver.resolve(request)
    assert out.strategy_used == ConflictResolutionStrategy.MOST_RESTRICTIVE
    assert out.winning_output == "toxic"


@pytest.mark.asyncio
async def test_tie_breaker_alphabetical_when_all_equal(conflict_scenarios):
    resolver = ConflictResolver()
    tie = conflict_scenarios["tie"]
    # Equal top confidences; ensure deterministic alphabetical fallback
    results = [
        _res(
            tie["detectors"][0]["detector"],
            tie["detectors"][0]["output"],
            tie["detectors"][0]["confidence"],
        ),
        _res(
            tie["detectors"][1]["detector"],
            tie["detectors"][1]["output"],
            tie["detectors"][1]["confidence"],
        ),
    ]
    out = await resolver.resolve(
        tenant_id="t1",
        policy_bundle="default",
        content_type=ContentType.IMAGE,
        detector_results=results,
    )
    # With equal metrics, fallback is alphabetical by output
    assert out.winning_output == "safe"
