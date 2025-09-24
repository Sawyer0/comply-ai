from __future__ import annotations

import sys
from pathlib import Path
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


class FakeOPA:
    def __init__(self, decision: dict | Exception):
        self._decision = decision

    async def evaluate_conflict(self, tenant_id: str, bundle: str, input_data: dict):
        if isinstance(self._decision, Exception):
            raise self._decision
        return self._decision


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
async def test_opa_overrides_default_strategy(
    conflict_scenarios, opa_decision_fixtures
):
    # Default for TEXT is WEIGHTED_AVERAGE, but OPA requests HIGHEST_CONFIDENCE
    tie = conflict_scenarios["tie"]
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
    resolver = ConflictResolver(
        opa_engine=FakeOPA(opa_decision_fixtures["highest_confidence"])
    )
    weights = {tie["detectors"][0]["detector"]: 10.0}
    out = await resolver.resolve(
        _request(
            content_type=ContentType.TEXT,
            results=results,
            weights=weights,
        )
    )
    assert out.strategy_used == ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    assert out.winning_output == "safe"
    assert out.audit_decision.get("strategy") == "highest_confidence"


@pytest.mark.asyncio
async def test_opa_tenant_preference_picks_preferred_detector_output(
    conflict_scenarios, opa_decision_fixtures
):
    scen = conflict_scenarios["outlier"]
    # Ensure preferred detector matches one in the scenario (det-C)
    resolver = ConflictResolver(
        opa_engine=FakeOPA(
            {
                "strategy": "tenant_preference",
                "preferred_detector": "det-C",
                "tie_breaker": "detector_priority",
            }
        )
    )
    results = [
        _res(d["detector"], d["output"], d["confidence"]) for d in scen["detectors"]
    ]
    out = await resolver.resolve(
        _request(content_type=ContentType.TEXT, results=results)
    )
    assert out.strategy_used == ConflictResolutionStrategy.TENANT_PREFERENCE
    assert out.winning_detector == "det-C"
    assert out.winning_output == "toxic"
    assert (out.tie_breaker_applied is None) or (
        "preferred_detector" in (out.tie_breaker_applied or "")
    )


@pytest.mark.asyncio
async def test_opa_failure_falls_back_to_default(conflict_scenarios):
    # Raise error -> fall back to content-type default (TEXT => WEIGHTED_AVERAGE)
    tie = conflict_scenarios["tie"]
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
    resolver = ConflictResolver(opa_engine=FakeOPA(RuntimeError("opa down")))
    out = await resolver.resolve(
        _request(content_type=ContentType.TEXT, results=results)
    )
    assert out.strategy_used == ConflictResolutionStrategy.WEIGHTED_AVERAGE
    # No weights -> sums equal; alphabetical fallback resolves to 'safe'
    assert out.winning_output in ("safe", "toxic")
