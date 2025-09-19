from __future__ import annotations

import sys
from pathlib import Path


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.aggregator import (  # type: ignore  # noqa: E402
    ResponseAggregator,
)
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    DetectorResult,
    DetectorStatus,
    RoutingPlan,
)


def _res(detector: str, status: DetectorStatus) -> DetectorResult:
    return DetectorResult(
        detector=detector, status=status, output=None, processing_time_ms=10
    )


def test_taxonomy_coverage_all_required_hit():
    agg = ResponseAggregator()
    plan = RoutingPlan(
        primary_detectors=["regex-pii", "deberta-toxicity"],
        coverage_method="taxonomy",
        required_taxonomy_categories=["PII", "HARM"],
    )
    results = [
        _res("regex-pii", DetectorStatus.SUCCESS),
        _res("deberta-toxicity", DetectorStatus.SUCCESS),
    ]
    payload, coverage = agg.aggregate(results, plan, tenant_id="t1")
    assert coverage == 1.0


def test_taxonomy_coverage_partial():
    agg = ResponseAggregator()
    plan = RoutingPlan(
        primary_detectors=["regex-pii", "deberta-toxicity"],
        coverage_method="taxonomy",
        required_taxonomy_categories=["PII", "HARM"],
    )
    results = [
        _res("regex-pii", DetectorStatus.SUCCESS),
        _res("deberta-toxicity", DetectorStatus.FAILED),
    ]
    payload, coverage = agg.aggregate(results, plan, tenant_id="t1")
    assert 0.0 < coverage < 1.0


def test_taxonomy_coverage_defaults_to_union_when_not_provided():
    agg = ResponseAggregator()
    plan = RoutingPlan(
        primary_detectors=["regex-pii", "deberta-toxicity"],
        coverage_method="taxonomy",
    )
    results = [
        _res("regex-pii", DetectorStatus.SUCCESS),
        _res("deberta-toxicity", DetectorStatus.FAILED),
    ]
    payload, coverage = agg.aggregate(results, plan, tenant_id="t1")
    # Union is {PII, HARM}, and only PII hit => 0.5
    assert coverage == 0.5
