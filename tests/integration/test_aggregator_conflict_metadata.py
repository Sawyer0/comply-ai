from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest
from fastapi.testclient import TestClient


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.api.main import app, settings  # type: ignore  # noqa: E402
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    RoutingPlan,
    RoutingDecision,
    DetectorResult,
    DetectorStatus,
)
from detector_orchestration.router import ContentRouter  # type: ignore  # noqa: E402
from detector_orchestration.coordinator import DetectorCoordinator  # type: ignore  # noqa: E402


@pytest.mark.integration
def test_aggregated_payload_contains_conflict_metadata(monkeypatch):
    # Ensure no external mapper call and no cache interference
    settings.config.auto_map_results = False
    settings.config.cache_enabled = False

    # Route two detectors; we will simulate conflicting outputs
    async def _fake_route_request(self, request) -> Tuple[RoutingPlan, RoutingDecision]:  # type: ignore[override]
        plan = RoutingPlan(
            primary_detectors=["det-A", "det-B"],
            parallel_groups=[["det-A", "det-B"]],
            timeout_config={"det-A": 1000, "det-B": 1000},
            retry_config={"det-A": 0, "det-B": 0},
            coverage_method="required_set",
        )
        decision = RoutingDecision(
            selected_detectors=["det-A", "det-B"],
            routing_reason="test",
            policy_applied=request.policy_bundle,
            coverage_requirements={"min_success_fraction": 1.0},
            health_status={"det-A": True, "det-B": True},
        )
        return plan, decision

    # Simulate detector results loaded from fixtures (tie scenario)
    async def _fake_exec(self, detectors, content, plan, meta):  # type: ignore[override]
        from typing import List
        from detector_orchestration.models import DetectorResult, DetectorStatus  # type: ignore
        # Import conflict_scenarios from pytest fixtures via global namespace
        try:
            import builtins  # noqa: F401
        except Exception:
            pass
        # Load fixture file directly
        import json
        from pathlib import Path
        scenarios = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "conflict_scenarios.json").read_text(encoding="utf-8"))
        tie = scenarios["tie"]
        out: List[DetectorResult] = []
        for idx, d in enumerate(tie["detectors"][:2]):
            out.append(DetectorResult(detector=detectors[idx], status=DetectorStatus.SUCCESS, output=d["output"], confidence=d["confidence"], processing_time_ms=10 + idx))
        return out

    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)
    monkeypatch.setattr(DetectorCoordinator, "execute_detector_group", _fake_exec)

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-conflict"}
    body = {
        "content": "dummy",
        "content_type": "image",  # default strategy => highest_confidence
        "tenant_id": "tenant-conflict",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }

    r = client.post("/orchestrate", json=body, headers=headers)
    assert r.status_code in (200, 206)
    data = r.json()

    assert data.get("aggregated_payload") is not None
    meta = data["aggregated_payload"]["metadata"]

    # Aggregation method should reflect conflict strategy used
    assert meta["aggregation_method"] == "highest_confidence"
    assert meta["conflict_resolution_applied"] is True

    # Conflict resolution block present and consistent
    cr = meta.get("conflict_resolution")
    assert cr is not None
    assert cr["strategy_used"] == "highest_confidence"
    assert cr["winning_output"] == "safe"
    assert cr["winning_detector"] in ("det-A", "det-B")

    # Normalized scores include both outputs and are within [0,1]
    ns = meta.get("normalized_scores", {})
    assert set(["toxic", "safe"]).issubset(set(ns.keys()))
    assert 0.0 <= ns["toxic"] <= 1.0
    assert 0.0 <= ns["safe"] <= 1.0
