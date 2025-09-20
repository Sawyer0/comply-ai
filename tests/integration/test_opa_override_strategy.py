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
from detector_orchestration.policy import OPAPolicyEngine  # type: ignore  # noqa: E402


@pytest.mark.integration
def test_opa_override_changes_strategy_and_metadata(monkeypatch):
    # Disable external mapper and cache
    settings.config.auto_map_results = False
    settings.config.cache_enabled = False

    # Monkeypatch route to run two detectors
    async def _fake_route_request(self, request) -> Tuple[RoutingPlan, RoutingDecision]:  # type: ignore[override]
        plan = RoutingPlan(
            primary_detectors=["det-A", "det-B"],
            parallel_groups=[["det-A", "det-B"]],
            timeout_config={"det-A": 1000, "det-B": 1000},
            retry_config={"det-A": 0, "det-B": 0},
            coverage_method="required_set",
            # Provide weights that would bias WEIGHTED_AVERAGE toward toxic if used
            # (not strictly used by coordinator here, but included for parity)
            
        )
        decision = RoutingDecision(
            selected_detectors=["det-A", "det-B"],
            routing_reason="test",
            policy_applied=request.policy_bundle,
            coverage_requirements={"min_success_fraction": 1.0, "weights": {"det-A": 10.0, "det-B": 1.0}},
            health_status={"det-A": True, "det-B": True},
        )
        return plan, decision

    # Monkeypatch detector execution to produce conflicting outputs (fixtures: tie)
    async def _fake_exec(self, detectors, content, plan, meta):  # type: ignore[override]
        import json
        from pathlib import Path
        scenarios = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "conflict_scenarios.json").read_text(encoding="utf-8"))
        tie = scenarios["tie"]
        return [
            DetectorResult(detector=detectors[0], status=DetectorStatus.SUCCESS, output=tie["detectors"][0]["output"], confidence=tie["detectors"][0]["confidence"], processing_time_ms=12),
            DetectorResult(detector=detectors[1], status=DetectorStatus.SUCCESS, output=tie["detectors"][1]["output"], confidence=tie["detectors"][1]["confidence"], processing_time_ms=10),
        ]

    # Force OPA to request highest_confidence regardless of default (TEXT defaults to WEIGHTED_AVERAGE)
    async def _fake_evaluate_conflict(self, tenant_id: str, bundle: str, input_data: dict):  # type: ignore[override]
        return {"strategy": "highest_confidence"}

    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)
    monkeypatch.setattr(DetectorCoordinator, "execute_detector_group", _fake_exec)
    monkeypatch.setattr(OPAPolicyEngine, "evaluate_conflict", _fake_evaluate_conflict)

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-opa-override"}
    body = {
        "content": "dummy",
        "content_type": "text",  # default would be weighted_average
        "tenant_id": "tenant-opa-override",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }

    r = client.post("/orchestrate", json=body, headers=headers)
    assert r.status_code in (200, 206)
    data = r.json()

    assert data.get("aggregated_payload") is not None
    meta = data["aggregated_payload"]["metadata"]

    # Should reflect OPA-selected strategy, not the default for TEXT
    assert meta["aggregation_method"] == "highest_confidence"
    cr = meta.get("conflict_resolution")
    assert cr is not None
    assert cr["strategy_used"] == "highest_confidence"
    assert cr["winning_output"] == "safe"