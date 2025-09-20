from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import pytest
from fastapi.testclient import TestClient
import httpx


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
@pytest.mark.parametrize(
    "exception_factory",
    [
        lambda: httpx.TimeoutException("timeout"),
        lambda: httpx.HTTPStatusError("server err", request=httpx.Request("POST", "http://localhost/"), response=httpx.Response(500)),
        lambda: RuntimeError("opa down"),
    ],
)
def test_opa_failure_modes_fall_back_to_default(monkeypatch, exception_factory):
    # Disable external mapper and cache
    settings.config.auto_map_results = False
    settings.config.cache_enabled = False

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

    async def _fake_exec(self, detectors, content, plan, meta):  # type: ignore[override]
        import json
        from pathlib import Path
        scenarios = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "conflict_scenarios.json").read_text(encoding="utf-8"))
        tie = scenarios["tie"]
        return [
            DetectorResult(detector=detectors[0], status=DetectorStatus.SUCCESS, output=tie["detectors"][0]["output"], confidence=tie["detectors"][0]["confidence"], processing_time_ms=12),
            DetectorResult(detector=detectors[1], status=DetectorStatus.SUCCESS, output=tie["detectors"][1]["output"], confidence=tie["detectors"][1]["confidence"], processing_time_ms=10),
        ]

    async def _fake_evaluate_conflict(self, tenant_id: str, bundle: str, input_data: dict):  # type: ignore[override]
        raise exception_factory()

    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)
    monkeypatch.setattr(DetectorCoordinator, "execute_detector_group", _fake_exec)
    monkeypatch.setattr(OPAPolicyEngine, "evaluate_conflict", _fake_evaluate_conflict)

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-opa-fail"}
    body = {
        "content": "dummy",
        "content_type": "text",  # default should be weighted_average on OPA failure
        "tenant_id": "tenant-opa-fail",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }

    r = client.post("/orchestrate", json=body, headers=headers)
    assert r.status_code in (200, 206)
    data = r.json()

    meta = data["aggregated_payload"]["metadata"]
    # Expect fallback to default (TEXT => weighted_average)
    assert meta["aggregation_method"] == "weighted_average"


@pytest.mark.integration
def test_opa_malformed_decision_falls_back_to_default(monkeypatch):
    settings.config.auto_map_results = False
    settings.config.cache_enabled = False

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

    async def _fake_exec(self, detectors, content, plan, meta):  # type: ignore[override]
        return [
            DetectorResult(detector=detectors[0], status=DetectorStatus.SUCCESS, output="toxic", confidence=0.80, processing_time_ms=12),
            DetectorResult(detector=detectors[1], status=DetectorStatus.SUCCESS, output="safe", confidence=0.90, processing_time_ms=10),
        ]

    async def _fake_evaluate_conflict(self, tenant_id: str, bundle: str, input_data: dict):  # type: ignore[override]
        return {"unexpected": "shape"}

    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)
    monkeypatch.setattr(DetectorCoordinator, "execute_detector_group", _fake_exec)
    monkeypatch.setattr(OPAPolicyEngine, "evaluate_conflict", _fake_evaluate_conflict)

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-opa-malformed"}
    body = {
        "content": "dummy",
        "content_type": "text",
        "tenant_id": "tenant-opa-malformed",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }

    r = client.post("/orchestrate", json=body, headers=headers)
    assert r.status_code in (200, 206)
    data = r.json()
    meta = data["aggregated_payload"]["metadata"]
    assert meta["aggregation_method"] == "weighted_average"
