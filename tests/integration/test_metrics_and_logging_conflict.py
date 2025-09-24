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

from detector_orchestration.api.main import (  # type: ignore  # noqa: E402
    app,
    metrics,
    settings,
)
from detector_orchestration.coordinator import (  # type: ignore  # noqa: E402
    DetectorCoordinator,
)
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    DetectorResult,
    DetectorStatus,
    RoutingDecision,
    RoutingPlan,
)
from detector_orchestration.router import ContentRouter  # type: ignore  # noqa: E402


class _MetricProbe:
    def __init__(self):
        self.detector_latency_calls = 0
        self.coverage_set = []
        self.request_calls = 0

    def record_detector_latency(
        self, detector: str, success: bool, duration_ms: float
    ) -> None:  # noqa: D401
        self.detector_latency_calls += 1

    def record_coverage(
        self, tenant: str, policy: str, coverage: float
    ) -> None:  # noqa: D401
        self.coverage_set.append((tenant, policy, coverage))

    def record_request(
        self, tenant: str, policy: str, status: str, duration_ms: float
    ) -> None:  # noqa: D401
        self.request_calls += 1


@pytest.mark.integration
def test_metrics_and_structured_metadata(monkeypatch):
    # Disable external mapper and cache
    settings.config.auto_map_results = False
    settings.config.cache_enabled = False

    # Route
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

    # Simulate conflicting outputs from fixtures (tie)
    async def _fake_exec(self, detectors, content, plan, meta):  # type: ignore[override]
        import json
        from pathlib import Path

        scenarios = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures"
                / "conflict_scenarios.json"
            ).read_text(encoding="utf-8")
        )
        tie = scenarios["tie"]
        return [
            DetectorResult(
                detector=detectors[0],
                status=DetectorStatus.SUCCESS,
                output=tie["detectors"][0]["output"],
                confidence=tie["detectors"][0]["confidence"],
                processing_time_ms=12,
            ),
            DetectorResult(
                detector=detectors[1],
                status=DetectorStatus.SUCCESS,
                output=tie["detectors"][1]["output"],
                confidence=tie["detectors"][1]["confidence"],
                processing_time_ms=10,
            ),
        ]

    probe = _MetricProbe()
    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)
    monkeypatch.setattr(DetectorCoordinator, "execute_detector_group", _fake_exec)
    # Patch metrics instance methods to route through probe
    monkeypatch.setattr(
        metrics, "record_detector_latency", probe.record_detector_latency
    )
    monkeypatch.setattr(metrics, "record_coverage", probe.record_coverage)
    monkeypatch.setattr(metrics, "record_request", probe.record_request)

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-metrics"}
    body = {
        "content": "dummy",
        "content_type": "image",
        "tenant_id": "tenant-metrics",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }

    r = client.post("/orchestrate", json=body, headers=headers)
    assert r.status_code in (200, 206)
    data = r.json()

    # Structured metadata: conflict_resolution block present
    meta = data["aggregated_payload"]["metadata"]
    cr = meta.get("conflict_resolution")
    assert cr is not None
    assert set(["strategy_used", "winning_output", "winning_detector"]).issubset(
        cr.keys()
    )

    # Metrics assertions
    assert probe.detector_latency_calls >= 2  # two detectors
    assert probe.request_calls >= 1
    assert any(
        t == "tenant-metrics" and p == "default" for (t, p, _c) in probe.coverage_set
    )
