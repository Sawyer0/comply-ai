from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
from fastapi.testclient import TestClient


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.api.main import app, settings  # type: ignore  # noqa: E402
from detector_orchestration.models import RoutingPlan, RoutingDecision  # type: ignore  # noqa: E402
from detector_orchestration.router import ContentRouter  # type: ignore  # noqa: E402


@pytest.mark.integration
def test_secondary_routing_improves_taxonomy_coverage(monkeypatch):
    # Enable secondary on coverage below threshold
    settings.config.secondary_on_coverage_below = True
    settings.config.secondary_min_coverage = 1.0

    # Monkeypatch router to force primary=['regex-pii'], secondary=['deberta-toxicity']
    async def _fake_route_request(self, request) -> Tuple[RoutingPlan, RoutingDecision]:  # type: ignore[override]
        plan = RoutingPlan(
            primary_detectors=["regex-pii"],
            secondary_detectors=["deberta-toxicity"],
            parallel_groups=[["regex-pii"]],
            timeout_config={"regex-pii": 1000, "deberta-toxicity": 1000},
            retry_config={"regex-pii": 0, "deberta-toxicity": 0},
            coverage_method="taxonomy",
            required_taxonomy_categories=["PII", "HARM"],
        )
        decision = RoutingDecision(
            selected_detectors=["regex-pii", "deberta-toxicity"],
            routing_reason="test",
            policy_applied=request.policy_bundle,
            coverage_requirements={"required_taxonomy_categories": ["PII", "HARM"]},
            health_status={"regex-pii": True, "deberta-toxicity": True},
        )
        return plan, decision

    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-cov"}
    body = {
        "content": "this is toxic",  # so toxicity detector succeeds
        "content_type": "text",
        "tenant_id": "tenant-cov",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }

    r = client.post("/orchestrate", json=body, headers=headers)
    assert r.status_code in (200, 206, 502)
    data = r.json()
    # Should have executed both detectors to reach full taxonomy coverage
    assert data["detectors_attempted"] == 2
    assert data["coverage_achieved"] >= 1.0 or abs(data["coverage_achieved"] - 1.0) < 1e-6

