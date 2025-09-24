from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
from fastapi.testclient import TestClient


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.api.main import app, settings  # type: ignore  # noqa: E402
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


@pytest.mark.integration
def test_orchestrate_determinism_under_out_of_order(monkeypatch):
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

    async def _exec_order(self, detectors, content, plan, meta, reverse: bool) -> List[DetectorResult]:  # type: ignore[override]
        # Simulate different completion orders by returning reversed list
        order = list(reversed(detectors)) if reverse else list(detectors)
        out: List[DetectorResult] = []
        for idx, d in enumerate(order):
            if d == "det-A":
                out.append(
                    DetectorResult(
                        detector=d,
                        status=DetectorStatus.SUCCESS,
                        output="safe",
                        confidence=0.90,
                        processing_time_ms=10 + idx,
                    )
                )
            else:
                out.append(
                    DetectorResult(
                        detector=d,
                        status=DetectorStatus.SUCCESS,
                        output="toxic",
                        confidence=0.80,
                        processing_time_ms=10 + idx,
                    )
                )
        return out

    # First run: normal order
    monkeypatch.setattr(ContentRouter, "route_request", _fake_route_request)
    monkeypatch.setattr(
        DetectorCoordinator,
        "execute_detector_group",
        lambda self, detectors, content, plan, meta: _exec_order(
            self, detectors, content, plan, meta, False
        ),
    )

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-determ"}
    body = {
        "content": "dummy",
        "content_type": "image",
        "tenant_id": "tenant-determ",
        "policy_bundle": "default",
        "processing_mode": "sync",
    }
    r1 = client.post("/orchestrate", json=body, headers=headers)
    assert r1.status_code in (200, 206)
    data1 = r1.json()
    win1 = data1["aggregated_payload"]["metadata"]["conflict_resolution"][
        "winning_output"
    ]

    # Second run: reversed order
    monkeypatch.setattr(
        DetectorCoordinator,
        "execute_detector_group",
        lambda self, detectors, content, plan, meta: _exec_order(
            self, detectors, content, plan, meta, True
        ),
    )
    r2 = client.post("/orchestrate", json=body, headers=headers)
    assert r2.status_code in (200, 206)
    data2 = r2.json()
    win2 = data2["aggregated_payload"]["metadata"]["conflict_resolution"][
        "winning_output"
    ]

    assert win1 == win2 == "safe"
