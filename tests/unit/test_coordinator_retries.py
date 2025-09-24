from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.coordinator import (  # type: ignore  # noqa: E402
    DetectorCoordinator,
)
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    DetectorResult,
    DetectorStatus,
    RoutingPlan,
)


class _FlakyClient:
    def __init__(self, failures_before_success: int) -> None:
        self.n = failures_before_success

    async def analyze(self, content: str, metadata: dict) -> DetectorResult:
        await asyncio.sleep(0.01)
        if self.n > 0:
            self.n -= 1
            raise RuntimeError("transient")
        return DetectorResult(
            detector="flaky",
            status=DetectorStatus.SUCCESS,
            output="ok",
            processing_time_ms=5,
        )


class _FailingClient:
    async def analyze(self, content: str, metadata: dict) -> DetectorResult:
        await asyncio.sleep(0.005)
        return DetectorResult(
            detector="A",
            status=DetectorStatus.FAILED,
            error="err",
            processing_time_ms=5,
        )


class _SuccessClient:
    async def analyze(self, content: str, metadata: dict) -> DetectorResult:
        await asyncio.sleep(0.005)
        return DetectorResult(
            detector="B",
            status=DetectorStatus.SUCCESS,
            output="ok",
            processing_time_ms=5,
        )


@pytest.mark.asyncio
async def test_coordinator_retries_until_success():
    coord = DetectorCoordinator(clients={"flaky": _FlakyClient(1)})
    plan = RoutingPlan(
        primary_detectors=["flaky"],
        retry_config={"flaky": 1},
        timeout_config={"flaky": 100},
    )
    res = await coord.execute_routing_plan("content", plan, request_id="req")
    assert len(res) == 1
    assert res[0].status.value == "success"


@pytest.mark.asyncio
async def test_coordinator_runs_secondary_on_failure():
    coord = DetectorCoordinator(clients={"A": _FailingClient(), "B": _SuccessClient()})
    plan = RoutingPlan(
        primary_detectors=["A"],
        secondary_detectors=["B"],
        retry_config={"A": 0, "B": 0},
        timeout_config={"A": 100, "B": 100},
    )
    res = await coord.execute_routing_plan("content", plan, request_id="req")
    assert len(res) == 2
    statuses = {r.detector: r.status.value for r in res}
    assert statuses["A"] in ("failed", "timeout")
    assert statuses["B"] == "success"
