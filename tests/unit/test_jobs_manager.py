from __future__ import annotations

import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

import pytest


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

from detector_orchestration.jobs import JobManager  # type: ignore  # noqa: E402
from detector_orchestration.models import (  # type: ignore  # noqa: E402
    OrchestrationRequest,
    OrchestrationResponse,
    RoutingDecision,
    RoutingPlan,
    ContentType,
    ProcessingMode,
    Priority,
)


def _mk_request(priority: Priority) -> OrchestrationRequest:
    return OrchestrationRequest(
        content="hello",
        content_type=ContentType.TEXT,
        tenant_id="tenant-u",
        policy_bundle="default",
        environment="dev",
        processing_mode=ProcessingMode.ASYNC,
        priority=priority,
    )


def _mk_decision(selected: Optional[list[str]] = None) -> RoutingDecision:
    return RoutingDecision(
        selected_detectors=selected or [],
        routing_reason="test",
        policy_applied="default",
        coverage_requirements={},
        health_status={},
    )


def _mk_plan() -> RoutingPlan:
    return RoutingPlan(
        primary_detectors=[],
        parallel_groups=[],
        sequential_dependencies={},
        timeout_config={},
        retry_config={},
        coverage_method="required_set",
        weights={},
        required_taxonomy_categories=[],
    )


def _mk_response(req: OrchestrationRequest, dec: RoutingDecision) -> OrchestrationResponse:
    return OrchestrationResponse(
        request_id="req",
        processing_mode=req.processing_mode,
        detector_results=[],
        aggregated_payload=None,
        mapping_result=None,
        total_processing_time_ms=0,
        detectors_attempted=0,
        detectors_succeeded=0,
        detectors_failed=0,
        coverage_achieved=0.0,
        routing_decision=dec,
        fallback_used=False,
        timestamp=datetime.now(timezone.utc),
        idempotency_key=None,
    )


@pytest.mark.asyncio
async def test_job_manager_priority_ordering():
    # Arrange: single worker to enforce ordering
    async def run_fn(req, idem, dec, plan, progress_cb: Optional[Callable[[float], None]]):
        if progress_cb:
            progress_cb(0.5)
        # Critical jobs finish faster to observe ordering
        await asyncio.sleep(0.15 if req.priority == Priority.CRITICAL else 0.3)
        return _mk_response(req, dec)

    jm = JobManager(run_fn=run_fn, workers=1)
    await jm.start()

    # Enqueue normal first, then critical. Critical should complete before normal even though enqueued second.
    normal_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan())
    critical_id = await jm.enqueue(_mk_request(Priority.CRITICAL), None, _mk_decision(), _mk_plan())

    # Act/Assert: after ~0.2s, critical should be completed, normal should not.
    await asyncio.sleep(0.2)
    s_crit = jm.get_status(critical_id)
    s_norm = jm.get_status(normal_id)
    assert s_crit is not None and s_crit.status.value == "completed"
    assert s_norm is not None and s_norm.status.value in {"pending", "running"}

    # Cleanup: wait both finished
    for _ in range(20):
        s_norm = jm.get_status(normal_id)
        if s_norm and s_norm.status.value == "completed":
            break
        await asyncio.sleep(0.05)
    await jm.stop()


@pytest.mark.asyncio
async def test_job_manager_progress_updates_and_completion():
    # Arrange
    async def run_fn(req, idem, dec, plan, progress_cb: Optional[Callable[[float], None]]):
        if progress_cb:
            progress_cb(0.2)
        await asyncio.sleep(0.05)
        if progress_cb:
            progress_cb(0.6)
        await asyncio.sleep(0.05)
        return _mk_response(req, dec)

    jm = JobManager(run_fn=run_fn, workers=1)
    await jm.start()

    job_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan())

    # Assert progress increases and then completes
    saw_mid_progress = False
    for _ in range(60):
        s = jm.get_status(job_id)
        assert s is not None
        if 0.5 <= s.progress < 1.0:
            saw_mid_progress = True
        if s.status.value == "completed":
            assert s.progress == 1.0
            break
        await asyncio.sleep(0.05)
    assert saw_mid_progress, "Expected to see progress update before completion"
    await jm.stop()


@pytest.mark.asyncio
async def test_job_manager_callback_invoked(monkeypatch):
    # Arrange: patch httpx.AsyncClient used in jobs module
    captured: list[tuple[str, dict]] = []

    class DummyResp:
        def __init__(self, status_code: int = 200) -> None:
            self.status_code = status_code

    class DummyClient:
        def __init__(self, timeout: float | int | None = None) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict | None = None, **kwargs):
            captured.append((url, json or {}))
            return DummyResp(200)

    import detector_orchestration.jobs as jobs_mod  # type: ignore

    monkeypatch.setattr(jobs_mod.httpx, "AsyncClient", DummyClient)

    async def run_fn(req, idem, dec, plan, progress_cb):
        return _mk_response(req, dec)

    jm = JobManager(run_fn=run_fn, workers=1)
    await jm.start()

    job_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan(), callback_url="http://localhost/callback")

    # Wait for completion and callback
    for _ in range(60):
        s = jm.get_status(job_id)
        if s and s.status.value == "completed":
            break
        await asyncio.sleep(0.05)

    await asyncio.sleep(0.05)
    await jm.stop()

    # Assert callback captured
    assert captured, "Expected callback to be invoked"
    url, payload = captured[-1]
    assert url == "http://localhost/callback"
    assert payload.get("job_id") == job_id
    assert payload.get("status") == "completed"
    assert isinstance(payload.get("result"), dict)


@pytest.mark.asyncio
async def test_job_manager_cancel_before_run():
    # run_fn sleeps so we can enqueue two jobs and cancel the second before it starts
    async def run_fn(req, idem, dec, plan, progress_cb):
        await asyncio.sleep(0.2)
        return _mk_response(req, dec)

    jm = JobManager(run_fn=run_fn, workers=1)
    await jm.start()

    first_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan())
    second_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan())

    # Cancel the second while the first is still running
    cancelled = jm.cancel(second_id)
    assert cancelled is True

    # Wait for first to complete
    for _ in range(20):
        s1 = jm.get_status(first_id)
        if s1 and s1.status.value == "completed":
            break
        await asyncio.sleep(0.05)

    # The second should remain cancelled
    s2 = jm.get_status(second_id)
    assert s2 is not None
    assert s2.status.value == "cancelled"
    await jm.stop()


@pytest.mark.asyncio
async def test_job_manager_failure_sets_failed_status():
    async def run_fn(req, idem, dec, plan, progress_cb):
        raise RuntimeError("boom")

    jm = JobManager(run_fn=run_fn, workers=1)
    await jm.start()

    job_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan())

    for _ in range(40):
        s = jm.get_status(job_id)
        if s and s.status.value == "failed":
            assert s.error and "boom" in s.error
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail("Job did not transition to failed state")

    await jm.stop()


@pytest.mark.asyncio
async def test_job_manager_status_transitions():
    # Use sleeps to observe transitions pending -> running -> completed
    async def run_fn(req, idem, dec, plan, progress_cb):
        await asyncio.sleep(0.1)
        return _mk_response(req, dec)

    jm = JobManager(run_fn=run_fn, workers=1)
    await jm.start()

    job_id = await jm.enqueue(_mk_request(Priority.NORMAL), None, _mk_decision(), _mk_plan())

    saw_pending = False
    saw_running = False

    for _ in range(40):
        s = jm.get_status(job_id)
        assert s is not None
        if s.status.value == "pending":
            saw_pending = True
        elif s.status.value == "running":
            saw_running = True
        elif s.status.value == "completed":
            break
        await asyncio.sleep(0.02)

    assert saw_pending
    assert saw_running

    await jm.stop()
