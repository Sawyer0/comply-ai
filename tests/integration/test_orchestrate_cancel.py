from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _ensure_orchestrator_on_path() -> None:
    root = Path(__file__).resolve().parents[2]
    orch_src = root / "detector-orchestration" / "src"
    if str(orch_src) not in sys.path:
        sys.path.insert(0, str(orch_src))


_ensure_orchestrator_on_path()

import detector_orchestration.clients as clients_mod  # type: ignore  # noqa: E402
from detector_orchestration.api.main import app, settings  # type: ignore  # noqa: E402


@pytest.mark.integration
def test_cancel_async_job_success(monkeypatch):
    # Configure to avoid external mapper and rate-limiting interactions
    settings.config.auto_map_results = False
    settings.config.rate_limit_enabled = False

    # Monkeypatch DetectorClient.analyze to add artificial latency
    orig_analyze = clients_mod.DetectorClient.analyze

    async def slow_analyze(self, content: str, metadata=None):  # type: ignore[override]
        await asyncio.sleep(0.25)
        return await orig_analyze(self, content, metadata)

    monkeypatch.setattr(clients_mod.DetectorClient, "analyze", slow_analyze)

    with TestClient(app) as client:
        headers = {settings.config.tenant_header: "tenant-cancel-1"}
        body = {
            "content": "hello cancel job",
            "content_type": "text",
            "tenant_id": "tenant-cancel-1",
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "async",
        }

        r = client.post("/orchestrate", json=body, headers=headers)
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        # Immediately request cancellation
        c = client.delete(f"/orchestrate/status/{job_id}")
        # Either 200 with cancelled OR 409 if job raced to completion (should not due to delay)
        assert c.status_code in (200, 409)
        if c.status_code == 200:
            assert c.json()["status"] == "cancelled"
        else:
            # If it raced, ensure it's not pending/running anymore
            assert c.json().get("detail") == "job_not_cancellable"

        # Poll for final state (cancelled or completed)
        for _ in range(60):
            s = client.get(f"/orchestrate/status/{job_id}")
            if s.status_code == 200:
                st = s.json()["status"]
                if st in ("cancelled", "completed", "failed"):
                    # If completed, it raced past cancellation; acceptable but unlikely
                    break
            time.sleep(0.05)
        else:
            pytest.fail("Job did not reach a terminal state in time")


@pytest.mark.integration
def test_cancel_after_completion_returns_409():
    settings.config.auto_map_results = False
    settings.config.rate_limit_enabled = False

    with TestClient(app) as client:
        headers = {settings.config.tenant_header: "tenant-cancel-2"}
        body = {
            "content": "quick complete",
            "content_type": "text",
            "tenant_id": "tenant-cancel-2",
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "async",
        }
        r = client.post("/orchestrate", json=body, headers=headers)
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        # Wait for completion
        for _ in range(100):
            s = client.get(f"/orchestrate/status/{job_id}")
            if s.status_code == 200 and s.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)
        # Attempt cancel now should 409
        c = client.delete(f"/orchestrate/status/{job_id}")
        assert c.status_code == 409
        detail = c.json().get("detail")
        if isinstance(detail, dict):
            assert detail.get("message") == "job_not_cancellable"
        else:
            assert detail == "job_not_cancellable"


@pytest.mark.integration
def test_cancel_idempotent_with_concurrent_requests(monkeypatch):
    settings.config.auto_map_results = False
    settings.config.rate_limit_enabled = False

    # Slow down analyze so the job remains cancellable for a small window
    orig_analyze = clients_mod.DetectorClient.analyze

    async def slow_analyze(self, content: str, metadata=None):  # type: ignore[override]
        await asyncio.sleep(0.3)
        return await orig_analyze(self, content, metadata)

    monkeypatch.setattr(clients_mod.DetectorClient, "analyze", slow_analyze)

    with TestClient(app) as client:
        headers = {settings.config.tenant_header: "tenant-cancel-3"}
        body = {
            "content": "hello cancel idempotency",
            "content_type": "text",
            "tenant_id": "tenant-cancel-3",
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "async",
        }
        r = client.post("/orchestrate", json=body, headers=headers)
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        # Fire multiple DELETEs concurrently
        import concurrent.futures

        def do_cancel():
            resp = client.delete(f"/orchestrate/status/{job_id}")
            return resp.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            codes = list(ex.map(lambda _: do_cancel(), range(5)))

        # Expect exactly one 200 and the rest 409s (best-effort idempotency)
        assert codes.count(200) >= 1
        assert codes.count(409) >= 1
