from __future__ import annotations

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

from detector_orchestration.api.main import app, settings  # type: ignore  # noqa: E402


@pytest.mark.integration
def test_orchestrate_threshold_converts_to_async_and_completes():
    # Configure to avoid external mapper and rate-limiting interactions
    settings.config.auto_map_results = False
    settings.config.rate_limit_enabled = False
    # Force sync->async conversion
    settings.config.sla.sync_to_async_threshold_ms = 1

    with TestClient(app) as client:
        headers = {settings.config.tenant_header: "tenant-async-1"}
        body = {
            "content": "hello async threshold",
            "content_type": "text",
            "tenant_id": "tenant-async-1",
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "sync",
        }

        r = client.post("/orchestrate", json=body, headers=headers)
        assert r.status_code == 202
        data = r.json()
        assert data.get("job_id")
        job_id = data["job_id"]

        # Poll status until completed
        for _ in range(200):
            s = client.get(f"/orchestrate/status/{job_id}")
            assert s.status_code in (200, 404)
            if s.status_code == 200:
                sj = s.json()
                if sj["status"] == "completed":
                    assert sj.get("result") is not None
                    assert sj.get("progress") == 1.0
                    break
            time.sleep(0.05)
        else:
            pytest.fail("Job did not complete in time")


@pytest.mark.integration
def test_orchestrate_async_mode_returns_202_and_completes():
    settings.config.auto_map_results = False
    settings.config.rate_limit_enabled = False

    with TestClient(app) as client:
        headers = {settings.config.tenant_header: "tenant-async-2"}
        body = {
            "content": "hello explicit async",
            "content_type": "text",
            "tenant_id": "tenant-async-2",
            "policy_bundle": "default",
            "environment": "dev",
            "processing_mode": "async",
        }

        r = client.post("/orchestrate", json=body, headers=headers)
        assert r.status_code == 202
        job_id = r.json()["job_id"]

        # Poll status until completed
        for _ in range(200):
            s = client.get(f"/orchestrate/status/{job_id}")
            if s.status_code == 200 and s.json()["status"] == "completed":
                assert s.json().get("result") is not None
                assert s.json().get("progress") == 1.0
                break
            time.sleep(0.05)
        else:
            pytest.fail("Async job did not complete in time")


@pytest.mark.integration
def test_status_404_for_unknown_job_id():
    with TestClient(app) as client:
        s = client.get("/orchestrate/status/job-nonexistent")
        assert s.status_code == 404
