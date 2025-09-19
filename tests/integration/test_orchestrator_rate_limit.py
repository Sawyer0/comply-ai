from __future__ import annotations

import sys
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
def test_orchestrate_rate_limit_works_and_returns_rate_limited_error_code():
    # Configure strict tenant limit for a small window so second request blocks
    settings.config.rate_limit_enabled = True
    settings.config.rate_limit_window_seconds = 60
    settings.config.rate_limit_tenant_limit = 1

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-test"}
    body = {
        "content": "hello",
        "content_type": "text",
        "tenant_id": "tenant-test",
        "policy_bundle": "default",
        "processing_mode": "async",
    }

    # First request should pass (async mode returns 202)
    r1 = client.post("/orchestrate", json=body, headers=headers)
    assert r1.status_code in (200, 202, 206, 502)

    # Second request should be rate-limited by middleware
    r2 = client.post("/orchestrate", json=body, headers=headers)
    assert r2.status_code == 403
    data = r2.json()
    # Middleware sets error and error_code to RATE_LIMITED
    assert data.get("error_code") == "RATE_LIMITED" or data.get("error") == "RATE_LIMITED"


@pytest.mark.integration
def test_orchestrate_batch_rate_limit_works():
    # Avoid mapper call inside _process_sync
    settings.config.auto_map_results = False
    settings.config.rate_limit_enabled = True
    settings.config.rate_limit_window_seconds = 60
    settings.config.rate_limit_tenant_limit = 1

    client = TestClient(app)
    headers = {settings.config.tenant_header: "tenant-test-batch"}
    batch = [
        {
            "content": "hello batch",
            "content_type": "text",
            "tenant_id": "tenant-test-batch",
            "policy_bundle": "default",
            "processing_mode": "sync",
        }
    ]

    # First request should pass
    r1 = client.post("/orchestrate/batch", json=batch, headers=headers)
    assert r1.status_code in (200, 206, 502)

    # Second should be rate-limited
    r2 = client.post("/orchestrate/batch", json=batch, headers=headers)
    assert r2.status_code == 403
    data = r2.json()
    assert data.get("error_code") == "RATE_LIMITED" or data.get("error") == "RATE_LIMITED"
