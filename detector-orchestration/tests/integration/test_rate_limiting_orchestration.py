"""Integration tests for orchestration service rate limiting.

This test targets the new orchestration service stack (orchestration.main)
 and verifies that tenant-based rate limiting returns HTTP 429 with the
 structured error payload when limits are exceeded.
"""

from __future__ import annotations

# pylint: disable=import-error

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from orchestration.app_state import service_container
from orchestration.main import app
from orchestration.resilience import RateLimiter


@pytest.fixture(name="orchestration_client")
def _orchestration_client() -> Iterator[TestClient]:
    """Provide a TestClient for the new orchestration service app."""
    with TestClient(app) as client:
        yield client


def _build_request_payload() -> dict:
    """Minimal valid orchestration request payload for the new API.

    Uses the shared.interfaces.orchestration.OrchestrationRequest shape:
    - content: str
    - detector_types: list[str]
    Other fields are optional for this test.
    """

    return {
        "content": "rate limiting test content",
        "detector_types": ["mock-detector"],
    }


def test_rate_limit_enforced_for_tenant(orchestration_client: TestClient) -> None:
    """Hammer /api/v1/orchestrate until HTTP 429 is returned.

    The test reconfigures the in-process RateLimiter component to a very low
    tenant limit and then issues repeated requests for the same tenant. It
    asserts that at least one response is a 429 with the structured
    RateLimitError payload (error_code == "RATE_LIMIT_ERROR").
    """

    # Configure a very low tenant limit for this test-only tenant.
    service = service_container.get_orchestration_service()
    assert service is not None

    tenant_id = "rate-limit-test-tenant"

    # Ensure the test tenant exists and is active so security checks pass
    tenant_manager = getattr(service.components, "tenant_manager", None)
    if tenant_manager and not tenant_manager.is_tenant_active(tenant_id):
        tenant_manager.create_tenant(tenant_id=tenant_id, name="Rate Limit Test Tenant")

    # Replace the rate limiter with a stricter instance for deterministic testing.
    service.components.rate_limiter = RateLimiter(
        tenant_limit=2,
        window_seconds=60,
        tenant_overrides={},
    )

    headers = {"X-Tenant-ID": tenant_id}
    payload = _build_request_payload()

    saw_429 = False
    statuses: list[int] = []
    first_error_body = None

    for _ in range(10):
        response = orchestration_client.post(
            "/api/v1/orchestrate",
            json=payload,
            headers=headers,
        )
        statuses.append(response.status_code)

        if response.status_code >= 400 and first_error_body is None:
            # Capture the first error body to understand why requests are failing.
            try:
                first_error_body = response.json()
            except Exception:  # pragma: no cover - defensive
                first_error_body = response.text

        if response.status_code == 429:
            saw_429 = True
            body = response.json()

            # Response should use the shared error envelope from BaseServiceException.
            assert body.get("error_code") == "RATE_LIMIT_ERROR"
            assert "message" in body
            # retry_after may or may not be present, but structure should be valid.
            break

    assert saw_429, (
        "Expected at least one 429 response due to rate limiting; "
        f"observed statuses={statuses}, first_error_body={first_error_body}"
    )
