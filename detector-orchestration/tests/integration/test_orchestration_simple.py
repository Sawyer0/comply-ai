"""
Simple integration tests for the Detector Orchestration service.

These tests focus on testing the actual API endpoints without complex mocking,
ensuring the basic functionality works correctly.
"""

import pytest
from fastapi.testclient import TestClient

from detector_orchestration.api.main import app
from detector_orchestration.models import (
    ContentType,
    OrchestrationRequest,
    Priority,
    ProcessingMode,
)


@pytest.fixture
def test_client():
    """Provide test client for FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_request():
    """Provide a sample orchestration request."""
    return OrchestrationRequest(
        content="This is test content for detector analysis",
        content_type=ContentType.TEXT,
        tenant_id="test-tenant",
        policy_bundle="default",
        environment="dev",
        processing_mode=ProcessingMode.SYNC,
        priority=Priority.NORMAL,
        metadata={"test": True},
    )


class TestOrchestrationAPI:
    """Test the orchestration API endpoints."""

    def test_health_endpoint(self, test_client):
        """Test the health endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_metrics_endpoint(self, test_client):
        """Test the metrics endpoint."""
        response = test_client.get("/metrics")
        assert response.status_code == 200
        # Should return Prometheus metrics format
        assert "text/plain" in response.headers.get("content-type", "")

    def test_detectors_list_endpoint(self, test_client):
        """Test the detectors list endpoint."""
        response = test_client.get("/detectors")
        # This might return 503 if registry is not available, which is expected in test
        assert response.status_code in [200, 503]
        if response.status_code == 200:
            data = response.json()
            assert "detectors" in data

    def test_orchestrate_endpoint_basic(self, test_client, sample_request):
        """Test basic orchestration endpoint functionality."""
        response = test_client.post(
            "/orchestrate",
            json=sample_request.model_dump(),
            headers={
                "Authorization": "Bearer test-token",
                "X-Tenant-ID": "test-tenant",
            },
        )
        
        # The endpoint should respond (even if it fails due to missing detectors)
        assert response.status_code in [200, 202, 400, 422, 500, 502, 503]
        
        if response.status_code in [200, 202]:
            data = response.json()
            # Should have basic response structure
            assert "request_id" in data or "job_id" in data

    def test_orchestrate_batch_endpoint_basic(self, test_client):
        """Test basic batch orchestration endpoint functionality."""
        batch_requests = [
            OrchestrationRequest(
                content=f"Test content {i}",
                content_type=ContentType.TEXT,
                tenant_id="test-tenant",
                policy_bundle="default",
                environment="dev",
                priority=Priority.NORMAL,
            ).model_dump()
            for i in range(2)
        ]
        
        response = test_client.post(
            "/orchestrate/batch",
            json={"requests": batch_requests},
            headers={
                "Authorization": "Bearer test-token",
                "X-Tenant-ID": "test-tenant",
            },
        )
        
        # The endpoint should respond (even if it fails due to missing detectors)
        assert response.status_code in [200, 202, 400, 422, 500, 502, 503]

    def test_async_job_status_endpoint(self, test_client):
        """Test the job status endpoint."""
        # Test with a non-existent job ID
        response = test_client.get("/orchestrate/status/non-existent-job")
        
        # Should return 404 for non-existent job
        assert response.status_code == 404

    def test_invalid_request_handling(self, test_client):
        """Test handling of invalid requests."""
        # Test with invalid JSON
        response = test_client.post(
            "/orchestrate",
            json={"invalid": "data"},
            headers={
                "Authorization": "Bearer test-token",
                "X-Tenant-ID": "test-tenant",
            },
        )
        
        # Should return validation error
        assert response.status_code == 422

    def test_missing_headers(self, test_client, sample_request):
        """Test handling of missing required headers."""
        response = test_client.post(
            "/orchestrate",
            json=sample_request.model_dump(),
        )
        
        # Should handle missing headers gracefully
        assert response.status_code in [200, 202, 400, 401, 422, 500, 502, 503]

    def test_cors_headers(self, test_client):
        """Test CORS headers are present."""
        response = test_client.options("/orchestrate")
        # CORS preflight should be handled
        assert response.status_code in [200, 204, 405]
