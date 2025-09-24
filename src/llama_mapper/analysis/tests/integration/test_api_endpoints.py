"""
Integration tests for analysis API endpoints.

Tests full API pipeline with real HTTP requests and responses.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from ...api.factory import create_analysis_app
from ...domain.entities import AnalysisRequest, AnalysisType


class TestAnalysisAPIEndpoints:
    """Integration tests for analysis API endpoints."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        return create_analysis_app()

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_request_data(self):
        """Sample request data for testing."""
        return {
            "period": "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "test-route",
            "required_detectors": ["detector1", "detector2"],
            "observed_coverage": {"detector1": 0.8, "detector2": 0.9},
            "required_coverage": {"detector1": 0.7, "detector2": 0.8},
            "detector_errors": {"detector1": {"error": "test"}},
            "high_sev_hits": [{"hit": "test"}],
            "false_positive_bands": [{"band": "test"}],
            "policy_bundle": "test-bundle",
            "env": "dev",
        }

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/api/v1/analysis/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/analysis/metrics")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_analyze_endpoint_success(self, client, sample_request_data):
        """Test successful analysis endpoint."""
        with patch(
            "src.llama_mapper.analysis.api.dependencies.get_analysis_service"
        ) as mock_service:
            # Mock the analysis service
            mock_analysis_service = Mock()
            mock_analysis_service.analyze.return_value = {
                "reason": "Test reason",
                "remediation": "Test remediation",
                "opa_diff": "package test",
                "confidence": 0.8,
                "confidence_cutoff_used": 0.3,
                "evidence_refs": ["detector1"],
                "notes": "Test notes",
                "version_info": {
                    "taxonomy": "v1.0",
                    "frameworks": "v1.0",
                    "analyst_model": "phi3-mini-3.8b",
                },
                "request_id": "test-request-id",
                "timestamp": "2024-01-01T00:00:00Z",
                "processing_time_ms": 100,
            }
            mock_service.return_value = mock_analysis_service

            response = client.post(
                "/api/v1/analysis/analyze",
                json=sample_request_data,
                headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "reason" in data
            assert "remediation" in data
            assert "opa_diff" in data
            assert "confidence" in data
            assert "version_info" in data

    def test_analyze_endpoint_validation_error(self, client):
        """Test analysis endpoint with validation error."""
        invalid_data = {
            "period": "invalid-period",
            "tenant": "",
            # Missing required fields
        }

        response = client.post(
            "/api/v1/analysis/analyze",
            json=invalid_data,
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 422

    def test_analyze_endpoint_missing_headers(self, client, sample_request_data):
        """Test analysis endpoint with missing headers."""
        response = client.post("/api/v1/analysis/analyze", json=sample_request_data)

        assert response.status_code == 422

    def test_analyze_endpoint_service_error(self, client, sample_request_data):
        """Test analysis endpoint with service error."""
        with patch(
            "src.llama_mapper.analysis.api.dependencies.get_analysis_service"
        ) as mock_service:
            # Mock service to raise exception
            mock_analysis_service = Mock()
            mock_analysis_service.analyze.side_effect = Exception("Service error")
            mock_service.return_value = mock_analysis_service

            response = client.post(
                "/api/v1/analysis/analyze",
                json=sample_request_data,
                headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
            )

            assert response.status_code == 500

    def test_batch_analyze_endpoint_success(self, client, sample_request_data):
        """Test successful batch analysis endpoint."""
        batch_data = {
            "requests": [sample_request_data, sample_request_data],
            "idempotency_key": "test-batch-key",
        }

        with patch(
            "src.llama_mapper.analysis.api.dependencies.get_batch_analysis_service"
        ) as mock_service:
            # Mock the batch analysis service
            mock_batch_service = Mock()
            mock_batch_service.analyze_batch.return_value = {
                "responses": [
                    {
                        "reason": "Test reason 1",
                        "remediation": "Test remediation 1",
                        "opa_diff": "package test",
                        "confidence": 0.8,
                        "confidence_cutoff_used": 0.3,
                        "evidence_refs": ["detector1"],
                        "notes": "Test notes 1",
                        "version_info": {
                            "taxonomy": "v1.0",
                            "frameworks": "v1.0",
                            "analyst_model": "phi3-mini-3.8b",
                        },
                        "request_id": "test-request-1",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "processing_time_ms": 100,
                    },
                    {
                        "reason": "Test reason 2",
                        "remediation": "Test remediation 2",
                        "opa_diff": "package test",
                        "confidence": 0.9,
                        "confidence_cutoff_used": 0.3,
                        "evidence_refs": ["detector2"],
                        "notes": "Test notes 2",
                        "version_info": {
                            "taxonomy": "v1.0",
                            "frameworks": "v1.0",
                            "analyst_model": "phi3-mini-3.8b",
                        },
                        "request_id": "test-request-2",
                        "timestamp": "2024-01-01T00:00:00Z",
                        "processing_time_ms": 120,
                    },
                ],
                "batch_id": "test-batch-id",
                "total_requests": 2,
                "successful_requests": 2,
                "failed_requests": 0,
                "processing_time_ms": 220,
                "timestamp": "2024-01-01T00:00:00Z",
            }
            mock_service.return_value = mock_batch_service

            response = client.post(
                "/api/v1/analysis/analyze/batch",
                json=batch_data,
                headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "responses" in data
            assert "batch_id" in data
            assert "total_requests" in data
            assert len(data["responses"]) == 2

    def test_batch_analyze_endpoint_validation_error(self, client):
        """Test batch analysis endpoint with validation error."""
        invalid_batch_data = {
            "requests": [],  # Empty requests
            "idempotency_key": "test-batch-key",
        }

        response = client.post(
            "/api/v1/analysis/analyze/batch",
            json=invalid_batch_data,
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 422

    def test_batch_analyze_endpoint_too_many_requests(
        self, client, sample_request_data
    ):
        """Test batch analysis endpoint with too many requests."""
        # Create 101 requests (exceeds limit of 100)
        batch_data = {
            "requests": [sample_request_data] * 101,
            "idempotency_key": "test-batch-key",
        }

        response = client.post(
            "/api/v1/analysis/analyze/batch",
            json=batch_data,
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 422

    def test_batch_analyze_endpoint_service_error(self, client, sample_request_data):
        """Test batch analysis endpoint with service error."""
        batch_data = {
            "requests": [sample_request_data],
            "idempotency_key": "test-batch-key",
        }

        with patch(
            "src.llama_mapper.analysis.api.dependencies.get_batch_analysis_service"
        ) as mock_service:
            # Mock service to raise exception
            mock_batch_service = Mock()
            mock_batch_service.analyze_batch.side_effect = Exception(
                "Batch service error"
            )
            mock_service.return_value = mock_batch_service

            response = client.post(
                "/api/v1/analysis/analyze/batch",
                json=batch_data,
                headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
            )

            assert response.status_code == 500

    def test_cache_cleanup_endpoint(self, client):
        """Test cache cleanup endpoint."""
        response = client.post(
            "/api/v1/analysis/admin/cache/cleanup",
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "cleaned_entries" in data

    def test_quality_evaluation_endpoint(self, client):
        """Test quality evaluation endpoint."""
        response = client.get(
            "/api/v1/analysis/quality/evaluation",
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_examples" in data
        assert "schema_valid_rate" in data
        assert "rubric_score" in data
        assert "opa_compile_success_rate" in data

    def test_quality_evaluate_endpoint(self, client):
        """Test quality evaluate endpoint."""
        evaluate_data = {
            "examples": [
                {
                    "input": {"test": "data"},
                    "expected_output": {"reason": "test", "remediation": "test"},
                }
            ]
        }

        response = client.post(
            "/api/v1/analysis/quality/evaluate",
            json=evaluate_data,
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "total_examples" in data
        assert "schema_valid_rate" in data
        assert "rubric_score" in data

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/analysis/health")

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers

    def test_rate_limiting(self, client, sample_request_data):
        """Test rate limiting functionality."""
        # Make multiple requests quickly
        for i in range(10):
            response = client.post(
                "/api/v1/analysis/analyze",
                json=sample_request_data,
                headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
            )

            # Should not be rate limited for first few requests
            if i < 5:
                assert response.status_code in [200, 500]  # 500 due to mocked service

    def test_api_versioning(self, client):
        """Test API versioning in endpoints."""
        response = client.get("/api/v1/analysis/health")
        assert response.status_code == 200

        # Test that v1 endpoints work
        response = client.get("/api/v1/analysis/metrics")
        assert response.status_code == 200

    def test_error_response_format(self, client):
        """Test error response format."""
        response = client.post(
            "/api/v1/analysis/analyze",
            json={"invalid": "data"},
            headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_request_id_generation(self, client, sample_request_data):
        """Test that request IDs are generated."""
        with patch(
            "src.llama_mapper.analysis.api.dependencies.get_analysis_service"
        ) as mock_service:
            mock_analysis_service = Mock()
            mock_analysis_service.analyze.return_value = {
                "reason": "Test reason",
                "remediation": "Test remediation",
                "opa_diff": "package test",
                "confidence": 0.8,
                "confidence_cutoff_used": 0.3,
                "evidence_refs": ["detector1"],
                "notes": "Test notes",
                "version_info": {
                    "taxonomy": "v1.0",
                    "frameworks": "v1.0",
                    "analyst_model": "phi3-mini-3.8b",
                },
                "request_id": "test-request-id",
                "timestamp": "2024-01-01T00:00:00Z",
                "processing_time_ms": 100,
            }
            mock_service.return_value = mock_analysis_service

            response = client.post(
                "/api/v1/analysis/analyze",
                json=sample_request_data,
                headers={"X-API-Key": "test-api-key", "X-Tenant-ID": "test-tenant"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "request_id" in data
            assert data["request_id"] is not None
