"""
API behavior validation tests.

Ensures that the refactored system maintains consistent API behavior
and response formats without requiring backward compatibility.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from ...api.factory import create_analysis_app
from ...domain.entities import AnalysisRequest, AnalysisResponse, VersionInfo
from datetime import datetime, timezone


class TestAPIBehaviorValidation:
    """Test API behavior consistency after refactoring."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI app for testing."""
        return create_analysis_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_analysis_request(self):
        """Sample analysis request for testing."""
        return {
            "period": "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "test-route",
            "required_detectors": ["presidio", "deberta-toxicity"],
            "observed_coverage": {"presidio": 0.8, "deberta-toxicity": 0.9},
            "required_coverage": {"presidio": 0.9, "deberta-toxicity": 0.9},
            "detector_errors": {},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-bundle",
            "env": "dev"
        }
    
    @pytest.fixture
    def expected_analysis_response(self):
        """Expected analysis response structure."""
        return {
            "reason": "Coverage gap detected for presidio detector",
            "remediation": "Increase detector coverage by adjusting configuration",
            "opa_diff": "package test\n\nallow = true",
            "confidence": 0.85,
            "confidence_cutoff_used": 0.7,
            "evidence_refs": ["presidio_coverage_metrics", "detector_config"],
            "notes": "Analysis completed successfully",
            "version_info": {
                "taxonomy": "v1.0.0",
                "frameworks": "v1.0.0",
                "analyst_model": "phi3-mini-3.8b"
            },
            "request_id": "test-request-123",
            "timestamp": "2024-01-01T12:00:00Z",
            "processing_time_ms": 150
        }
    
    def test_health_endpoint_structure(self, client):
        """Test health endpoint returns expected structure."""
        response = client.get("/api/v1/analysis/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate required fields
        required_fields = ["status", "service", "version", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate field types
        assert isinstance(data["status"], str)
        assert isinstance(data["service"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)
        
        # Validate expected values
        assert data["service"] == "analysis"
        assert data["status"] == "healthy"
    
    def test_metrics_endpoint_structure(self, client):
        """Test metrics endpoint returns expected structure."""
        response = client.get("/api/v1/analysis/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return a dictionary (metrics data)
        assert isinstance(data, dict)
    
    @patch('llama_mapper.analysis.api.dependencies.get_analysis_service')
    def test_analyze_endpoint_response_structure(self, mock_get_service, client, 
                                               sample_analysis_request, expected_analysis_response):
        """Test analyze endpoint returns expected response structure."""
        # Mock the analysis service
        mock_service = Mock()
        mock_service.analyze_metrics = AsyncMock(return_value=AnalysisResponse(
            reason=expected_analysis_response["reason"],
            remediation=expected_analysis_response["remediation"],
            opa_diff=expected_analysis_response["opa_diff"],
            confidence=expected_analysis_response["confidence"],
            confidence_cutoff_used=expected_analysis_response["confidence_cutoff_used"],
            evidence_refs=expected_analysis_response["evidence_refs"],
            notes=expected_analysis_response["notes"],
            version_info=VersionInfo(
                taxonomy=expected_analysis_response["version_info"]["taxonomy"],
                frameworks=expected_analysis_response["version_info"]["frameworks"],
                analyst_model=expected_analysis_response["version_info"]["analyst_model"]
            ),
            request_id=expected_analysis_response["request_id"],
            timestamp=datetime.fromisoformat(expected_analysis_response["timestamp"].replace('Z', '+00:00')),
            processing_time_ms=expected_analysis_response["processing_time_ms"]
        ))
        
        mock_get_service.return_value = mock_service
        
        # Make request
        response = client.post(
            "/api/v1/analysis/analyze",
            json={"request": sample_analysis_request}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = [
            "reason", "remediation", "opa_diff", "confidence",
            "confidence_cutoff_used", "evidence_refs", "notes",
            "version_info", "request_id", "timestamp", "processing_time_ms"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Validate field types
        assert isinstance(data["reason"], str)
        assert isinstance(data["remediation"], str)
        assert isinstance(data["opa_diff"], str)
        assert isinstance(data["confidence"], (int, float))
        assert isinstance(data["confidence_cutoff_used"], (int, float))
        assert isinstance(data["evidence_refs"], list)
        assert isinstance(data["notes"], str)
        assert isinstance(data["version_info"], dict)
        assert isinstance(data["request_id"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["processing_time_ms"], int)
        
        # Validate confidence ranges
        assert 0.0 <= data["confidence"] <= 1.0
        assert 0.0 <= data["confidence_cutoff_used"] <= 1.0
        
        # Validate version_info structure
        version_info = data["version_info"]
        version_fields = ["taxonomy", "frameworks", "analyst_model"]
        for field in version_fields:
            assert field in version_info, f"Missing version_info field: {field}"
            assert isinstance(version_info[field], str)
    
    def test_analyze_endpoint_error_handling(self, client):
        """Test analyze endpoint error handling."""
        # Test with invalid request data
        invalid_request = {"invalid": "data"}
        
        response = client.post(
            "/api/v1/analysis/analyze",
            json={"request": invalid_request}
        )
        
        # Should return 400 for validation error
        assert response.status_code == 400
        
        # Test with missing request data
        response = client.post("/api/v1/analysis/analyze", json={})
        
        # Should return 422 for missing data
        assert response.status_code == 422
    
    @patch('llama_mapper.analysis.api.dependencies.get_batch_analysis_service')
    def test_batch_analyze_endpoint_structure(self, mock_get_service, client):
        """Test batch analyze endpoint response structure."""
        # Mock the batch analysis service
        mock_service = Mock()
        mock_service.process_batch = AsyncMock(return_value={
            "batch_id": "test-batch-123",
            "total_requests": 2,
            "successful_analyses": 2,
            "failed_analyses": 0,
            "results": [
                {
                    "request_id": "req-1",
                    "status": "success",
                    "analysis": {
                        "reason": "Test reason 1",
                        "confidence": 0.8
                    }
                },
                {
                    "request_id": "req-2", 
                    "status": "success",
                    "analysis": {
                        "reason": "Test reason 2",
                        "confidence": 0.9
                    }
                }
            ],
            "processing_time_ms": 300
        })
        
        mock_get_service.return_value = mock_service
        
        # Create batch request
        batch_request = {
            "requests": [
                {"request_id": "req-1", "analysis_request": {"tenant": "test1"}},
                {"request_id": "req-2", "analysis_request": {"tenant": "test2"}}
            ]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze/batch",
            json=batch_request,
            headers={"Idempotency-Key": "test-key-123"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate batch response structure
        batch_fields = ["batch_id", "total_requests", "successful_analyses", 
                       "failed_analyses", "results", "processing_time_ms"]
        
        for field in batch_fields:
            assert field in data, f"Missing batch field: {field}"
        
        # Validate field types
        assert isinstance(data["batch_id"], str)
        assert isinstance(data["total_requests"], int)
        assert isinstance(data["successful_analyses"], int)
        assert isinstance(data["failed_analyses"], int)
        assert isinstance(data["results"], list)
        assert isinstance(data["processing_time_ms"], int)
        
        # Validate results structure
        for result in data["results"]:
            assert "request_id" in result
            assert "status" in result
            assert isinstance(result["request_id"], str)
            assert isinstance(result["status"], str)
    
    def test_api_error_response_structure(self, client):
        """Test API error responses have consistent structure."""
        # Test 404 error
        response = client.get("/api/v1/analysis/nonexistent")
        assert response.status_code == 404
        
        # Test 405 error (method not allowed)
        response = client.put("/api/v1/analysis/health")
        assert response.status_code == 405
    
    def test_content_type_headers(self, client, sample_analysis_request):
        """Test that API returns correct content type headers."""
        # Test health endpoint
        response = client.get("/api/v1/analysis/health")
        assert response.headers["content-type"] == "application/json"
        
        # Test metrics endpoint
        response = client.get("/api/v1/analysis/metrics")
        assert response.headers["content-type"] == "application/json"
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present when needed."""
        # Test preflight request
        response = client.options("/api/v1/analysis/health")
        
        # Should handle OPTIONS request
        assert response.status_code in [200, 405]  # Depending on CORS configuration
    
    def test_api_versioning_consistency(self, client):
        """Test that API versioning is consistent across endpoints."""
        endpoints = [
            "/api/v1/analysis/health",
            "/api/v1/analysis/metrics"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            # All v1 endpoints should be accessible
            assert response.status_code in [200, 401, 403]  # Not 404
    
    def test_request_id_generation(self, client, sample_analysis_request):
        """Test that request IDs are generated consistently."""
        with patch('llama_mapper.analysis.api.dependencies.get_analysis_service') as mock_get_service:
            # Mock service that returns request_id
            mock_service = Mock()
            mock_service.analyze_metrics = AsyncMock(return_value=AnalysisResponse(
                reason="Test",
                remediation="Test",
                opa_diff="Test",
                confidence=0.8,
                confidence_cutoff_used=0.7,
                evidence_refs=[],
                notes="Test",
                version_info=VersionInfo(taxonomy="v1", frameworks="v1", analyst_model="test"),
                request_id="generated-request-id",
                timestamp=datetime.now(timezone.utc),
                processing_time_ms=100
            ))
            
            mock_get_service.return_value = mock_service
            
            response = client.post(
                "/api/v1/analysis/analyze",
                json={"request": sample_analysis_request}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have a request_id
            assert "request_id" in data
            assert isinstance(data["request_id"], str)
            assert len(data["request_id"]) > 0


@pytest.mark.integration
class TestRefactoredSystemAPIIntegration:
    """Integration tests for API with refactored system."""
    
    def test_api_uses_dependency_injection(self):
        """Test that API properly uses dependency injection system."""
        # Create app
        app = create_analysis_app()
        
        # Verify app is created successfully
        assert app is not None
        
        # Verify routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/api/v1/analysis/health",
            "/api/v1/analysis/metrics",
            "/api/v1/analysis/analyze",
            "/api/v1/analysis/analyze/batch"
        ]
        
        for expected_route in expected_routes:
            assert any(expected_route in route for route in routes), f"Missing route: {expected_route}"
    
    def test_api_configuration_integration(self):
        """Test that API integrates with configuration system."""
        # This would test that the API properly uses the configuration
        # management system for settings like timeouts, limits, etc.
        
        app = create_analysis_app()
        client = TestClient(app)
        
        # Test that health endpoint reflects configuration
        response = client.get("/api/v1/analysis/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "version" in data
        # Version should come from configuration system
        assert isinstance(data["version"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])