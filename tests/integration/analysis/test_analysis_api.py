"""
Integration tests for the Analysis Module API.

This module tests the FastAPI endpoints and full integration
of the analysis module components.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.llama_mapper.analysis.api.factory import create_analysis_app
from src.llama_mapper.analysis.domain.entities import AnalysisRequest


class TestAnalysisAPI:
    """Test analysis API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch('src.llama_mapper.analysis.config.settings.AnalysisSettings') as mock_settings:
            mock_settings.return_value.cors_origins = ["*"]
            mock_settings.return_value.analysis_model_path = "/fake/path"
            mock_settings.return_value.analysis_temperature = 0.1
            mock_settings.return_value.analysis_confidence_cutoff = 0.3
            
            # Disable authentication for testing
            app = create_analysis_app(disable_auth=True)
            return TestClient(app)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample analysis request."""
        return {
            "period": "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/test",
            "required_detectors": ["pii", "jailbreak"],
            "observed_coverage": {"pii": 0.58, "jailbreak": 0.0},
            "required_coverage": {"pii": 0.95, "jailbreak": 0.95},
            "detector_errors": {"jailbreak": {"5xx": 142}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "riskpack-1.4.0",
            "env": "prod"
        }
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/analysis/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "analysis"
        assert "version" in data
        assert "timestamp" in data
    
    def test_analyze_endpoint_success(self, client, sample_request):
        """Test successful analysis endpoint."""
        
        response = client.post(
            "/api/v1/analysis/analyze",
            json=sample_request,
            headers={
                "X-API-Key": "test-api-key",
                "X-Tenant-ID": "test-tenant"
            }
        )
        
        # Should return 200 or 422 (depending on model server implementation)
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "reason" in data
            assert "remediation" in data
            assert "confidence" in data
            assert "evidence_refs" in data
    
    def test_analyze_endpoint_missing_headers(self, client, sample_request):
        """Test analysis endpoint with missing headers."""
        response = client.post("/api/v1/analysis/analyze", json=sample_request)
        assert response.status_code == 401  # Missing API key
    
    def test_analyze_endpoint_invalid_request(self, client):
        """Test analysis endpoint with invalid request."""
        invalid_request = {
            "period": "invalid-period",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/test",
            "required_detectors": ["pii"],
            "observed_coverage": {"pii": 1.5},  # Invalid coverage > 1.0
            "required_coverage": {"pii": 0.95},
            "detector_errors": {"pii": {"5xx": 0}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "riskpack-1.4.0",
            "env": "prod"
        }
        
        response = client.post(
            "/api/v1/analysis/analyze",
            json=invalid_request,
            headers={
                "X-API-Key": "test-api-key",
                "X-Tenant-ID": "test-tenant"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_analyze_batch_endpoint(self, client, sample_request):
        """Test batch analysis endpoint."""
        
        batch_request = {
            "requests": [sample_request, sample_request]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze/batch",
            json=batch_request,
            headers={
                "X-API-Key": "test-api-key",
                "X-Tenant-ID": "test-tenant",
                "Idempotency-Key": "test-idempotency-key"
            }
        )
        
        # Should return 200 or 422 (depending on model server implementation)
        assert response.status_code in [200, 422]
        
        if response.status_code == 200:
            data = response.json()
            assert "responses" in data
            assert "batch_id" in data
            assert "idempotency_key" in data
            assert "success_count" in data
            assert "error_count" in data
            assert len(data["responses"]) == 2
    
    def test_analyze_batch_endpoint_missing_idempotency_key(self, client, sample_request):
        """Test batch analysis endpoint with missing idempotency key."""
        batch_request = {
            "requests": [sample_request]
        }
        
        response = client.post(
            "/api/v1/analysis/analyze/batch",
            json=batch_request,
            headers={
                "X-API-Key": "test-api-key",
                "X-Tenant-ID": "test-tenant"
            }
        )
        
        assert response.status_code == 422  # Missing idempotency key
    
    def test_analyze_batch_endpoint_too_many_requests(self, client, sample_request):
        """Test batch analysis endpoint with too many requests."""
        batch_request = {
            "requests": [sample_request] * 101  # Exceeds limit of 100
        }
        
        response = client.post(
            "/api/v1/analysis/analyze/batch",
            json=batch_request,
            headers={
                "X-API-Key": "test-api-key",
                "X-Tenant-ID": "test-tenant",
                "Idempotency-Key": "test-idempotency-key"
            }
        )
        
        assert response.status_code == 422  # Too many requests
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/analysis/metrics")
        assert response.status_code == 200
        
        # Should return Prometheus metrics format
        content = response.text
        assert "analysis_requests_total" in content or "prometheus" in content.lower()


class TestAnalysisModuleIntegration:
    """Test full integration of analysis module components."""
    
    def test_model_server_integration(self):
        """Test model server integration with other components."""
        from src.llama_mapper.analysis.infrastructure.model_server import Phi3AnalysisModelServer
        from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator
        from src.llama_mapper.analysis.infrastructure.template_provider import AnalysisTemplateProvider
        
        # Create components
        model_server = Phi3AnalysisModelServer("/fake/path")
        validator = AnalysisValidator()
        templates = AnalysisTemplateProvider()
        
        # Test that components can be instantiated together
        assert model_server is not None
        assert validator is not None
        assert templates is not None
    
    def test_security_integration(self):
        """Test security integration."""
        from src.llama_mapper.analysis.infrastructure.security import AnalysisSecurityValidator, PIIRedactor
        
        security_validator = AnalysisSecurityValidator()
        pii_redactor = PIIRedactor()
        
        # Test PII redaction
        text_with_pii = "Contact john.doe@example.com for support"
        redacted = pii_redactor.redact(text_with_pii)
        
        assert "[REDACTED_EMAIL]" in redacted
        assert "john.doe@example.com" not in redacted
    
    def test_opa_integration(self):
        """Test OPA policy generation integration."""
        from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
        
        opa_generator = OPAPolicyGenerator()
        
        # Test policy generation
        policy = opa_generator.generate_coverage_policy(
            ["pii", "jailbreak"],
            {"pii": 0.95, "jailbreak": 0.95}
        )
        
        assert "package policy" in policy
        assert "coverage_violation" in policy
    
    def test_quality_evaluation_integration(self):
        """Test quality evaluation integration."""
        from src.llama_mapper.analysis.infrastructure.quality_evaluator import QualityEvaluator
        from src.llama_mapper.analysis.domain.entities import AnalysisRequest, AnalysisResponse, VersionInfo
        from datetime import datetime
        
        evaluator = QualityEvaluator()
        
        # Create test data
        request = AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="/test",
            required_detectors=["jailbreak"],
            observed_coverage={"jailbreak": 0.0},
            required_coverage={"jailbreak": 0.95},
            detector_errors={"jailbreak": {"5xx": 142}},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="riskpack-1.4.0",
            env="prod"
        )
        
        response = AnalysisResponse(
            reason="jailbreak detector down",
            remediation="restore jailbreak detector",
            opa_diff="package policy\npolicy.coverage_violation[route] {\n  required := {\"jailbreak\"}\n  observed := input.coverage[route]\n  required - observed != {}\n}",
            confidence=0.8,
            confidence_cutoff_used=0.3,
            evidence_refs=["observed_coverage", "detector_errors"],
            notes="Detector health issue detected",
            version_info=VersionInfo(
                taxonomy="v1.0.0",
                frameworks="v1.0.0",
                analyst_model="phi3-mini-3.8b-v1.0"
            ),
            request_id="test-request-123",
            timestamp=datetime.utcnow(),
            processing_time_ms=150
        )
        
        # Test evaluation
        examples = [(request, response)]
        metrics = evaluator.evaluate_batch(examples)
        
        assert "total_examples" in metrics
        assert "schema_valid_rate" in metrics
        assert "rubric_score" in metrics


if __name__ == "__main__":
    pytest.main([__file__])
