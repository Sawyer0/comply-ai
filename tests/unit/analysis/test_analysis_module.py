"""
Unit tests for the Analysis Module.

This module tests the core functionality of the analysis module including
model server, templates, validation, and security features.
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.llama_mapper.analysis.models import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    VersionInfo,
    AnalysisType
)
from src.llama_mapper.analysis.infrastructure.model_server import Phi3AnalysisModelServer
from src.llama_mapper.analysis.infrastructure.template_provider import AnalysisTemplateProvider
from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator
from src.llama_mapper.analysis.infrastructure.security import AnalysisSecurityValidator, PIIRedactor
from src.llama_mapper.analysis.infrastructure.opa_generator import OPAPolicyGenerator
from src.llama_mapper.analysis.infrastructure.quality_evaluator import QualityEvaluator


class TestAnalysisModels:
    """Test analysis data models."""
    
    def test_analysis_request_validation(self):
        """Test AnalysisRequest validation."""
        # Valid request
        request = AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="/test",
            required_detectors=["pii", "jailbreak"],
            observed_coverage={"pii": 0.8, "jailbreak": 0.9},
            required_coverage={"pii": 0.95, "jailbreak": 0.95},
            detector_errors={"pii": {"5xx": 0}, "jailbreak": {"5xx": 0}},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="riskpack-1.4.0",
            env="prod"
        )
        
        assert request.tenant == "test-tenant"
        assert len(request.required_detectors) == 2
    
    def test_analysis_request_coverage_validation(self):
        """Test coverage value validation."""
        with pytest.raises(ValueError, match="Coverage value.*must be between 0.0 and 1.0"):
            AnalysisRequest(
                period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
                tenant="test-tenant",
                app="test-app",
                route="/test",
                required_detectors=["pii"],
                observed_coverage={"pii": 1.5},  # Invalid coverage > 1.0
                required_coverage={"pii": 0.95},
                detector_errors={"pii": {"5xx": 0}},
                high_sev_hits=[],
                false_positive_bands=[],
                policy_bundle="riskpack-1.4.0",
                env="prod"
            )


class TestAnalysisModelServer:
    """Test analysis model server."""
    
    @pytest.fixture
    def model_server(self):
        """Create model server instance."""
        return Phi3AnalysisModelServer(
            model_path="/fake/path",
            temperature=0.1,
            confidence_cutoff=0.3
        )
    
    @pytest.fixture
    def sample_request(self):
        """Create sample analysis request."""
        return AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="/test",
            required_detectors=["pii", "jailbreak"],
            observed_coverage={"pii": 0.58, "jailbreak": 0.0},
            required_coverage={"pii": 0.95, "jailbreak": 0.95},
            detector_errors={"jailbreak": {"5xx": 142}},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="riskpack-1.4.0",
            env="prod"
        )
    
    def test_model_server_initialization(self, model_server):
        """Test model server initialization."""
        assert model_server.temperature == 0.1
        assert model_server.confidence_cutoff == 0.3
        assert model_server.version_info is not None
    
    def test_build_prompt(self, model_server, sample_request):
        """Test prompt building."""
        prompt = model_server._build_prompt(sample_request)
        
        assert "You are an internal compliance analyst" in prompt
        assert "pii" in prompt
        assert "jailbreak" in prompt
        assert "0.58" in prompt
        assert "0.0" in prompt
    
    @pytest.mark.asyncio
    async def test_analyze_coverage_gap(self, model_server, sample_request):
        """Test analysis for coverage gap scenario."""
        result = await model_server.analyze(sample_request)
        
        assert "reason" in result
        assert "remediation" in result
        assert "confidence" in result
        assert "evidence_refs" in result
        assert result["confidence"] >= 0.0
        assert result["confidence"] <= 1.0


class TestAnalysisTemplates:
    """Test analysis templates."""
    
    @pytest.fixture
    def templates(self):
        """Create templates instance."""
        return AnalysisTemplates()
    
    @pytest.fixture
    def coverage_gap_request(self):
        """Create coverage gap request."""
        return AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="/test",
            required_detectors=["pii", "jailbreak"],
            observed_coverage={"pii": 0.58, "jailbreak": 0.0},
            required_coverage={"pii": 0.95, "jailbreak": 0.95},
            detector_errors={"jailbreak": {"5xx": 142}},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="riskpack-1.4.0",
            env="prod"
        )
    
    def test_coverage_gap_template(self, templates, coverage_gap_request):
        """Test coverage gap template."""
        response = templates._coverage_gap_template(coverage_gap_request)
        
        assert "reason" in response
        assert "remediation" in response
        assert "confidence" in response
        assert "evidence_refs" in response
        assert len(response["reason"]) <= 120
        assert len(response["remediation"]) <= 120
        assert response["confidence"] > 0.0
    
    def test_template_selection(self, templates, coverage_gap_request):
        """Test template selection logic."""
        analysis_type = templates.select_analysis_type(coverage_gap_request)
        assert analysis_type == AnalysisType.COVERAGE_GAP
    
    def test_opa_policy_generation(self, templates):
        """Test OPA policy generation."""
        required_detectors = ["pii", "jailbreak"]
        required_coverage = {"pii": 0.95, "jailbreak": 0.95}
        
        policy = templates._generate_coverage_policy(required_detectors, required_coverage)
        
        assert "package policy" in policy
        assert "coverage_violation" in policy
        assert "pii" in policy
        assert "jailbreak" in policy


class TestAnalysisValidator:
    """Test analysis validator."""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return AnalysisValidator()
    
    def test_schema_validation_success(self, validator):
        """Test successful schema validation."""
        valid_output = {
            "reason": "jailbreak detector down",
            "remediation": "restore jailbreak detector",
            "opa_diff": "package policy\npolicy.coverage_violation[route] {\n  required := {\"jailbreak\"}\n  observed := input.coverage[route]\n  required - observed != {}\n}",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage", "detector_errors"]
        }
        
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
        
        result = validator.validate_and_fallback(valid_output, request)
        assert result == valid_output
    
    def test_schema_validation_failure(self, validator):
        """Test schema validation failure with fallback."""
        invalid_output = {
            "reason": "jailbreak detector down",
            # Missing required fields
        }
        
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
        
        result = validator.validate_and_fallback(invalid_output, request)
        assert "_template_fallback" in result
        assert "reason" in result
        assert "remediation" in result


class TestSecurityValidator:
    """Test security validator."""
    
    @pytest.fixture
    def security_validator(self):
        """Create security validator instance."""
        return AnalysisSecurityValidator()
    
    def test_pii_redaction(self, security_validator):
        """Test PII redaction."""
        text_with_pii = "Contact john.doe@example.com or call 555-123-4567"
        redacted = security_validator.redact_pii(text_with_pii)
        
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED_PHONE]" in redacted
        assert "john.doe@example.com" not in redacted
        assert "555-123-4567" not in redacted
    
    def test_response_security_validation(self, security_validator):
        """Test response security validation."""
        response = {
            "reason": "Contact admin@example.com for issues",
            "remediation": "Check logs at /var/log/secret.log",
            "notes": "API key: abc123def456"
        }
        
        redacted_response = security_validator.validate_response_security(response)
        
        assert "[REDACTED_EMAIL]" in redacted_response["reason"]
        assert "[REDACTED_PATH]" in redacted_response["remediation"]
        assert "[REDACTED_TOKEN]" in redacted_response["notes"]


class TestOPAPolicyGenerator:
    """Test OPA policy generator."""
    
    @pytest.fixture
    def opa_generator(self):
        """Create OPA generator instance."""
        return OPAPolicyGenerator()
    
    def test_coverage_policy_generation(self, opa_generator):
        """Test coverage policy generation."""
        required_detectors = ["pii", "jailbreak"]
        required_coverage = {"pii": 0.95, "jailbreak": 0.95}
        
        policy = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
        
        assert "package policy" in policy
        assert "coverage_violation" in policy
        assert "pii" in policy
        assert "jailbreak" in policy
        assert "0.95" in policy
    
    def test_threshold_policy_generation(self, opa_generator):
        """Test threshold policy generation."""
        policy = opa_generator.generate_threshold_policy("pii", 0.75)
        
        assert "package policy" in policy
        assert "allow" in policy
        assert "pii" in policy
        assert "0.75" in policy


class TestQualityEvaluator:
    """Test quality evaluator."""
    
    @pytest.fixture
    def evaluator(self):
        """Create quality evaluator instance."""
        return QualityEvaluator()
    
    @pytest.fixture
    def sample_response(self):
        """Create sample analysis response."""
        return AnalysisResponse(
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
    
    def test_schema_validation(self, evaluator, sample_response):
        """Test schema validation."""
        is_valid = evaluator._validate_schema(sample_response)
        assert is_valid
    
    def test_rubric_scoring(self, evaluator):
        """Test rubric scoring."""
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
        
        score = evaluator._calculate_rubric_score(request, response)
        assert 0.0 <= score <= 5.0
        assert score > 0.0  # Should have some positive score


if __name__ == "__main__":
    pytest.main([__file__])
