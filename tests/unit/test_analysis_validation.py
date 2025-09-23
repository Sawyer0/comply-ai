"""
Unit tests for analysis module input validation and schema management.

Tests the AnalysisRequest model validation, evidence reference validation,
and schema compliance as specified in task 2.
"""

import pytest
from pydantic import ValidationError
from typing import Dict, List, Any

from src.llama_mapper.analysis import (
    AnalysisRequest, 
    AnalysisResponse,
    VersionInfo
)
from src.llama_mapper.analysis.validation.evidence_refs import (
    ALLOWED_EVIDENCE_REFS,
    validate_evidence_refs,
    validate_evidence_refs_against_input
)
from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator


class TestAnalysisRequestValidation:
    """Test AnalysisRequest model validation and field constraints."""
    
    def test_valid_analysis_request(self):
        """Test that a valid analysis request passes validation."""
        request_data = {
            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": ["detector1", "detector2"],
            "observed_coverage": {"detector1": 0.8, "detector2": 0.9},
            "required_coverage": {"detector1": 0.7, "detector2": 0.8},
            "detector_errors": {"detector1": {"5xx": 0}, "detector2": {"5xx": 1}},
            "high_sev_hits": [{"taxonomy": "PII", "score": 0.9}],
            "false_positive_bands": [{"detector": "detector1", "fp_rate": 0.3}],
            "policy_bundle": "test-policy-v1.0",
            "env": "prod"
        }
        
        request = AnalysisRequest(**request_data)
        assert request.tenant == "test-tenant"
        assert request.required_detectors == ["detector1", "detector2"]
        assert request.observed_coverage["detector1"] == 0.8
    
    def test_invalid_period_format(self):
        """Test that invalid period format raises ValidationError."""
        request_data = {
            "period": "invalid-period-format",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": ["detector1"],
            "observed_coverage": {"detector1": 0.8},
            "required_coverage": {"detector1": 0.7},
            "detector_errors": {"detector1": {"5xx": 0}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-policy",
            "env": "prod"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        assert "period" in str(exc_info.value)
    
    def test_coverage_values_bounds_checking(self):
        """Test that coverage values are properly bounds-checked."""
        request_data = {
            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": ["detector1"],
            "observed_coverage": {"detector1": 1.5},  # Invalid: > 1.0
            "required_coverage": {"detector1": -0.1},  # Invalid: < 0.0
            "detector_errors": {"detector1": {"5xx": 0}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-policy",
            "env": "prod"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        error_messages = str(exc_info.value)
        assert "must be between 0.0 and 1.0" in error_messages
    
    def test_detector_keys_consistency(self):
        """Test that coverage keys match required_detectors."""
        request_data = {
            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": ["detector1"],
            "observed_coverage": {"detector1": 0.8, "detector2": 0.9},  # detector2 not in required
            "required_coverage": {"detector1": 0.7},
            "detector_errors": {"detector1": {"5xx": 0}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-policy",
            "env": "prod"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        assert "not in required_detectors list" in str(exc_info.value)
    
    def test_field_length_constraints(self):
        """Test field length constraints are enforced."""
        # Test tenant max length
        request_data = {
            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            "tenant": "a" * 65,  # Exceeds max_length=64
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": ["detector1"],
            "observed_coverage": {"detector1": 0.8},
            "required_coverage": {"detector1": 0.7},
            "detector_errors": {"detector1": {"5xx": 0}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-policy",
            "env": "prod"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        assert "at most 64 characters" in str(exc_info.value) or "String should have at most 64 characters" in str(exc_info.value)
    
    def test_required_detectors_constraints(self):
        """Test required_detectors field constraints."""
        # Test min_items
        request_data = {
            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": [],  # Empty list violates min_items=1
            "observed_coverage": {},
            "required_coverage": {},
            "detector_errors": {},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-policy",
            "env": "prod"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        assert "at least 1 items" in str(exc_info.value) or "List should have at least 1 item" in str(exc_info.value)
        
        # Test max_items
        request_data["required_detectors"] = [f"detector{i}" for i in range(21)]  # 21 items violates max_items=20
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        assert "at most 20 items" in str(exc_info.value) or "List should have at most 20 items" in str(exc_info.value)
    
    def test_environment_enum_validation(self):
        """Test that env field only accepts valid enum values."""
        request_data = {
            "period": "2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            "tenant": "test-tenant",
            "app": "test-app",
            "route": "/api/test",
            "required_detectors": ["detector1"],
            "observed_coverage": {"detector1": 0.8},
            "required_coverage": {"detector1": 0.7},
            "detector_errors": {"detector1": {"5xx": 0}},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-policy",
            "env": "invalid-env"  # Invalid enum value
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisRequest(**request_data)
        
        assert "Input should be 'dev', 'stage' or 'prod'" in str(exc_info.value)


class TestEvidenceReferenceValidation:
    """Test evidence reference validation system."""
    
    def test_allowed_evidence_refs_constant(self):
        """Test that ALLOWED_EVIDENCE_REFS contains expected fields."""
        expected_refs = [
            "required_detectors", "observed_coverage", "required_coverage",
            "detector_errors", "high_sev_hits", "false_positive_bands",
            "policy_bundle", "env", "period", "tenant", "app", "route"
        ]
        
        assert set(ALLOWED_EVIDENCE_REFS) == set(expected_refs)
        assert len(ALLOWED_EVIDENCE_REFS) == len(expected_refs)
    
    def test_evidence_refs_validation_valid(self):
        """Test that valid evidence references pass validation."""
        # Test with valid evidence refs
        evidence_refs = ["observed_coverage", "required_coverage", "detector_errors"]
        
        # Should not raise exception
        is_valid, errors = validate_evidence_refs(evidence_refs)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_evidence_refs_validation_invalid(self):
        """Test that invalid evidence references are detected."""
        # Test with invalid evidence ref
        evidence_refs = ["invalid_ref", "observed_coverage"]
        
        is_valid, errors = validate_evidence_refs(evidence_refs)
        assert is_valid is False
        assert len(errors) > 0
        assert any("invalid_ref" in error for error in errors)
    
    def test_evidence_refs_edge_cases(self):
        """Test evidence reference validation edge cases."""
        # Test empty evidence_refs
        evidence_refs = []
        
        # Empty list should be valid (no invalid refs)
        is_valid, errors = validate_evidence_refs(evidence_refs)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test case sensitivity
        evidence_refs = ["Observed_Coverage"]  # Wrong case
        
        is_valid, errors = validate_evidence_refs(evidence_refs)
        assert is_valid is False
        assert len(errors) > 0
        assert any("Observed_Coverage" in error for error in errors)


class TestSchemaValidation:
    """Test JSON schema validation and fallback system."""
    
    def test_schema_validation_success(self):
        """Test successful schema validation."""
        validator = AnalysisValidator()
        
        valid_output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage", "required_coverage"],
            "opa_diff": "package policy\n\nallow { input.score > 0.7 }"
        }
        
        result = validator.validate_schema_compliance(valid_output)
        assert result is True
    
    def test_schema_validation_failure(self):
        """Test schema validation failure handling."""
        validator = AnalysisValidator()
        
        # Missing required field
        invalid_output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            # Missing evidence_refs
            "opa_diff": ""
        }
        
        result = validator.validate_schema_compliance(invalid_output)
        assert result is False
    
    def test_field_constraints_validation(self):
        """Test additional field constraints validation."""
        validator = AnalysisValidator()
        
        # Test word count constraint (reason > 20 words)
        output = {
            "reason": " ".join(["word"] * 21),  # 21 words
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": ""
        }
        
        with pytest.raises(Exception) as exc_info:
            validator._validate_field_constraints(output)
        
        assert "must be ≤20 words" in str(exc_info.value)
        
        # Test confidence range
        output["reason"] = "coverage gap detected"
        output["confidence"] = 1.5  # Invalid: > 1.0
        
        with pytest.raises(Exception) as exc_info:
            validator._validate_field_constraints(output)
        
        assert "must be between 0.0 and 1.0" in str(exc_info.value)
    
    def test_validate_and_fallback_success(self):
        """Test validate_and_fallback with successful validation."""
        validator = AnalysisValidator()
        
        valid_output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage", "required_coverage"],
            "opa_diff": ""
        }
        
        request = AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            tenant="test-tenant",
            app="test-app",
            route="/api/test",
            required_detectors=["detector1"],
            observed_coverage={"detector1": 0.8},
            required_coverage={"detector1": 0.7},
            detector_errors={"detector1": {"5xx": 0}},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="test-policy",
            env="prod"
        )
        
        result = validator.validate_and_fallback(valid_output, request)
        assert result == valid_output
        assert "_template_fallback" not in result
    
    def test_validate_and_fallback_failure(self):
        """Test validate_and_fallback with validation failure."""
        validator = AnalysisValidator()
        
        invalid_output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["invalid_ref"],  # Invalid evidence ref
            "opa_diff": ""
        }
        
        request = AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-01T23:59:59Z",
            tenant="test-tenant",
            app="test-app",
            route="/api/test",
            required_detectors=["detector1"],
            observed_coverage={"detector1": 0.8},
            required_coverage={"detector1": 0.7},
            detector_errors={"detector1": {"5xx": 0}},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="test-policy",
            env="prod"
        )
        
        result = validator.validate_and_fallback(invalid_output, request)
        
        # Should return template fallback
        assert "_template_fallback" in result
        assert result["_template_fallback"] is True
        assert "_fallback_reason" in result
        assert "Invalid evidence reference" in result["_fallback_reason"]
    
    def test_get_validation_errors(self):
        """Test getting detailed validation errors."""
        validator = AnalysisValidator()
        
        invalid_output = {
            "reason": " ".join(["word"] * 21),  # Too many words
            "remediation": "add secondary detector",
            "confidence": 1.5,  # Invalid confidence
            "evidence_refs": ["invalid_ref"],  # Invalid evidence ref
            "opa_diff": ""
        }
        
        errors = validator.get_validation_errors(invalid_output)
        
        assert len(errors) > 0
        assert any("must be ≤20 words" in error for error in errors)
        assert any("greater than the maximum of 1.0" in error for error in errors)
        assert any("Invalid evidence reference" in error for error in errors)


class TestAnalysisResponseValidation:
    """Test AnalysisResponse model validation."""
    
    def test_valid_analysis_response(self):
        """Test that a valid analysis response passes validation."""
        version_info = VersionInfo(
            taxonomy="v1.0",
            frameworks="SOC2-v2.0",
            analyst_model="phi3-mini-v1.0"
        )
        
        response_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "opa_diff": "package policy\n\nallow { input.score > 0.7 }",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["observed_coverage", "required_coverage"],
            "notes": "Analysis based on coverage metrics",
            "version_info": version_info,
            "processing_time_ms": 150
        }
        
        response = AnalysisResponse(**response_data)
        assert response.reason == "coverage gap detected"
        assert response.confidence == 0.8
        assert response.version_info.taxonomy == "v1.0"
    
    def test_response_field_constraints(self):
        """Test AnalysisResponse field constraints."""
        version_info = VersionInfo(
            taxonomy="v1.0",
            frameworks="SOC2-v2.0",
            analyst_model="phi3-mini-v1.0"
        )
        
        # Test reason max length
        response_data = {
            "reason": "a" * 121,  # Exceeds max_length=120
            "remediation": "add secondary detector",
            "opa_diff": "",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["observed_coverage"],
            "notes": "Analysis notes",
            "version_info": version_info,
            "processing_time_ms": 150
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse(**response_data)
        
        assert "at most 120 characters" in str(exc_info.value)
        
        # Test confidence range
        response_data["reason"] = "coverage gap detected"
        response_data["confidence"] = 1.5  # Invalid: > 1.0
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse(**response_data)
        
        assert "less than or equal to 1.0" in str(exc_info.value) or "Input should be less than or equal to 1" in str(exc_info.value)
    
    def test_evidence_refs_min_items(self):
        """Test that evidence_refs requires at least 1 item."""
        version_info = VersionInfo(
            taxonomy="v1.0",
            frameworks="SOC2-v2.0",
            analyst_model="phi3-mini-v1.0"
        )
        
        response_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "opa_diff": "",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": [],  # Empty list violates min_items=1
            "notes": "Analysis notes",
            "version_info": version_info,
            "processing_time_ms": 150
        }
        
        with pytest.raises(ValidationError) as exc_info:
            AnalysisResponse(**response_data)
        
        assert "at least 1 items" in str(exc_info.value) or "List should have at least 1 item" in str(exc_info.value)
