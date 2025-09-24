"""
Unit tests for the analysis validator.

Tests schema validation, template fallback logic, and evidence reference validation.
"""

import json
from unittest.mock import Mock, patch

import pytest

from ...domain.entities import AnalysisRequest, AnalysisType
from ...infrastructure.validator import AnalysisValidator
from ...validation.evidence_refs import ALLOWED_EVIDENCE_REFS, validate_evidence_refs


class TestAnalysisValidator:
    """Test cases for AnalysisValidator."""

    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing."""
        return AnalysisValidator()

    @pytest.fixture
    def sample_request(self):
        """Create a sample analysis request."""
        return AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="test-route",
            required_detectors=["detector1", "detector2"],
            observed_coverage={"detector1": 0.8, "detector2": 0.9},
            required_coverage={"detector1": 0.7, "detector2": 0.8},
            detector_errors={"detector1": {"error": "test"}},
            high_sev_hits=[{"hit": "test"}],
            false_positive_bands=[{"band": "test"}],
            policy_bundle="test-bundle",
            env="dev",
        )

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert validator.json_validator is not None

    def test_validate_and_fallback_valid_output(self, validator, sample_request):
        """Test validation with valid output."""
        valid_output = {
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1", "detector2"],
            "notes": "Test notes",
        }

        with patch.object(
            validator.json_validator, "validate", return_value=(True, [])
        ):
            result = validator.validate_and_fallback(valid_output, sample_request)

            assert result == valid_output

    def test_validate_and_fallback_invalid_output(self, validator, sample_request):
        """Test validation with invalid output triggers fallback."""
        invalid_output = {
            "reason": "Test reason",
            "remediation": "Test remediation",
            # Missing required fields
        }

        with patch.object(
            validator.json_validator,
            "validate",
            return_value=(False, ["Missing required field: opa_diff"]),
        ):
            result = validator.validate_and_fallback(invalid_output, sample_request)

            assert result != invalid_output
            assert "reason" in result
            assert "remediation" in result
            assert "opa_diff" in result
            assert "confidence" in result
            assert "confidence_cutoff_used" in result
            assert "evidence_refs" in result
            assert "notes" in result

    def test_validate_and_fallback_exception(self, validator, sample_request):
        """Test validation with exception triggers fallback."""
        invalid_output = {"invalid": "data"}

        with patch.object(
            validator.json_validator,
            "validate",
            side_effect=Exception("Validation error"),
        ):
            result = validator.validate_and_fallback(invalid_output, sample_request)

            assert "reason" in result
            assert "remediation" in result
            assert "opa_diff" in result
            assert "confidence" in result
            assert "confidence_cutoff_used" in result
            assert "evidence_refs" in result
            assert "notes" in result

    def test_get_template_fallback_coverage_gap(self, validator, sample_request):
        """Test template fallback for coverage gap analysis."""
        sample_request.observed_coverage = {"detector1": 0.5}
        sample_request.required_coverage = {"detector1": 0.8}

        result = validator._get_template_fallback(sample_request, "Test error")

        assert "reason" in result
        assert "remediation" in result
        assert "opa_diff" in result
        assert "confidence" in result
        assert "confidence_cutoff_used" in result
        assert "evidence_refs" in result
        assert "notes" in result
        assert "coverage" in result["reason"].lower()

    def test_get_template_fallback_false_positive(self, validator, sample_request):
        """Test template fallback for false positive tuning."""
        sample_request.false_positive_bands = [{"band": "high"}]
        sample_request.high_sev_hits = []
        sample_request.detector_errors = {}

        result = validator._get_template_fallback(sample_request, "Test error")

        assert "reason" in result
        assert "remediation" in result
        assert "opa_diff" in result
        assert "confidence" in result
        assert "confidence_cutoff_used" in result
        assert "evidence_refs" in result
        assert "notes" in result
        assert "false positive" in result["reason"].lower()

    def test_get_template_fallback_incident_summary(self, validator, sample_request):
        """Test template fallback for incident summary."""
        sample_request.high_sev_hits = [{"severity": "high"}]
        sample_request.detector_errors = {"detector1": {"error": "timeout"}}

        result = validator._get_template_fallback(sample_request, "Test error")

        assert "reason" in result
        assert "remediation" in result
        assert "opa_diff" in result
        assert "confidence" in result
        assert "confidence_cutoff_used" in result
        assert "evidence_refs" in result
        assert "notes" in result
        assert "incident" in result["reason"].lower()

    def test_get_template_fallback_insufficient_data(self, validator, sample_request):
        """Test template fallback for insufficient data."""
        sample_request.observed_coverage = {}
        sample_request.required_coverage = {}
        sample_request.high_sev_hits = []
        sample_request.detector_errors = {}
        sample_request.false_positive_bands = []

        result = validator._get_template_fallback(sample_request, "Test error")

        assert "reason" in result
        assert "remediation" in result
        assert "opa_diff" in result
        assert "confidence" in result
        assert "confidence_cutoff_used" in result
        assert "evidence_refs" in result
        assert "notes" in result
        assert "insufficient" in result["reason"].lower()

    def test_template_fallback_field_constraints(self, validator, sample_request):
        """Test that template fallback respects field constraints."""
        result = validator._get_template_fallback(sample_request, "Test error")

        # Check field length constraints
        assert len(result["reason"]) <= 120
        assert len(result["remediation"]) <= 120

        # Check confidence range
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["confidence_cutoff_used"] <= 1.0

        # Check evidence_refs is a list
        assert isinstance(result["evidence_refs"], list)

    def test_template_fallback_deterministic(self, validator, sample_request):
        """Test that template fallback is deterministic."""
        result1 = validator._get_template_fallback(sample_request, "Test error")
        result2 = validator._get_template_fallback(sample_request, "Test error")

        assert result1 == result2

    def test_template_fallback_includes_error_info(self, validator, sample_request):
        """Test that template fallback includes error information."""
        error_message = "Schema validation failed: Missing required field"
        result = validator._get_template_fallback(sample_request, error_message)

        assert error_message in result["notes"]


class TestEvidenceRefsValidation:
    """Test cases for evidence reference validation."""

    def test_allowed_evidence_refs_constant(self):
        """Test that ALLOWED_EVIDENCE_REFS contains expected fields."""
        expected_fields = {
            "period",
            "tenant",
            "app",
            "route",
            "required_detectors",
            "observed_coverage",
            "required_coverage",
            "detector_errors",
            "high_sev_hits",
            "false_positive_bands",
            "policy_bundle",
            "env",
        }

        assert ALLOWED_EVIDENCE_REFS == expected_fields

    def test_validate_evidence_refs_valid(self):
        """Test validation with valid evidence references."""
        valid_refs = [
            "detector1",
            "detector2",
            "observed_coverage",
            "required_coverage",
        ]

        is_valid, errors = validate_evidence_refs(valid_refs)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_evidence_refs_invalid(self):
        """Test validation with invalid evidence references."""
        invalid_refs = ["detector1", "invalid_field", "another_invalid_field"]

        is_valid, errors = validate_evidence_refs(invalid_refs)

        assert is_valid is False
        assert len(errors) == 2
        assert "invalid_field" in errors[0]
        assert "another_invalid_field" in errors[1]

    def test_validate_evidence_refs_mixed(self):
        """Test validation with mixed valid and invalid references."""
        mixed_refs = ["detector1", "invalid_field", "observed_coverage"]

        is_valid, errors = validate_evidence_refs(mixed_refs)

        assert is_valid is False
        assert len(errors) == 1
        assert "invalid_field" in errors[0]

    def test_validate_evidence_refs_empty(self):
        """Test validation with empty evidence references."""
        empty_refs = []

        is_valid, errors = validate_evidence_refs(empty_refs)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_evidence_refs_non_string(self):
        """Test validation with non-string evidence references."""
        non_string_refs = ["detector1", 123, {"invalid": "type"}]

        is_valid, errors = validate_evidence_refs(non_string_refs)

        assert is_valid is False
        assert len(errors) == 2
        assert "int" in errors[0]
        assert "dict" in errors[1]

    def test_validate_evidence_refs_against_input_valid(self):
        """Test validation against input data with valid references."""
        from ...validation.evidence_refs import validate_evidence_refs_against_input

        evidence_refs = ["detector1", "detector2"]
        input_data = {
            "detector1": "value1",
            "detector2": "value2",
            "other_field": "value3",
        }

        is_valid, errors = validate_evidence_refs_against_input(
            evidence_refs, input_data
        )

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_evidence_refs_against_input_missing(self):
        """Test validation against input data with missing references."""
        from ...validation.evidence_refs import validate_evidence_refs_against_input

        evidence_refs = ["detector1", "missing_field"]
        input_data = {"detector1": "value1", "other_field": "value3"}

        is_valid, errors = validate_evidence_refs_against_input(
            evidence_refs, input_data
        )

        assert is_valid is False
        assert len(errors) == 1
        assert "missing_field" in errors[0]

    def test_extract_field_references(self):
        """Test extraction of field names from evidence references."""
        from ...validation.evidence_refs import extract_field_references

        evidence_refs = ["detector1", "invalid_field", "observed_coverage"]

        referenced_fields = extract_field_references(evidence_refs)

        expected_fields = {"detector1", "observed_coverage"}
        assert referenced_fields == expected_fields
