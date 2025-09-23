"""
Unit tests for analysis module JSON schema validation.

Tests the AnalystInput.schema.json file and schema validation edge cases
as specified in task 2.1.
"""

import json
import pytest
from pathlib import Path
from jsonschema import Draft202012Validator, ValidationError

from src.llama_mapper.analysis.models import AnalysisRequest
from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator


class TestAnalystInputSchema:
    """Test the AnalystInput.schema.json file validation."""
    
    def test_schema_file_exists(self):
        """Test that the schema file exists and is valid JSON."""
        schema_path = Path("schemas/analyst_output.json")
        assert schema_path.exists(), "Schema file should exist"
        
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Validate it's a proper JSON schema
        assert "$schema" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
    
    def test_schema_required_fields(self):
        """Test that schema defines all required fields."""
        schema_path = Path("schemas/analyst_output.json")
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        required_fields = schema["required"]
        expected_required = [
            "reason", "remediation", "confidence", "evidence_refs", "opa_diff"
        ]
        
        assert set(required_fields) == set(expected_required)
    
    def test_schema_field_constraints(self):
        """Test that schema defines proper field constraints."""
        schema_path = Path("schemas/analyst_output.json")
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        properties = schema["properties"]
        
        # Test reason field constraints
        assert "reason" in properties
        assert properties["reason"]["type"] == "string"
        assert properties["reason"]["maxLength"] == 120
        
        # Test remediation field constraints
        assert "remediation" in properties
        assert properties["remediation"]["type"] == "string"
        assert properties["remediation"]["maxLength"] == 120
        
        # Test confidence field constraints
        assert "confidence" in properties
        assert properties["confidence"]["type"] == "number"
        assert properties["confidence"]["minimum"] == 0.0
        assert properties["confidence"]["maximum"] == 1.0
        
        # Test evidence_refs field constraints
        assert "evidence_refs" in properties
        assert properties["evidence_refs"]["type"] == "array"
        assert properties["evidence_refs"]["minItems"] == 1
        assert properties["evidence_refs"]["items"]["type"] == "string"
        
        # Test opa_diff field constraints
        assert "opa_diff" in properties
        assert properties["opa_diff"]["type"] == "string"
        assert properties["opa_diff"]["maxLength"] == 2000
        
        # Test notes field constraints (optional)
        assert "notes" in properties
        assert properties["notes"]["type"] == "string"
        assert properties["notes"]["maxLength"] == 500
    
    def test_schema_validation_with_valid_data(self):
        """Test schema validation with valid data."""
        validator = AnalysisValidator()
        
        valid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage", "required_coverage"],
            "opa_diff": "package policy\n\nallow { input.score > 0.7 }",
            "notes": "Analysis based on coverage metrics"
        }
        
        # Should not raise exception
        validator.validator.validate(valid_data)
    
    def test_schema_validation_with_invalid_data(self):
        """Test schema validation with invalid data."""
        validator = AnalysisValidator()
        
        # Missing required field
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            # Missing evidence_refs
            "opa_diff": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "evidence_refs" in str(exc_info.value)
    
    def test_schema_field_type_validation(self):
        """Test schema field type validation."""
        validator = AnalysisValidator()
        
        # Wrong type for confidence
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": "0.8",  # String instead of number
            "evidence_refs": ["observed_coverage"],
            "opa_diff": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "confidence" in str(exc_info.value)
        assert "string" in str(exc_info.value)
    
    def test_schema_field_length_validation(self):
        """Test schema field length validation."""
        validator = AnalysisValidator()
        
        # Reason too long
        invalid_data = {
            "reason": "a" * 121,  # Exceeds maxLength=120
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "reason" in str(exc_info.value)
        assert "121" in str(exc_info.value)
    
    def test_schema_confidence_range_validation(self):
        """Test schema confidence range validation."""
        validator = AnalysisValidator()
        
        # Confidence out of range
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 1.5,  # Exceeds maximum=1.0
            "evidence_refs": ["observed_coverage"],
            "opa_diff": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "confidence" in str(exc_info.value)
        assert "1.5" in str(exc_info.value)
    
    def test_schema_evidence_refs_validation(self):
        """Test schema evidence_refs validation."""
        validator = AnalysisValidator()
        
        # Empty evidence_refs array
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": [],  # Violates minItems=1
            "opa_diff": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "evidence_refs" in str(exc_info.value)
        assert "minItems" in str(exc_info.value)
    
    def test_schema_opa_diff_length_validation(self):
        """Test schema opa_diff length validation."""
        validator = AnalysisValidator()
        
        # OPA diff too long
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": "a" * 2001  # Exceeds maxLength=2000
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "opa_diff" in str(exc_info.value)
        assert "2001" in str(exc_info.value)
    
    def test_schema_notes_length_validation(self):
        """Test schema notes length validation."""
        validator = AnalysisValidator()
        
        # Notes too long
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": "",
            "notes": "a" * 501  # Exceeds maxLength=500
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "notes" in str(exc_info.value)
        assert "501" in str(exc_info.value)
    
    def test_schema_additional_properties_validation(self):
        """Test that schema rejects additional properties."""
        validator = AnalysisValidator()
        
        # Extra field not in schema
        invalid_data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": "",
            "extra_field": "not allowed"  # Additional property
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validator.validate(invalid_data)
        
        assert "extra_field" in str(exc_info.value)
        assert "additional properties" in str(exc_info.value)


class TestSchemaEdgeCases:
    """Test schema validation edge cases."""
    
    def test_empty_string_fields(self):
        """Test validation with empty string fields."""
        validator = AnalysisValidator()
        
        data = {
            "reason": "",  # Empty string
            "remediation": "",  # Empty string
            "confidence": 0.0,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": ""  # Empty string
        }
        
        # Should pass validation (empty strings are valid)
        validator.validator.validate(data)
    
    def test_boundary_values(self):
        """Test validation with boundary values."""
        validator = AnalysisValidator()
        
        # Test minimum confidence
        data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.0,  # Minimum value
            "evidence_refs": ["observed_coverage"],
            "opa_diff": ""
        }
        validator.validator.validate(data)
        
        # Test maximum confidence
        data["confidence"] = 1.0  # Maximum value
        validator.validator.validate(data)
        
        # Test maximum length strings
        data["reason"] = "a" * 120  # Maximum length
        data["remediation"] = "a" * 120  # Maximum length
        validator.validator.validate(data)
    
    def test_unicode_strings(self):
        """Test validation with unicode strings."""
        validator = AnalysisValidator()
        
        data = {
            "reason": "coverage gap detected with unicode: 🚨",
            "remediation": "add secondary detector with unicode: ✅",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": "package policy\n\n# Unicode comment: 🔒\nallow { input.score > 0.7 }"
        }
        
        # Should pass validation
        validator.validator.validate(data)
    
    def test_evidence_refs_with_special_characters(self):
        """Test evidence_refs with special characters."""
        validator = AnalysisValidator()
        
        data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage", "required_coverage", "detector_errors"],
            "opa_diff": ""
        }
        
        # Should pass validation
        validator.validator.validate(data)
    
    def test_opa_diff_with_complex_rego(self):
        """Test opa_diff with complex Rego code."""
        validator = AnalysisValidator()
        
        complex_rego = """
package policy

import rego.v1

default allow := false

allow if {
    input.score >= 0.7
    input.detector in ["detector1", "detector2"]
    count(input.evidence) > 0
}

violation[msg] {
    not allow
    msg := "Access denied: insufficient confidence"
}
"""
        
        data = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": ["observed_coverage"],
            "opa_diff": complex_rego
        }
        
        # Should pass validation
        validator.validator.validate(data)
