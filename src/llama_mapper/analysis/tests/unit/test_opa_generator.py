"""
Unit tests for the OPA policy generator.

Tests policy generation, Rego validation, and compilation checking.
"""

import os
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest

from ...infrastructure.opa_generator import OPAPolicyGenerator


class TestOPAPolicyGenerator:
    """Test cases for OPAPolicyGenerator."""

    @pytest.fixture
    def opa_generator(self):
        """Create an OPA generator instance for testing."""
        return OPAPolicyGenerator()

    def test_initialization(self, opa_generator):
        """Test OPA generator initialization."""
        assert opa_generator is not None

    def test_validate_rego_valid_policy(self, opa_generator):
        """Test validation with valid Rego policy."""
        valid_policy = """
        package test
        
        allow {
            input.user == "admin"
        }
        """

        result = opa_generator.validate_rego(valid_policy)

        assert result is True

    def test_validate_rego_invalid_policy(self, opa_generator):
        """Test validation with invalid Rego policy."""
        invalid_policy = """
        package test
        
        allow {
            invalid syntax here
        }
        """

        result = opa_generator.validate_rego(invalid_policy)

        assert result is False

    def test_validate_rego_empty_policy(self, opa_generator):
        """Test validation with empty policy."""
        empty_policy = ""

        result = opa_generator.validate_rego(empty_policy)

        assert result is False

    def test_validate_rego_malformed_policy(self, opa_generator):
        """Test validation with malformed policy."""
        malformed_policy = "this is not valid rego syntax at all"

        result = opa_generator.validate_rego(malformed_policy)

        assert result is False

    @patch("subprocess.run")
    def test_validate_rego_subprocess_success(self, mock_run, opa_generator):
        """Test validation with successful subprocess execution."""
        mock_run.return_value.returncode = 0

        policy = "package test\n\nallow { true }"
        result = opa_generator.validate_rego(policy)

        assert result is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_validate_rego_subprocess_failure(self, mock_run, opa_generator):
        """Test validation with failed subprocess execution."""
        mock_run.return_value.returncode = 1

        policy = "package test\n\ninvalid syntax"
        result = opa_generator.validate_rego(policy)

        assert result is False
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_validate_rego_subprocess_exception(self, mock_run, opa_generator):
        """Test validation with subprocess exception."""
        mock_run.side_effect = Exception("Subprocess error")

        policy = "package test\n\nallow { true }"
        result = opa_generator.validate_rego(policy)

        assert result is False

    def test_generate_coverage_policy(self, opa_generator):
        """Test coverage policy generation."""
        required_detectors = ["detector1", "detector2", "detector3"]
        required_coverage = {"detector1": 0.8, "detector2": 0.9, "detector3": 0.7}

        policy = opa_generator.generate_coverage_policy(
            required_detectors, required_coverage
        )

        assert isinstance(policy, str)
        assert len(policy) > 0
        assert "package" in policy
        assert "detector1" in policy
        assert "detector2" in policy
        assert "detector3" in policy
        assert "0.8" in policy
        assert "0.9" in policy
        assert "0.7" in policy

    def test_generate_coverage_policy_structure(self, opa_generator):
        """Test coverage policy structure."""
        required_detectors = ["detector1"]
        required_coverage = {"detector1": 0.8}

        policy = opa_generator.generate_coverage_policy(
            required_detectors, required_coverage
        )

        # Check for required Rego elements
        assert "package" in policy
        assert "allow" in policy
        assert "input" in policy
        assert "coverage" in policy.lower()

    def test_generate_coverage_policy_deterministic(self, opa_generator):
        """Test that coverage policy generation is deterministic."""
        required_detectors = ["detector1", "detector2"]
        required_coverage = {"detector1": 0.8, "detector2": 0.9}

        policy1 = opa_generator.generate_coverage_policy(
            required_detectors, required_coverage
        )
        policy2 = opa_generator.generate_coverage_policy(
            required_detectors, required_coverage
        )

        assert policy1 == policy2

    def test_generate_threshold_policy(self, opa_generator):
        """Test threshold policy generation."""
        detector_name = "test_detector"
        current_threshold = 0.7
        suggested_threshold = 0.9

        policy = opa_generator.generate_threshold_policy(
            detector_name, current_threshold, suggested_threshold
        )

        assert isinstance(policy, str)
        assert len(policy) > 0
        assert "package" in policy
        assert detector_name in policy
        assert str(current_threshold) in policy
        assert str(suggested_threshold) in policy

    def test_generate_threshold_policy_structure(self, opa_generator):
        """Test threshold policy structure."""
        detector_name = "test_detector"
        current_threshold = 0.7
        suggested_threshold = 0.9

        policy = opa_generator.generate_threshold_policy(
            detector_name, current_threshold, suggested_threshold
        )

        # Check for required Rego elements
        assert "package" in policy
        assert "allow" in policy
        assert "input" in policy
        assert "threshold" in policy.lower()

    def test_generate_threshold_policy_deterministic(self, opa_generator):
        """Test that threshold policy generation is deterministic."""
        detector_name = "test_detector"
        current_threshold = 0.7
        suggested_threshold = 0.9

        policy1 = opa_generator.generate_threshold_policy(
            detector_name, current_threshold, suggested_threshold
        )
        policy2 = opa_generator.generate_threshold_policy(
            detector_name, current_threshold, suggested_threshold
        )

        assert policy1 == policy2

    def test_generate_false_positive_policy(self, opa_generator):
        """Test false positive policy generation."""
        detector_name = "test_detector"
        false_positive_rate = 0.15
        suggested_threshold = 0.85

        policy = opa_generator.generate_false_positive_policy(
            detector_name, false_positive_rate, suggested_threshold
        )

        assert isinstance(policy, str)
        assert len(policy) > 0
        assert "package" in policy
        assert detector_name in policy
        assert str(false_positive_rate) in policy
        assert str(suggested_threshold) in policy

    def test_generate_false_positive_policy_structure(self, opa_generator):
        """Test false positive policy structure."""
        detector_name = "test_detector"
        false_positive_rate = 0.15
        suggested_threshold = 0.85

        policy = opa_generator.generate_false_positive_policy(
            detector_name, false_positive_rate, suggested_threshold
        )

        # Check for required Rego elements
        assert "package" in policy
        assert "allow" in policy
        assert "input" in policy
        assert "false" in policy.lower()
        assert "positive" in policy.lower()

    def test_generate_incident_response_policy(self, opa_generator):
        """Test incident response policy generation."""
        incident_type = "high_severity"
        response_actions = ["escalate", "notify", "quarantine"]

        policy = opa_generator.generate_incident_response_policy(
            incident_type, response_actions
        )

        assert isinstance(policy, str)
        assert len(policy) > 0
        assert "package" in policy
        assert incident_type in policy
        for action in response_actions:
            assert action in policy

    def test_generate_incident_response_policy_structure(self, opa_generator):
        """Test incident response policy structure."""
        incident_type = "high_severity"
        response_actions = ["escalate", "notify"]

        policy = opa_generator.generate_incident_response_policy(
            incident_type, response_actions
        )

        # Check for required Rego elements
        assert "package" in policy
        assert "allow" in policy
        assert "input" in policy
        assert "incident" in policy.lower()

    def test_generate_policy_with_validation(self, opa_generator):
        """Test policy generation with automatic validation."""
        required_detectors = ["detector1"]
        required_coverage = {"detector1": 0.8}

        policy = opa_generator.generate_coverage_policy(
            required_detectors, required_coverage
        )

        # The generated policy should be valid
        is_valid = opa_generator.validate_rego(policy)
        assert is_valid is True

    def test_generate_policy_edge_cases(self, opa_generator):
        """Test policy generation with edge cases."""
        # Empty detector list
        policy1 = opa_generator.generate_coverage_policy([], {})
        assert isinstance(policy1, str)
        assert len(policy1) > 0

        # Single detector
        policy2 = opa_generator.generate_coverage_policy(
            ["detector1"], {"detector1": 1.0}
        )
        assert isinstance(policy2, str)
        assert "detector1" in policy2
        assert "1.0" in policy2

        # Zero threshold
        policy3 = opa_generator.generate_threshold_policy("detector1", 0.0, 0.0)
        assert isinstance(policy3, str)
        assert "0.0" in policy3

    def test_generate_policy_special_characters(self, opa_generator):
        """Test policy generation with special characters in detector names."""
        detector_name = "detector-with-dashes_and_underscores"
        current_threshold = 0.7
        suggested_threshold = 0.9

        policy = opa_generator.generate_threshold_policy(
            detector_name, current_threshold, suggested_threshold
        )

        assert isinstance(policy, str)
        assert detector_name in policy

    def test_generate_policy_unicode_support(self, opa_generator):
        """Test policy generation with unicode characters."""
        detector_name = "detector_测试"
        current_threshold = 0.7
        suggested_threshold = 0.9

        policy = opa_generator.generate_threshold_policy(
            detector_name, current_threshold, suggested_threshold
        )

        assert isinstance(policy, str)
        assert detector_name in policy
