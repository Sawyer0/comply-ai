"""
Integration tests for OPA policy generation and validation.

Tests OPA compilation validation in CI pipeline and policy generation accuracy.
"""

import pytest
import tempfile
import os
import subprocess
from unittest.mock import Mock, patch

from ...infrastructure.opa_generator import OPAPolicyGenerator


class TestOPAIntegration:
    """Integration tests for OPA functionality."""
    
    @pytest.fixture
    def opa_generator(self):
        """Create OPA generator for testing."""
        return OPAPolicyGenerator()
    
    def test_opa_installation_check(self):
        """Test that OPA is installed and accessible."""
        try:
            result = subprocess.run(['opa', 'version'], capture_output=True, text=True)
            assert result.returncode == 0
            assert "Open Policy Agent" in result.stdout
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping OPA integration tests")
    
    def test_opa_check_command_validation(self, opa_generator):
        """Test OPA check command validation with real OPA."""
        try:
            # Test with valid policy
            valid_policy = """
            package test
            
            allow {
                input.user == "admin"
            }
            """
            
            result = opa_generator.validate_rego(valid_policy)
            assert result is True
            
            # Test with invalid policy
            invalid_policy = """
            package test
            
            allow {
                invalid syntax here
            }
            """
            
            result = opa_generator.validate_rego(invalid_policy)
            assert result is False
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping OPA check tests")
    
    def test_generated_coverage_policy_compilation(self, opa_generator):
        """Test that generated coverage policies compile correctly."""
        try:
            required_detectors = ["detector1", "detector2", "detector3"]
            required_coverage = {
                "detector1": 0.8,
                "detector2": 0.9,
                "detector3": 0.7
            }
            
            policy = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
            
            # Validate the generated policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
            # Test with sample input
            test_input = {
                "detector1": {"coverage": 0.85},
                "detector2": {"coverage": 0.95},
                "detector3": {"coverage": 0.75}
            }
            
            # Write policy to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
                f.write(policy)
                policy_file = f.name
            
            try:
                # Test policy evaluation
                result = subprocess.run([
                    'opa', 'eval', 
                    '-d', policy_file,
                    'data.test.allow',
                    '--input', f'{{"input": {test_input}}}'
                ], capture_output=True, text=True)
                
                assert result.returncode == 0
                
            finally:
                os.unlink(policy_file)
                
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping policy compilation tests")
    
    def test_generated_threshold_policy_compilation(self, opa_generator):
        """Test that generated threshold policies compile correctly."""
        try:
            detector_name = "test_detector"
            current_threshold = 0.7
            suggested_threshold = 0.9
            
            policy = opa_generator.generate_threshold_policy(
                detector_name, current_threshold, suggested_threshold
            )
            
            # Validate the generated policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
            # Test with sample input
            test_input = {
                "detector": detector_name,
                "current_threshold": current_threshold,
                "suggested_threshold": suggested_threshold
            }
            
            # Write policy to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
                f.write(policy)
                policy_file = f.name
            
            try:
                # Test policy evaluation
                result = subprocess.run([
                    'opa', 'eval', 
                    '-d', policy_file,
                    'data.test.allow',
                    '--input', f'{{"input": {test_input}}}'
                ], capture_output=True, text=True)
                
                assert result.returncode == 0
                
            finally:
                os.unlink(policy_file)
                
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping threshold policy tests")
    
    def test_generated_false_positive_policy_compilation(self, opa_generator):
        """Test that generated false positive policies compile correctly."""
        try:
            detector_name = "test_detector"
            false_positive_rate = 0.15
            suggested_threshold = 0.85
            
            policy = opa_generator.generate_false_positive_policy(
                detector_name, false_positive_rate, suggested_threshold
            )
            
            # Validate the generated policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping false positive policy tests")
    
    def test_generated_incident_response_policy_compilation(self, opa_generator):
        """Test that generated incident response policies compile correctly."""
        try:
            incident_type = "high_severity"
            response_actions = ["escalate", "notify", "quarantine"]
            
            policy = opa_generator.generate_incident_response_policy(
                incident_type, response_actions
            )
            
            # Validate the generated policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping incident response policy tests")
    
    def test_policy_generation_accuracy(self, opa_generator):
        """Test policy generation accuracy with real-world scenarios."""
        # Test coverage gap scenario
        required_detectors = ["pii_detector", "malware_detector", "phishing_detector"]
        required_coverage = {
            "pii_detector": 0.95,
            "malware_detector": 0.90,
            "phishing_detector": 0.85
        }
        
        policy = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
        
        # Check that all detectors are included
        for detector in required_detectors:
            assert detector in policy
        
        # Check that coverage values are included
        for detector, coverage in required_coverage.items():
            assert str(coverage) in policy
        
        # Validate the policy
        is_valid = opa_generator.validate_rego(policy)
        assert is_valid is True
    
    def test_policy_generation_edge_cases(self, opa_generator):
        """Test policy generation with edge cases."""
        try:
            # Test with special characters in detector names
            detector_name = "detector-with-special_chars.and.dots"
            current_threshold = 0.0
            suggested_threshold = 1.0
            
            policy = opa_generator.generate_threshold_policy(
                detector_name, current_threshold, suggested_threshold
            )
            
            assert detector_name in policy
            assert "0.0" in policy
            assert "1.0" in policy
            
            # Validate the policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping edge case tests")
    
    def test_policy_generation_performance(self, opa_generator):
        """Test policy generation performance with large inputs."""
        import time
        
        # Test with many detectors
        required_detectors = [f"detector_{i}" for i in range(100)]
        required_coverage = {detector: 0.8 for detector in required_detectors}
        
        start_time = time.time()
        policy = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 1.0  # Less than 1 second
        
        # Validate the policy
        is_valid = opa_generator.validate_rego(policy)
        assert is_valid is True
    
    def test_policy_generation_deterministic(self, opa_generator):
        """Test that policy generation is deterministic."""
        required_detectors = ["detector1", "detector2"]
        required_coverage = {"detector1": 0.8, "detector2": 0.9}
        
        # Generate policy multiple times
        policy1 = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
        policy2 = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
        policy3 = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
        
        # All policies should be identical
        assert policy1 == policy2 == policy3
    
    def test_policy_generation_with_unicode(self, opa_generator):
        """Test policy generation with unicode characters."""
        try:
            detector_name = "detector_测试_unicode"
            current_threshold = 0.7
            suggested_threshold = 0.9
            
            policy = opa_generator.generate_threshold_policy(
                detector_name, current_threshold, suggested_threshold
            )
            
            assert detector_name in policy
            
            # Validate the policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping unicode tests")
    
    def test_policy_generation_with_complex_inputs(self, opa_generator):
        """Test policy generation with complex input structures."""
        try:
            # Test with complex detector names and coverage values
            required_detectors = [
                "pii-detector-v2.1",
                "malware_detector_advanced",
                "phishing-detector-with-ml"
            ]
            required_coverage = {
                "pii-detector-v2.1": 0.95,
                "malware_detector_advanced": 0.90,
                "phishing-detector-with-ml": 0.85
            }
            
            policy = opa_generator.generate_coverage_policy(required_detectors, required_coverage)
            
            # Check that complex names are handled correctly
            for detector in required_detectors:
                assert detector in policy
            
            # Validate the policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping complex input tests")
    
    def test_policy_generation_with_empty_inputs(self, opa_generator):
        """Test policy generation with empty inputs."""
        try:
            # Test with empty detector list
            policy = opa_generator.generate_coverage_policy([], {})
            
            assert isinstance(policy, str)
            assert len(policy) > 0
            
            # Validate the policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping empty input tests")
    
    def test_policy_generation_with_zero_thresholds(self, opa_generator):
        """Test policy generation with zero thresholds."""
        try:
            detector_name = "test_detector"
            current_threshold = 0.0
            suggested_threshold = 0.0
            
            policy = opa_generator.generate_threshold_policy(
                detector_name, current_threshold, suggested_threshold
            )
            
            assert "0.0" in policy
            
            # Validate the policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping zero threshold tests")
    
    def test_policy_generation_with_maximum_thresholds(self, opa_generator):
        """Test policy generation with maximum thresholds."""
        try:
            detector_name = "test_detector"
            current_threshold = 1.0
            suggested_threshold = 1.0
            
            policy = opa_generator.generate_threshold_policy(
                detector_name, current_threshold, suggested_threshold
            )
            
            assert "1.0" in policy
            
            # Validate the policy
            is_valid = opa_generator.validate_rego(policy)
            assert is_valid is True
            
        except FileNotFoundError:
            pytest.skip("OPA not installed - skipping maximum threshold tests")
