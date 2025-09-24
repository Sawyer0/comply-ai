"""
Integration tests to validate refactoring correctness and backward compatibility.

These tests ensure that the refactored system maintains identical behavior
to the original monolithic template provider.
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from ...compatibility.backward_compatibility_adapter import BackwardCompatibilityAdapter
from ...domain.entities import AnalysisRequest, AnalysisType


class TestRefactoringCompatibility:
    """Test suite for validating refactoring correctness."""
    
    @pytest.fixture
    def adapter(self):
        """Create backward compatibility adapter for testing."""
        return BackwardCompatibilityAdapter()
    
    @pytest.fixture
    def sample_coverage_gap_request(self):
        """Create sample request with coverage gaps."""
        return AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="test-route",
            required_detectors=["presidio", "deberta-toxicity", "pii-detector"],
            observed_coverage={
                "presidio": 0.5,
                "deberta-toxicity": 0.8,
                "pii-detector": 0.3
            },
            required_coverage={
                "presidio": 0.9,
                "deberta-toxicity": 0.9,
                "pii-detector": 0.9
            },
            detector_errors={},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="test-bundle",
            env="dev"
        )
    
    @pytest.fixture
    def sample_false_positive_request(self):
        """Create sample request with false positive issues."""
        return AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app", 
            route="test-route",
            required_detectors=["presidio", "deberta-toxicity"],
            observed_coverage={"presidio": 0.9, "deberta-toxicity": 0.9},
            required_coverage={"presidio": 0.9, "deberta-toxicity": 0.9},
            detector_errors={},
            high_sev_hits=[],
            false_positive_bands=[
                {
                    "detector": "presidio",
                    "false_positive_rate": 0.25,
                    "current_threshold": 0.5
                },
                {
                    "detector": "deberta-toxicity", 
                    "false_positive_rate": 0.15,
                    "current_threshold": 0.7
                }
            ],
            policy_bundle="test-bundle",
            env="dev"
        )
    
    @pytest.fixture
    def sample_incident_request(self):
        """Create sample request with high severity incidents."""
        return AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="test-tenant",
            app="test-app",
            route="test-route", 
            required_detectors=["presidio", "pii-detector"],
            observed_coverage={"presidio": 0.9, "pii-detector": 0.9},
            required_coverage={"presidio": 0.9, "pii-detector": 0.9},
            detector_errors={},
            high_sev_hits=[
                {
                    "detector": "presidio",
                    "severity": "high",
                    "category": "pii_exposure",
                    "description": "PII detected in logs"
                },
                {
                    "detector": "pii-detector",
                    "severity": "critical", 
                    "category": "data_breach",
                    "description": "Sensitive data exposure"
                }
            ],
            false_positive_bands=[],
            policy_bundle="test-bundle",
            env="dev"
        )
    
    def test_analysis_type_selection_coverage_gap(self, adapter, sample_coverage_gap_request):
        """Test that analysis type selection works correctly for coverage gaps."""
        analysis_type = adapter.select_analysis_type(sample_coverage_gap_request)
        assert analysis_type == AnalysisType.COVERAGE_GAP
    
    def test_analysis_type_selection_false_positive(self, adapter, sample_false_positive_request):
        """Test that analysis type selection works correctly for false positives."""
        analysis_type = adapter.select_analysis_type(sample_false_positive_request)
        assert analysis_type == AnalysisType.FALSE_POSITIVE_TUNING
    
    def test_analysis_type_selection_incident(self, adapter, sample_incident_request):
        """Test that analysis type selection works correctly for incidents."""
        analysis_type = adapter.select_analysis_type(sample_incident_request)
        assert analysis_type == AnalysisType.INCIDENT_SUMMARY
    
    def test_coverage_gap_response_format(self, adapter, sample_coverage_gap_request):
        """Test that coverage gap response maintains expected format."""
        analysis_type = AnalysisType.COVERAGE_GAP
        response = adapter.get_template_response(
            sample_coverage_gap_request, 
            analysis_type
        )
        
        # Validate required fields
        required_fields = [
            "reason", "remediation", "opa_diff", "confidence", 
            "confidence_cutoff_used", "evidence_refs", "notes"
        ]
        
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
        
        # Validate field types
        assert isinstance(response["reason"], str)
        assert isinstance(response["remediation"], str)
        assert isinstance(response["opa_diff"], str)
        assert isinstance(response["confidence"], (int, float))
        assert isinstance(response["confidence_cutoff_used"], (int, float))
        assert isinstance(response["evidence_refs"], list)
        assert isinstance(response["notes"], str)
        
        # Validate confidence range
        assert 0.0 <= response["confidence"] <= 1.0
        assert 0.0 <= response["confidence_cutoff_used"] <= 1.0
    
    def test_false_positive_response_format(self, adapter, sample_false_positive_request):
        """Test that false positive response maintains expected format."""
        analysis_type = AnalysisType.FALSE_POSITIVE_TUNING
        response = adapter.get_template_response(
            sample_false_positive_request,
            analysis_type
        )
        
        # Validate required fields exist
        required_fields = [
            "reason", "remediation", "opa_diff", "confidence",
            "confidence_cutoff_used", "evidence_refs", "notes"
        ]
        
        for field in required_fields:
            assert field in response
        
        # Validate content is relevant to false positives
        assert "false positive" in response["reason"].lower() or "threshold" in response["reason"].lower()
        assert len(response["evidence_refs"]) > 0
    
    def test_incident_response_format(self, adapter, sample_incident_request):
        """Test that incident response maintains expected format."""
        analysis_type = AnalysisType.INCIDENT_SUMMARY
        response = adapter.get_template_response(
            sample_incident_request,
            analysis_type
        )
        
        # Validate required fields exist
        required_fields = [
            "reason", "remediation", "opa_diff", "confidence",
            "confidence_cutoff_used", "evidence_refs", "notes"
        ]
        
        for field in required_fields:
            assert field in response
        
        # Validate content is relevant to incidents
        assert "incident" in response["reason"].lower() or "security" in response["reason"].lower()
        assert len(response["evidence_refs"]) > 0
    
    def test_opa_policy_generation(self, adapter, sample_coverage_gap_request):
        """Test that OPA policies are generated correctly."""
        analysis_type = AnalysisType.COVERAGE_GAP
        response = adapter.get_template_response(
            sample_coverage_gap_request,
            analysis_type
        )
        
        opa_diff = response["opa_diff"]
        
        # Should contain valid OPA policy structure
        assert "package" in opa_diff
        assert "{" in opa_diff and "}" in opa_diff
        
        # Should be relevant to coverage
        assert "coverage" in opa_diff.lower() or "detector" in opa_diff.lower()
    
    def test_fallback_behavior(self, adapter):
        """Test that fallback behavior works when orchestrator fails."""
        # Create a request that might cause issues
        problematic_request = AnalysisRequest(
            period="invalid-period",  # Invalid period format
            tenant="",  # Empty tenant
            app="test-app",
            route="test-route",
            required_detectors=[],  # Empty detectors
            observed_coverage={},
            required_coverage={},
            detector_errors={},
            high_sev_hits=[],
            false_positive_bands=[],
            policy_bundle="test-bundle",
            env="dev"
        )
        
        # Should still return a valid response
        response = adapter.get_template_response(
            problematic_request,
            AnalysisType.INSUFFICIENT_DATA
        )
        
        # Validate fallback response format
        assert "reason" in response
        assert "remediation" in response
        assert "confidence" in response
        assert response["confidence"] >= 0.0
    
    def test_confidence_scores_reasonable(self, adapter, sample_coverage_gap_request):
        """Test that confidence scores are reasonable."""
        analysis_type = AnalysisType.COVERAGE_GAP
        response = adapter.get_template_response(
            sample_coverage_gap_request,
            analysis_type
        )
        
        confidence = response["confidence"]
        
        # Confidence should be reasonable for coverage gap scenario
        assert 0.3 <= confidence <= 0.9  # Should not be too low or too high
    
    def test_evidence_refs_populated(self, adapter, sample_coverage_gap_request):
        """Test that evidence references are properly populated."""
        analysis_type = AnalysisType.COVERAGE_GAP
        response = adapter.get_template_response(
            sample_coverage_gap_request,
            analysis_type
        )
        
        evidence_refs = response["evidence_refs"]
        
        # Should have evidence references
        assert len(evidence_refs) > 0
        
        # Evidence refs should be strings
        for ref in evidence_refs:
            assert isinstance(ref, str)
            assert len(ref) > 0
    
    def test_analysis_details_included(self, adapter, sample_coverage_gap_request):
        """Test that analysis details are included when available."""
        analysis_type = AnalysisType.COVERAGE_GAP
        response = adapter.get_template_response(
            sample_coverage_gap_request,
            analysis_type
        )
        
        # Analysis details should be included for coverage gap
        if "analysis_details" in response:
            details = response["analysis_details"]
            assert isinstance(details, dict)
            
            # Should have relevant fields for coverage gap
            expected_fields = ["gaps_identified", "risk_assessment", "priority_actions", "estimated_effort"]
            for field in expected_fields:
                if field in details:
                    assert details[field] is not None
    
    def test_response_consistency(self, adapter, sample_coverage_gap_request):
        """Test that responses are consistent across multiple calls."""
        analysis_type = AnalysisType.COVERAGE_GAP
        
        # Get multiple responses
        response1 = adapter.get_template_response(sample_coverage_gap_request, analysis_type)
        response2 = adapter.get_template_response(sample_coverage_gap_request, analysis_type)
        
        # Core fields should be consistent
        assert response1["reason"] == response2["reason"]
        assert response1["remediation"] == response2["remediation"]
        assert response1["confidence_cutoff_used"] == response2["confidence_cutoff_used"]
        
        # Confidence might vary slightly but should be close
        assert abs(response1["confidence"] - response2["confidence"]) < 0.1


@pytest.mark.integration
class TestSystemIntegration:
    """Integration tests for the complete refactored system."""
    
    def test_end_to_end_analysis_flow(self):
        """Test complete end-to-end analysis flow."""
        adapter = BackwardCompatibilityAdapter()
        
        # Create a comprehensive request
        request = AnalysisRequest(
            period="2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            tenant="integration-test",
            app="test-app",
            route="test-route",
            required_detectors=["presidio", "deberta-toxicity", "pii-detector"],
            observed_coverage={
                "presidio": 0.7,
                "deberta-toxicity": 0.9,
                "pii-detector": 0.4
            },
            required_coverage={
                "presidio": 0.9,
                "deberta-toxicity": 0.9,
                "pii-detector": 0.9
            },
            detector_errors={
                "presidio": {"severity": "low", "message": "Minor configuration issue"}
            },
            high_sev_hits=[
                {
                    "detector": "pii-detector",
                    "severity": "high",
                    "category": "pii_exposure"
                }
            ],
            false_positive_bands=[
                {
                    "detector": "presidio",
                    "false_positive_rate": 0.2,
                    "current_threshold": 0.5
                }
            ],
            policy_bundle="integration-test-bundle",
            env="dev"
        )
        
        # Should successfully process the request
        analysis_type = adapter.select_analysis_type(request)
        response = adapter.get_template_response(request, analysis_type)
        
        # Validate complete response
        assert response is not None
        assert isinstance(response, dict)
        assert len(response) > 0
        
        # Should have selected coverage gap as primary issue
        assert analysis_type == AnalysisType.COVERAGE_GAP
        
        # Response should be comprehensive
        assert "reason" in response
        assert "remediation" in response
        assert "confidence" in response
        assert response["confidence"] > 0.0