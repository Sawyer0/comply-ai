"""
Unit tests for the analysis model server.

Tests prompt generation consistency, confidence computation, and model inference.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ...domain.entities import AnalysisRequest, AnalysisType, VersionInfo
from ...infrastructure.model_server import Phi3AnalysisModelServer


class TestPhi3AnalysisModelServer:
    """Test cases for Phi3AnalysisModelServer."""
    
    @pytest.fixture
    def model_server(self):
        """Create a model server instance for testing."""
        return Phi3AnalysisModelServer(
            model_path="test-model-path",
            temperature=0.1,
            confidence_cutoff=0.3
        )
    
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
            env="dev"
        )
    
    def test_initialization(self, model_server):
        """Test model server initialization."""
        assert model_server.model_path == "test-model-path"
        assert model_server.temperature == 0.1
        assert model_server.confidence_cutoff == 0.3
        assert model_server.version_info is not None
        assert isinstance(model_server.version_info, VersionInfo)
    
    def test_version_info(self, model_server):
        """Test version info structure."""
        version_info = model_server.version_info
        assert hasattr(version_info, 'taxonomy')
        assert hasattr(version_info, 'frameworks')
        assert hasattr(version_info, 'analyst_model')
        assert version_info.analyst_model == "phi3-mini-3.8b"
    
    def test_determine_analysis_type_coverage_gap(self, model_server, sample_request):
        """Test analysis type determination for coverage gap."""
        # Modify request to indicate coverage gap
        sample_request.observed_coverage = {"detector1": 0.5, "detector2": 0.6}
        sample_request.required_coverage = {"detector1": 0.8, "detector2": 0.9}
        
        analysis_type = model_server._determine_analysis_type(sample_request)
        assert analysis_type == AnalysisType.COVERAGE_GAP
    
    def test_determine_analysis_type_false_positive(self, model_server, sample_request):
        """Test analysis type determination for false positive tuning."""
        # Modify request to indicate false positive tuning
        sample_request.false_positive_bands = [{"band": "high", "rate": 0.8}]
        sample_request.high_sev_hits = []
        sample_request.detector_errors = {}
        
        analysis_type = model_server._determine_analysis_type(sample_request)
        assert analysis_type == AnalysisType.FALSE_POSITIVE_TUNING
    
    def test_determine_analysis_type_incident_summary(self, model_server, sample_request):
        """Test analysis type determination for incident summary."""
        # Modify request to indicate incident summary
        sample_request.high_sev_hits = [{"severity": "high", "count": 10}]
        sample_request.detector_errors = {"detector1": {"error": "timeout"}}
        
        analysis_type = model_server._determine_analysis_type(sample_request)
        assert analysis_type == AnalysisType.INCIDENT_SUMMARY
    
    def test_determine_analysis_type_insufficient_data(self, model_server, sample_request):
        """Test analysis type determination for insufficient data."""
        # Modify request to indicate insufficient data
        sample_request.observed_coverage = {}
        sample_request.required_coverage = {}
        sample_request.high_sev_hits = []
        sample_request.detector_errors = {}
        sample_request.false_positive_bands = []
        
        analysis_type = model_server._determine_analysis_type(sample_request)
        assert analysis_type == AnalysisType.INSUFFICIENT_DATA
    
    def test_build_coverage_gap_prompt(self, model_server, sample_request):
        """Test coverage gap prompt building."""
        prompt = model_server._build_coverage_gap_prompt(sample_request)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "coverage gap" in prompt.lower()
        assert "detector1" in prompt
        assert "detector2" in prompt
        assert "0.8" in prompt  # observed coverage
        assert "0.7" in prompt  # required coverage
    
    def test_build_false_positive_prompt(self, model_server, sample_request):
        """Test false positive tuning prompt building."""
        prompt = model_server._build_false_positive_prompt(sample_request)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "false positive" in prompt.lower()
        assert "tuning" in prompt.lower()
    
    def test_build_incident_summary_prompt(self, model_server, sample_request):
        """Test incident summary prompt building."""
        prompt = model_server._build_incident_summary_prompt(sample_request)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "incident" in prompt.lower()
        assert "summary" in prompt.lower()
    
    def test_build_insufficient_data_prompt(self, model_server, sample_request):
        """Test insufficient data prompt building."""
        prompt = model_server._build_insufficient_data_prompt(sample_request)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "insufficient" in prompt.lower()
        assert "data" in prompt.lower()
    
    def test_compute_confidence_deterministic(self, model_server):
        """Test confidence computation is deterministic."""
        # Test with same input multiple times
        test_data = {
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1"],
            "notes": "Test notes"
        }
        
        confidence1 = model_server._compute_confidence(test_data)
        confidence2 = model_server._compute_confidence(test_data)
        
        assert confidence1 == confidence2
        assert 0.0 <= confidence1 <= 1.0
    
    def test_compute_confidence_with_evidence(self, model_server):
        """Test confidence computation with evidence references."""
        test_data_with_evidence = {
            "reason": "Test reason with evidence",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1", "detector2", "detector3"],
            "notes": "Test notes"
        }
        
        test_data_without_evidence = {
            "reason": "Test reason without evidence",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": [],
            "notes": "Test notes"
        }
        
        confidence_with = model_server._compute_confidence(test_data_with_evidence)
        confidence_without = model_server._compute_confidence(test_data_without_evidence)
        
        # Confidence should be higher with more evidence
        assert confidence_with > confidence_without
    
    def test_compute_confidence_with_opa_diff(self, model_server):
        """Test confidence computation with OPA diff."""
        test_data_with_opa = {
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "package test\n\nallow { true }",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1"],
            "notes": "Test notes"
        }
        
        test_data_without_opa = {
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1"],
            "notes": "Test notes"
        }
        
        confidence_with = model_server._compute_confidence(test_data_with_opa)
        confidence_without = model_server._compute_confidence(test_data_without_opa)
        
        # Confidence should be higher with OPA diff
        assert confidence_with > confidence_without
    
    @pytest.mark.asyncio
    async def test_analyze_success(self, model_server, sample_request):
        """Test successful analysis."""
        with patch.object(model_server, '_simulate_phi3_inference', return_value={
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1"],
            "notes": "Test notes"
        }):
            result = await model_server.analyze(sample_request)
            
            assert isinstance(result, dict)
            assert "reason" in result
            assert "remediation" in result
            assert "opa_diff" in result
            assert "confidence" in result
            assert "confidence_cutoff_used" in result
            assert "evidence_refs" in result
            assert "notes" in result
            assert "processing_time_ms" in result
            assert result["confidence"] >= 0.0
            assert result["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_analyze_low_confidence_fallback(self, model_server, sample_request):
        """Test analysis with low confidence triggers fallback."""
        with patch.object(model_server, '_simulate_phi3_inference', return_value={
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.1,  # Below cutoff
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1"],
            "notes": "Test notes"
        }):
            result = await model_server.analyze(sample_request)
            
            assert result["_template_fallback"] is True
            assert result["_fallback_reason"] == "Low confidence"
    
    @pytest.mark.asyncio
    async def test_analyze_model_error_fallback(self, model_server, sample_request):
        """Test analysis with model error triggers fallback."""
        with patch.object(model_server, '_simulate_phi3_inference', side_effect=Exception("Model error")):
            result = await model_server.analyze(sample_request)
            
            assert result["_template_fallback"] is True
            assert "Model error" in result["_fallback_reason"]
    
    @pytest.mark.asyncio
    async def test_batch_analyze(self, model_server, sample_request):
        """Test batch analysis."""
        requests = [sample_request, sample_request]
        
        with patch.object(model_server, 'analyze', return_value={
            "reason": "Test reason",
            "remediation": "Test remediation",
            "opa_diff": "package test",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": ["detector1"],
            "notes": "Test notes",
            "processing_time_ms": 100
        }):
            results = await model_server.analyze_batch(requests)
            
            assert isinstance(results, list)
            assert len(results) == 2
            for result in results:
                assert "reason" in result
                assert "remediation" in result
    
    def test_simulate_phi3_inference_deterministic(self, model_server, sample_request):
        """Test that simulated inference is deterministic."""
        # Test with same input multiple times
        result1 = model_server._simulate_phi3_inference("test prompt", sample_request)
        result2 = model_server._simulate_phi3_inference("test prompt", sample_request)
        
        # Results should be identical for deterministic behavior
        assert result1 == result2
    
    def test_simulate_phi3_inference_structure(self, model_server, sample_request):
        """Test simulated inference output structure."""
        result = model_server._simulate_phi3_inference("test prompt", sample_request)
        
        assert isinstance(result, dict)
        assert "reason" in result
        assert "remediation" in result
        assert "opa_diff" in result
        assert "confidence" in result
        assert "confidence_cutoff_used" in result
        assert "evidence_refs" in result
        assert "notes" in result
        
        # Check field constraints
        assert len(result["reason"]) <= 120
        assert len(result["remediation"]) <= 120
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["evidence_refs"], list)
    
    def test_simulate_phi3_inference_analysis_type_specific(self, model_server, sample_request):
        """Test that simulated inference varies by analysis type."""
        # Test coverage gap
        sample_request.observed_coverage = {"detector1": 0.5}
        sample_request.required_coverage = {"detector1": 0.8}
        result1 = model_server._simulate_phi3_inference("coverage gap prompt", sample_request)
        
        # Test false positive
        sample_request.false_positive_bands = [{"band": "high"}]
        result2 = model_server._simulate_phi3_inference("false positive prompt", sample_request)
        
        # Results should be different for different analysis types
        assert result1["reason"] != result2["reason"]
        assert result1["remediation"] != result2["remediation"]
