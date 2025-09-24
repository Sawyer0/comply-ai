"""
Unit tests for compliance guardrails.

Tests all guardrail functionality including citation requirements,
risk assessment validation, and regulatory accuracy checks.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any

from src.llama_mapper.rag.guardrails.compliance_guardrails import (
    ComplianceGuardrails, GuardrailResult, ComplianceLevel, RiskLevel
)
from src.llama_mapper.rag.integration.model_enhancement import ExpertResponse, RAGContext
from src.llama_mapper.rag.core.vector_store import Document


class TestComplianceGuardrails:
    """Test compliance guardrails functionality."""
    
    @pytest.fixture
    def guardrails(self):
        """Create guardrails instance for testing."""
        return ComplianceGuardrails()
    
    @pytest.fixture
    def sample_expert_response(self):
        """Create sample expert response for testing."""
        return ExpertResponse(
            analysis="Issue: Data processing requirements\nRule: GDPR Article 6\nAnalysis: Legal basis required\nConclusion: Implement consent management",
            recommendations=[
                "Implement consent management system",
                "Conduct data protection impact assessment",
                "Establish data retention policies"
            ],
            risk_assessment={
                "regulatory_risk": "High",
                "implementation_risk": "Medium",
                "timeline_risk": "Low"
            },
            regulatory_citations=[
                {"title": "GDPR Article 6", "citation": "GDPR Art. 6", "date": "2024-01-01"},
                {"title": "GDPR Article 35", "citation": "GDPR Art. 35", "date": "2024-01-01"}
            ],
            confidence_score=0.85,
            evidence_required=[
                "Data processing records",
                "Consent documentation",
                "DPIA report"
            ],
            next_actions=[
                {"owner": "Compliance Officer", "due_by": "7 days", "evidence": "Consent audit"},
                {"owner": "Legal Team", "due_by": "14 days", "evidence": "Legal review"}
            ],
            jurisdictional_scope=["EU", "GDPR"],
            effective_dates=["2024-01-01"],
            implementation_complexity="High",
            cost_impact="High"
        )
    
    @pytest.fixture
    def sample_rag_context(self):
        """Create sample RAG context for testing."""
        return RAGContext(
            query="What are GDPR data processing requirements?",
            retrieved_documents=[
                Document(
                    content="GDPR Article 6 legal basis",
                    source="gdpr_art6.txt",
                    regulatory_framework="GDPR"
                ),
                Document(
                    content="GDPR Article 35 DPIA requirements",
                    source="gdpr_art35.txt",
                    regulatory_framework="GDPR"
                )
            ],
            regulatory_framework="GDPR",
            industry="technology"
        )
    
    @pytest.mark.asyncio
    async def test_evaluate_response_success(self, guardrails, sample_expert_response, sample_rag_context):
        """Test successful guardrail evaluation."""
        result = await guardrails.evaluate_response(sample_expert_response, sample_rag_context)
        
        assert isinstance(result, GuardrailResult)
        assert result.passed is True
        assert result.confidence_score > 0.0
        assert len(result.violations) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_response_missing_citations(self, guardrails, sample_rag_context):
        """Test evaluation with missing citations."""
        response = ExpertResponse(
            analysis="Some analysis without proper structure",
            recommendations=["Do something"],
            risk_assessment={},
            regulatory_citations=[],  # No citations
            confidence_score=0.5,
            evidence_required=[],
            next_actions=[],
            jurisdictional_scope=[],
            effective_dates=[],
            implementation_complexity="Unknown",
            cost_impact="Unknown"
        )
        
        result = await guardrails.evaluate_response(response, sample_rag_context)
        
        assert result.passed is False
        assert len(result.violations) > 0
        assert any("citations" in violation.lower() for violation in result.violations)
    
    @pytest.mark.asyncio
    async def test_evaluate_response_missing_risk_assessment(self, guardrails, sample_rag_context):
        """Test evaluation with missing risk assessment."""
        response = ExpertResponse(
            analysis="Analysis without risk assessment",
            recommendations=["Do something"],
            risk_assessment=None,  # No risk assessment
            regulatory_citations=[
                {"title": "GDPR Article 6", "citation": "GDPR Art. 6", "date": "2024-01-01"}
            ],
            confidence_score=0.5,
            evidence_required=[],
            next_actions=[],
            jurisdictional_scope=["EU"],
            effective_dates=["2024-01-01"],
            implementation_complexity="Medium",
            cost_impact="Medium"
        )
        
        result = await guardrails.evaluate_response(response, sample_rag_context)
        
        assert result.passed is False
        assert len(result.violations) > 0
        assert any("risk assessment" in violation.lower() for violation in result.violations)
    
    @pytest.mark.asyncio
    async def test_evaluate_response_missing_evidence_requirements(self, guardrails, sample_expert_response, sample_rag_context):
        """Test evaluation with missing evidence requirements."""
        sample_expert_response.evidence_required = []  # No evidence required
        
        result = await guardrails.evaluate_response(sample_expert_response, sample_rag_context)
        
        assert result.passed is False
        assert len(result.violations) > 0
        assert any("evidence" in violation.lower() for violation in result.violations)
    
    @pytest.mark.asyncio
    async def test_citation_quality_assessment(self, guardrails):
        """Test citation quality assessment."""
        # High quality citations
        good_citations = [
            {"title": "GDPR Article 6", "citation": "GDPR Art. 6", "date": "2024-01-01"},
            {"title": "GDPR Article 35", "citation": "GDPR Art. 35", "date": "2024-01-01"}
        ]
        quality = guardrails._assess_citation_quality(good_citations)
        assert quality > 0.8
        
        # Poor quality citations
        poor_citations = [
            {"title": "", "citation": "", "date": ""},
            {"title": "Something", "citation": "", "date": ""}
        ]
        quality = guardrails._assess_citation_quality(poor_citations)
        assert quality < 0.5
    
    @pytest.mark.asyncio
    async def test_citation_coverage_assessment(self, guardrails):
        """Test citation coverage assessment."""
        citations = [
            {"title": "gdpr_art6.txt", "citation": "GDPR Art. 6", "date": "2024-01-01"}
        ]
        documents = [
            Document(content="test", source="gdpr_art6.txt"),
            Document(content="test", source="gdpr_art35.txt")
        ]
        
        coverage = guardrails._assess_citation_coverage(citations, documents)
        assert coverage == 0.5  # 1 out of 2 documents cited
    
    @pytest.mark.asyncio
    async def test_risk_assessment_quality(self, guardrails):
        """Test risk assessment quality evaluation."""
        # Good risk assessment
        good_risk = {
            "regulatory_risk": "High",
            "operational_risk": "Medium",
            "mitigation": "Implement controls"
        }
        quality = guardrails._assess_risk_assessment_quality(good_risk)
        assert quality > 0.5
        
        # Poor risk assessment
        poor_risk = {}
        quality = guardrails._assess_risk_assessment_quality(poor_risk)
        assert quality == 0.0
    
    def test_extract_risk_levels(self, guardrails):
        """Test risk level extraction from analysis."""
        analysis = "This presents a high risk situation with medium severity impact."
        levels = guardrails._extract_risk_levels(analysis)
        assert "high" in levels
        assert "medium" in levels
    
    def test_regulatory_language_assessment(self, guardrails):
        """Test regulatory language assessment."""
        # Good regulatory language
        good_analysis = "This compliance requirement mandates audit controls and governance procedures."
        score = guardrails._assess_regulatory_language(good_analysis)
        assert score > 0.3
        
        # Poor regulatory language
        poor_analysis = "This is just some random text without regulatory terms."
        score = guardrails._assess_regulatory_language(poor_analysis)
        assert score < 0.1
    
    def test_evidence_specificity_assessment(self, guardrails):
        """Test evidence specificity assessment."""
        # Specific evidence
        specific_evidence = [
            "Signed consent documents dated within 30 days",
            "Certified DPIA report approved by legal team"
        ]
        score = guardrails._assess_evidence_specificity(specific_evidence)
        assert score > 0.5
        
        # Vague evidence
        vague_evidence = [
            "Some documents",
            "Various records"
        ]
        score = guardrails._assess_evidence_specificity(vague_evidence)
        assert score < 0.3
    
    def test_uncertainty_handling_assessment(self, guardrails):
        """Test uncertainty handling assessment."""
        # Good uncertainty handling
        good_analysis = "Based on available information, this may require legal counsel review and should be verified."
        score = guardrails._assess_uncertainty_handling(good_analysis)
        assert score > 0.2
        
        # Poor uncertainty handling
        poor_analysis = "This definitely requires immediate action without any doubt."
        score = guardrails._assess_uncertainty_handling(poor_analysis)
        assert score < 0.1
    
    def test_overconfidence_assessment(self, guardrails):
        """Test overconfidence assessment."""
        # Overconfident analysis
        overconfident = "This is definitely the right approach and guaranteed to work absolutely."
        score = guardrails._assess_overconfidence(overconfident)
        assert score > 0.3
        
        # Appropriate confidence
        appropriate = "This approach may be suitable and could help address the requirements."
        score = guardrails._assess_overconfidence(appropriate)
        assert score < 0.1
    
    @pytest.mark.asyncio
    async def test_conservative_risk_assessment(self, guardrails):
        """Test conservative risk assessment evaluation."""
        # Conservative risk assessment
        conservative_risk = {
            "approach": "conservative and cautious",
            "risk_level": "high and critical assessment"
        }
        score = guardrails._assess_conservative_risk_assessment(conservative_risk)
        assert score > 0.3
        
        # Non-conservative risk assessment
        aggressive_risk = {
            "approach": "aggressive",
            "risk_level": "low"
        }
        score = guardrails._assess_conservative_risk_assessment(aggressive_risk)
        assert score < 0.1


class TestGuardrailResult:
    """Test GuardrailResult dataclass."""
    
    def test_guardrail_result_creation(self):
        """Test creating guardrail result."""
        result = GuardrailResult(
            passed=True,
            confidence_score=0.85,
            violations=[],
            warnings=["Minor warning"],
            recommendations=["Improve citations"],
            required_actions=["Review evidence"]
        )
        
        assert result.passed is True
        assert result.confidence_score == 0.85
        assert len(result.violations) == 0
        assert len(result.warnings) == 1
        assert len(result.recommendations) == 1
        assert len(result.required_actions) == 1
    
    def test_guardrail_result_to_dict(self):
        """Test converting guardrail result to dictionary."""
        result = GuardrailResult(
            passed=False,
            confidence_score=0.5,
            violations=["Missing citations"],
            warnings=["Risk assessment incomplete"],
            recommendations=["Add citations"],
            required_actions=["Provide evidence"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["passed"] is False
        assert result_dict["confidence_score"] == 0.5
        assert "Missing citations" in result_dict["violations"]
        assert "Risk assessment incomplete" in result_dict["warnings"]
        assert "Add citations" in result_dict["recommendations"]
        assert "Provide evidence" in result_dict["required_actions"]
