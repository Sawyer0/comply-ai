"""
Compliance guardrails extending the existing guardrails system.

Integrates compliance-specific guardrails into the existing cost guardrails framework
to ensure expert-level compliance guidance with proper citations and risk assessment.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ...cost_monitoring.guardrails import (
    GuardrailAction, GuardrailSeverity, CostGuardrail, 
    CostGuardrails, GuardrailViolation
)

logger = logging.getLogger(__name__)


class ComplianceMetricType(str, Enum):
    """Types of compliance metrics to monitor."""
    CITATION_ACCURACY = "citation_accuracy"
    REGULATORY_COVERAGE = "regulatory_coverage"
    RISK_ASSESSMENT_QUALITY = "risk_assessment_quality"
    RESPONSE_COMPLETENESS = "response_completeness"
    EXPERTISE_LEVEL = "expertise_level"


class ComplianceLevel(Enum):
    """Compliance level indicators."""
    MANDATORY = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    UNKNOWN = "unknown"


class RiskLevel(Enum):
    """Risk level indicators."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceGuardrailResult:
    """Result of compliance guardrail evaluation."""
    
    passed: bool
    confidence_score: float
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    required_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "confidence_score": self.confidence_score,
            "violations": self.violations,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "required_actions": self.required_actions
        }


class ComplianceGuardrailsExtension:
    """
    Compliance guardrails extension for the existing guardrails system.
    
    Extends the cost guardrails framework with compliance-specific monitoring
    and validation capabilities.
    """
    
    def __init__(self, base_guardrails: Optional[CostGuardrails] = None):
        """Initialize compliance guardrails extension.
        
        Args:
            base_guardrails: Existing cost guardrails system to extend (optional)
        """
        self.base_guardrails = base_guardrails
        self.logger = logging.getLogger(__name__)
        
        # Add compliance-specific guardrails if base system is available
        if self.base_guardrails:
            self._setup_compliance_guardrails()
    
    def _setup_compliance_guardrails(self):
        """Set up compliance-specific guardrails."""
        
        # Citation accuracy guardrail
        citation_guardrail = CostGuardrail(
            guardrail_id="compliance_citation_accuracy",
            name="Citation Accuracy Requirement",
            description="Ensures responses include accurate regulatory citations",
            metric_type=ComplianceMetricType.CITATION_ACCURACY.value,
            threshold=0.95,  # 95% citation accuracy required
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT, GuardrailAction.THROTTLE],
            cooldown_minutes=30
        )
        self.base_guardrails.add_guardrail(citation_guardrail)
        
        # Regulatory coverage guardrail
        coverage_guardrail = CostGuardrail(
            guardrail_id="compliance_regulatory_coverage",
            name="Regulatory Coverage Requirement",
            description="Ensures comprehensive regulatory framework coverage",
            metric_type=ComplianceMetricType.REGULATORY_COVERAGE.value,
            threshold=0.8,  # 80% coverage required
            severity=GuardrailSeverity.MEDIUM,
            actions=[GuardrailAction.ALERT],
            cooldown_minutes=60
        )
        self.base_guardrails.add_guardrail(coverage_guardrail)
        
        # Risk assessment guardrail
        risk_guardrail = CostGuardrail(
            guardrail_id="compliance_risk_assessment",
            name="Risk Assessment Quality",
            description="Ensures comprehensive risk assessment in responses",
            metric_type=ComplianceMetricType.RISK_ASSESSMENT_QUALITY.value,
            threshold=0.9,  # 90% risk assessment quality required
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT, GuardrailAction.BLOCK_REQUESTS],
            cooldown_minutes=15
        )
        self.base_guardrails.add_guardrail(risk_guardrail)
    
    async def evaluate_response(self, response: Dict[str, Any], 
                              context: Dict[str, Any]) -> ComplianceGuardrailResult:
        """
        Evaluate response against compliance guardrails.
        
        Args:
            response: RAG response to evaluate
            context: RAG context used for response generation
            
        Returns:
            Compliance guardrail evaluation result
        """
        violations = []
        warnings = []
        recommendations = []
        required_actions = []
        
        # Evaluate citation requirements
        citation_score = self._evaluate_citations(response, context)
        if citation_score < 0.95:
            violations.append(f"Citation accuracy below threshold: {citation_score:.2f}")
            required_actions.append("Include specific regulatory citations with section numbers")
        
        # Evaluate risk assessment
        risk_score = self._evaluate_risk_assessment(response)
        if risk_score < 0.9:
            violations.append(f"Risk assessment quality below threshold: {risk_score:.2f}")
            required_actions.append("Provide comprehensive risk assessment with mitigation strategies")
        
        # Evaluate regulatory coverage
        coverage_score = self._evaluate_regulatory_coverage(response, context)
        if coverage_score < 0.8:
            warnings.append(f"Regulatory coverage below optimal: {coverage_score:.2f}")
            recommendations.append("Consider additional regulatory frameworks")
        
        # Evaluate response completeness
        completeness_score = self._evaluate_completeness(response)
        if completeness_score < 0.8:
            warnings.append(f"Response completeness could be improved: {completeness_score:.2f}")
            recommendations.append("Include implementation guidance and next steps")
        
        # Calculate overall confidence
        confidence_score = (citation_score + risk_score + coverage_score + completeness_score) / 4
        
        # Determine if guardrails passed
        passed = len(violations) == 0
        
        # Log metrics to base guardrails system
        await self._log_compliance_metrics(citation_score, risk_score, coverage_score, completeness_score)
        
        return ComplianceGuardrailResult(
            passed=passed,
            confidence_score=confidence_score,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            required_actions=required_actions
        )
    
    def _evaluate_citations(self, response: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate citation accuracy and completeness."""
        
        # Check if response includes regulatory citations
        citations = response.get("regulatory_citations", [])
        if not citations:
            return 0.0
        
        # Check citation format and specificity
        citation_score = 0.0
        for citation in citations:
            score = 0.0
            
            # Check if citation includes specific section/article
            if any(keyword in citation.lower() for keyword in ["article", "section", "part", "chapter"]):
                score += 0.3
            
            # Check if citation includes date/version
            if re.search(r'\d{4}', citation):
                score += 0.2
            
            # Check if citation includes framework name
            frameworks = ["gdpr", "hipaa", "sox", "iso", "pci", "ccpa"]
            if any(fw in citation.lower() for fw in frameworks):
                score += 0.3
            
            # Check if citation is properly formatted
            if citation.strip() and len(citation) > 10:
                score += 0.2
            
            citation_score += min(score, 1.0)
        
        return min(citation_score / len(citations), 1.0) if citations else 0.0
    
    def _evaluate_risk_assessment(self, response: Dict[str, Any]) -> float:
        """Evaluate risk assessment quality."""
        
        risk_assessment = response.get("risk_assessment", {})
        if not risk_assessment:
            return 0.0
        
        score = 0.0
        
        # Check if risk level is specified
        if "risk_level" in risk_assessment or "overall_risk" in risk_assessment:
            score += 0.3
        
        # Check if specific risks are identified
        if "risk_factors" in risk_assessment or "risks" in risk_assessment:
            score += 0.3
        
        # Check if mitigation strategies are provided
        if "mitigation" in risk_assessment or "mitigation_strategies" in risk_assessment:
            score += 0.4
        
        return min(score, 1.0)
    
    def _evaluate_regulatory_coverage(self, response: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate regulatory framework coverage."""
        
        # Check if response addresses the requested regulatory framework
        requested_framework = context.get("regulatory_framework")
        if not requested_framework:
            return 0.8  # Neutral score if no specific framework requested
        
        analysis = response.get("analysis", "").lower()
        citations = response.get("regulatory_citations", [])
        
        # Check if framework is mentioned in response
        if requested_framework.lower() in analysis:
            score = 0.5
        else:
            score = 0.0
        
        # Check if framework appears in citations
        framework_in_citations = any(requested_framework.lower() in str(citation).lower() 
                                   for citation in citations)
        if framework_in_citations:
            score += 0.5
        
        return min(score, 1.0)
    
    def _evaluate_completeness(self, response: Dict[str, Any]) -> float:
        """Evaluate response completeness."""
        
        score = 0.0
        required_elements = [
            "analysis",
            "recommendations", 
            "regulatory_citations",
            "risk_assessment"
        ]
        
        for element in required_elements:
            if element in response and response[element]:
                score += 0.25
        
        # Bonus for additional helpful elements
        bonus_elements = [
            "next_actions",
            "implementation_complexity",
            "cost_impact",
            "jurisdictional_scope"
        ]
        
        bonus_score = 0.0
        for element in bonus_elements:
            if element in response and response[element]:
                bonus_score += 0.05
        
        return min(score + bonus_score, 1.0)
    
    async def _log_compliance_metrics(self, citation_score: float, risk_score: float, 
                                    coverage_score: float, completeness_score: float):
        """Log compliance metrics to the base guardrails system."""
        
        # This would integrate with the metrics collector from the base system
        # For now, just log the metrics
        self.logger.info(
            "Compliance metrics recorded",
            citation_accuracy=citation_score,
            risk_assessment_quality=risk_score,
            regulatory_coverage=coverage_score,
            response_completeness=completeness_score
        )


# Convenience alias for backward compatibility
ComplianceGuardrails = ComplianceGuardrailsExtension