"""
Template Fallbacks for Low Confidence Compliance Analysis
Provides deterministic, conservative outputs when model confidence is low or grounding fails.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class FallbackTrigger:
    """Conditions that trigger template fallback."""
    low_confidence: bool = False
    citation_validation_failed: bool = False
    cross_regulatory_conflict: bool = False
    insufficient_evidence: bool = False
    temporal_validation_failed: bool = False
    

class ComplianceTemplateFallbacks:
    """Deterministic template fallbacks for compliance analysis."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        
    def should_use_fallback(self, 
                           output: Dict[str, Any], 
                           grounding_validated: bool,
                           citation_errors: List[str]) -> FallbackTrigger:
        """Determine if template fallback should be used."""
        
        triggers = FallbackTrigger()
        
        # Low confidence trigger
        confidence = output.get('confidence', 0.0)
        if confidence < self.confidence_threshold:
            triggers.low_confidence = True
            
        # Citation validation failure
        if not grounding_validated or citation_errors:
            triggers.citation_validation_failed = True
            
        # Cross-regulatory conflict detection
        frameworks = output.get('frameworks_assessed', [])
        if len(frameworks) > 1 and confidence < 0.8:
            triggers.cross_regulatory_conflict = True
            
        # Insufficient evidence
        evidence_gaps = output.get('evidence_gaps', [])
        if len(evidence_gaps) > 2:
            triggers.insufficient_evidence = True
            
        # Temporal validation issues
        temporal_issues = output.get('temporal_validation', {}).get('stale_sources_flagged', [])
        if temporal_issues:
            triggers.temporal_validation_failed = True
            
        return triggers
    
    def generate_gap_analysis_fallback(self, 
                                     triggers: FallbackTrigger,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback gap analysis template."""
        
        uncertainty_reasons = []
        if triggers.low_confidence:
            uncertainty_reasons.append("insufficient model confidence")
        if triggers.citation_validation_failed:
            uncertainty_reasons.append("citation validation issues")
        if triggers.cross_regulatory_conflict:
            uncertainty_reasons.append("cross-regulatory framework conflicts")
        if triggers.insufficient_evidence:
            uncertainty_reasons.append("insufficient evidence")
            
        return {
            "analysis_type": "gap_analysis",
            "jurisdictions": [
                {
                    "code": "UNKNOWN",
                    "name": "Jurisdiction requires clarification",
                    "effective_date": date.today().strftime("%Y-%m-%d")
                }
            ],
            "effective_dates": [date.today().strftime("%Y-%m-%d")],
            "citations": [
                {
                    "chunk_text": "Analysis requires additional regulatory research to provide specific citations",
                    "citation": "PENDING - Regulatory research required",
                    "pub_date": date.today().strftime("%Y-%m-%d"),
                    "source_id": "FALLBACK-TEMPLATE",
                    "authority": "Multiple authorities potentially applicable",
                    "section_granularity": "To be determined"
                }
            ],
            "risk_rationale": {
                "level": "high",
                "justification": f"Conservative assessment applied due to: {', '.join(uncertainty_reasons)}. Additional analysis required to determine actual risk level.",
                "evidence_based": False,
                "confidence": 0.3,
                "factors": [
                    "Incomplete information available",
                    "Multiple regulatory frameworks potentially applicable",
                    "Requires subject matter expert review"
                ],
                "mitigation_urgency": "immediate"
            },
            "next_actions": [
                {
                    "action": "Engage regulatory compliance expert for detailed analysis",
                    "owner": "Compliance Team Lead",
                    "due_date": (date.today() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    "priority": "critical",
                    "estimated_effort": "1-2 weeks",
                    "dependencies": ["Regulatory research", "Framework identification"]
                },
                {
                    "action": "Identify applicable regulatory frameworks and jurisdictions",
                    "owner": "Legal Team",
                    "due_date": (date.today() + timedelta(days=5)).strftime("%Y-%m-%d"),
                    "priority": "high",
                    "estimated_effort": "3-5 days",
                    "dependencies": []
                },
                {
                    "action": "Collect additional evidence and documentation",
                    "owner": "Operations Team",
                    "due_date": (date.today() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "priority": "high",
                    "estimated_effort": "1 week",
                    "dependencies": ["Framework identification"]
                }
            ],
            "confidence": 0.3,
            "grounding_validated": False,
            "uncertainty_notice": f"⚠️ CONSERVATIVE ASSESSMENT: This analysis uses template fallback due to {', '.join(uncertainty_reasons)}. Human expert review required.",
            "evidence_gaps": [
                "Specific regulatory framework identification needed",
                "Jurisdiction and applicability clarification required",
                "Current compliance state documentation needed",
                "Regulatory citations require validation"
            ],
            "conservative_approach_applied": True,
            "gaps_identified": [
                {
                    "gap_description": "Comprehensive compliance analysis pending expert review and additional evidence",
                    "regulatory_requirement": "To be determined through expert analysis",
                    "current_state": "Insufficient information to assess current compliance state",
                    "target_state": "Full compliance with applicable regulatory requirements",
                    "gap_severity": "critical",
                    "compliance_deadline": (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
                }
            ],
            "compliance_percentage": 0.0,
            "frameworks_assessed": ["REQUIRES_IDENTIFICATION"],
            "template_fallback_applied": True,
            "fallback_reason": uncertainty_reasons
        }
    
    def generate_risk_rating_fallback(self, 
                                    triggers: FallbackTrigger,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback risk rating template."""
        
        uncertainty_reasons = []
        if triggers.low_confidence:
            uncertainty_reasons.append("insufficient model confidence")
        if triggers.citation_validation_failed:
            uncertainty_reasons.append("citation validation issues")
            
        return {
            "analysis_type": "risk_rating",
            "jurisdictions": [
                {
                    "code": "UNKNOWN",
                    "name": "Multiple jurisdictions potentially applicable",
                    "effective_date": date.today().strftime("%Y-%m-%d")
                }
            ],
            "effective_dates": [date.today().strftime("%Y-%m-%d")],
            "citations": [
                {
                    "chunk_text": "Risk assessment requires regulatory expert review for accurate citation",
                    "citation": "PENDING - Expert review required",
                    "pub_date": date.today().strftime("%Y-%m-%d"),
                    "source_id": "FALLBACK-TEMPLATE",
                    "authority": "Multiple authorities potentially applicable"
                }
            ],
            "risk_rationale": {
                "level": "critical",
                "justification": f"Conservative critical risk rating applied due to uncertainty in: {', '.join(uncertainty_reasons)}",
                "evidence_based": False,
                "confidence": 0.2,
                "factors": [
                    "Insufficient information for accurate risk assessment",
                    "Multiple regulatory frameworks potentially applicable",
                    "Expert review required for proper risk calibration"
                ],
                "mitigation_urgency": "immediate"
            },
            "next_actions": [
                {
                    "action": "Conduct comprehensive risk assessment with regulatory expert",
                    "owner": "Risk Management Team",
                    "due_date": (date.today() + timedelta(days=2)).strftime("%Y-%m-%d"),
                    "priority": "critical",
                    "estimated_effort": "1 week"
                }
            ],
            "confidence": 0.2,
            "grounding_validated": False,
            "uncertainty_notice": f"⚠️ CRITICAL CONSERVATIVE RATING: Template fallback applied due to {', '.join(uncertainty_reasons)}. Immediate expert review required.",
            "evidence_gaps": [
                "Comprehensive risk factor identification needed",
                "Regulatory framework scope clarification required",
                "Impact and likelihood assessment data needed"
            ],
            "conservative_approach_applied": True,
            "risk_scores": {
                "overall_score": 9.0,  # Conservative high score
                "category_scores": {
                    "privacy": 9.0,
                    "security": 9.0,
                    "operational": 8.0,
                    "financial": 8.0,
                    "reputational": 9.0
                }
            },
            "risk_factors": [
                {
                    "factor": "Regulatory uncertainty requiring expert clarification",
                    "impact": "critical",
                    "likelihood": "high"
                },
                {
                    "factor": "Insufficient evidence for accurate assessment",
                    "impact": "high",
                    "likelihood": "very_high"
                }
            ],
            "risk_appetite_alignment": "exceeds_appetite",
            "template_fallback_applied": True,
            "fallback_reason": uncertainty_reasons
        }
    
    def generate_remediation_plan_fallback(self,
                                         triggers: FallbackTrigger,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback remediation plan template."""
        
        uncertainty_reasons = []
        if triggers.low_confidence:
            uncertainty_reasons.append("insufficient model confidence")
        if triggers.insufficient_evidence:
            uncertainty_reasons.append("insufficient evidence")
            
        return {
            "analysis_type": "remediation_plan",
            "jurisdictions": [
                {
                    "code": "UNKNOWN",
                    "name": "Jurisdiction assessment pending",
                    "effective_date": date.today().strftime("%Y-%m-%d")
                }
            ],
            "effective_dates": [date.today().strftime("%Y-%m-%d")],
            "citations": [
                {
                    "chunk_text": "Specific remediation requirements pending regulatory analysis",
                    "citation": "PENDING - Regulatory analysis required",
                    "pub_date": date.today().strftime("%Y-%m-%d"),
                    "source_id": "FALLBACK-TEMPLATE"
                }
            ],
            "risk_rationale": {
                "level": "high",
                "justification": f"Conservative approach applied pending detailed analysis. Uncertainty in: {', '.join(uncertainty_reasons)}",
                "evidence_based": False,
                "confidence": 0.3
            },
            "next_actions": [
                {
                    "action": "Engage compliance expert for detailed remediation planning",
                    "owner": "Compliance Officer",
                    "due_date": (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "priority": "critical"
                }
            ],
            "confidence": 0.3,
            "grounding_validated": False,
            "uncertainty_notice": f"⚠️ INTERIM PLAN: Template fallback applied due to {', '.join(uncertainty_reasons)}. Expert consultation required.",
            "evidence_gaps": [
                "Specific regulatory requirements identification",
                "Current compliance state assessment",
                "Resource availability confirmation"
            ],
            "conservative_approach_applied": True,
            "remediation_steps": [
                {
                    "step_number": 1,
                    "description": "Immediate halt of potentially non-compliant activities pending expert review",
                    "owner": "Operations Manager",
                    "target_date": date.today().strftime("%Y-%m-%d"),
                    "success_criteria": "All potentially non-compliant activities documented and paused",
                    "resources_required": ["Operations team", "Legal review"],
                    "risk_if_delayed": "Continued potential non-compliance exposure"
                },
                {
                    "step_number": 2,
                    "description": "Engage regulatory compliance expert for comprehensive assessment",
                    "owner": "Compliance Officer",
                    "target_date": (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "success_criteria": "Expert engaged and assessment initiated",
                    "resources_required": ["External consultant", "Internal SME"],
                    "risk_if_delayed": "Prolonged uncertainty and potential violations"
                },
                {
                    "step_number": 3,
                    "description": "Develop comprehensive remediation plan based on expert analysis",
                    "owner": "Compliance Team",
                    "target_date": (date.today() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    "success_criteria": "Detailed remediation plan with specific actions and timelines",
                    "resources_required": ["Compliance team", "Legal team", "Operations team"],
                    "risk_if_delayed": "Continued non-compliance risk"
                }
            ],
            "implementation_timeline": {
                "start_date": date.today().strftime("%Y-%m-%d"),
                "target_completion": (date.today() + timedelta(days=30)).strftime("%Y-%m-%d"),
                "milestones": [
                    {
                        "milestone": "Expert assessment complete",
                        "date": (date.today() + timedelta(days=7)).strftime("%Y-%m-%d")
                    },
                    {
                        "milestone": "Detailed plan approved",
                        "date": (date.today() + timedelta(days=14)).strftime("%Y-%m-%d")
                    }
                ]
            },
            "success_metrics": [
                "Expert assessment completion within 7 days",
                "Zero compliance violations during transition",
                "Stakeholder approval of remediation plan"
            ],
            "template_fallback_applied": True,
            "fallback_reason": uncertainty_reasons
        }
    
    def generate_evidence_request_fallback(self,
                                         triggers: FallbackTrigger,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback evidence request template."""
        
        return {
            "analysis_type": "evidence_request",
            "jurisdictions": [
                {
                    "code": "UNKNOWN",
                    "name": "Multiple jurisdictions potentially applicable",
                    "effective_date": date.today().strftime("%Y-%m-%d")
                }
            ],
            "effective_dates": [date.today().strftime("%Y-%m-%d")],
            "citations": [
                {
                    "chunk_text": "Evidence requirements pending regulatory framework identification",
                    "citation": "PENDING - Framework analysis required",
                    "pub_date": date.today().strftime("%Y-%m-%d"),
                    "source_id": "FALLBACK-TEMPLATE"
                }
            ],
            "risk_rationale": {
                "level": "high",
                "justification": "Conservative approach applied pending evidence collection and regulatory clarification",
                "evidence_based": False,
                "confidence": 0.3
            },
            "next_actions": [
                {
                    "action": "Initiate comprehensive evidence collection process",
                    "owner": "Compliance Team",
                    "due_date": (date.today() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "priority": "critical"
                }
            ],
            "confidence": 0.3,
            "grounding_validated": False,
            "uncertainty_notice": "⚠️ COMPREHENSIVE EVIDENCE REQUEST: Template applied due to uncertainty. Immediate evidence collection required.",
            "evidence_gaps": [
                "Regulatory framework identification",
                "Applicable jurisdiction determination",
                "Current compliance documentation status"
            ],
            "conservative_approach_applied": True,
            "evidence_items": [
                {
                    "evidence_type": "policy_document",
                    "description": "Current compliance policies and procedures for all potentially applicable frameworks",
                    "regulatory_basis": "To be determined through framework analysis",
                    "urgency": "immediate",
                    "specific_requirements": [
                        "All privacy and data protection policies",
                        "Security policies and procedures",
                        "Risk management frameworks",
                        "Incident response procedures"
                    ],
                    "responsible_party": "Policy Management Team"
                },
                {
                    "evidence_type": "assessment_report",
                    "description": "Recent compliance assessments and audit findings",
                    "regulatory_basis": "General compliance validation",
                    "urgency": "high",
                    "specific_requirements": [
                        "Internal audit reports (last 12 months)",
                        "External assessment reports",
                        "Penetration testing results",
                        "Compliance gap analyses"
                    ],
                    "responsible_party": "Audit Team"
                }
            ],
            "evidence_gap_impact": "Critical - Cannot proceed with accurate compliance analysis without comprehensive evidence collection",
            "collection_deadline": (date.today() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "template_fallback_applied": True,
            "fallback_reason": ["insufficient_evidence", "regulatory_uncertainty"]
        }
    
    def generate_fallback(self, 
                         analysis_type: str,
                         triggers: FallbackTrigger,
                         context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate fallback template based on analysis type."""
        
        fallback_generators = {
            "gap_analysis": self.generate_gap_analysis_fallback,
            "risk_rating": self.generate_risk_rating_fallback,
            "remediation_plan": self.generate_remediation_plan_fallback,
            "evidence_request": self.generate_evidence_request_fallback
        }
        
        if analysis_type not in fallback_generators:
            logger.error(f"No fallback generator for analysis type: {analysis_type}")
            # Return basic fallback
            return self.generate_gap_analysis_fallback(triggers, context)
            
        return fallback_generators[analysis_type](triggers, context)


if __name__ == "__main__":
    # Example usage
    fallbacks = ComplianceTemplateFallbacks()
    
    # Simulate triggers
    triggers = FallbackTrigger(
        low_confidence=True,
        citation_validation_failed=True
    )
    
    context = {"frameworks": ["GDPR", "HIPAA"]}
    
    # Generate fallback
    fallback_output = fallbacks.generate_fallback("gap_analysis", triggers, context)
    
    print("Fallback Template Generated:")
    print(json.dumps(fallback_output, indent=2))
