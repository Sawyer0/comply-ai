"""
Backward Compatibility Adapter for maintaining API compatibility.

This adapter ensures that the refactored analysis system maintains
identical external API behavior and response formats.
"""

import logging
from typing import Any, Dict, Optional

from ..domain import ITemplateProvider, AnalysisType
from ..domain.entities import AnalysisRequest
from ..engines import TemplateOrchestrator
from ..factories import create_default_factory

logger = logging.getLogger(__name__)


class BackwardCompatibilityAdapter(ITemplateProvider):
    """
    Adapter that maintains backward compatibility with the original template provider.
    
    This adapter wraps the new orchestrated analysis system while preserving
    the exact same external API and response formats as the original system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backward compatibility adapter.
        
        Args:
            config: Optional configuration for the analysis system
        """
        self.config = config or {}
        self.factory = create_default_factory(config)
        self.orchestrator: Optional[TemplateOrchestrator] = None
        self._initialized = False
        
        # Initialize the orchestrator
        self._initialize_orchestrator()
    
    def _initialize_orchestrator(self) -> None:
        """Initialize the template orchestrator."""
        try:
            self.orchestrator = self.factory.create_orchestrator()
            self._initialized = True
            logger.info("Backward compatibility adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            self._initialized = False
    
    def get_template_response(
        self,
        request: AnalysisRequest,
        analysis_type: AnalysisType,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get template response for the specified analysis type.
        
        This method maintains the exact same signature and behavior as the
        original template provider while using the new orchestrated system.
        
        Args:
            request: Analysis request
            analysis_type: Type of analysis to perform
            fallback_reason: Reason for using template fallback
            
        Returns:
            Template response dictionary in original format
        """
        if not self._initialized or not self.orchestrator:
            return self._create_fallback_response(request, fallback_reason or "System not initialized")
        
        try:
            # Use the orchestrated analysis system
            response = asyncio.run(self.orchestrator.orchestrate_analysis(request))
            
            # Convert to original template format
            return self._convert_to_template_format(response, analysis_type, fallback_reason)
            
        except Exception as e:
            logger.error(f"Orchestrated analysis failed: {e}")
            return self._create_fallback_response(request, f"Analysis failed: {str(e)}")
    
    def select_analysis_type(self, request: AnalysisRequest) -> AnalysisType:
        """
        Select the most appropriate analysis type based on request content.
        
        This method maintains the original logic for analysis type selection
        to ensure backward compatibility.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis type to use
        """
        # Original logic from the monolithic template provider
        
        # Check for coverage gaps
        coverage_gaps = []
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            if observed < required:
                coverage_gaps.append(detector)

        if coverage_gaps:
            return AnalysisType.COVERAGE_GAP

        # Check for false positive tuning opportunities
        if request.false_positive_bands:
            return AnalysisType.FALSE_POSITIVE_TUNING

        # Check for incident summary
        if request.high_sev_hits:
            return AnalysisType.INCIDENT_SUMMARY

        return AnalysisType.INSUFFICIENT_DATA
    
    def _convert_to_template_format(self, response: Any, analysis_type: AnalysisType, 
                                  fallback_reason: Optional[str]) -> Dict[str, Any]:
        """
        Convert orchestrated response to original template format.
        
        Args:
            response: Response from orchestrated analysis
            analysis_type: Analysis type that was requested
            fallback_reason: Optional fallback reason
            
        Returns:
            Response in original template format
        """
        # Extract key information from orchestrated response
        reason = getattr(response, 'reason', 'Analysis completed')
        remediation = getattr(response, 'remediation', 'Review findings and take appropriate action')
        opa_diff = getattr(response, 'opa_diff', '')
        confidence = getattr(response, 'confidence', 0.7)
        evidence_refs = getattr(response, 'evidence_refs', [])
        notes = getattr(response, 'notes', '')
        
        # Add fallback reason to notes if provided
        if fallback_reason:
            notes = f"{notes} - {fallback_reason}" if notes else fallback_reason
        
        # Create response in original format
        template_response = {
            "reason": reason,
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": confidence,
            "confidence_cutoff_used": 0.3,  # Original default
            "evidence_refs": evidence_refs,
            "notes": notes
        }
        
        # Add analysis-specific details if available
        if hasattr(response, 'metadata') and response.metadata:
            # Extract analysis details from metadata
            analysis_details = self._extract_analysis_details(response.metadata, analysis_type)
            if analysis_details:
                template_response["analysis_details"] = analysis_details
        
        return template_response
    
    def _extract_analysis_details(self, metadata: Dict[str, Any], 
                                analysis_type: AnalysisType) -> Optional[Dict[str, Any]]:
        """Extract analysis details from metadata based on analysis type."""
        if analysis_type == AnalysisType.COVERAGE_GAP:
            return {
                "gaps_identified": metadata.get("gaps_count", 0),
                "risk_assessment": "medium",  # Default
                "priority_actions": [],
                "estimated_effort": {"hours": 8, "complexity": "medium"}
            }
        elif analysis_type == AnalysisType.FALSE_POSITIVE_TUNING:
            return {
                "patterns_identified": [],
                "threshold_recommendations": {},
                "expected_reduction": 0.2,
                "validation_approach": "A/B test recommended"
            }
        elif analysis_type == AnalysisType.INCIDENT_SUMMARY:
            return {
                "incident_breakdown": {},
                "severity_analysis": "mixed severity levels",
                "response_timeline": {},
                "escalation_required": False
            }
        
        return None
    
    def _create_fallback_response(self, request: AnalysisRequest, 
                                fallback_reason: str) -> Dict[str, Any]:
        """
        Create a fallback response when the orchestrated system fails.
        
        This ensures the system always returns a response in the expected format.
        """
        # Determine analysis type for fallback
        analysis_type = self.select_analysis_type(request)
        
        if analysis_type == AnalysisType.COVERAGE_GAP:
            return {
                "reason": "Coverage gaps detected in security monitoring",
                "remediation": "Enable missing detectors and adjust coverage thresholds",
                "opa_diff": self._generate_basic_coverage_policy(),
                "confidence": 0.6,
                "confidence_cutoff_used": 0.3,
                "evidence_refs": request.required_detectors[:3],
                "notes": f"Fallback analysis - {fallback_reason}",
                "analysis_details": {
                    "gaps_identified": len([d for d in request.required_detectors 
                                          if request.observed_coverage.get(d, 0) < 
                                             request.required_coverage.get(d, 0)]),
                    "risk_assessment": "medium",
                    "priority_actions": ["Enable missing detectors"],
                    "estimated_effort": {"hours": 4, "complexity": "low"}
                }
            }
        elif analysis_type == AnalysisType.FALSE_POSITIVE_TUNING:
            return {
                "reason": "False positive patterns detected requiring threshold adjustment",
                "remediation": "Adjust detector thresholds to reduce false positive rates",
                "opa_diff": self._generate_basic_threshold_policy(),
                "confidence": 0.5,
                "confidence_cutoff_used": 0.3,
                "evidence_refs": [band.get('detector', 'unknown') for band in request.false_positive_bands[:3]],
                "notes": f"Fallback analysis - {fallback_reason}",
                "analysis_details": {
                    "patterns_identified": len(request.false_positive_bands),
                    "threshold_recommendations": {},
                    "expected_reduction": 0.3,
                    "validation_approach": "Monitor for 24 hours after adjustment"
                }
            }
        elif analysis_type == AnalysisType.INCIDENT_SUMMARY:
            return {
                "reason": "Security incidents detected requiring investigation",
                "remediation": "Investigate high severity incidents and implement containment measures",
                "opa_diff": self._generate_basic_incident_policy(),
                "confidence": 0.7,
                "confidence_cutoff_used": 0.3,
                "evidence_refs": [hit.get('detector', 'unknown') for hit in request.high_sev_hits[:3]],
                "notes": f"Fallback analysis - {fallback_reason}",
                "analysis_details": {
                    "incident_breakdown": {"high_severity": len(request.high_sev_hits)},
                    "severity_analysis": f"{len(request.high_sev_hits)} high severity incidents",
                    "response_timeline": {"immediate": "Begin investigation"},
                    "escalation_required": len(request.high_sev_hits) > 5
                }
            }
        else:
            return {
                "reason": "Insufficient data for detailed analysis",
                "remediation": "Collect more comprehensive security metrics",
                "opa_diff": "",
                "confidence": 0.1,
                "confidence_cutoff_used": 0.3,
                "evidence_refs": request.required_detectors[:3],
                "notes": f"Fallback analysis - {fallback_reason}"
            }
    
    def _generate_basic_coverage_policy(self) -> str:
        """Generate basic OPA policy for coverage gaps."""
        return """package coverage_policy

# Basic coverage enforcement
coverage_violation[msg] {
    detector := input.detectors[_]
    observed := input.observed_coverage[detector]
    required := input.required_coverage[detector]
    observed < required
    msg := sprintf("Coverage gap: %v has %v%% coverage, requires %v%%", [detector, observed * 100, required * 100])
}"""
    
    def _generate_basic_threshold_policy(self) -> str:
        """Generate basic OPA policy for threshold adjustment."""
        return """package threshold_policy

# Basic threshold enforcement
threshold_violation[msg] {
    detector := input.detectors[_]
    fp_rate := input.false_positive_rates[detector]
    fp_rate > 0.15
    msg := sprintf("High false positive rate: %v at %v%%", [detector, fp_rate * 100])
}"""
    
    def _generate_basic_incident_policy(self) -> str:
        """Generate basic OPA policy for incident response."""
        return """package incident_policy

# Basic incident response
incident_escalation[msg] {
    count(input.high_severity_incidents) > 5
    msg := "Multiple high severity incidents detected - escalation required"
}"""


# Import asyncio at module level to avoid issues
import asyncio