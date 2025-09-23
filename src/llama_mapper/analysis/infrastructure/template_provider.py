"""
Infrastructure implementation of the template provider for the Analysis Module.

This module contains the concrete implementation of the ITemplateProvider interface
for providing deterministic template responses.
"""

import logging
from typing import Any, Dict, Optional

from ..domain.entities import AnalysisRequest, AnalysisType
from ..domain.interfaces import ITemplateProvider

logger = logging.getLogger(__name__)


class AnalysisTemplateProvider(ITemplateProvider):
    """
    Analysis template provider implementation.
    
    Provides concrete implementation of the ITemplateProvider interface
    for generating deterministic template responses.
    """
    
    def __init__(self):
        """Initialize the analysis template provider."""
        logger.info("Initialized Analysis Template Provider")
    
    def get_template_response(
        self, 
        request: AnalysisRequest, 
        analysis_type: AnalysisType,
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get template response for the specified analysis type.
        
        Args:
            request: Analysis request
            analysis_type: Type of analysis to perform
            fallback_reason: Reason for using template fallback
            
        Returns:
            Template response dictionary
        """
        if analysis_type == AnalysisType.COVERAGE_GAP:
            return self._get_coverage_gap_template(request, fallback_reason)
        elif analysis_type == AnalysisType.FALSE_POSITIVE_TUNING:
            return self._get_false_positive_template(request, fallback_reason)
        elif analysis_type == AnalysisType.INCIDENT_SUMMARY:
            return self._get_incident_summary_template(request, fallback_reason)
        else:
            return self._get_insufficient_data_template(request, fallback_reason)
    
    def select_analysis_type(self, request: AnalysisRequest) -> AnalysisType:
        """
        Select the most appropriate analysis type based on request content.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis type to use
        """
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
    
    def _get_coverage_gap_template(
        self, 
        request: AnalysisRequest, 
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get coverage gap template response."""
        # Identify specific coverage gaps
        coverage_gaps = []
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            if observed < required:
                coverage_gaps.append(f"{detector} ({observed:.1%} < {required:.1%})")
        
        reason = f"coverage gap detected - {', '.join(coverage_gaps)}"
        remediation = "enable missing detectors or adjust coverage requirements"
        
        # Generate OPA policy for coverage enforcement
        opa_diff = self._generate_coverage_opa_policy(request.required_detectors)
        
        return {
            "reason": reason,
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Template fallback for coverage gap analysis{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _get_false_positive_template(
        self, 
        request: AnalysisRequest, 
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get false positive tuning template response."""
        # Analyze false positive bands
        detectors_with_fp = [band.get("detector", "unknown") for band in request.false_positive_bands]
        
        reason = f"false positive patterns detected in {len(detectors_with_fp)} detectors"
        remediation = "adjust detector thresholds based on false positive analysis"
        
        # Generate OPA policy for threshold enforcement
        opa_diff = self._generate_threshold_opa_policy(detectors_with_fp)
        
        return {
            "reason": reason,
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": detectors_with_fp,
            "notes": f"Template fallback for false positive tuning{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _get_incident_summary_template(
        self, 
        request: AnalysisRequest, 
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get incident summary template response."""
        # Analyze high severity hits
        high_sev_count = len(request.high_sev_hits)
        detectors_with_incidents = [hit.get("detector", "unknown") for hit in request.high_sev_hits]
        
        reason = f"high severity security incidents detected ({high_sev_count} incidents)"
        remediation = "investigate and remediate high severity findings immediately"
        
        # Generate OPA policy for incident response
        opa_diff = self._generate_incident_opa_policy()
        
        return {
            "reason": reason,
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": detectors_with_incidents,
            "notes": f"Template fallback for incident summary{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _get_insufficient_data_template(
        self, 
        request: AnalysisRequest, 
        fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get insufficient data template response."""
        reason = "insufficient data for detailed analysis"
        remediation = "collect more comprehensive security metrics"
        
        return {
            "reason": reason,
            "remediation": remediation,
            "opa_diff": "",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Template fallback for insufficient data{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _generate_coverage_opa_policy(self, required_detectors: list) -> str:
        """Generate OPA policy for coverage enforcement."""
        return f"""package coverage

# Coverage enforcement policy
violation[msg] {{
    detector := input.required_detectors[_]
    not input.observed_coverage[detector]
    msg := sprintf("Missing coverage for detector: %v", [detector])
}}

# Coverage threshold enforcement
violation[msg] {{
    detector := input.required_detectors[_]
    observed := input.observed_coverage[detector]
    required := input.required_coverage[detector]
    observed < required
    msg := sprintf("Insufficient coverage for %v: %v < %v", [detector, observed, required])
}}"""
    
    def _generate_threshold_opa_policy(self, detectors: list) -> str:
        """Generate OPA policy for threshold enforcement."""
        return f"""package thresholds

# Threshold enforcement policy
violation[msg] {{
    detector := input.false_positive_bands[_].detector
    detector in {detectors}
    recommended_threshold := input.false_positive_bands[detector].recommended_threshold
    msg := sprintf("Adjust threshold for %v to %v", [detector, recommended_threshold])
}}"""
    
    def _generate_incident_opa_policy(self) -> str:
        """Generate OPA policy for incident response."""
        return """package incidents

# Incident response policy
violation[msg] {
    hit := input.high_sev_hits[_]
    hit.severity == "high"
    msg := sprintf("High severity incident: %v", [hit.description])
}

# Incident escalation policy
violation[msg] {
    hit := input.high_sev_hits[_]
    hit.severity == "critical"
    msg := sprintf("Critical incident requiring immediate attention: %v", [hit.description])
}"""
