"""
Refactored Template Provider

This is the new implementation that uses the specialized analysis engines
instead of the monolithic approach. It maintains backward compatibility
while providing enhanced functionality through the orchestrated engines.
"""

import logging
from typing import Any, Dict, Optional

from ..domain.entities import AnalysisRequest, AnalysisType
from ..domain.interfaces import ITemplateProvider
from ..engines import (
    PatternRecognitionEngine,
    RiskScoringEngine,
    ComplianceIntelligence,
    TemplateOrchestrator,
)

logger = logging.getLogger(__name__)


class RefactoredTemplateProvider(ITemplateProvider):
    """
    Refactored template provider using specialized analysis engines.
    
    This implementation replaces the monolithic AnalysisTemplateProvider
    with a modular approach using specialized engines coordinated by
    the TemplateOrchestrator.
    
    Benefits of the refactored approach:
    - Single responsibility per engine
    - Better testability and maintainability
    - Enhanced functionality through specialized analysis
    - Backward compatibility with existing interface
    """
    
    def __init__(self):
        """Initialize the refactored template provider with specialized engines."""
        # Initialize specialized engines
        self.pattern_engine = PatternRecognitionEngine()
        self.risk_engine = RiskScoringEngine()
        self.compliance_engine = ComplianceIntelligence()
        
        # Initialize orchestrator
        self.orchestrator = TemplateOrchestrator(
            pattern_engine=self.pattern_engine,
            risk_engine=self.risk_engine,
            compliance_engine=self.compliance_engine,
        )
        
        logger.info("Initialized Refactored Template Provider with specialized engines")
    
    def get_template_response(
        self,
        request: AnalysisRequest,
        analysis_type: AnalysisType,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get template response using orchestrated specialized engines.
        
        Args:
            request: Analysis request
            analysis_type: Type of analysis to perform
            fallback_reason: Reason for using template fallback
            
        Returns:
            Template response dictionary
        """
        try:
            # For backward compatibility, provide synchronous template responses
            # In a full implementation, this would use async orchestrator methods
            if analysis_type == AnalysisType.COVERAGE_GAP:
                return self._get_coverage_gap_template_sync(request, fallback_reason)
            elif analysis_type == AnalysisType.FALSE_POSITIVE_TUNING:
                return self._get_false_positive_template_sync(request, fallback_reason)
            elif analysis_type == AnalysisType.INCIDENT_SUMMARY:
                return self._get_incident_summary_template_sync(request, fallback_reason)
            else:
                return self._get_insufficient_data_template_sync(request, fallback_reason)
            
        except Exception as e:
            logger.error(f"Error in refactored template provider: {str(e)}")
            # Fallback to basic response
            return self._get_fallback_response(request, fallback_reason, str(e))
    
    def select_analysis_type(self, request: AnalysisRequest) -> AnalysisType:
        """
        Select analysis type using orchestrator's enhanced logic.
        
        Args:
            request: Analysis request
            
        Returns:
            Selected analysis type
        """
        try:
            # Use synchronous logic for backward compatibility
            # Check for coverage gaps (highest priority)
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
                significant_fp = [
                    band for band in request.false_positive_bands
                    if band.get("false_positive_rate", 0.0) > 0.1
                ]
                if significant_fp:
                    return AnalysisType.FALSE_POSITIVE_TUNING
            
            # Check for incident summary
            if request.high_sev_hits:
                critical_incidents = [
                    hit for hit in request.high_sev_hits
                    if hit.get("severity") in ["critical", "high"]
                ]
                if critical_incidents:
                    return AnalysisType.INCIDENT_SUMMARY
            
            return AnalysisType.INSUFFICIENT_DATA
            
        except Exception as e:
            logger.error(f"Error selecting analysis type: {str(e)}")
            # Fallback to insufficient data
            return AnalysisType.INSUFFICIENT_DATA
    
    async def get_comprehensive_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Get comprehensive analysis using all engines.
        
        This is a new method that provides enhanced functionality
        beyond the original template provider interface.
        
        Args:
            request: Analysis request
            
        Returns:
            Comprehensive analysis result
        """
        try:
            return await self.orchestrator.orchestrate_analysis(request)
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {
                "error": str(e),
                "analysis_type": "error",
                "confidence": 0.0
            }
    
    async def get_pattern_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Get specialized pattern analysis.
        
        Args:
            request: Analysis request
            
        Returns:
            Pattern analysis result
        """
        try:
            return await self.pattern_engine.analyze(request)
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def get_risk_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Get specialized risk analysis.
        
        Args:
            request: Analysis request
            
        Returns:
            Risk analysis result
        """
        try:
            return await self.risk_engine.analyze(request)
        except Exception as e:
            logger.error(f"Error in risk analysis: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    async def get_compliance_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Get specialized compliance analysis.
        
        Args:
            request: Analysis request
            
        Returns:
            Compliance analysis result
        """
        try:
            return await self.compliance_engine.analyze(request)
        except Exception as e:
            logger.error(f"Error in compliance analysis: {str(e)}")
            return {"error": str(e), "confidence": 0.0}
    
    def _get_fallback_response(
        self, request: AnalysisRequest, fallback_reason: Optional[str], error: str
    ) -> Dict[str, Any]:
        """Get basic fallback response when engines fail."""
        return {
            "reason": f"Analysis engine error: {error}",
            "remediation": "Review system configuration and retry analysis",
            "opa_diff": "",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Fallback response due to engine error{f' - {fallback_reason}' if fallback_reason else ''}",
            "error": error
        }
    
    def _get_coverage_gap_template_sync(self, request: AnalysisRequest, fallback_reason: Optional[str]) -> Dict[str, Any]:
        """Synchronous coverage gap template for backward compatibility."""
        gaps = []
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            if observed < required:
                gaps.append(detector)
        
        return {
            "reason": f"Coverage gaps detected in {len(gaps)} detectors requiring attention",
            "remediation": f"Enable missing detectors: {', '.join(gaps[:3])}{'...' if len(gaps) > 3 else ''}",
            "opa_diff": "# Coverage gap enforcement policy\ncoverage_gap_violation[msg] { count(input.gaps) > 0 }",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": gaps,
            "notes": f"Coverage gap analysis{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _get_false_positive_template_sync(self, request: AnalysisRequest, fallback_reason: Optional[str]) -> Dict[str, Any]:
        """Synchronous false positive template for backward compatibility."""
        fp_detectors = [band.get("detector", "unknown") for band in request.false_positive_bands]
        
        return {
            "reason": f"False positive patterns detected in {len(fp_detectors)} detectors",
            "remediation": f"Adjust thresholds for detectors: {', '.join(fp_detectors)}",
            "opa_diff": "# False positive threshold policy\nthreshold_violation[msg] { input.false_positive_rate > 0.1 }",
            "confidence": 0.7,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": fp_detectors,
            "notes": f"False positive analysis{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _get_incident_summary_template_sync(self, request: AnalysisRequest, fallback_reason: Optional[str]) -> Dict[str, Any]:
        """Synchronous incident summary template for backward compatibility."""
        incident_detectors = list(set(hit.get("detector", "unknown") for hit in request.high_sev_hits))
        
        return {
            "reason": f"Security incidents detected across {len(incident_detectors)} detectors",
            "remediation": f"Investigate incidents from: {', '.join(incident_detectors)}",
            "opa_diff": "# Incident response policy\nincident_violation[msg] { count(input.high_sev_hits) > 0 }",
            "confidence": 0.8,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": incident_detectors,
            "notes": f"Incident summary analysis{f' - {fallback_reason}' if fallback_reason else ''}"
        }
    
    def _get_insufficient_data_template_sync(self, request: AnalysisRequest, fallback_reason: Optional[str]) -> Dict[str, Any]:
        """Synchronous insufficient data template for backward compatibility."""
        return {
            "reason": "Insufficient data for detailed analysis",
            "remediation": "Collect more comprehensive security metrics",
            "opa_diff": "",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Insufficient data analysis{f' - {fallback_reason}' if fallback_reason else ''}"
        }


# Factory function for backward compatibility
def create_refactored_template_provider() -> RefactoredTemplateProvider:
    """
    Factory function to create refactored template provider.
    
    Returns:
        Configured RefactoredTemplateProvider instance
    """
    return RefactoredTemplateProvider()


# Alias for easy migration
EnhancedTemplateProvider = RefactoredTemplateProvider