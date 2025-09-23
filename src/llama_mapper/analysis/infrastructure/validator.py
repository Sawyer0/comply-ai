"""
Infrastructure implementation of the validator for the Analysis Module.

This module contains the concrete implementation of the IValidator interface
for schema validation and template fallback.
"""

import json
import logging
from typing import Any, Dict, List

from ..domain.entities import AnalysisRequest
from ..domain.interfaces import IValidator
from ...serving.json_validator import JSONValidator
from ..validation.evidence_refs import validate_evidence_refs

logger = logging.getLogger(__name__)


class AnalysisValidator(IValidator):
    """
    Analysis validator implementation.
    
    Provides concrete implementation of the IValidator interface
    for validating analysis outputs and providing template fallback.
    """
    
    def __init__(self, schema_path: str = "schemas/analyst_output.json"):
        """
        Initialize the analysis validator.
        
        Args:
            schema_path: Path to the JSON schema file
        """
        self.schema_path = schema_path
        self.json_validator = JSONValidator(schema_path)
        
        logger.info(f"Initialized Analysis Validator with schema: {schema_path}")
    
    def validate_and_fallback(
        self, 
        model_output: Dict[str, Any], 
        request: AnalysisRequest
    ) -> Dict[str, Any]:
        """
        Validate output and fallback to templates on failure.
        
        Args:
            model_output: Model output to validate
            request: Original analysis request
            
        Returns:
            Validated output or template fallback
        """
        try:
            # Convert dict to JSON string for validation
            json_output = json.dumps(model_output)
            # Validate against schema
            is_valid, errors = self.json_validator.validate(json_output)
            
            if not is_valid:
                logger.warning(f"Model output failed schema validation: {errors}")
                # Return template fallback
                return self._get_template_fallback(request, f"Schema validation failed: {', '.join(errors)}")
            
            # Additional validation for evidence references
            self._validate_evidence_refs(model_output)
            
            # Additional validation for field constraints
            self._validate_field_constraints(model_output)
            
            logger.debug("Model output passed all validation")
            return model_output
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            # Return template fallback
            return self._get_template_fallback(request, f"Validation error: {str(e)}")
    
    def validate_schema_compliance(self, output: Dict[str, Any]) -> bool:
        """
        Check if output complies with schema without raising exceptions.
        
        Args:
            output: Output to validate
            
        Returns:
            True if compliant, False otherwise
        """
        try:
            # Convert dict to JSON string for validation
            json_output = json.dumps(output)
            is_valid, _ = self.json_validator.validate(json_output)
            return is_valid
        except Exception as e:
            logger.error(f"Schema compliance check error: {e}")
            return False
    
    def get_validation_errors(self, output: Dict[str, Any]) -> List[str]:
        """
        Get list of validation errors for output.
        
        Args:
            output: Output to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        try:
            # Convert dict to JSON string for validation
            json_output = json.dumps(output)
            _, schema_errors = self.json_validator.validate(json_output)
            if schema_errors:
                errors.extend(schema_errors)
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            errors.append(f"Schema validation error: {str(e)}")
        
        # Additional validation for evidence references
        try:
            self._validate_evidence_refs(output)
        except Exception as e:
            errors.append(str(e))
        
        # Additional validation for field constraints
        try:
            self._validate_field_constraints(output)
        except Exception as e:
            errors.append(str(e))
        
        return errors
    
    def _validate_evidence_refs(self, output: Dict[str, Any]) -> None:
        """
        Validate that evidence references are from allowed list.
        
        Args:
            output: Output to validate
            
        Raises:
            ValidationError: If evidence refs are invalid
        """
        evidence_refs = output.get("evidence_refs", [])
        
        is_valid, errors = validate_evidence_refs(evidence_refs)
        if not is_valid:
            raise ValueError(f"Invalid evidence references: {', '.join(errors)}")
    
    def _validate_field_constraints(self, output: Dict[str, Any]) -> None:
        """
        Validate additional field constraints.
        
        Args:
            output: Output to validate
            
        Raises:
            ValidationError: If constraints are violated
        """
        # Validate reason and remediation word count (≤20 words each)
        reason = output.get("reason", "")
        remediation = output.get("remediation", "")
        
        if len(reason.split()) > 20:
            raise ValueError("Reason must be ≤20 words")
        
        if len(remediation.split()) > 20:
            raise ValueError("Remediation must be ≤20 words")
        
        # Validate confidence range
        confidence = output.get("confidence", 0.0)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def _get_template_fallback(self, request: AnalysisRequest, reason: str) -> Dict[str, Any]:
        """
        Get template fallback response.
        
        Args:
            request: Original analysis request
            reason: Reason for fallback
            
        Returns:
            Template fallback response
        """
        # Determine analysis type
        analysis_type = self._determine_analysis_type(request)
        
        # Get template based on analysis type
        if analysis_type == "coverage_gap":
            return self._get_coverage_gap_template(request, reason)
        elif analysis_type == "false_positive_tuning":
            return self._get_false_positive_template(request, reason)
        elif analysis_type == "incident_summary":
            return self._get_incident_summary_template(request, reason)
        else:
            return self._get_insufficient_data_template(request, reason)
    
    def _determine_analysis_type(self, request: AnalysisRequest) -> str:
        """
        Determine the type of analysis needed based on request content.
        
        Args:
            request: Analysis request
            
        Returns:
            Analysis type string
        """
        # Check for coverage gaps
        coverage_gaps = []
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            if observed < required:
                coverage_gaps.append(detector)
        
        if coverage_gaps:
            return "coverage_gap"
        
        # Check for false positive tuning opportunities
        if request.false_positive_bands:
            return "false_positive_tuning"
        
        # Check for incident summary
        if request.high_sev_hits:
            return "incident_summary"
        
        return "insufficient_data"
    
    def _get_coverage_gap_template(self, request: AnalysisRequest, reason: str) -> Dict[str, Any]:
        """Get coverage gap template response."""
        return {
            "reason": "coverage gap detected - insufficient detector coverage",
            "remediation": "enable missing detectors or adjust coverage requirements",
            "opa_diff": "package coverage\n\nviolation[msg] {\n    detector := input.detectors[_]\n    not input.observed_coverage[detector]\n    msg := sprintf(\"Missing coverage for detector: %v\", [detector])\n}",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Template fallback due to: {reason}",
            "_template_fallback": True,
            "_fallback_reason": reason
        }
    
    def _get_false_positive_template(self, request: AnalysisRequest, reason: str) -> Dict[str, Any]:
        """Get false positive tuning template response."""
        return {
            "reason": "false positive patterns detected - threshold tuning needed",
            "remediation": "adjust detector thresholds based on false positive analysis",
            "opa_diff": "package thresholds\n\nviolation[msg] {\n    detector := input.detectors[_]\n    threshold := input.false_positive_bands[detector].recommended_threshold\n    msg := sprintf(\"Adjust threshold for %v to %v\", [detector, threshold])\n}",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": [band.get("detector", "unknown") for band in request.false_positive_bands],
            "notes": f"Template fallback due to: {reason}",
            "_template_fallback": True,
            "_fallback_reason": reason
        }
    
    def _get_incident_summary_template(self, request: AnalysisRequest, reason: str) -> Dict[str, Any]:
        """Get incident summary template response."""
        return {
            "reason": "high severity security incidents detected",
            "remediation": "investigate and remediate high severity findings immediately",
            "opa_diff": "package incidents\n\nviolation[msg] {\n    hit := input.high_sev_hits[_]\n    hit.severity == \"high\"\n    msg := sprintf(\"High severity incident: %v\", [hit.description])\n}",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": [hit.get("detector", "unknown") for hit in request.high_sev_hits],
            "notes": f"Template fallback due to: {reason}",
            "_template_fallback": True,
            "_fallback_reason": reason
        }
    
    def _get_insufficient_data_template(self, request: AnalysisRequest, reason: str) -> Dict[str, Any]:
        """Get insufficient data template response."""
        return {
            "reason": "insufficient data for detailed analysis",
            "remediation": "collect more comprehensive security metrics",
            "opa_diff": "",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Template fallback due to: {reason}",
            "_template_fallback": True,
            "_fallback_reason": reason
        }