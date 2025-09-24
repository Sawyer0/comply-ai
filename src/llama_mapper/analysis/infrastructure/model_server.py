"""
Infrastructure implementation of the model server for the Analysis Module.

This module contains the concrete implementation of the IModelServer interface
using Phi-3 Mini for analysis generation.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

from ...config.settings import Settings
from ..domain.entities import AnalysisRequest, VersionInfo
from ..domain.interfaces import IModelServer

logger = logging.getLogger(__name__)


class Phi3AnalysisModelServer(IModelServer):
    """
    Phi-3 Mini model server implementation for analysis generation.

    Provides concrete implementation of the IModelServer interface
    using Phi-3 Mini for generating analysis insights.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.1,
        confidence_cutoff: float = 0.3,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize the Phi-3 analysis model server.

        Args:
            model_path: Path to the Phi-3 Mini model
            temperature: Generation temperature
            confidence_cutoff: Confidence cutoff for fallback
            settings: Application settings
        """
        self.model_path = model_path
        self.temperature = temperature
        self.confidence_cutoff = confidence_cutoff
        self.settings = settings or Settings()

        # Model loading would happen here in production
        self._model_loaded = False
        self._version_info = VersionInfo(
            taxonomy="v1.0.0", frameworks="v1.0.0", analyst_model="phi3-mini-v1.0.0"
        )

        logger.info(
            "Initialized Phi-3 Analysis Model Server with model: %s", model_path
        )

    @property
    def version_info(self) -> VersionInfo:
        """
        Get version information for the model server.

        Returns:
            VersionInfo object containing version information
        """
        return self._version_info

    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Generate analysis using the Phi-3 Mini model.

        Args:
            request: Analysis request with structured metrics

        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()

        try:
            # Build prompt from request
            prompt = self._build_analysis_prompt(request)

            # Generate analysis (in production, this would call the actual model)
            analysis_output = await self._generate_analysis(prompt, request)

            # Compute confidence score
            confidence = self._compute_confidence(analysis_output, request)

            # Apply confidence governance
            if confidence < self.confidence_cutoff:
                analysis_output["_template_fallback"] = True
                analysis_output["_fallback_reason"] = (
                    f"Low confidence: {confidence:.3f} < {self.confidence_cutoff}"
                )

            # Add metadata
            analysis_output.update(
                {
                    "confidence": confidence,
                    "confidence_cutoff_used": self.confidence_cutoff,
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "version_info": self._version_info.dict(),
                }
            )

            return analysis_output

        except Exception as e:
            logger.error("Model analysis error: %s", e)
            # Return fallback response
            return {
                "reason": "analysis error - unable to process request",
                "remediation": "check request format and retry",
                "opa_diff": "",
                "confidence": 0.1,
                "confidence_cutoff_used": self.confidence_cutoff,
                "evidence_refs": ["required_detectors"],
                "notes": f"Model analysis failed: {str(e)}",
                "_template_fallback": True,
                "_fallback_reason": f"Model error: {str(e)}",
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "version_info": self._version_info.dict(),
            }

    async def analyze_batch(
        self, requests: List[AnalysisRequest], idempotency_key: str
    ) -> List[Dict[str, Any]]:
        """
        Process batch requests with idempotency support.

        Args:
            requests: List of analysis requests
            idempotency_key: Idempotency key for caching

        Returns:
            List of analysis results
        """
        # Process requests concurrently
        tasks = [self.analyze(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Batch analysis error for request %s: %s", i, result)
                processed_results.append(
                    {
                        "reason": "batch processing error",
                        "remediation": "retry individual request",
                        "opa_diff": "",
                        "confidence": 0.1,
                        "confidence_cutoff_used": self.confidence_cutoff,
                        "evidence_refs": ["required_detectors"],
                        "notes": f"Batch processing failed: {str(result)}",
                        "_template_fallback": True,
                        "_fallback_reason": f"Batch error: {str(result)}",
                        "processing_time_ms": 0,
                        "version_info": self._version_info.dict(),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    def _build_analysis_prompt(self, request: AnalysisRequest) -> str:
        """
        Build analysis prompt from request.

        Args:
            request: Analysis request

        Returns:
            Formatted prompt string
        """
        # Determine analysis type based on request content
        analysis_type = self._determine_analysis_type(request)

        # Build context from request data
        context = {
            "period": request.period,
            "tenant": request.tenant,
            "app": request.app,
            "route": request.route,
            "required_detectors": request.required_detectors,
            "observed_coverage": request.observed_coverage,
            "required_coverage": request.required_coverage,
            "detector_errors": request.detector_errors,
            "high_sev_hits": request.high_sev_hits,
            "false_positive_bands": request.false_positive_bands,
            "policy_bundle": request.policy_bundle,
            "env": request.env,
        }

        # Build type-specific prompt
        if analysis_type == "coverage_gap":
            return self._build_coverage_gap_prompt(context)
        elif analysis_type == "false_positive_tuning":
            return self._build_false_positive_prompt(context)
        elif analysis_type == "incident_summary":
            return self._build_incident_summary_prompt(context)
        else:
            return self._build_generic_prompt(context)

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

    def _build_coverage_gap_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for coverage gap analysis."""
        return f"""
Analyze the following coverage gap scenario and provide insights:

Context:
- Period: {context['period']}
- Tenant: {context['tenant']}
- App: {context['app']}
- Route: {context['route']}
- Environment: {context['env']}

Coverage Analysis:
- Required Detectors: {context['required_detectors']}
- Observed Coverage: {context['observed_coverage']}
- Required Coverage: {context['required_coverage']}

Detector Errors:
{json.dumps(context['detector_errors'], indent=2)}

Provide:
1. Reason: Concise explanation of the coverage gap issue
2. Remediation: Suggested remediation steps
3. OPA Diff: OPA/Rego policy to enforce coverage requirements
4. Evidence Refs: List of relevant detector names
5. Notes: Additional context or recommendations

Format as JSON with keys: reason, remediation, opa_diff, evidence_refs, notes
"""

    def _build_false_positive_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for false positive tuning analysis."""
        return f"""
Analyze the following false positive tuning scenario:

Context:
- Period: {context['period']}
- Tenant: {context['tenant']}
- App: {context['app']}
- Route: {context['route']}
- Environment: {context['env']}

False Positive Bands:
{json.dumps(context['false_positive_bands'], indent=2)}

Detector Errors:
{json.dumps(context['detector_errors'], indent=2)}

Provide:
1. Reason: Explanation of false positive patterns
2. Remediation: Threshold adjustment recommendations
3. OPA Diff: OPA/Rego policy for threshold enforcement
4. Evidence Refs: List of relevant detector names
5. Notes: Additional tuning recommendations

Format as JSON with keys: reason, remediation, opa_diff, evidence_refs, notes
"""

    def _build_incident_summary_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for incident summary analysis."""
        return f"""
Analyze the following incident summary scenario:

Context:
- Period: {context['period']}
- Tenant: {context['tenant']}
- App: {context['app']}
- Route: {context['route']}
- Environment: {context['env']}

High Severity Hits:
{json.dumps(context['high_sev_hits'], indent=2)}

Detector Errors:
{json.dumps(context['detector_errors'], indent=2)}

Provide:
1. Reason: Summary of the incident
2. Remediation: Recommended response actions
3. OPA Diff: OPA/Rego policy for incident response
4. Evidence Refs: List of relevant detector names
5. Notes: Additional incident context

Format as JSON with keys: reason, remediation, opa_diff, evidence_refs, notes
"""

    def _build_generic_prompt(self, context: Dict[str, Any]) -> str:
        """Build generic analysis prompt."""
        return f"""
Analyze the following security metrics scenario:

Context:
- Period: {context['period']}
- Tenant: {context['tenant']}
- App: {context['app']}
- Route: {context['route']}
- Environment: {context['env']}

Metrics:
- Required Detectors: {context['required_detectors']}
- Observed Coverage: {context['observed_coverage']}
- Required Coverage: {context['required_coverage']}
- Detector Errors: {json.dumps(context['detector_errors'], indent=2)}

Provide:
1. Reason: General analysis of the security posture
2. Remediation: General improvement recommendations
3. OPA Diff: OPA/Rego policy for general enforcement
4. Evidence Refs: List of relevant detector names
5. Notes: Additional observations

Format as JSON with keys: reason, remediation, opa_diff, evidence_refs, notes
"""

    async def _generate_analysis(
        self, prompt: str, request: AnalysisRequest
    ) -> Dict[str, Any]:
        """
        Generate analysis using the model.

        Args:
            prompt: Analysis prompt
            request: Original request

        Returns:
            Analysis output dictionary
        """
        # In production, this would call the actual Phi-3 Mini model
        # For now, we'll simulate the model response

        await asyncio.sleep(0.1)  # Simulate model processing time

        # Simulate model output based on analysis type
        analysis_type = self._determine_analysis_type(request)

        if analysis_type == "coverage_gap":
            # Generate comprehensive OPA policy with threshold enforcement
            opa_policy = f"""package coverage

# Coverage enforcement policy - missing detectors
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
    msg := sprintf("Insufficient coverage for %v: observed %v < required %v", [detector, observed, required])
}}"""

            return {
                "reason": "coverage gap detected - insufficient detector coverage",
                "remediation": "enable missing detectors or adjust coverage requirements",
                "opa_diff": opa_policy,
                "evidence_refs": [
                    "required_detectors",
                    "observed_coverage",
                    "required_coverage",
                ],
                "notes": "Coverage gap analysis based on required vs observed detector coverage",
            }
        elif analysis_type == "false_positive_tuning":
            return {
                "reason": "false positive patterns detected - threshold tuning needed",
                "remediation": "adjust detector thresholds based on false positive analysis",
                "opa_diff": 'package thresholds\n\nviolation[msg] {\n    detector := input.detectors[_]\n    threshold := input.false_positive_bands[detector].recommended_threshold\n    msg := sprintf("Adjust threshold for %v to %v", [detector, threshold])\n}',
                "evidence_refs": ["false_positive_bands", "required_detectors"],
                "notes": "False positive tuning recommendations based on historical patterns",
            }
        elif analysis_type == "incident_summary":
            return {
                "reason": "high severity security incidents detected",
                "remediation": "investigate and remediate high severity findings immediately",
                "opa_diff": 'package incidents\n\nviolation[msg] {\n    hit := input.high_sev_hits[_]\n    hit.severity == "high"\n    msg := sprintf("High severity incident: %v", [hit.description])\n}',
                "evidence_refs": ["high_sev_hits", "required_detectors"],
                "notes": "Incident summary based on high severity security findings",
            }
        else:
            return {
                "reason": "insufficient data for detailed analysis",
                "remediation": "collect more comprehensive security metrics",
                "opa_diff": "",
                "evidence_refs": ["required_detectors"],
                "notes": "Generic analysis due to insufficient specific data",
            }

    def _compute_confidence(
        self, analysis_output: Dict[str, Any], request: AnalysisRequest
    ) -> float:
        """
        Compute confidence score for the analysis output.

        Args:
            analysis_output: Analysis output dictionary
            request: Original request

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence
        confidence = 0.8

        # Adjust based on data quality
        if not request.high_sev_hits and not request.false_positive_bands:
            confidence -= 0.2

        if not request.detector_errors:
            confidence += 0.1

        # Adjust based on analysis type
        analysis_type = self._determine_analysis_type(request)
        if analysis_type == "insufficient_data":
            confidence -= 0.3

        # Ensure confidence is within bounds
        return max(0.0, min(1.0, confidence))
