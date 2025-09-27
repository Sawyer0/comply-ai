"""
Core Analysis Engine for the Analysis Service.

Orchestrates all analysis engines following SRP.
This is the main entry point for analysis operations.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from ..shared_integration import get_shared_logger
from ..validation import InputValidator, OutputValidator

logger = get_shared_logger(__name__)


@dataclass
class AnalysisRequest:
    """Analysis request structure."""

    tenant_id: str
    analysis_types: List[str]
    findings: Dict[str, Any]
    context: Optional[str] = None
    frameworks: Optional[List[str]] = None


@dataclass
class AnalysisResult:
    """Analysis result structure."""

    request_id: str
    status: str
    results: Dict[str, Any]
    processing_time_ms: float
    confidence: Optional[float] = None


class AnalysisEngine:
    """Main analysis engine that orchestrates all analysis components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger.bind(component="analysis_engine")

        # Initialize validators
        self._validators = {
            "input": InputValidator(),
            "output": OutputValidator(),
        }

        self.logger.info("Analysis engine initialized")

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform comprehensive analysis based on request."""
        start_time = datetime.utcnow()
        request_id = f"analysis_{start_time.timestamp()}"

        try:
            # Validate input
            validation_result = self._validators["input"].validate_analysis_request(
                request.__dict__
            )

            if not validation_result.is_valid:
                return AnalysisResult(
                    request_id=request_id,
                    status="failed",
                    results={"errors": validation_result.errors},
                    processing_time_ms=0,
                )

            # Process analysis types
            results = await self._process_analysis_types(request)

            # Calculate processing time and confidence
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            confidence = self._calculate_overall_confidence(results)

            return AnalysisResult(
                request_id=request_id,
                status="completed",
                results=results,
                processing_time_ms=processing_time,
                confidence=confidence,
            )

        except ValueError as e:
            self.logger.error(
                "Analysis validation failed", request_id=request_id, error=str(e)
            )
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AnalysisResult(
                request_id=request_id,
                status="failed",
                results={"error": str(e)},
                processing_time_ms=processing_time,
            )

    async def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all engines."""
        return {
            "validators_loaded": list(self._validators.keys()),
            "config_keys": list(self.config.keys()),
            "status": "operational",
        }

    async def _process_analysis_types(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process different analysis types."""
        results = {}

        for analysis_type in request.analysis_types:
            if analysis_type == "risk_assessment":
                results["risk_assessment"] = await self._perform_risk_assessment(
                    request
                )
            elif analysis_type == "pattern_analysis":
                results["pattern_analysis"] = await self._perform_pattern_analysis(
                    request
                )
            elif analysis_type == "compliance_mapping":
                results["compliance_mapping"] = await self._perform_compliance_mapping(
                    request
                )

        return results

    def _calculate_overall_confidence(self, results: Dict[str, Any]) -> Optional[float]:
        """Calculate overall confidence from results."""
        confidences = []
        for result in results.values():
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])

        return sum(confidences) / len(confidences) if confidences else None

    async def _perform_risk_assessment(
        self, request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Perform risk assessment analysis."""
        try:
            # Simplified risk assessment using rule-based approach
            findings = request.findings

            # Calculate basic risk score
            risk_score = 0.0
            risk_factors = []

            # Check for high severity findings
            if "high_severity" in findings and findings["high_severity"]:
                risk_score += 0.8
                risk_factors.append("high_severity_findings")

            # Check for medium severity findings
            if "medium_severity" in findings and findings["medium_severity"]:
                risk_score += 0.5
                risk_factors.append("medium_severity_findings")

            # Check for low severity findings
            if "low_severity" in findings and findings["low_severity"]:
                risk_score += 0.2
                risk_factors.append("low_severity_findings")

            # Normalize risk score
            risk_score = min(1.0, risk_score)

            # Determine risk level
            if risk_score >= 0.8:
                risk_level = "HIGH"
            elif risk_score >= 0.5:
                risk_level = "MEDIUM"
            elif risk_score >= 0.2:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"

            return {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "confidence": 0.8,
                "recommendations": self._generate_risk_recommendations(risk_level),
            }

        except (ValueError, AttributeError) as e:
            self.logger.error("Risk assessment failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}

    async def _perform_pattern_analysis(
        self, request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Perform pattern analysis."""
        try:
            findings = request.findings

            # Simple pattern detection
            patterns_detected = []

            # Check for temporal patterns
            if "events" in findings and len(findings["events"]) > 1:
                patterns_detected.append(
                    {
                        "type": "temporal_sequence",
                        "confidence": 0.7,
                        "description": "Multiple events detected in sequence",
                    }
                )

            # Check for frequency patterns
            if "frequency" in findings and findings["frequency"] > 5:
                patterns_detected.append(
                    {
                        "type": "high_frequency",
                        "confidence": 0.8,
                        "description": "High frequency activity detected",
                    }
                )

            return {
                "patterns": patterns_detected,
                "pattern_count": len(patterns_detected),
                "confidence": 0.7,
            }

        except (ValueError, AttributeError) as e:
            self.logger.error("Pattern analysis failed", error=str(e))
            return {"error": str(e), "confidence": 0.0}

    async def _perform_compliance_mapping(
        self, request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Perform compliance mapping analysis."""
        try:
            frameworks = request.frameworks or ["General"]
            findings = request.findings

            compliance_results = []

            for framework in frameworks:
                # Simple compliance mapping
                compliance_score = 0.8  # Default compliance score

                # Adjust based on findings
                if "violations" in findings and findings["violations"]:
                    compliance_score -= 0.3

                if "gaps" in findings and findings["gaps"]:
                    compliance_score -= 0.2

                compliance_score = max(0.0, compliance_score)

                compliance_results.append(
                    {
                        "framework": framework,
                        "compliance_score": compliance_score,
                        "status": (
                            "compliant" if compliance_score >= 0.7 else "non_compliant"
                        ),
                    }
                )

            return {
                "compliance_mappings": compliance_results,
                "recommendations": self._generate_compliance_recommendations(
                    compliance_results
                ),
            }

        except (ValueError, AttributeError) as e:
            self.logger.error("Compliance mapping failed", error=str(e))
            return {"error": str(e)}

    def _generate_risk_recommendations(self, risk_level: str) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = {
            "HIGH": [
                "Implement immediate access restrictions",
                "Enable comprehensive audit logging",
                "Conduct security review within 24 hours",
            ],
            "MEDIUM": [
                "Review access permissions",
                "Enable monitoring and alerting",
                "Consider additional security controls",
            ],
            "LOW": ["Monitor for unusual activity", "Review data handling procedures"],
            "MINIMAL": ["Continue regular monitoring"],
        }

        return recommendations.get(risk_level, ["Review security posture"])

    def _generate_compliance_recommendations(
        self, compliance_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        for result in compliance_results:
            if result["status"] == "non_compliant":
                framework = result["framework"]
                recommendations.append(f"Address compliance gaps for {framework}")
                recommendations.append(f"Review {framework} requirements")

        if not recommendations:
            recommendations.append("Maintain current compliance posture")

        return recommendations
