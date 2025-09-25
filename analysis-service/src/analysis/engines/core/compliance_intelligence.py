"""
Compliance Intelligence Engine

This engine provides compliance analysis and intelligence capabilities,
mapping security findings to compliance frameworks and requirements.
Follows SRP by focusing solely on compliance analysis and mapping.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from ...schemas.analysis_schemas import AnalysisRequest, AnalysisResult

logger = logging.getLogger(__name__)


class ComplianceIntelligenceEngine:
    """
    Compliance intelligence engine for framework mapping and analysis.

    Provides compliance analysis capabilities including:
    - Framework requirement mapping
    - Compliance gap analysis
    - Regulatory risk assessment
    - Audit trail generation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compliance_config = config.get("compliance", {})

        # Supported frameworks
        self.frameworks = {
            "soc2": self._initialize_soc2_mappings(),
            "iso27001": self._initialize_iso27001_mappings(),
            "hipaa": self._initialize_hipaa_mappings(),
            "gdpr": self._initialize_gdpr_mappings(),
        }

        logger.info("Compliance Intelligence Engine initialized")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the compliance intelligence engine."""
        return {
            "engine_type": "compliance_intelligence",
            "version": "1.0.0",
            "supported_frameworks": list(self.frameworks.keys()),
            "capabilities": [
                "framework_mapping",
                "compliance_gap_analysis",
                "regulatory_risk_assessment",
                "audit_trail_generation",
            ],
        }

    async def analyze_compliance(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Analyze compliance requirements and gaps.

        Args:
            request: Analysis request containing security findings

        Returns:
            AnalysisResult with compliance analysis
        """
        try:
            # Extract compliance-relevant data
            compliance_data = self._extract_compliance_data(request)

            # Analyze against all frameworks
            framework_analyses = {}
            for framework_name in self.frameworks.keys():
                analysis = await self._analyze_framework_compliance(
                    compliance_data, framework_name
                )
                framework_analyses[framework_name] = analysis

            # Identify compliance gaps
            compliance_gaps = self._identify_compliance_gaps(framework_analyses)

            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(framework_analyses)

            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(
                compliance_gaps, framework_analyses
            )

            # Create result
            result = AnalysisResult(
                analysis_type="compliance_intelligence",
                confidence=self._calculate_confidence(framework_analyses),
                compliance_assessment={
                    "overall_score": compliance_score,
                    "framework_analyses": framework_analyses,
                    "compliance_gaps": compliance_gaps,
                    "recommendations": recommendations,
                },
                evidence=self._create_compliance_evidence(compliance_data),
                metadata={
                    "frameworks_analyzed": list(self.frameworks.keys()),
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                    "engine": "compliance_intelligence",
                },
            )

            logger.info(
                "Compliance analysis completed",
                compliance_score=compliance_score,
                gaps_found=len(compliance_gaps),
            )

            return result

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Compliance analysis failed", error=str(e))
            return self._create_error_result(str(e))

    def _initialize_soc2_mappings(self) -> Dict[str, Any]:
        """Initialize SOC 2 framework mappings."""
        return {
            "controls": {
                "CC6.1": {
                    "name": "Logical Access Controls",
                    "requirements": [
                        "access_control",
                        "authentication",
                        "authorization",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
                "CC6.7": {
                    "name": "Data Transmission Controls",
                    "requirements": [
                        "encryption",
                        "data_protection",
                        "transmission_security",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
                "CC7.1": {
                    "name": "System Boundaries",
                    "requirements": [
                        "boundary_controls",
                        "network_security",
                        "perimeter_defense",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
            }
        }

    def _initialize_iso27001_mappings(self) -> Dict[str, Any]:
        """Initialize ISO 27001 framework mappings."""
        return {
            "controls": {
                "A.8.2.1": {
                    "name": "Data Classification",
                    "requirements": [
                        "data_classification",
                        "information_labeling",
                        "handling_procedures",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
                "A.13.2.1": {
                    "name": "Information Transfer Policies",
                    "requirements": [
                        "transfer_policies",
                        "secure_transfer",
                        "data_protection",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
            }
        }

    def _initialize_hipaa_mappings(self) -> Dict[str, Any]:
        """Initialize HIPAA framework mappings."""
        return {
            "safeguards": {
                "administrative": {
                    "name": "Administrative Safeguards",
                    "requirements": [
                        "access_management",
                        "workforce_training",
                        "incident_procedures",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
                "physical": {
                    "name": "Physical Safeguards",
                    "requirements": [
                        "facility_access",
                        "workstation_security",
                        "device_controls",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
                "technical": {
                    "name": "Technical Safeguards",
                    "requirements": [
                        "access_control",
                        "audit_controls",
                        "integrity",
                        "transmission_security",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
            }
        }

    def _initialize_gdpr_mappings(self) -> Dict[str, Any]:
        """Initialize GDPR framework mappings."""
        return {
            "principles": {
                "lawfulness": {
                    "name": "Lawfulness, Fairness and Transparency",
                    "requirements": ["legal_basis", "transparency", "fair_processing"],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
                "data_minimization": {
                    "name": "Data Minimization",
                    "requirements": [
                        "purpose_limitation",
                        "data_minimization",
                        "storage_limitation",
                    ],
                    "severity_mapping": {"critical": 1.0, "high": 0.8, "medium": 0.5},
                },
            }
        }

    def _extract_compliance_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Extract compliance-relevant data from request."""
        return {
            "security_findings": request.high_sev_hits + request.medium_sev_hits,
            "data_types": self._identify_data_types(request),
            "access_patterns": self._analyze_access_patterns(request),
            "control_gaps": self._identify_control_gaps(request),
        }

    def _identify_data_types(self, request: AnalysisRequest) -> List[str]:
        """Identify data types from security findings."""
        data_types = set()
        for finding in request.high_sev_hits + request.medium_sev_hits:
            detector = finding.get("detector", "").lower()
            if "pii" in detector:
                data_types.add("personal_data")
            if "financial" in detector:
                data_types.add("financial_data")
            if "health" in detector:
                data_types.add("health_data")
        return list(data_types)

    def _analyze_access_patterns(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze access patterns from request."""
        return {
            "unusual_access": len(
                [f for f in request.high_sev_hits if "access" in f.get("type", "")]
            ),
            "failed_authentication": len(
                [f for f in request.high_sev_hits if "auth" in f.get("type", "")]
            ),
            "privilege_escalation": len(
                [f for f in request.high_sev_hits if "privilege" in f.get("type", "")]
            ),
        }

    def _identify_control_gaps(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Identify control gaps from detector errors and coverage."""
        gaps = []

        # Coverage gaps
        for detector, observed in request.observed_coverage.items():
            required = request.required_coverage.get(detector, 1.0)
            if observed < required:
                gaps.append(
                    {
                        "type": "coverage_gap",
                        "detector": detector,
                        "gap_size": required - observed,
                        "severity": "high" if required - observed > 0.3 else "medium",
                    }
                )

        # Detector errors
        for detector, errors in request.detector_errors.items():
            if errors:
                gaps.append(
                    {
                        "type": "detector_error",
                        "detector": detector,
                        "error_count": len(errors) if isinstance(errors, list) else 1,
                        "severity": "critical",
                    }
                )

        return gaps

    async def _analyze_framework_compliance(
        self, compliance_data: Dict[str, Any], framework: str
    ) -> Dict[str, Any]:
        """Analyze compliance against a specific framework."""
        framework_config = self.frameworks.get(framework, {})

        # Analyze each control/requirement
        control_analyses = {}
        for control_id, control_config in framework_config.get("controls", {}).items():
            analysis = self._analyze_control_compliance(compliance_data, control_config)
            control_analyses[control_id] = analysis

        # Calculate framework score
        framework_score = self._calculate_framework_score(control_analyses)

        return {
            "framework": framework,
            "overall_score": framework_score,
            "control_analyses": control_analyses,
            "compliance_status": self._determine_compliance_status(framework_score),
        }

    def _analyze_control_compliance(
        self, compliance_data: Dict[str, Any], control_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze compliance for a specific control."""
        requirements = control_config.get("requirements", [])
        severity_mapping = control_config.get("severity_mapping", {})

        # Check findings against requirements
        relevant_findings = []
        for finding in compliance_data.get("security_findings", []):
            finding_type = finding.get("type", "").lower()
            if any(req in finding_type for req in requirements):
                relevant_findings.append(finding)

        # Calculate compliance score
        if not relevant_findings:
            compliance_score = 1.0  # No issues found
        else:
            # Weight by severity
            total_impact = 0.0
            for finding in relevant_findings:
                severity = finding.get("severity", "medium").lower()
                impact = severity_mapping.get(severity, 0.5)
                total_impact += impact

            # Normalize score (lower is better for compliance)
            compliance_score = max(0.0, 1.0 - (total_impact / len(relevant_findings)))

        return {
            "compliance_score": compliance_score,
            "relevant_findings": len(relevant_findings),
            "status": "compliant" if compliance_score > 0.8 else "non_compliant",
            "findings": relevant_findings[:5],  # Limit for brevity
        }

    def _calculate_framework_score(self, control_analyses: Dict[str, Any]) -> float:
        """Calculate overall framework compliance score."""
        if not control_analyses:
            return 0.0

        scores = [
            analysis["compliance_score"] for analysis in control_analyses.values()
        ]
        return sum(scores) / len(scores)

    def _determine_compliance_status(self, score: float) -> str:
        """Determine compliance status from score."""
        if score >= 0.9:
            return "fully_compliant"
        if score >= 0.7:
            return "mostly_compliant"
        if score >= 0.5:
            return "partially_compliant"
        return "non_compliant"

    def _identify_compliance_gaps(
        self, framework_analyses: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify compliance gaps across frameworks."""
        gaps = []

        for framework, analysis in framework_analyses.items():
            framework_score = analysis["overall_score"]
            if framework_score < 0.8:  # Below compliance threshold
                gaps.append(
                    {
                        "framework": framework,
                        "gap_score": 0.8 - framework_score,
                        "severity": "critical" if framework_score < 0.5 else "high",
                        "affected_controls": [
                            control_id
                            for control_id, control_analysis in analysis[
                                "control_analyses"
                            ].items()
                            if control_analysis["compliance_score"] < 0.8
                        ],
                    }
                )

        return gaps

    def _calculate_compliance_score(self, framework_analyses: Dict[str, Any]) -> float:
        """Calculate overall compliance score."""
        if not framework_analyses:
            return 0.0

        scores = [analysis["overall_score"] for analysis in framework_analyses.values()]
        return sum(scores) / len(scores)

    def _generate_compliance_recommendations(
        self, gaps: List[Dict[str, Any]], framework_analyses: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate compliance recommendations."""
        recommendations = []

        for gap in gaps:
            if gap["severity"] == "critical":
                recommendations.append(
                    {
                        "priority": "critical",
                        "framework": gap["framework"],
                        "title": f"Address Critical {gap['framework'].upper()} Compliance Gap",
                        "description": f"Compliance score of {gap['gap_score']:.2f} below threshold",
                        "affected_controls": gap["affected_controls"],
                    }
                )

        return recommendations[:10]  # Limit recommendations

    def _calculate_confidence(self, framework_analyses: Dict[str, Any]) -> float:
        """Calculate confidence in compliance analysis."""
        if not framework_analyses:
            return 0.0

        # Base confidence on data quality and coverage
        total_findings = sum(
            len(analysis["control_analyses"])
            for analysis in framework_analyses.values()
        )

        # More data = higher confidence
        confidence = min(1.0, total_findings / 10)
        return max(0.1, confidence)

    def _create_compliance_evidence(
        self, compliance_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create evidence for compliance analysis."""
        return [
            {
                "type": "compliance_analysis",
                "security_findings_analyzed": len(
                    compliance_data.get("security_findings", [])
                ),
                "data_types_identified": len(compliance_data.get("data_types", [])),
                "control_gaps_found": len(compliance_data.get("control_gaps", [])),
            }
        ]

    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error result when compliance analysis fails."""
        return AnalysisResult(
            analysis_type="compliance_intelligence",
            confidence=0.0,
            compliance_assessment={"overall_score": 0.0, "error": error_message},
            evidence=[{"type": "error", "message": error_message}],
            metadata={
                "error": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
