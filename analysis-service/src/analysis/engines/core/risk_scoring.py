"""
Consolidated Risk Scoring Engine

This engine merges all risk scoring implementations into a single comprehensive system,
consolidating technical, business, regulatory, and temporal risk assessment capabilities.
Follows SRP by focusing solely on risk calculation and scoring.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from ...schemas.analysis_schemas import AnalysisRequest, AnalysisResult
from ...taxonomy.risk_taxonomy import RiskTaxonomy
from ..statistical import CompoundRiskCalculator
from ..optimization import RiskFactorAnalyzer

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """
    Consolidated risk scoring engine merging all risk scoring implementations.

    Consolidates capabilities from:
    - Technical risk assessment
    - Business impact scoring
    - Regulatory compliance risk
    - Temporal risk factors
    - Compound risk calculations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get("risk_scoring", {})
        self.taxonomy = RiskTaxonomy()

        # Risk calculation weights
        self.weights = {
            "technical": self.risk_config.get("technical_weight", 0.3),
            "business": self.risk_config.get("business_weight", 0.25),
            "regulatory": self.risk_config.get("regulatory_weight", 0.25),
            "temporal": self.risk_config.get("temporal_weight", 0.2),
        }

        # Initialize sophisticated calculators
        self.compound_calculator = CompoundRiskCalculator(self.risk_config)
        self.risk_factor_analyzer = RiskFactorAnalyzer(self.risk_config)

        # Risk thresholds
        self.thresholds = {
            "critical": self.risk_config.get("critical_threshold", 0.8),
            "high": self.risk_config.get("high_threshold", 0.6),
            "medium": self.risk_config.get("medium_threshold", 0.3),
            "low": self.risk_config.get("low_threshold", 0.0),
        }

        logger.info("Risk Scoring Engine initialized with consolidated capabilities")

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the risk scoring engine."""
        return {
            "engine_type": "consolidated_risk_scoring",
            "version": "1.0.0",
            "weights": self.weights,
            "thresholds": self.thresholds,
            "capabilities": [
                "technical_risk_assessment",
                "business_impact_scoring",
                "regulatory_compliance_risk",
                "temporal_risk_factors",
                "compound_risk_calculations",
            ],
        }

    async def calculate_risk_score(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Calculate comprehensive risk score for the analysis request.

        Args:
            request: Analysis request containing security findings

        Returns:
            AnalysisResult with comprehensive risk assessment
        """
        try:
            # Extract risk factors from request
            risk_factors = self._extract_risk_factors(request)

            # Calculate individual risk dimensions
            technical_risk = await self._calculate_technical_risk(risk_factors)
            business_risk = await self._calculate_business_risk(risk_factors)
            regulatory_risk = await self._calculate_regulatory_risk(risk_factors)
            temporal_risk = await self._calculate_temporal_risk(risk_factors)

            # Calculate compound risk using sophisticated algorithms
            compound_risk = await self._calculate_compound_risk(
                technical_risk,
                business_risk,
                regulatory_risk,
                temporal_risk,
            )

            # Calculate composite score
            composite_score = self._calculate_composite_score(
                technical_risk,
                business_risk,
                regulatory_risk,
                temporal_risk,
                compound_risk,
            )

            # Determine risk level
            risk_level = self._determine_risk_level(composite_score)

            # Create risk breakdown
            risk_breakdown = self._create_risk_breakdown(
                technical_risk,
                business_risk,
                regulatory_risk,
                temporal_risk,
                compound_risk,
            )

            # Calculate confidence
            confidence = self._calculate_confidence(risk_factors, risk_breakdown)

            # Create comprehensive result
            result = AnalysisResult(
                analysis_type="risk_scoring",
                confidence=confidence,
                risk_score={
                    "composite_score": composite_score,
                    "risk_level": risk_level,
                    "breakdown": risk_breakdown,
                    "compound_factors": compound_risk,
                },
                evidence=self._create_evidence(risk_factors, risk_breakdown),
                metadata=self._create_metadata(composite_score, risk_level),
            )

            logger.info(
                "Risk scoring completed",
                composite_score=composite_score,
                risk_level=risk_level,
                confidence=confidence,
            )

            return result

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Risk scoring failed", error=str(e))
            return self._create_error_result(str(e))

    async def _calculate_technical_risk(
        self, risk_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate technical risk based on security findings and detector performance."""
        try:
            # Severity-based risk
            severity_scores = []
            for hit in risk_factors.get("high_severity_hits", []):
                severity = hit.get("severity", "medium").lower()
                severity_score = {
                    "critical": 1.0,
                    "high": 0.8,
                    "medium": 0.5,
                    "low": 0.2,
                }.get(severity, 0.5)
                severity_scores.append(severity_score)

            avg_severity = (
                sum(severity_scores) / len(severity_scores) if severity_scores else 0.0
            )

            # Detector error impact
            error_count = len(risk_factors.get("detector_errors", {}))
            error_impact = min(
                1.0, error_count * 0.1
            )  # Each error adds 10% risk, capped at 100%

            # Coverage gap impact
            coverage_gaps = risk_factors.get("coverage_gaps", [])
            gap_impact = (
                sum(gap.get("impact", 0.0) for gap in coverage_gaps)
                / len(coverage_gaps)
                if coverage_gaps
                else 0.0
            )

            # Calculate technical risk score
            technical_score = (
                (avg_severity * 0.5) + (error_impact * 0.3) + (gap_impact * 0.2)
            )

            return {
                "score": min(1.0, technical_score),
                "components": {
                    "severity_risk": avg_severity,
                    "error_risk": error_impact,
                    "coverage_risk": gap_impact,
                },
                "contributing_factors": [
                    f"High severity findings: {len(severity_scores)}",
                    f"Detector errors: {error_count}",
                    f"Coverage gaps: {len(coverage_gaps)}",
                ],
            }

        except (ValueError, KeyError, ZeroDivisionError) as e:
            logger.error("Technical risk calculation failed", error=str(e))
            return {"score": 0.5, "error": str(e)}

    async def _calculate_business_risk(
        self, risk_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate business impact risk."""
        try:
            # Data sensitivity impact
            sensitive_data_types = risk_factors.get("sensitive_data_types", [])
            sensitivity_score = (
                len(sensitive_data_types) * 0.2
            )  # Each type adds 20% risk

            # Business process impact
            affected_processes = risk_factors.get("affected_business_processes", [])
            process_impact = (
                len(affected_processes) * 0.15
            )  # Each process adds 15% risk

            # Compliance requirements impact
            compliance_requirements = risk_factors.get("compliance_requirements", [])
            compliance_impact = (
                len(compliance_requirements) * 0.25
            )  # Each requirement adds 25% risk

            # Calculate business risk score
            business_score = sensitivity_score + process_impact + compliance_impact

            return {
                "score": min(1.0, business_score),
                "components": {
                    "data_sensitivity": sensitivity_score,
                    "process_impact": process_impact,
                    "compliance_impact": compliance_impact,
                },
                "contributing_factors": [
                    f"Sensitive data types: {len(sensitive_data_types)}",
                    f"Affected processes: {len(affected_processes)}",
                    f"Compliance requirements: {len(compliance_requirements)}",
                ],
            }

        except (ValueError, KeyError) as e:
            logger.error("Business risk calculation failed", error=str(e))
            return {"score": 0.5, "error": str(e)}

    async def _calculate_regulatory_risk(
        self, risk_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate regulatory compliance risk."""
        try:
            # Framework violations
            framework_violations = risk_factors.get("framework_violations", {})
            violation_score = 0.0

            framework_weights = {
                "hipaa": 0.3,
                "gdpr": 0.3,
                "pci_dss": 0.25,
                "soc2": 0.2,
                "iso27001": 0.15,
            }

            for framework, violations in framework_violations.items():
                weight = framework_weights.get(framework.lower(), 0.1)
                violation_count = (
                    len(violations) if isinstance(violations, list) else violations
                )
                violation_score += weight * min(1.0, violation_count * 0.2)

            # Audit findings impact
            audit_findings = risk_factors.get("audit_findings", [])
            audit_impact = len(audit_findings) * 0.15

            # Calculate regulatory risk score
            regulatory_score = violation_score + audit_impact

            return {
                "score": min(1.0, regulatory_score),
                "components": {
                    "framework_violations": violation_score,
                    "audit_findings": audit_impact,
                },
                "contributing_factors": [
                    f"Framework violations: {len(framework_violations)}",
                    f"Audit findings: {len(audit_findings)}",
                ],
            }

        except (ValueError, KeyError) as e:
            logger.error("Regulatory risk calculation failed", error=str(e))
            return {"score": 0.5, "error": str(e)}

    async def _calculate_temporal_risk(
        self, risk_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate temporal risk factors."""
        try:
            # Time-based risk escalation
            current_time = datetime.now(timezone.utc)

            # Recent incident frequency
            recent_incidents = risk_factors.get("recent_incidents", [])
            incident_frequency = (
                len(recent_incidents) / 7
            )  # Incidents per day over last week
            frequency_risk = min(1.0, incident_frequency * 0.1)

            # Trend analysis
            trend_direction = risk_factors.get("trend_direction", "stable")
            trend_risk = {"increasing": 0.3, "stable": 0.1, "decreasing": 0.0}.get(
                trend_direction, 0.1
            )

            # Time since last assessment
            last_assessment = risk_factors.get("last_assessment_time")
            if last_assessment:
                time_delta = current_time - last_assessment
                staleness_risk = min(
                    0.3, time_delta.days * 0.01
                )  # 1% per day, capped at 30%
            else:
                staleness_risk = 0.3  # High risk if never assessed

            # Calculate temporal risk score
            temporal_score = frequency_risk + trend_risk + staleness_risk

            return {
                "score": min(1.0, temporal_score),
                "components": {
                    "frequency_risk": frequency_risk,
                    "trend_risk": trend_risk,
                    "staleness_risk": staleness_risk,
                },
                "contributing_factors": [
                    f"Recent incidents: {len(recent_incidents)}",
                    f"Trend: {trend_direction}",
                    f"Assessment staleness: {staleness_risk:.2f}",
                ],
            }

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Temporal risk calculation failed", error=str(e))
            return {"score": 0.5, "error": str(e)}

    async def _calculate_compound_risk(
        self,
        technical_risk: Dict[str, Any],
        business_risk: Dict[str, Any],
        regulatory_risk: Dict[str, Any],
        temporal_risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate compound risk using sophisticated algorithms."""
        try:
            # Use the sophisticated compound risk calculator
            compound_result = await self.compound_calculator.calculate_compound_risk(
                patterns=[],  # Would convert risk factors to patterns in production
                pattern_relationships=[],
                context_data=None,  # Would create appropriate context in production
            )

            # Extract compound factors
            risk_amplification = self._calculate_risk_amplification(
                technical_risk, business_risk, regulatory_risk, temporal_risk
            )

            risk_correlation = self._calculate_risk_correlation(
                technical_risk, business_risk, regulatory_risk, temporal_risk
            )

            return {
                "amplification_factor": risk_amplification,
                "correlation_factor": risk_correlation,
                "compound_score": compound_result.get("risk_level", "medium"),
                "interaction_effects": self._analyze_risk_interactions(
                    technical_risk, business_risk, regulatory_risk, temporal_risk
                ),
            }

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Compound risk calculation failed", error=str(e))
            return {
                "amplification_factor": 1.0,
                "correlation_factor": 0.0,
                "error": str(e),
            }

    def _calculate_risk_amplification(self, *risk_dimensions) -> float:
        """Calculate risk amplification factor when multiple high risks are present."""
        high_risk_count = sum(
            1 for risk in risk_dimensions if risk.get("score", 0) > 0.7
        )

        if high_risk_count >= 3:
            return 1.3  # 30% amplification for 3+ high risks
        if high_risk_count == 2:
            return 1.2  # 20% amplification for 2 high risks
        return 1.0  # No amplification

    def _calculate_risk_correlation(self, *risk_dimensions) -> float:
        """Calculate correlation between risk dimensions."""
        scores = [risk.get("score", 0) for risk in risk_dimensions]

        # Simple correlation calculation
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

        # Higher variance indicates lower correlation
        correlation = max(0.0, 1.0 - (variance * 2))
        return correlation

    def _analyze_risk_interactions(self, *risk_dimensions) -> List[Dict[str, Any]]:
        """Analyze interactions between risk dimensions."""
        interactions = []

        dimension_names = ["technical", "business", "regulatory", "temporal"]

        for i, risk1 in enumerate(risk_dimensions):
            for j, risk2 in enumerate(risk_dimensions[i + 1 :], i + 1):
                score1 = risk1.get("score", 0)
                score2 = risk2.get("score", 0)

                if score1 > 0.6 and score2 > 0.6:
                    interaction_strength = (score1 + score2) / 2
                    interactions.append(
                        {
                            "dimension1": dimension_names[i],
                            "dimension2": dimension_names[j],
                            "interaction_strength": interaction_strength,
                            "effect": (
                                "amplification"
                                if interaction_strength > 0.8
                                else "correlation"
                            ),
                        }
                    )

        return interactions

    def _calculate_composite_score(
        self,
        technical_risk: Dict[str, Any],
        business_risk: Dict[str, Any],
        regulatory_risk: Dict[str, Any],
        temporal_risk: Dict[str, Any],
        compound_risk: Dict[str, Any],
    ) -> float:
        """Calculate weighted composite risk score."""
        base_score = (
            technical_risk.get("score", 0) * self.weights["technical"]
            + business_risk.get("score", 0) * self.weights["business"]
            + regulatory_risk.get("score", 0) * self.weights["regulatory"]
            + temporal_risk.get("score", 0) * self.weights["temporal"]
        )

        # Apply compound risk amplification
        amplification = compound_risk.get("amplification_factor", 1.0)
        composite_score = min(1.0, base_score * amplification)

        return composite_score

    def _determine_risk_level(self, composite_score: float) -> str:
        """Determine categorical risk level from composite score."""
        if composite_score >= self.thresholds["critical"]:
            return "critical"
        if composite_score >= self.thresholds["high"]:
            return "high"
        if composite_score >= self.thresholds["medium"]:
            return "medium"
        return "low"

    def _create_risk_breakdown(
        self,
        technical_risk: Dict[str, Any],
        business_risk: Dict[str, Any],
        regulatory_risk: Dict[str, Any],
        temporal_risk: Dict[str, Any],
        compound_risk: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create detailed risk breakdown."""
        return {
            "technical": technical_risk,
            "business": business_risk,
            "regulatory": regulatory_risk,
            "temporal": temporal_risk,
            "compound": compound_risk,
            "weights": self.weights,
        }

    def _calculate_confidence(
        self, risk_factors: Dict[str, Any], risk_breakdown: Dict[str, Any]
    ) -> float:
        """Calculate confidence in the risk assessment."""
        # Base confidence from data quality
        data_quality_factors = [
            len(risk_factors.get("high_severity_hits", [])) > 0,
            len(risk_factors.get("detector_errors", {})) >= 0,
            risk_factors.get("last_assessment_time") is not None,
            len(risk_factors.get("coverage_gaps", [])) >= 0,
        ]

        data_quality_score = sum(data_quality_factors) / len(data_quality_factors)

        # Adjust based on risk calculation errors
        error_count = sum(
            1
            for risk in risk_breakdown.values()
            if isinstance(risk, dict) and "error" in risk
        )
        error_penalty = error_count * 0.1

        confidence = max(0.1, data_quality_score - error_penalty)
        return confidence

    def _extract_risk_factors(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Extract risk factors from analysis request."""
        return {
            "high_severity_hits": request.high_sev_hits,
            "detector_errors": request.detector_errors,
            "coverage_gaps": [
                {
                    "detector": detector,
                    "observed": observed,
                    "required": request.required_coverage.get(detector, 0.0),
                    "impact": max(
                        0.0, request.required_coverage.get(detector, 0.0) - observed
                    ),
                }
                for detector, observed in request.observed_coverage.items()
            ],
            "sensitive_data_types": self._extract_sensitive_data_types(request),
            "affected_business_processes": self._extract_business_processes(request),
            "compliance_requirements": self._extract_compliance_requirements(request),
            "framework_violations": self._extract_framework_violations(request),
            "audit_findings": self._extract_audit_findings(request),
            "recent_incidents": self._extract_recent_incidents(request),
            "trend_direction": self._analyze_trend_direction(request),
            "last_assessment_time": datetime.now(timezone.utc)
            - timedelta(days=1),  # Mock data
        }

    def _extract_sensitive_data_types(self, request: AnalysisRequest) -> List[str]:
        """Extract sensitive data types from request."""
        data_types = set()
        for hit in request.high_sev_hits:
            if "pii" in hit.get("detector", "").lower():
                data_types.add("pii")
            if "financial" in hit.get("detector", "").lower():
                data_types.add("financial")
            if "health" in hit.get("detector", "").lower():
                data_types.add("health")
        return list(data_types)

    def _extract_business_processes(self, request: AnalysisRequest) -> List[str]:
        """Extract affected business processes from request."""
        # Mock implementation - would analyze request context in production
        return (
            ["data_processing", "user_authentication"] if request.high_sev_hits else []
        )

    def _extract_compliance_requirements(self, request: AnalysisRequest) -> List[str]:
        """Extract compliance requirements from request."""
        # Mock implementation - would analyze request context in production
        requirements = []
        if any(
            "pii" in hit.get("detector", "").lower() for hit in request.high_sev_hits
        ):
            requirements.extend(["gdpr", "hipaa"])
        return requirements

    def _extract_framework_violations(
        self, request: AnalysisRequest
    ) -> Dict[str, List[str]]:
        """Extract framework violations from request."""
        # Mock implementation - would analyze findings against frameworks in production
        violations = {}
        if request.high_sev_hits:
            violations["gdpr"] = ["data_processing_violation"]
        return violations

    def _extract_audit_findings(self, request: AnalysisRequest) -> List[str]:
        """Extract audit findings from request."""
        # Mock implementation - would analyze audit context in production
        return ["access_control_gap"] if len(request.detector_errors) > 2 else []

    def _extract_recent_incidents(
        self, request: AnalysisRequest
    ) -> List[Dict[str, Any]]:
        """Extract recent incidents from request."""
        # Mock implementation - would analyze incident history in production
        return (
            [{"type": "security_alert", "timestamp": datetime.now(timezone.utc)}]
            if request.high_sev_hits
            else []
        )

    def _analyze_trend_direction(self, request: AnalysisRequest) -> str:
        """Analyze trend direction from request."""
        # Mock implementation - would analyze historical trends in production
        if len(request.high_sev_hits) > 3:
            return "increasing"
        if len(request.high_sev_hits) == 0:
            return "decreasing"
        return "stable"

    def _create_evidence(
        self, risk_factors: Dict[str, Any], risk_breakdown: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create evidence for the risk assessment."""
        return [
            {
                "type": "risk_assessment",
                "high_severity_findings": len(
                    risk_factors.get("high_severity_hits", [])
                ),
                "detector_errors": len(risk_factors.get("detector_errors", {})),
                "coverage_gaps": len(risk_factors.get("coverage_gaps", [])),
                "risk_dimensions_analyzed": len(
                    [k for k in risk_breakdown.keys() if k != "weights"]
                ),
            }
        ]

    def _create_metadata(
        self, composite_score: float, risk_level: str
    ) -> Dict[str, Any]:
        """Create metadata for the risk assessment."""
        return {
            "engine": "consolidated_risk_scoring",
            "version": "1.0.0",
            "composite_score": composite_score,
            "risk_level": risk_level,
            "calculation_method": "weighted_composite_with_compound_factors",
            "weights_used": self.weights,
            "thresholds_used": self.thresholds,
            "assessment_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error result when risk scoring fails."""
        return AnalysisResult(
            analysis_type="risk_scoring",
            confidence=0.0,
            risk_score={
                "composite_score": 0.5,
                "risk_level": "medium",
                "breakdown": {"error": error_message},
                "compound_factors": {"error": error_message},
            },
            evidence=[{"type": "error", "message": error_message}],
            metadata={
                "error": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
