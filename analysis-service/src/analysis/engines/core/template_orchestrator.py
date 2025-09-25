"""
Template Orchestrator Engine

This engine coordinates analysis across multiple engines and provides
unified template response generation. Follows SRP by focusing solely
on orchestration and coordination of analysis engines.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from ...schemas.analysis_schemas import AnalysisRequest, AnalysisResult
from .pattern_recognition import PatternRecognitionEngine
from .risk_scoring import RiskScoringEngine
from .compliance_intelligence import ComplianceIntelligenceEngine

logger = logging.getLogger(__name__)


class TemplateOrchestrator:
    """
    Orchestrates analysis across multiple engines and coordinates template responses.

    Coordinates:
    - Pattern recognition analysis
    - Risk scoring assessment
    - Compliance intelligence mapping
    - Template response generation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestration_config = config.get("template_orchestrator", {})

        # Initialize analysis engines
        self.pattern_engine = PatternRecognitionEngine(config)
        self.risk_engine = RiskScoringEngine(config)
        self.compliance_engine = ComplianceIntelligenceEngine(config)

        # Analysis strategy configuration
        self.analysis_strategies = {
            "comprehensive": [
                "pattern_recognition",
                "risk_scoring",
                "compliance_intelligence",
            ],
            "security_focused": ["pattern_recognition", "risk_scoring"],
            "compliance_focused": ["compliance_intelligence", "risk_scoring"],
            "pattern_focused": ["pattern_recognition"],
            "risk_focused": ["risk_scoring"],
        }

        logger.info("Template Orchestrator initialized with consolidated engines")

    def get_orchestrator_info(self) -> Dict[str, Any]:
        """Get information about the template orchestrator."""
        return {
            "orchestrator_type": "template_orchestrator",
            "version": "1.0.0",
            "available_engines": [
                "pattern_recognition",
                "risk_scoring",
                "compliance_intelligence",
            ],
            "analysis_strategies": list(self.analysis_strategies.keys()),
            "capabilities": [
                "multi_engine_coordination",
                "adaptive_analysis_strategy",
                "template_response_generation",
                "cross_engine_correlation",
            ],
        }

    async def orchestrate_analysis(self, request: AnalysisRequest) -> AnalysisResult:
        """
        Orchestrate comprehensive analysis across all engines.

        Args:
            request: Analysis request to process

        Returns:
            AnalysisResult with coordinated analysis from all engines
        """
        try:
            # Select analysis strategy
            strategy = self._select_analysis_strategy(request)

            # Execute analysis engines based on strategy
            analysis_results = await self._execute_analysis_engines(request, strategy)

            # Coordinate and merge results
            coordinated_result = await self._coordinate_results(
                analysis_results, request
            )

            # Generate template response
            template_response = await self._generate_template_response(
                coordinated_result
            )

            # Create comprehensive orchestrated result
            result = AnalysisResult(
                analysis_type="orchestrated_analysis",
                confidence=coordinated_result["overall_confidence"],
                patterns=coordinated_result.get("patterns", []),
                risk_score=coordinated_result.get("risk_assessment"),
                compliance_assessment=coordinated_result.get("compliance_assessment"),
                evidence=self._create_evidence(analysis_results, coordinated_result),
                recommendations=coordinated_result.get("recommendations", []),
                metadata=self._create_metadata(
                    strategy, analysis_results, coordinated_result, template_response
                ),
            )

            logger.info(
                "Analysis orchestration completed",
                strategy=strategy,
                engines_executed=len(analysis_results),
                overall_confidence=coordinated_result["overall_confidence"],
            )

            return result

        except (ValueError, KeyError, AttributeError) as e:
            logger.error("Analysis orchestration failed", error=str(e))
            return self._create_error_result(str(e))

    def _select_analysis_strategy(self, request: AnalysisRequest) -> str:
        """Select the most appropriate analysis strategy based on request characteristics."""
        # Analyze request characteristics
        has_high_severity = len(request.high_sev_hits) > 0
        has_detector_errors = len(request.detector_errors) > 0
        has_coverage_gaps = any(
            request.required_coverage.get(detector, 0) > observed
            for detector, observed in request.observed_coverage.items()
        )

        # Strategy selection logic
        if has_high_severity and has_detector_errors and has_coverage_gaps:
            return "comprehensive"  # Full analysis needed
        if has_high_severity or has_detector_errors:
            return "security_focused"  # Focus on security analysis
        if has_coverage_gaps:
            return "compliance_focused"  # Focus on compliance
        return "pattern_focused"  # Look for patterns in normal operations

    async def _execute_analysis_engines(
        self, request: AnalysisRequest, strategy: str
    ) -> Dict[str, AnalysisResult]:
        """Execute analysis engines based on selected strategy."""
        engines_to_run = self.analysis_strategies.get(strategy, ["comprehensive"])
        results = {}

        # Execute pattern recognition if needed
        if "pattern_recognition" in engines_to_run:
            try:
                pattern_result = await self.pattern_engine.analyze_patterns(request)
                results["pattern_recognition"] = pattern_result
            except (ValueError, KeyError, AttributeError) as e:
                logger.error("Pattern recognition failed", error=str(e))
                results["pattern_recognition"] = self._create_engine_error_result(
                    "pattern_recognition", str(e)
                )

        # Execute risk scoring if needed
        if "risk_scoring" in engines_to_run:
            try:
                risk_result = await self.risk_engine.calculate_risk_score(request)
                results["risk_scoring"] = risk_result
            except (ValueError, KeyError, AttributeError) as e:
                logger.error("Risk scoring failed", error=str(e))
                results["risk_scoring"] = self._create_engine_error_result(
                    "risk_scoring", str(e)
                )

        # Execute compliance intelligence if needed
        if "compliance_intelligence" in engines_to_run:
            try:
                compliance_result = await self.compliance_engine.analyze_compliance(
                    request
                )
                results["compliance_intelligence"] = compliance_result
            except (ValueError, KeyError, AttributeError) as e:
                logger.error("Compliance intelligence failed", error=str(e))
                results["compliance_intelligence"] = self._create_engine_error_result(
                    "compliance_intelligence", str(e)
                )

        return results

    async def _coordinate_results(
        self, analysis_results: Dict[str, AnalysisResult], request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Coordinate and merge results from multiple analysis engines."""
        coordinated = {
            "strategy_used": self._select_analysis_strategy(request),
            "engines_executed": list(analysis_results.keys()),
            "individual_results": {},
            "cross_engine_correlations": {},
            "overall_confidence": 0.0,
            "consolidated_insights": {},
            "recommendations": [],
        }

        # Process individual results
        confidences = []
        for engine_name, result in analysis_results.items():
            coordinated["individual_results"][engine_name] = {
                "confidence": result.confidence,
                "analysis_type": result.analysis_type,
                "key_findings": self._extract_key_findings(result),
                "evidence_count": (
                    len(result.evidence) if hasattr(result, "evidence") else 0
                ),
            }
            confidences.append(result.confidence)

        # Calculate overall confidence
        coordinated["overall_confidence"] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        # Identify cross-engine correlations
        coordinated["cross_engine_correlations"] = (
            await self._identify_cross_engine_correlations(analysis_results)
        )

        # Generate consolidated insights
        coordinated["consolidated_insights"] = (
            await self._generate_consolidated_insights(analysis_results)
        )

        # Generate coordinated recommendations
        coordinated["recommendations"] = (
            await self._generate_coordinated_recommendations(analysis_results)
        )

        return coordinated

    async def _identify_cross_engine_correlations(
        self, analysis_results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """Identify correlations between results from different engines."""
        correlations = {
            "pattern_risk_correlation": None,
            "risk_compliance_correlation": None,
            "pattern_compliance_correlation": None,
            "overall_correlation_strength": 0.0,
        }

        # Pattern-Risk correlation
        if (
            "pattern_recognition" in analysis_results
            and "risk_scoring" in analysis_results
        ):
            pattern_confidence = analysis_results["pattern_recognition"].confidence
            risk_confidence = analysis_results["risk_scoring"].confidence

            # Simple correlation based on confidence alignment
            correlation_strength = 1.0 - abs(pattern_confidence - risk_confidence)
            correlations["pattern_risk_correlation"] = {
                "strength": correlation_strength,
                "interpretation": (
                    "high"
                    if correlation_strength > 0.8
                    else "medium" if correlation_strength > 0.5 else "low"
                ),
            }

        # Risk-Compliance correlation
        if (
            "risk_scoring" in analysis_results
            and "compliance_intelligence" in analysis_results
        ):
            risk_confidence = analysis_results["risk_scoring"].confidence
            compliance_confidence = analysis_results[
                "compliance_intelligence"
            ].confidence

            correlation_strength = 1.0 - abs(risk_confidence - compliance_confidence)
            correlations["risk_compliance_correlation"] = {
                "strength": correlation_strength,
                "interpretation": (
                    "high"
                    if correlation_strength > 0.8
                    else "medium" if correlation_strength > 0.5 else "low"
                ),
            }

        # Pattern-Compliance correlation
        if (
            "pattern_recognition" in analysis_results
            and "compliance_intelligence" in analysis_results
        ):
            pattern_confidence = analysis_results["pattern_recognition"].confidence
            compliance_confidence = analysis_results[
                "compliance_intelligence"
            ].confidence

            correlation_strength = 1.0 - abs(pattern_confidence - compliance_confidence)
            correlations["pattern_compliance_correlation"] = {
                "strength": correlation_strength,
                "interpretation": (
                    "high"
                    if correlation_strength > 0.8
                    else "medium" if correlation_strength > 0.5 else "low"
                ),
            }

        # Calculate overall correlation strength
        correlation_values = [
            corr["strength"]
            for corr in correlations.values()
            if isinstance(corr, dict) and "strength" in corr
        ]
        correlations["overall_correlation_strength"] = (
            sum(correlation_values) / len(correlation_values)
            if correlation_values
            else 0.0
        )

        return correlations

    async def _generate_consolidated_insights(
        self, analysis_results: Dict[str, AnalysisResult]
    ) -> Dict[str, Any]:
        """Generate consolidated insights from all analysis results."""
        insights = {
            "security_posture": "unknown",
            "compliance_status": "unknown",
            "risk_level": "unknown",
            "pattern_significance": "unknown",
            "key_concerns": [],
            "positive_indicators": [],
        }

        # Analyze security posture from pattern and risk results
        if (
            "pattern_recognition" in analysis_results
            and "risk_scoring" in analysis_results
        ):
            pattern_confidence = analysis_results["pattern_recognition"].confidence
            risk_confidence = analysis_results["risk_scoring"].confidence

            avg_confidence = (pattern_confidence + risk_confidence) / 2
            if avg_confidence > 0.8:
                insights["security_posture"] = "strong"
                insights["positive_indicators"].append(
                    "High confidence in security analysis"
                )
            elif avg_confidence > 0.5:
                insights["security_posture"] = "moderate"
            else:
                insights["security_posture"] = "weak"
                insights["key_concerns"].append("Low confidence in security analysis")

        # Analyze compliance status
        if "compliance_intelligence" in analysis_results:
            compliance_result = analysis_results["compliance_intelligence"]
            if hasattr(compliance_result, "compliance_state"):
                compliance_state = getattr(compliance_result, "compliance_state", {})
                insights["compliance_status"] = compliance_state.get(
                    "overall_status", "unknown"
                )

                if compliance_state.get("overall_status") == "compliant":
                    insights["positive_indicators"].append("Good compliance posture")
                elif compliance_state.get("overall_status") in [
                    "non_compliant",
                    "partially_compliant",
                ]:
                    insights["key_concerns"].append("Compliance gaps identified")

        # Analyze risk level
        if "risk_scoring" in analysis_results:
            risk_result = analysis_results["risk_scoring"]
            if hasattr(risk_result, "risk_score"):
                risk_score = getattr(risk_result, "risk_score", {})
                insights["risk_level"] = risk_score.get("risk_level", "unknown")

                if risk_score.get("risk_level") in ["critical", "high"]:
                    insights["key_concerns"].append(
                        f"High risk level: {risk_score.get('risk_level')}"
                    )
                elif risk_score.get("risk_level") == "low":
                    insights["positive_indicators"].append("Low risk level identified")

        # Analyze pattern significance
        if "pattern_recognition" in analysis_results:
            pattern_result = analysis_results["pattern_recognition"]
            if hasattr(pattern_result, "patterns"):
                patterns = getattr(pattern_result, "patterns", [])
                if len(patterns) > 5:
                    insights["pattern_significance"] = "high"
                    insights["key_concerns"].append(
                        "Multiple significant patterns detected"
                    )
                elif len(patterns) > 0:
                    insights["pattern_significance"] = "moderate"
                else:
                    insights["pattern_significance"] = "low"
                    insights["positive_indicators"].append(
                        "No concerning patterns detected"
                    )

        return insights

    async def _generate_coordinated_recommendations(
        self, analysis_results: Dict[str, AnalysisResult]
    ) -> List[Dict[str, Any]]:
        """Generate coordinated recommendations based on all analysis results."""
        recommendations = []

        # Security-focused recommendations
        if (
            "pattern_recognition" in analysis_results
            and "risk_scoring" in analysis_results
        ):
            pattern_result = analysis_results["pattern_recognition"]
            risk_result = analysis_results["risk_scoring"]

            if pattern_result.confidence > 0.7 and hasattr(pattern_result, "patterns"):
                patterns = getattr(pattern_result, "patterns", [])
                if len(patterns) > 3:
                    recommendations.append(
                        {
                            "type": "security",
                            "priority": "high",
                            "title": "Address Multiple Security Patterns",
                            "description": (
                                f"Multiple security patterns detected ({len(patterns)}). "
                                "Investigate and remediate."
                            ),
                            "source_engines": ["pattern_recognition", "risk_scoring"],
                        }
                    )

            if hasattr(risk_result, "risk_score"):
                risk_score = getattr(risk_result, "risk_score", {})
                if risk_score.get("risk_level") in ["critical", "high"]:
                    recommendations.append(
                        {
                            "type": "risk_mitigation",
                            "priority": (
                                "critical"
                                if risk_score.get("risk_level") == "critical"
                                else "high"
                            ),
                            "title": f"Mitigate {risk_score.get('risk_level').title()} Risk",
                            "description": (
                                f"Risk level assessed as {risk_score.get('risk_level')}. "
                                "Immediate action required."
                            ),
                            "source_engines": ["risk_scoring"],
                        }
                    )

        # Compliance-focused recommendations
        if "compliance_intelligence" in analysis_results:
            compliance_result = analysis_results["compliance_intelligence"]

            if hasattr(compliance_result, "compliance_gaps"):
                gaps = getattr(compliance_result, "compliance_gaps", [])
                if len(gaps) > 0:
                    critical_gaps = [g for g in gaps if g.get("severity") == "critical"]
                    if critical_gaps:
                        recommendations.append(
                            {
                                "type": "compliance",
                                "priority": "critical",
                                "title": "Address Critical Compliance Gaps",
                                "description": (
                                    f"{len(critical_gaps)} critical compliance gaps identified. "
                                    "Immediate remediation required."
                                ),
                                "source_engines": ["compliance_intelligence"],
                            }
                        )
                    else:
                        recommendations.append(
                            {
                                "type": "compliance",
                                "priority": "medium",
                                "title": "Address Compliance Gaps",
                                "description": (
                                    f"{len(gaps)} compliance gaps identified. "
                                    "Plan remediation activities."
                                ),
                                "source_engines": ["compliance_intelligence"],
                            }
                        )

        # Cross-engine recommendations
        if len(analysis_results) > 1:
            overall_confidence = sum(
                result.confidence for result in analysis_results.values()
            ) / len(analysis_results)
            if overall_confidence < 0.5:
                recommendations.append(
                    {
                        "type": "data_quality",
                        "priority": "medium",
                        "title": "Improve Data Quality",
                        "description": (
                            "Low confidence across multiple analysis engines. "
                            "Consider improving data quality and coverage."
                        ),
                        "source_engines": list(analysis_results.keys()),
                    }
                )

        # Sort recommendations by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

        return recommendations

    async def _generate_template_response(
        self, coordinated_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate template response based on coordinated analysis results."""
        template_response = {
            "analysis_summary": self._create_analysis_summary(coordinated_result),
            "key_findings": self._extract_key_findings_from_coordinated_result(
                coordinated_result
            ),
            "risk_assessment": self._create_risk_assessment_summary(coordinated_result),
            "compliance_status": self._create_compliance_status_summary(
                coordinated_result
            ),
            "recommendations": coordinated_result.get("recommendations", []),
            "next_steps": self._generate_next_steps(coordinated_result),
            "confidence_assessment": self._create_confidence_assessment(
                coordinated_result
            ),
        }

        return template_response

    def _create_analysis_summary(self, coordinated_result: Dict[str, Any]) -> str:
        """Create high-level analysis summary."""
        strategy = coordinated_result.get("strategy_used", "unknown")
        engines_count = len(coordinated_result.get("engines_executed", []))
        overall_confidence = coordinated_result.get("overall_confidence", 0.0)

        confidence_level = (
            "high"
            if overall_confidence > 0.7
            else "medium" if overall_confidence > 0.4 else "low"
        )

        return (
            f"Comprehensive analysis completed using {strategy} strategy across "
            f"{engines_count} analysis engines. "
            f"Overall confidence level: {confidence_level} ({overall_confidence:.2f}). "
            f"Analysis identified key security, risk, and compliance considerations."
        )

    def _extract_key_findings_from_coordinated_result(
        self, coordinated_result: Dict[str, Any]
    ) -> List[str]:
        """Extract key findings from coordinated analysis result."""
        findings = []

        insights = coordinated_result.get("consolidated_insights", {})

        # Security findings
        security_posture = insights.get("security_posture", "unknown")
        if security_posture != "unknown":
            findings.append(f"Security posture assessed as: {security_posture}")

        # Risk findings
        risk_level = insights.get("risk_level", "unknown")
        if risk_level != "unknown":
            findings.append(f"Risk level identified as: {risk_level}")

        # Compliance findings
        compliance_status = insights.get("compliance_status", "unknown")
        if compliance_status != "unknown":
            findings.append(f"Compliance status: {compliance_status}")

        # Pattern findings
        pattern_significance = insights.get("pattern_significance", "unknown")
        if pattern_significance != "unknown":
            findings.append(f"Pattern significance: {pattern_significance}")

        # Key concerns
        key_concerns = insights.get("key_concerns", [])
        for concern in key_concerns[:3]:  # Limit to top 3 concerns
            findings.append(f"Concern: {concern}")

        return findings

    def _create_risk_assessment_summary(
        self, coordinated_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create risk assessment summary."""
        individual_results = coordinated_result.get("individual_results", {})
        risk_result = individual_results.get("risk_scoring", {})

        return {
            "overall_risk_level": coordinated_result.get(
                "consolidated_insights", {}
            ).get("risk_level", "unknown"),
            "confidence": risk_result.get("confidence", 0.0),
            "key_risk_factors": self._extract_risk_factors(coordinated_result),
            "risk_trend": "stable",  # Would be calculated from historical data in production
        }

    def _create_compliance_status_summary(
        self, coordinated_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create compliance status summary."""
        individual_results = coordinated_result.get("individual_results", {})
        compliance_result = individual_results.get("compliance_intelligence", {})

        return {
            "overall_status": coordinated_result.get("consolidated_insights", {}).get(
                "compliance_status", "unknown"
            ),
            "confidence": compliance_result.get("confidence", 0.0),
            "frameworks_assessed": [
                "soc2",
                "iso27001",
                "hipaa",
                "gdpr",
            ],  # Would be dynamic in production
            "gaps_identified": compliance_result.get("key_findings", {}).get(
                "gaps_count", 0
            ),
        }

    def _generate_next_steps(self, coordinated_result: Dict[str, Any]) -> List[str]:
        """Generate recommended next steps."""
        next_steps = []

        recommendations = coordinated_result.get("recommendations", [])

        # Add steps based on critical recommendations
        critical_recs = [r for r in recommendations if r.get("priority") == "critical"]
        for rec in critical_recs[:2]:  # Top 2 critical recommendations
            next_steps.append(f"URGENT: {rec.get('title', 'Address critical issue')}")

        # Add steps based on high priority recommendations
        high_recs = [r for r in recommendations if r.get("priority") == "high"]
        for rec in high_recs[:2]:  # Top 2 high priority recommendations
            next_steps.append(
                f"HIGH: {rec.get('title', 'Address high priority issue')}"
            )

        # Add general next steps
        if not next_steps:
            next_steps.extend(
                [
                    "Continue monitoring security posture",
                    "Review and update compliance controls",
                    "Schedule regular risk assessments",
                ]
            )

        return next_steps

    def _create_confidence_assessment(
        self, coordinated_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create confidence assessment for the analysis."""
        overall_confidence = coordinated_result.get("overall_confidence", 0.0)
        individual_results = coordinated_result.get("individual_results", {})

        confidence_breakdown = {}
        for engine, result in individual_results.items():
            confidence_breakdown[engine] = result.get("confidence", 0.0)

        return {
            "overall_confidence": overall_confidence,
            "confidence_level": (
                "high"
                if overall_confidence > 0.7
                else "medium" if overall_confidence > 0.4 else "low"
            ),
            "engine_breakdown": confidence_breakdown,
            "factors_affecting_confidence": self._identify_confidence_factors(
                coordinated_result
            ),
        }

    def _extract_key_findings(self, result: AnalysisResult) -> Dict[str, Any]:
        """Extract key findings from an individual analysis result."""
        findings = {
            "confidence": result.confidence,
            "analysis_type": result.analysis_type,
        }

        # Extract type-specific findings
        if hasattr(result, "patterns"):
            findings["patterns_count"] = len(getattr(result, "patterns", []))

        if hasattr(result, "risk_score"):
            risk_score = getattr(result, "risk_score", {})
            findings["risk_level"] = risk_score.get("risk_level", "unknown")
            findings["composite_score"] = risk_score.get("composite_score", 0.0)

        if hasattr(result, "compliance_gaps"):
            gaps = getattr(result, "compliance_gaps", [])
            findings["gaps_count"] = len(gaps)
            findings["critical_gaps"] = len(
                [g for g in gaps if g.get("severity") == "critical"]
            )

        return findings

    def _extract_risk_factors(self, coordinated_result: Dict[str, Any]) -> List[str]:
        """Extract key risk factors from coordinated results."""
        risk_factors = []

        insights = coordinated_result.get("consolidated_insights", {})
        key_concerns = insights.get("key_concerns", [])

        for concern in key_concerns:
            risk_factors.append(concern)

        return risk_factors

    def _identify_confidence_factors(
        self, coordinated_result: Dict[str, Any]
    ) -> List[str]:
        """Identify factors affecting confidence in the analysis."""
        factors = []

        individual_results = coordinated_result.get("individual_results", {})

        # Check for low confidence engines
        low_confidence_engines = [
            engine
            for engine, result in individual_results.items()
            if result.get("confidence", 0) < 0.5
        ]

        if low_confidence_engines:
            factors.append(f"Low confidence in: {', '.join(low_confidence_engines)}")

        # Check for missing engines
        expected_engines = [
            "pattern_recognition",
            "risk_scoring",
            "compliance_intelligence",
        ]
        missing_engines = [
            engine for engine in expected_engines if engine not in individual_results
        ]

        if missing_engines:
            factors.append(f"Missing analysis from: {', '.join(missing_engines)}")

        # Check correlation strength
        correlations = coordinated_result.get("cross_engine_correlations", {})
        overall_correlation = correlations.get("overall_correlation_strength", 0.0)

        if overall_correlation < 0.5:
            factors.append("Low correlation between analysis engines")

        return factors

    def _create_engine_error_result(
        self, engine_name: str, error_message: str
    ) -> AnalysisResult:
        """Create error result for a failed engine."""
        return AnalysisResult(
            analysis_type=engine_name,
            confidence=0.0,
            evidence=[{"type": "error", "message": error_message}],
            metadata={"error": error_message, "engine": engine_name},
        )

    def _create_evidence(
        self,
        analysis_results: Dict[str, AnalysisResult],
        coordinated_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Create evidence for orchestrated analysis."""
        return [
            {
                "type": "orchestrated_analysis",
                "engines_executed": len(analysis_results),
                "strategy_used": coordinated_result.get("strategy_used", "unknown"),
                "overall_confidence": coordinated_result.get("overall_confidence", 0.0),
                "recommendations_generated": len(
                    coordinated_result.get("recommendations", [])
                ),
                "cross_engine_correlations": coordinated_result.get(
                    "cross_engine_correlations", {}
                ).get("overall_correlation_strength", 0.0),
            }
        ]

    def _create_metadata(
        self,
        strategy: str,
        analysis_results: Dict[str, AnalysisResult],
        coordinated_result: Dict[str, Any],
        template_response: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create metadata for orchestrated analysis."""
        return {
            "engine": "template_orchestrator",
            "version": "1.0.0",
            "strategy_used": strategy,
            "engines_executed": list(analysis_results.keys()),
            "coordination_method": "confidence_weighted_aggregation",
            "correlation_analysis": coordinated_result.get(
                "cross_engine_correlations", {}
            ),
            "template_response": template_response,
            "orchestration_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _create_error_result(self, error_message: str) -> AnalysisResult:
        """Create error result when orchestration fails."""
        return AnalysisResult(
            analysis_type="orchestrated_analysis",
            confidence=0.0,
            evidence=[{"type": "error", "message": error_message}],
            metadata={
                "error": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
