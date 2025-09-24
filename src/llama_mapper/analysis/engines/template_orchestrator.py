"""
Template Orchestrator

Coordinates the specialized analysis engines to provide comprehensive
analysis results. This replaces the monolithic template provider approach
with a modular, orchestrated system.
"""

import logging
from typing import Any, Dict, List, Optional
from ..domain.entities import AnalysisRequest, AnalysisType
from .interfaces import (
    ITemplateOrchestrator,
    IPatternRecognitionEngine,
    IRiskScoringEngine,
    IComplianceIntelligence,
)

logger = logging.getLogger(__name__)


class TemplateOrchestrator(ITemplateOrchestrator):
    """
    Orchestrates analysis across specialized engines.
    
    This orchestrator coordinates the refactored analysis engines to provide
    comprehensive analysis results while maintaining the same external interface
    as the original monolithic template provider.
    
    The orchestrator:
    - Selects appropriate analysis strategies
    - Coordinates multiple engines
    - Combines results intelligently
    - Maintains backward compatibility
    """
    
    def __init__(
        self,
        pattern_engine: IPatternRecognitionEngine,
        risk_engine: IRiskScoringEngine,
        compliance_engine: IComplianceIntelligence,
    ):
        """
        Initialize the template orchestrator.
        
        Args:
            pattern_engine: Pattern recognition engine
            risk_engine: Risk scoring engine
            compliance_engine: Compliance intelligence engine
        """
        self.pattern_engine = pattern_engine
        self.risk_engine = risk_engine
        self.compliance_engine = compliance_engine
        logger.info("Initialized Template Orchestrator with specialized engines")
    
    async def orchestrate_analysis(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Orchestrate comprehensive analysis across all engines.
        
        Args:
            request: Analysis request
            
        Returns:
            Comprehensive analysis result combining all engines
        """
        # Run all engines in parallel for comprehensive analysis
        pattern_result = await self.pattern_engine.analyze(request)
        risk_result = await self.risk_engine.analyze(request)
        compliance_result = await self.compliance_engine.analyze(request)
        
        # Combine results intelligently
        combined_result = self._combine_engine_results(
            pattern_result, risk_result, compliance_result, request
        )
        
        return combined_result
    
    async def select_analysis_strategy(self, request: AnalysisRequest) -> AnalysisType:
        """
        Select the most appropriate analysis strategy based on request content.
        
        Args:
            request: Analysis request
            
        Returns:
            Selected analysis type
        """
        # Check for coverage gaps (highest priority)
        coverage_gaps = []
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            if observed < required:
                coverage_gaps.append(detector)
        
        if coverage_gaps:
            logger.info(f"Selected COVERAGE_GAP analysis for {len(coverage_gaps)} gaps")
            return AnalysisType.COVERAGE_GAP
        
        # Check for false positive tuning opportunities
        if request.false_positive_bands:
            significant_fp = [
                band for band in request.false_positive_bands
                if band.get("false_positive_rate", 0.0) > 0.1
            ]
            if significant_fp:
                logger.info(f"Selected FALSE_POSITIVE_TUNING analysis for {len(significant_fp)} detectors")
                return AnalysisType.FALSE_POSITIVE_TUNING
        
        # Check for incident summary
        if request.high_sev_hits:
            critical_incidents = [
                hit for hit in request.high_sev_hits
                if hit.get("severity") in ["critical", "high"]
            ]
            if critical_incidents:
                logger.info(f"Selected INCIDENT_SUMMARY analysis for {len(critical_incidents)} incidents")
                return AnalysisType.INCIDENT_SUMMARY
        
        logger.info("Selected INSUFFICIENT_DATA analysis - no clear patterns detected")
        return AnalysisType.INSUFFICIENT_DATA
    
    async def get_template_response(
        self,
        request: AnalysisRequest,
        analysis_type: AnalysisType,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get template response using coordinated engines.
        
        Args:
            request: Analysis request
            analysis_type: Type of analysis to perform
            fallback_reason: Reason for using template fallback
            
        Returns:
            Template response dictionary
        """
        if analysis_type == AnalysisType.COVERAGE_GAP:
            return await self._get_coverage_gap_response(request, fallback_reason)
        elif analysis_type == AnalysisType.FALSE_POSITIVE_TUNING:
            return await self._get_false_positive_response(request, fallback_reason)
        elif analysis_type == AnalysisType.INCIDENT_SUMMARY:
            return await self._get_incident_summary_response(request, fallback_reason)
        else:
            return await self._get_insufficient_data_response(request, fallback_reason)
    
    async def _get_coverage_gap_response(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced coverage gap analysis response using orchestrated engines."""
        # Get pattern analysis focused on coverage gaps
        pattern_result = await self.pattern_engine.analyze(request)
        coverage_patterns = pattern_result["patterns"]["coverage_gap_patterns"]
        
        # Get risk assessment for coverage gaps
        risk_result = await self.risk_engine.analyze(request)
        
        # Get compliance impact
        compliance_result = await self.compliance_engine.analyze(request)
        
        # Generate comprehensive analysis
        analysis = self._analyze_coverage_gaps_comprehensive(
            coverage_patterns, risk_result, compliance_result, request
        )
        
        # Generate specific remediation
        remediation = self._generate_coverage_remediation_comprehensive(analysis)
        
        # Create advanced OPA policy
        opa_diff = compliance_result.get("compliance_policy", "")
        
        # Calculate confidence based on engine results
        confidence = self._calculate_combined_confidence([
            pattern_result["confidence"],
            risk_result["confidence"],
            compliance_result["confidence"]
        ])
        
        return {
            "reason": analysis["summary"],
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": confidence,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": analysis["affected_detectors"],
            "notes": f"Advanced coverage gap analysis - {analysis['risk_level']} risk scenario{f' - {fallback_reason}' if fallback_reason else ''}",
            "analysis_details": {
                "gaps_identified": analysis["gaps"],
                "risk_assessment": analysis["risk_assessment"],
                "compliance_impact": analysis["compliance_impact"],
                "priority_actions": analysis["priority_actions"],
                "estimated_effort": analysis["estimated_effort"]
            }
        }
    
    async def _get_false_positive_response(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced false positive tuning analysis response."""
        # Get pattern analysis focused on false positives
        pattern_result = await self.pattern_engine.analyze(request)
        fp_patterns = pattern_result["patterns"]["false_positive_patterns"]
        
        # Get risk assessment
        risk_result = await self.risk_engine.analyze(request)
        
        # Generate comprehensive analysis
        analysis = self._analyze_false_positives_comprehensive(fp_patterns, risk_result, request)
        
        # Generate specific remediation
        remediation = self._generate_threshold_remediation_comprehensive(analysis)
        
        # Create advanced OPA policy
        opa_diff = self._generate_advanced_threshold_opa_policy(analysis)
        
        # Calculate confidence
        confidence = self._calculate_combined_confidence([
            pattern_result["confidence"],
            risk_result["confidence"]
        ])
        
        return {
            "reason": analysis["summary"],
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": confidence,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": analysis["affected_detectors"],
            "notes": f"Advanced false positive analysis - {analysis['pattern_strength']} pattern confidence{f' - {fallback_reason}' if fallback_reason else ''}",
            "analysis_details": {
                "patterns_identified": analysis["patterns"],
                "threshold_recommendations": analysis["threshold_recommendations"],
                "expected_reduction": analysis["expected_fp_reduction"],
                "risk_impact": analysis["risk_impact"],
                "validation_approach": analysis["validation_approach"]
            }
        }
    
    async def _get_incident_summary_response(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced incident summary analysis response."""
        # Get pattern analysis focused on incidents
        pattern_result = await self.pattern_engine.analyze(request)
        incident_patterns = pattern_result["patterns"]["incident_patterns"]
        
        # Get risk assessment
        risk_result = await self.risk_engine.analyze(request)
        
        # Get compliance impact
        compliance_result = await self.compliance_engine.analyze(request)
        
        # Generate comprehensive analysis
        analysis = self._analyze_security_incidents_comprehensive(
            incident_patterns, risk_result, compliance_result, request
        )
        
        # Generate specific remediation
        remediation = self._generate_incident_remediation_comprehensive(analysis)
        
        # Create comprehensive OPA policy
        opa_diff = compliance_result.get("compliance_policy", "")
        
        # Calculate confidence
        confidence = self._calculate_combined_confidence([
            pattern_result["confidence"],
            risk_result["confidence"],
            compliance_result["confidence"]
        ])
        
        return {
            "reason": analysis["summary"],
            "remediation": remediation,
            "opa_diff": opa_diff,
            "confidence": confidence,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": analysis["affected_detectors"],
            "notes": f"Advanced incident analysis - {analysis['severity_distribution']} severity pattern{f' - {fallback_reason}' if fallback_reason else ''}",
            "analysis_details": {
                "incident_breakdown": analysis["incident_breakdown"],
                "severity_analysis": analysis["severity_distribution"],
                "risk_assessment": analysis["risk_assessment"],
                "compliance_impact": analysis["compliance_impact"],
                "response_timeline": analysis["response_timeline"],
                "escalation_required": analysis["escalation_required"]
            }
        }
    
    async def _get_insufficient_data_response(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get insufficient data template response."""
        reason = "Insufficient data for detailed analysis - consider collecting more comprehensive security metrics"
        remediation = "Implement additional detectors and increase coverage monitoring to enable advanced analysis"
        
        return {
            "reason": reason,
            "remediation": remediation,
            "opa_diff": "",
            "confidence": 0.1,
            "confidence_cutoff_used": 0.3,
            "evidence_refs": request.required_detectors,
            "notes": f"Template fallback for insufficient data{f' - {fallback_reason}' if fallback_reason else ''}",
        }
    
    def _combine_engine_results(
        self,
        pattern_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        compliance_result: Dict[str, Any],
        request: AnalysisRequest,
    ) -> Dict[str, Any]:
        """Combine results from all engines into comprehensive analysis."""
        return {
            "analysis_type": "comprehensive",
            "pattern_analysis": pattern_result,
            "risk_analysis": risk_result,
            "compliance_analysis": compliance_result,
            "overall_confidence": self._calculate_combined_confidence([
                pattern_result.get("confidence", 0.5),
                risk_result.get("confidence", 0.5),
                compliance_result.get("confidence", 0.5)
            ]),
            "recommendations": self._generate_combined_recommendations(
                pattern_result, risk_result, compliance_result
            ),
            "summary": self._generate_executive_summary(
                pattern_result, risk_result, compliance_result
            )
        }
    
    def _analyze_coverage_gaps_comprehensive(
        self,
        coverage_patterns: list,
        risk_result: Dict[str, Any],
        compliance_result: Dict[str, Any],
        request: AnalysisRequest,
    ) -> Dict[str, Any]:
        """Comprehensive coverage gap analysis using all engines."""
        gaps = coverage_patterns
        risk_assessment = risk_result.get("risk_score", {})
        framework_mappings = compliance_result.get("framework_mappings", {})
        compliance_impact = framework_mappings.get("compliance_score", 100)
        
        # Determine overall risk level
        risk_level = risk_assessment.get("risk_level", "low")
        
        # Generate priority actions combining risk and compliance insights
        priority_actions = self._generate_priority_actions_comprehensive(gaps, risk_assessment, compliance_result)
        
        # Estimate effort using risk factors
        estimated_effort = self._estimate_remediation_effort_comprehensive(gaps, risk_assessment)
        
        # Create summary
        summary = self._create_coverage_summary_comprehensive(gaps, risk_level, compliance_impact)
        
        return {
            "gaps": gaps,
            "risk_assessment": risk_assessment,
            "compliance_impact": compliance_impact,
            "risk_level": risk_level,
            "priority_actions": priority_actions,
            "estimated_effort": estimated_effort,
            "summary": summary,
            "affected_detectors": [gap["detector"] for gap in gaps],
        }
    
    def _analyze_false_positives_comprehensive(
        self, fp_patterns: list, risk_result: Dict[str, Any], request: AnalysisRequest
    ) -> Dict[str, Any]:
        """Comprehensive false positive analysis."""
        patterns = fp_patterns
        risk_impact = risk_result["risk_analysis"]["risk_score"]
        
        # Generate threshold recommendations with risk context
        threshold_recommendations = {}
        for pattern in patterns:
            detector = pattern["detector"]
            threshold_recommendations[detector] = {
                "current": pattern["current_threshold"],
                "recommended": pattern["recommended_threshold"],
                "justification": f"Reduce {pattern['current_fp_rate']:.1%} false positive rate by {pattern['expected_reduction']:.1%}",
                "risk_impact": self._assess_threshold_risk_impact(pattern, risk_impact)
            }
        
        # Calculate expected overall reduction
        total_expected_reduction = sum(p["expected_reduction"] for p in patterns) / len(patterns) if patterns else 0
        
        # Determine pattern strength
        pattern_strength = self._determine_overall_pattern_strength(patterns)
        
        # Create summary
        summary = f"False positive patterns detected in {len(patterns)} detectors with {pattern_strength} signal strength and {risk_impact['risk_level']} risk impact"
        
        return {
            "patterns": patterns,
            "threshold_recommendations": threshold_recommendations,
            "pattern_strength": pattern_strength,
            "expected_fp_reduction": total_expected_reduction,
            "risk_impact": risk_impact,
            "summary": summary,
            "affected_detectors": [p["detector"] for p in patterns],
            "validation_approach": self._recommend_validation_approach_comprehensive(pattern_strength, risk_impact)
        }
    
    def _analyze_security_incidents_comprehensive(
        self,
        incident_patterns: list,
        risk_result: Dict[str, Any],
        compliance_result: Dict[str, Any],
        request: AnalysisRequest,
    ) -> Dict[str, Any]:
        """Comprehensive security incident analysis."""
        patterns = incident_patterns
        risk_assessment = risk_result.get("risk_score", {})
        compliance_impact = compliance_result.get("framework_mappings", {})
        
        # Analyze incident breakdown
        incident_breakdown = {}
        total_incidents = 0
        escalation_required = False
        
        for pattern in patterns:
            detector = pattern["detector"]
            incident_breakdown[detector] = pattern
            total_incidents += pattern["incident_count"]
            if pattern["requires_escalation"]:
                escalation_required = True
        
        # Generate response timeline based on risk and compliance
        response_timeline = self._generate_response_timeline_comprehensive(
            patterns, risk_assessment, compliance_impact, escalation_required
        )
        
        # Create severity distribution summary
        severity_dist = self._create_severity_distribution_summary(patterns)
        
        # Create summary
        risk_level = risk_assessment.get("risk_level", "low")
        if escalation_required:
            summary = f"Critical security incidents detected requiring immediate escalation - {risk_level} risk level with compliance impact"
        else:
            summary = f"Security incidents detected: {total_incidents} incidents across multiple detectors with {risk_level} risk level"
        
        return {
            "patterns": patterns,
            "incident_breakdown": incident_breakdown,
            "severity_distribution": severity_dist,
            "risk_assessment": risk_assessment,
            "compliance_impact": compliance_impact,
            "escalation_required": escalation_required,
            "response_timeline": response_timeline,
            "summary": summary,
            "affected_detectors": list(incident_breakdown.keys()),
            "total_count": total_incidents
        }
    
    def _generate_coverage_remediation_comprehensive(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive coverage remediation."""
        if not analysis["gaps"]:
            return "No remediation required - coverage targets are met"
        
        priority_actions = analysis["priority_actions"]
        if not priority_actions:
            return "Review detector configurations and adjust coverage requirements"
        
        top_action = priority_actions[0]
        effort = analysis["estimated_effort"]
        risk_level = analysis["risk_level"]
        
        return f"Priority: {top_action['action']} (Risk Level: {risk_level}). Estimated effort: {effort['estimated_days']} days with {effort['complexity']} complexity. Compliance impact: {analysis['compliance_impact']:.1f}% score."
    
    def _generate_threshold_remediation_comprehensive(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive threshold remediation."""
        if not analysis["patterns"]:
            return "No false positive patterns detected - current thresholds are appropriate"
        
        top_pattern = max(analysis["patterns"], key=lambda x: x["current_fp_rate"])
        total_detectors = len(analysis["patterns"])
        expected_reduction = analysis["expected_fp_reduction"]
        risk_impact = analysis["risk_impact"]["risk_level"]
        
        return f"Adjust {top_pattern['detector']} threshold from {top_pattern['current_threshold']:.2f} to {top_pattern['recommended_threshold']:.2f}. Expected {expected_reduction:.1%} false positive reduction across {total_detectors} detectors. Risk impact: {risk_impact}. Use {analysis['validation_approach']}."
    
    def _generate_incident_remediation_comprehensive(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive incident remediation."""
        if analysis["escalation_required"]:
            risk_level = analysis["risk_assessment"]["risk_level"]
            compliance_score = analysis["compliance_impact"]["compliance_score"]
            return f"IMMEDIATE ESCALATION: Critical incidents detected with {risk_level} risk level. Compliance score: {compliance_score:.1f}%. Notify security team within 1 hour. Begin containment procedures. Estimated resolution time: 24-72 hours with security team involvement."
        else:
            total_count = analysis["total_count"]
            affected_detectors = len(analysis["affected_detectors"])
            risk_level = analysis["risk_assessment"]["risk_level"]
            return f"Standard incident response: {total_count} incidents across {affected_detectors} detectors with {risk_level} risk level. Investigate patterns, adjust thresholds, monitor for 24 hours. Estimated resolution: 4-8 hours."
    
    def _generate_advanced_threshold_opa_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate advanced OPA policy for threshold enforcement."""
        policy_parts = [
            "package threshold_enforcement",
            "",
            "# Advanced threshold enforcement with false positive optimization",
            "# Generated by Template Orchestrator with risk and pattern analysis",
            ""
        ]
        
        # Generate detector-specific policies
        for pattern in analysis["patterns"]:
            detector = pattern["detector"]
            recommended = pattern["recommended_threshold"]
            current_fp = pattern["current_fp_rate"]
            
            policy_parts.extend([
                f"# {detector} threshold enforcement with risk context",
                f"{detector.replace('-', '_')}_threshold_violation[msg] {{",
                f"    input.detector == \"{detector}\"",
                f"    input.threshold < {recommended}",
                f"    input.false_positive_rate > {current_fp * 0.5}  # Target 50% reduction",
                f'    msg := "Adjust {detector} threshold to {recommended:.2f} (currently experiencing {current_fp:.1%} false positives)"',
                "}",
                ""
            ])
        
        return "\n".join(policy_parts)
    
    def _calculate_combined_confidence(self, confidences: list) -> float:
        """Calculate combined confidence from multiple engines."""
        if not confidences:
            return 0.1
        
        # Use weighted average with higher weight for higher confidences
        weights = [c ** 2 for c in confidences]  # Square to emphasize higher confidences
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        weight_sum = sum(weights)
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.1
    
    def _generate_combined_recommendations(
        self,
        pattern_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        compliance_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate combined recommendations from all engines."""
        recommendations = []
        
        # Add pattern-based recommendations
        pattern_strength = pattern_result.get("pattern_strength", "weak")
        if pattern_strength in ["moderate", "strong"]:
            recommendations.append({
                "type": "pattern_optimization",
                "priority": "high" if pattern_strength == "strong" else "medium",
                "description": f"Address {pattern_strength} patterns detected in security data",
                "source": "pattern_engine"
            })
        
        # Add risk-based recommendations
        risk_score_data = risk_result.get("risk_score", {})
        risk_level = risk_score_data.get("risk_level", "low")
        if risk_level in ["high", "critical"]:
            recommendations.append({
                "type": "risk_mitigation",
                "priority": "critical" if risk_level == "critical" else "high",
                "description": f"Mitigate {risk_level} risk factors identified",
                "source": "risk_engine"
            })
        
        # Add compliance recommendations
        framework_mappings = compliance_result.get("framework_mappings", {})
        compliance_score = framework_mappings.get("compliance_score", 100)
        if compliance_score < 90:
            recommendations.append({
                "type": "compliance_improvement",
                "priority": "high" if compliance_score < 70 else "medium",
                "description": f"Improve compliance score from {compliance_score:.1f}%",
                "source": "compliance_engine"
            })
        
        return recommendations
    
    def _generate_executive_summary(
        self,
        pattern_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        compliance_result: Dict[str, Any],
    ) -> str:
        """Generate executive summary combining all engine insights."""
        pattern_strength = pattern_result.get("pattern_strength", "weak")
        risk_score_data = risk_result.get("risk_score", {})
        risk_level = risk_score_data.get("risk_level", "low")
        framework_mappings = compliance_result.get("framework_mappings", {})
        compliance_score = framework_mappings.get("compliance_score", 100)
        
        return f"Security analysis reveals {pattern_strength} patterns with {risk_level} risk level and {compliance_score:.1f}% compliance score. Coordinated remediation across pattern optimization, risk mitigation, and compliance improvement recommended."
    
    # Helper methods for comprehensive analysis
    def _generate_priority_actions_comprehensive(
        self, gaps: list, risk_assessment: Dict[str, Any], compliance_result: Dict[str, Any]
    ) -> list:
        """Generate priority actions with risk and compliance context."""
        actions = []
        
        for i, gap in enumerate(gaps[:5]):  # Top 5 priorities
            # Get risk context for this detector
            detector_risk = self._get_detector_risk_context(gap["detector"], risk_assessment)
            
            # Get compliance context
            compliance_impact = self._get_detector_compliance_context(gap["detector"], compliance_result)
            
            actions.append({
                "priority": i + 1,
                "action": f"Enable {gap['detector']} detector",
                "detector": gap["detector"],
                "business_impact": gap["business_impact"],
                "risk_context": detector_risk,
                "compliance_impact": compliance_impact,
                "effort": "medium" if gap["gap_severity"] > 0.5 else "low",
                "timeline": "immediate" if gap["business_impact"] == "critical" else "1-2 weeks"
            })
        
        return actions
    
    def _estimate_remediation_effort_comprehensive(
        self, gaps: list, risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate effort with risk context."""
        total_gaps = len(gaps)
        critical_gaps = len([g for g in gaps if g["business_impact"] == "critical"])
        high_gaps = len([g for g in gaps if g["business_impact"] == "high"])
        
        # Adjust effort based on risk level
        risk_level = risk_assessment.get("risk_level", "low")
        risk_multiplier = {"critical": 1.5, "high": 1.2, "medium": 1.0, "low": 0.8}[risk_level]
        
        base_hours = (critical_gaps * 8) + (high_gaps * 4) + ((total_gaps - critical_gaps - high_gaps) * 2)
        estimated_hours = int(base_hours * risk_multiplier)
        estimated_days = max(1, estimated_hours // 8)
        
        return {
            "estimated_hours": estimated_hours,
            "estimated_days": estimated_days,
            "complexity": "high" if critical_gaps >= 2 else "medium" if high_gaps >= 2 else "low",
            "risk_adjusted": True,
            "risk_multiplier": risk_multiplier,
            "resources_needed": ["security_engineer", "devops_engineer"] if critical_gaps > 0 else ["security_engineer"]
        }
    
    def _create_coverage_summary_comprehensive(
        self, gaps: list, risk_level: str, compliance_score: float
    ) -> str:
        """Create comprehensive coverage summary."""
        if not gaps:
            return f"All required detectors have adequate coverage. Risk level: {risk_level}, Compliance: {compliance_score:.1f}%"
        
        critical_count = len([g for g in gaps if g["business_impact"] == "critical"])
        high_count = len([g for g in gaps if g["business_impact"] == "high"])
        
        if critical_count > 0:
            return f"Critical coverage gaps detected in {critical_count} high-impact detectors requiring immediate attention. Risk level: {risk_level}, Compliance impact: {compliance_score:.1f}%"
        elif high_count > 0:
            return f"Significant coverage gaps identified in {high_count} detectors with high business impact. Risk level: {risk_level}, Compliance: {compliance_score:.1f}%"
        else:
            return f"Coverage gaps detected in {len(gaps)} detectors with moderate business impact. Risk level: {risk_level}, Compliance: {compliance_score:.1f}%"
    
    def _assess_threshold_risk_impact(self, pattern: Dict[str, Any], risk_impact: Dict[str, Any]) -> str:
        """Assess risk impact of threshold changes."""
        fp_rate = pattern["current_fp_rate"]
        risk_level = risk_impact.get("risk_level", "low")
        
        if fp_rate > 0.2 and risk_level in ["high", "critical"]:
            return "high_operational_risk"
        elif fp_rate > 0.1:
            return "medium_operational_risk"
        else:
            return "low_operational_risk"
    
    def _determine_overall_pattern_strength(self, patterns: list) -> str:
        """Determine overall pattern strength."""
        if not patterns:
            return "weak"
        
        strong_patterns = len([p for p in patterns if p.get("pattern_strength") == "strong"])
        moderate_patterns = len([p for p in patterns if p.get("pattern_strength") == "moderate"])
        
        if strong_patterns >= 2:
            return "strong"
        elif strong_patterns >= 1 or moderate_patterns >= 3:
            return "moderate"
        else:
            return "weak"
    
    def _recommend_validation_approach_comprehensive(
        self, pattern_strength: str, risk_impact: Dict[str, Any]
    ) -> str:
        """Recommend validation approach with risk context."""
        risk_level = risk_impact.get("risk_level", "low")
        
        if pattern_strength == "strong" and risk_level in ["high", "critical"]:
            return "A/B test with 5% traffic split for 2 weeks with enhanced monitoring due to high risk"
        elif pattern_strength == "strong":
            return "A/B test with 10% traffic split for 1 week to validate threshold changes"
        elif pattern_strength == "moderate":
            return "Gradual rollout with 5% traffic for 3 days, then full deployment"
        else:
            return "Monitor for 24 hours after threshold adjustment"
    
    def _generate_response_timeline_comprehensive(
        self, patterns: list, risk_assessment: Dict[str, Any], compliance_impact: Dict[str, Any], escalation_required: bool
    ) -> Dict[str, str]:
        """Generate response timeline with risk and compliance context."""
        risk_level = risk_assessment.get("risk_level", "low")
        compliance_score = compliance_impact.get("compliance_score", 100)
        
        if escalation_required or risk_level == "critical" or compliance_score < 50:
            return {
                "immediate": "Notify security team and stakeholders - critical risk/compliance impact",
                "1_hour": "Begin incident containment procedures with compliance considerations",
                "4_hours": "Complete initial impact assessment including regulatory implications",
                "24_hours": "Implement remediation measures with compliance validation",
                "72_hours": "Conduct post-incident review and compliance audit"
            }
        else:
            return {
                "immediate": "Log and categorize incidents with risk context",
                "2_hours": "Investigate incident patterns and compliance implications",
                "8_hours": "Implement threshold adjustments with risk validation",
                "24_hours": "Monitor for pattern changes and compliance impact"
            }
    
    def _create_severity_distribution_summary(self, patterns: list) -> str:
        """Create severity distribution summary."""
        if not patterns:
            return "No incident patterns detected"
        
        total_incidents = sum(p["incident_count"] for p in patterns)
        escalation_count = len([p for p in patterns if p["requires_escalation"]])
        
        return f"{total_incidents} total incidents, {escalation_count} requiring escalation"
    
    def _get_detector_risk_context(self, detector: str, risk_assessment: Dict[str, Any]) -> str:
        """Get risk context for specific detector."""
        # Simplified - in real implementation would analyze risk factors for specific detector
        return risk_assessment.get("risk_level", "low")
    
    def _get_detector_compliance_context(self, detector: str, compliance_result: Dict[str, Any]) -> str:
        """Get compliance context for specific detector."""
        # Simplified - in real implementation would check detector compliance mappings
        framework_mappings = compliance_result.get("framework_mappings", {})
        compliance_score = framework_mappings.get("compliance_score", 100)
        if compliance_score < 70:
            return "high_compliance_impact"
        elif compliance_score < 90:
            return "medium_compliance_impact"
        else:
            return "low_compliance_impact"