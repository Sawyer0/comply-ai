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
        fallback_reason: Optional[str] = None,
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
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced coverage gap analysis response."""
        # Perform sophisticated coverage gap analysis
        analysis = self._analyze_coverage_gaps(request)
        
        # Generate specific, actionable remediation
        remediation = self._generate_coverage_remediation(analysis)
        
        # Create comprehensive OPA policy
        opa_diff = self._generate_advanced_coverage_opa_policy(analysis)
        
        # Calculate confidence based on data quality and analysis depth
        confidence = self._calculate_coverage_confidence(analysis)

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
                "risk_assessment": analysis["risk_level"],
                "priority_actions": analysis["priority_actions"],
                "estimated_effort": analysis["estimated_effort"]
            }
        }

    def _get_false_positive_template(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced false positive tuning analysis response."""
        # Perform sophisticated false positive analysis
        analysis = self._analyze_false_positives(request)
        
        # Generate specific threshold recommendations
        remediation = self._generate_threshold_remediation(analysis)
        
        # Create advanced OPA policy with threshold enforcement
        opa_diff = self._generate_advanced_threshold_opa_policy(analysis)
        
        # Calculate confidence based on false positive pattern strength
        confidence = self._calculate_fp_confidence(analysis)

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
                "validation_approach": analysis["validation_approach"]
            }
        }
    
    def _analyze_false_positives(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Analyze false positive patterns and generate recommendations."""
        patterns = []
        threshold_recommendations = {}
        total_fp_score = 0.0
        
        for band in request.false_positive_bands:
            detector = band.get("detector", "unknown")
            fp_rate = band.get("false_positive_rate", 0.0)
            current_threshold = band.get("current_threshold", 0.5)
            
            # Analyze pattern strength
            if fp_rate > 0.2:  # High false positive rate
                pattern_strength = "strong"
                recommended_threshold = min(current_threshold + 0.2, 0.9)
                expected_reduction = min(fp_rate * 0.7, 0.8)  # Expect 70% reduction
            elif fp_rate > 0.1:  # Medium false positive rate
                pattern_strength = "moderate"
                recommended_threshold = min(current_threshold + 0.1, 0.8)
                expected_reduction = min(fp_rate * 0.5, 0.6)  # Expect 50% reduction
            else:  # Low false positive rate
                pattern_strength = "weak"
                recommended_threshold = min(current_threshold + 0.05, 0.7)
                expected_reduction = min(fp_rate * 0.3, 0.4)  # Expect 30% reduction
            
            patterns.append({
                "detector": detector,
                "current_fp_rate": fp_rate,
                "pattern_strength": pattern_strength,
                "recommended_threshold": recommended_threshold,
                "expected_reduction": expected_reduction
            })
            
            threshold_recommendations[detector] = {
                "current": current_threshold,
                "recommended": recommended_threshold,
                "justification": f"Reduce {fp_rate:.1%} false positive rate by {expected_reduction:.1%}"
            }
            
            total_fp_score += fp_rate
        
        # Determine overall pattern strength
        avg_fp_rate = total_fp_score / len(request.false_positive_bands) if request.false_positive_bands else 0
        overall_pattern_strength = "strong" if avg_fp_rate > 0.15 else "moderate" if avg_fp_rate > 0.05 else "weak"
        
        # Calculate expected overall reduction
        total_expected_reduction = sum(p["expected_reduction"] for p in patterns) / len(patterns) if patterns else 0
        
        # Create summary
        summary = f"False positive patterns detected in {len(patterns)} detectors with {overall_pattern_strength} signal strength"
        
        return {
            "patterns": patterns,
            "threshold_recommendations": threshold_recommendations,
            "pattern_strength": overall_pattern_strength,
            "expected_fp_reduction": total_expected_reduction,
            "summary": summary,
            "affected_detectors": [p["detector"] for p in patterns],
            "validation_approach": self._recommend_validation_approach(overall_pattern_strength)
        }
    
    def _recommend_validation_approach(self, pattern_strength: str) -> str:
        """Recommend validation approach based on pattern strength."""
        if pattern_strength == "strong":
            return "A/B test with 10% traffic split for 1 week to validate threshold changes"
        elif pattern_strength == "moderate":
            return "Gradual rollout with 5% traffic for 3 days, then full deployment"
        else:
            return "Monitor for 24 hours after threshold adjustment"
    
    def _generate_threshold_remediation(self, analysis: Dict[str, Any]) -> str:
        """Generate specific threshold remediation steps."""
        if not analysis["patterns"]:
            return "No false positive patterns detected - current thresholds are appropriate"
        
        top_pattern = max(analysis["patterns"], key=lambda x: x["current_fp_rate"])
        total_detectors = len(analysis["patterns"])
        expected_reduction = analysis["expected_fp_reduction"]
        
        return f"Adjust {top_pattern['detector']} threshold from {top_pattern.get('current_threshold', 0.5):.2f} to {top_pattern['recommended_threshold']:.2f}. Expected {expected_reduction:.1%} false positive reduction across {total_detectors} detectors. Use {analysis['validation_approach']}."
    
    def _generate_advanced_threshold_opa_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate advanced OPA policy for threshold enforcement."""
        policy_parts = [
            "package threshold_enforcement",
            "",
            "# Advanced threshold enforcement with false positive optimization",
            ""
        ]
        
        # Generate detector-specific policies
        for pattern in analysis["patterns"]:
            detector = pattern["detector"]
            recommended = pattern["recommended_threshold"]
            current_fp = pattern["current_fp_rate"]
            
            policy_parts.extend([
                f"# {detector} threshold enforcement",
                f"{detector.replace('-', '_')}_threshold_violation[msg] {{",
                f"    input.detector == \"{detector}\"",
                f"    input.threshold < {recommended}",
                f"    input.false_positive_rate > {current_fp * 0.5}  # Target 50% reduction",
                f'    msg := "Adjust {detector} threshold to {recommended:.2f} (currently experiencing {current_fp:.1%} false positives)"',
                "}",
                ""
            ])
        
        # Overall policy
        policy_parts.extend([
            "# Overall false positive enforcement",
            "overall_fp_violation[msg] {",
            "    detector := input.detectors[_]",
            "    fp_rate := input.false_positive_rates[detector]",
            "    fp_rate > 0.15  # Alert if any detector has >15% false positives",
            '    msg := sprintf("High false positive rate detected: %v at %v%%", [detector, fp_rate * 100])',
            "}"
        ])
        
        return "\n".join(policy_parts)
    
    def _calculate_fp_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence for false positive analysis."""
        base_confidence = 0.6
        
        # Higher confidence for stronger patterns
        if analysis["pattern_strength"] == "strong":
            base_confidence += 0.2
        elif analysis["pattern_strength"] == "moderate":
            base_confidence += 0.1
        
        # Higher confidence for more data points
        if len(analysis["patterns"]) >= 3:
            base_confidence += 0.1
        
        return min(0.9, base_confidence)

    def _get_incident_summary_template(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get enhanced incident summary analysis response."""
        # Perform sophisticated incident analysis
        analysis = self._analyze_security_incidents(request)
        
        # Generate specific incident response plan
        remediation = self._generate_incident_remediation(analysis)
        
        # Create comprehensive incident response OPA policy
        opa_diff = self._generate_advanced_incident_opa_policy(analysis)
        
        # Calculate confidence based on incident pattern analysis
        confidence = self._calculate_incident_confidence(analysis)

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
                "response_timeline": analysis["response_timeline"],
                "escalation_required": analysis["escalation_required"]
            }
        }
    
    def _analyze_security_incidents(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Perform sophisticated security incident analysis."""
        incidents = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        detector_incident_map = {}
        
        for hit in request.high_sev_hits:
            detector = hit.get("detector", "unknown")
            severity = hit.get("severity", "medium")
            incident_type = hit.get("type", "unknown")
            
            # Classify incident impact
            business_impact = self._classify_incident_impact(detector, incident_type, severity)
            
            incident_info = {
                "detector": detector,
                "severity": severity,
                "type": incident_type,
                "business_impact": business_impact,
                "requires_escalation": business_impact in ["critical", "high"] and severity in ["critical", "high"]
            }
            
            incidents.append(incident_info)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if detector not in detector_incident_map:
                detector_incident_map[detector] = []
            detector_incident_map[detector].append(incident_info)
        
        # Analyze patterns
        total_incidents = len(incidents)
        critical_incidents = [i for i in incidents if i["business_impact"] == "critical"]
        escalation_required = len(critical_incidents) > 0 or severity_counts.get("critical", 0) > 2
        
        # Generate response timeline
        response_timeline = self._generate_response_timeline(incidents, escalation_required)
        
        # Create severity distribution summary
        severity_dist = f"{severity_counts.get('critical', 0)} critical, {severity_counts.get('high', 0)} high, {severity_counts.get('medium', 0)} medium"
        
        # Create summary
        if critical_incidents:
            summary = f"Critical security incidents detected: {len(critical_incidents)} critical incidents requiring immediate escalation"
        elif severity_counts.get("high", 0) >= 3:
            summary = f"Multiple high-severity incidents detected: {severity_counts['high']} incidents requiring urgent attention"
        else:
            summary = f"Security incidents detected: {total_incidents} incidents across multiple detectors"
        
        return {
            "incidents": incidents,
            "incident_breakdown": detector_incident_map,
            "severity_distribution": severity_dist,
            "escalation_required": escalation_required,
            "response_timeline": response_timeline,
            "summary": summary,
            "affected_detectors": list(detector_incident_map.keys()),
            "critical_count": len(critical_incidents),
            "total_count": total_incidents
        }
    
    def _classify_incident_impact(self, detector: str, incident_type: str, severity: str) -> str:
        """Classify business impact of security incident."""
        # High-impact detectors and incident types
        critical_detectors = ["pii-detector", "gdpr-scanner", "hipaa-validator", "financial-data"]
        critical_types = ["data_breach", "pii_exposure", "compliance_violation"]
        
        if detector in critical_detectors and incident_type in critical_types:
            return "critical"
        elif severity == "critical" or incident_type in critical_types:
            return "high"
        elif severity == "high" or detector in critical_detectors:
            return "medium"
        else:
            return "low"
    
    def _generate_response_timeline(self, incidents: list, escalation_required: bool) -> Dict[str, str]:
        """Generate incident response timeline."""
        if escalation_required:
            return {
                "immediate": "Notify security team and stakeholders",
                "1_hour": "Begin incident containment procedures",
                "4_hours": "Complete initial impact assessment",
                "24_hours": "Implement remediation measures",
                "72_hours": "Conduct post-incident review"
            }
        else:
            return {
                "immediate": "Log and categorize incidents",
                "2_hours": "Investigate incident patterns",
                "8_hours": "Implement threshold adjustments",
                "24_hours": "Monitor for pattern changes"
            }
    
    def _generate_incident_remediation(self, analysis: Dict[str, Any]) -> str:
        """Generate specific incident remediation plan."""
        if analysis["escalation_required"]:
            critical_count = analysis["critical_count"]
            return f"IMMEDIATE ESCALATION: {critical_count} critical incidents detected. Notify security team within 1 hour. Begin containment procedures. Estimated resolution time: 24-72 hours with security team involvement."
        else:
            total_count = analysis["total_count"]
            affected_detectors = len(analysis["affected_detectors"])
            return f"Standard incident response: {total_count} incidents across {affected_detectors} detectors. Investigate patterns, adjust thresholds, monitor for 24 hours. Estimated resolution: 4-8 hours."
    
    def _generate_advanced_incident_opa_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive incident response OPA policy."""
        policy_parts = [
            "package incident_response",
            "",
            "# Advanced incident response with escalation logic",
            ""
        ]
        
        # Critical incident escalation
        if analysis["escalation_required"]:
            policy_parts.extend([
                "# Critical incident escalation policy",
                "critical_escalation[msg] {",
                "    incident := input.incidents[_]",
                "    incident.business_impact == \"critical\"",
                "    incident.severity in [\"critical\", \"high\"]",
                '    msg := sprintf("ESCALATE: Critical %v incident from %v detector", [incident.type, incident.detector])',
                "}",
                ""
            ])
        
        # Detector-specific incident policies
        for detector, incidents in analysis["incident_breakdown"].items():
            incident_count = len(incidents)
            if incident_count >= 2:  # Multiple incidents from same detector
                policy_parts.extend([
                    f"# {detector} incident pattern enforcement",
                    f"{detector.replace('-', '_')}_pattern_violation[msg] {{",
                    f"    count([incident | incident := input.incidents[_]; incident.detector == \"{detector}\"]) >= {incident_count}",
                    f'    msg := "Pattern detected: {incident_count} incidents from {detector} - investigate detector configuration"',
                    "}",
                    ""
                ])
        
        # Response timeline enforcement
        policy_parts.extend([
            "# Response timeline enforcement",
            "response_timeline_violation[msg] {",
            "    incident := input.incidents[_]",
            "    incident.business_impact == \"critical\"",
            "    time.now_ns() - incident.timestamp_ns > 3600000000000  # 1 hour in nanoseconds",
            '    msg := sprintf("SLA BREACH: Critical incident %v not addressed within 1 hour", [incident.id])',
            "}"
        ])
        
        return "\n".join(policy_parts)
    
    def _calculate_incident_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence for incident analysis."""
        base_confidence = 0.7
        
        # Higher confidence for clear escalation scenarios
        if analysis["escalation_required"]:
            base_confidence += 0.1
        
        # Higher confidence for pattern detection
        if len(analysis["affected_detectors"]) >= 2:
            base_confidence += 0.1
        
        # Higher confidence for more incidents (better pattern analysis)
        if analysis["total_count"] >= 5:
            base_confidence += 0.1
        
        return min(0.9, base_confidence)

    def _get_insufficient_data_template(
        self, request: AnalysisRequest, fallback_reason: Optional[str] = None
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
            "notes": f"Template fallback for insufficient data{f' - {fallback_reason}' if fallback_reason else ''}",
        }

    def _analyze_coverage_gaps(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Perform sophisticated coverage gap analysis."""
        gaps = []
        critical_gaps = []
        total_gap_score = 0.0
        
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            
            if observed < required:
                gap_severity = (required - observed) / required
                gap_info = {
                    "detector": detector,
                    "observed": observed,
                    "required": required,
                    "gap_percentage": gap_severity,
                    "severity": self._classify_gap_severity(gap_severity),
                    "business_impact": self._assess_business_impact(detector, gap_severity)
                }
                gaps.append(gap_info)
                total_gap_score += gap_severity
                
                if gap_severity > 0.5:  # Critical gap
                    critical_gaps.append(detector)
        
        # Determine overall risk level
        avg_gap_score = total_gap_score / len(request.required_detectors) if request.required_detectors else 0
        risk_level = self._determine_risk_level(avg_gap_score, len(critical_gaps))
        
        # Generate priority actions
        priority_actions = self._generate_priority_actions(gaps, risk_level)
        
        # Estimate remediation effort
        estimated_effort = self._estimate_remediation_effort(gaps)
        
        # Create summary
        summary = self._create_coverage_summary(gaps, risk_level)
        
        return {
            "gaps": gaps,
            "critical_gaps": critical_gaps,
            "risk_level": risk_level,
            "priority_actions": priority_actions,
            "estimated_effort": estimated_effort,
            "summary": summary,
            "affected_detectors": [gap["detector"] for gap in gaps],
            "total_detectors": len(request.required_detectors),
            "gaps_count": len(gaps)
        }
    
    def _classify_gap_severity(self, gap_percentage: float) -> str:
        """Classify gap severity based on percentage."""
        if gap_percentage >= 0.7:
            return "critical"
        elif gap_percentage >= 0.4:
            return "high"
        elif gap_percentage >= 0.2:
            return "medium"
        else:
            return "low"
    
    def _assess_business_impact(self, detector: str, gap_severity: float) -> str:
        """Assess business impact of coverage gap."""
        # High-impact detectors
        high_impact_detectors = ["presidio", "pii-detector", "gdpr-scanner", "hipaa-validator"]
        medium_impact_detectors = ["toxicity-detector", "hate-speech", "content-moderation"]
        
        base_impact = "high" if detector in high_impact_detectors else "medium" if detector in medium_impact_detectors else "low"
        
        # Adjust based on gap severity
        if gap_severity >= 0.7 and base_impact in ["medium", "high"]:
            return "critical"
        elif gap_severity >= 0.4:
            return "high" if base_impact != "low" else "medium"
        else:
            return base_impact
    
    def _determine_risk_level(self, avg_gap_score: float, critical_gaps_count: int) -> str:
        """Determine overall risk level."""
        if critical_gaps_count >= 3 or avg_gap_score >= 0.6:
            return "critical"
        elif critical_gaps_count >= 1 or avg_gap_score >= 0.3:
            return "high"
        elif avg_gap_score >= 0.1:
            return "medium"
        else:
            return "low"
    
    def _generate_priority_actions(self, gaps: list, risk_level: str) -> list:
        """Generate prioritized action items."""
        actions = []
        
        # Sort gaps by business impact and severity
        sorted_gaps = sorted(gaps, key=lambda x: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[x["business_impact"]],
            x["gap_percentage"]
        ), reverse=True)
        
        for i, gap in enumerate(sorted_gaps[:5]):  # Top 5 priorities
            if gap["business_impact"] in ["critical", "high"]:
                actions.append({
                    "priority": i + 1,
                    "action": f"Enable {gap['detector']} detector",
                    "detector": gap["detector"],
                    "impact": gap["business_impact"],
                    "effort": "medium" if gap["gap_percentage"] > 0.5 else "low",
                    "timeline": "immediate" if gap["business_impact"] == "critical" else "1-2 weeks"
                })
        
        return actions
    
    def _estimate_remediation_effort(self, gaps: list) -> Dict[str, Any]:
        """Estimate effort required for remediation."""
        total_gaps = len(gaps)
        critical_gaps = len([g for g in gaps if g["business_impact"] == "critical"])
        high_gaps = len([g for g in gaps if g["business_impact"] == "high"])
        
        # Estimate based on gap complexity
        estimated_hours = (critical_gaps * 8) + (high_gaps * 4) + ((total_gaps - critical_gaps - high_gaps) * 2)
        estimated_days = max(1, estimated_hours // 8)
        
        return {
            "estimated_hours": estimated_hours,
            "estimated_days": estimated_days,
            "complexity": "high" if critical_gaps >= 2 else "medium" if high_gaps >= 2 else "low",
            "resources_needed": ["security_engineer", "devops_engineer"] if critical_gaps > 0 else ["security_engineer"]
        }
    
    def _create_coverage_summary(self, gaps: list, risk_level: str) -> str:
        """Create human-readable summary."""
        if not gaps:
            return "All required detectors have adequate coverage"
        
        critical_count = len([g for g in gaps if g["business_impact"] == "critical"])
        high_count = len([g for g in gaps if g["business_impact"] == "high"])
        
        if critical_count > 0:
            return f"Critical coverage gaps detected in {critical_count} high-impact detectors requiring immediate attention"
        elif high_count > 0:
            return f"Significant coverage gaps identified in {high_count} detectors with high business impact"
        else:
            return f"Coverage gaps detected in {len(gaps)} detectors with moderate business impact"
    
    def _generate_coverage_remediation(self, analysis: Dict[str, Any]) -> str:
        """Generate specific, actionable remediation steps."""
        if not analysis["gaps"]:
            return "No remediation required - coverage targets are met"
        
        priority_actions = analysis["priority_actions"]
        if not priority_actions:
            return "Review detector configurations and adjust coverage requirements"
        
        top_action = priority_actions[0]
        effort = analysis["estimated_effort"]
        
        return f"Priority: {top_action['action']} ({top_action['timeline']}). Estimated effort: {effort['estimated_days']} days with {effort['complexity']} complexity. Resources needed: {', '.join(effort['resources_needed'])}."
    
    def _generate_advanced_coverage_opa_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate advanced OPA policy for coverage enforcement."""
        policy_parts = [
            "package coverage_enforcement",
            "",
            "# Advanced coverage enforcement with business impact analysis",
            ""
        ]
        
        # Generate detector-specific policies
        for gap in analysis["gaps"]:
            detector = gap["detector"]
            required = gap["required"]
            business_impact = gap["business_impact"]
            
            policy_parts.extend([
                f"# {detector} coverage enforcement",
                f"{detector.replace('-', '_')}_coverage_violation[msg] {{",
                f"    input.detector == \"{detector}\"",
                f"    input.observed_coverage < {required}",
                f"    input.business_impact == \"{business_impact}\"",
                f'    msg := "Coverage gap: {detector} at {{input.observed_coverage}}% (required: {required}%) - {business_impact} business impact"',
                "}",
                ""
            ])
        
        # Overall coverage policy
        policy_parts.extend([
            "# Overall coverage enforcement",
            "critical_coverage_violation[msg] {",
            "    detector := input.detectors[_]",
            "    coverage := input.coverage_levels[detector]",
            "    required := input.required_levels[detector]",
            "    business_impact := input.business_impacts[detector]",
            "    coverage < required",
            "    business_impact in [\"critical\", \"high\"]",
            '    msg := sprintf("Critical coverage gap: %v at %v%% (required: %v%%)", [detector, coverage * 100, required * 100])',
            "}"
        ])
        
        return "\n".join(policy_parts)
    
    def _calculate_coverage_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence for coverage analysis."""
        base_confidence = 0.8  # High confidence for rule-based analysis
        
        # Higher confidence for more comprehensive data
        if analysis["total_detectors"] >= 5:
            base_confidence += 0.1
        
        # Lower confidence for insufficient data
        if analysis["gaps_count"] == 0:
            base_confidence -= 0.2  # Less confident when no gaps found
        
        return min(0.9, max(0.3, base_confidence))
    
    def _generate_advanced_coverage_opa_policy(self, analysis: Dict[str, Any]) -> str:
        """Generate comprehensive OPA policy based on analysis."""
        policy_parts = [
            "package coverage_enforcement",
            "",
            "# Advanced coverage enforcement with business impact assessment",
            ""
        ]
        
        # Critical detector enforcement
        critical_detectors = [gap["detector"] for gap in analysis["gaps"] if gap["business_impact"] == "critical"]
        if critical_detectors:
            policy_parts.extend([
                "# Critical detectors - must have 100% coverage",
                "critical_violation[msg] {",
                f"    detector := {critical_detectors}[_]",
                "    observed := input.observed_coverage[detector]",
                "    observed < 1.0",
                '    msg := sprintf("CRITICAL: %v detector coverage %v%% below required 100%%", [detector, observed * 100])',
                "}",
                ""
            ])
        
        # High-impact detector enforcement
        high_impact_detectors = [gap["detector"] for gap in analysis["gaps"] if gap["business_impact"] == "high"]
        if high_impact_detectors:
            policy_parts.extend([
                "# High-impact detectors - enforce strict thresholds",
                "high_impact_violation[msg] {",
                f"    detector := {high_impact_detectors}[_]",
                "    observed := input.observed_coverage[detector]",
                "    required := input.required_coverage[detector]",
                "    observed < required",
                '    msg := sprintf("HIGH IMPACT: %v detector coverage %v%% below required %v%%", [detector, observed * 100, required * 100])',
                "}",
                ""
            ])
        
        # Risk-based enforcement
        if analysis["risk_level"] in ["critical", "high"]:
            policy_parts.extend([
                f"# {analysis['risk_level'].upper()} risk scenario - enhanced monitoring",
                "risk_based_violation[msg] {",
                "    count(critical_violation) > 0",
                f'    msg := "ALERT: {analysis["risk_level"].upper()} risk coverage scenario detected - immediate action required"',
                "}",
                ""
            ])
        
        return "\n".join(policy_parts)
    
    def _calculate_coverage_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence score based on analysis quality."""
        base_confidence = 0.7  # Start with good confidence for rule-based analysis
        
        # Adjust based on data quality
        if analysis["total_detectors"] >= 5:
            base_confidence += 0.1  # More detectors = better analysis
        
        if analysis["gaps_count"] > 0:
            base_confidence += 0.1  # Clear gaps identified = confident analysis
        
        # Adjust based on risk assessment clarity
        if analysis["risk_level"] in ["critical", "high"]:
            base_confidence += 0.1  # Clear risk assessment
        
        # Cap at reasonable maximum for rule-based system
        return min(0.9, base_confidence)

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
