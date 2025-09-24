"""
Risk Scoring Engine

Extracted from the monolithic template provider to provide specialized
risk calculation and scoring capabilities with business context.
"""

import logging
from typing import Any, Dict, List
from ..domain.entities import AnalysisRequest
from .interfaces import IRiskScoringEngine

logger = logging.getLogger(__name__)


class RiskScoringEngine(IRiskScoringEngine):
    """
    Specialized engine for calculating intelligent risk scores.
    
    This engine was extracted from the monolithic AnalysisTemplateProvider
    to provide focused risk scoring capabilities including:
    - Multi-dimensional risk calculation
    - Business impact assessment
    - Regulatory compliance weighting
    - Temporal risk decay functions
    """
    
    def __init__(self):
        """Initialize the risk scoring engine."""
        self.risk_weights = {
            "technical": 0.3,
            "business": 0.4,
            "regulatory": 0.2,
            "temporal": 0.1
        }
        logger.info("Initialized Risk Scoring Engine")
    
    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis.
        
        Args:
            request: Analysis request containing security data
            
        Returns:
            Risk analysis results
        """
        risk_score = await self.calculate_risk_score(request)
        risk_breakdown = await self.get_risk_breakdown(risk_score)
        
        return {
            "risk_score": risk_score,
            "risk_breakdown": risk_breakdown,
            "confidence": self.get_confidence(risk_score),
            "analysis_type": "risk_scoring"
        }
    
    def get_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for risk analysis.
        
        Args:
            result: Risk analysis result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not result:
            return 0.1
        
        base_confidence = 0.7  # High confidence for rule-based risk scoring
        
        # Adjust based on data completeness
        data_completeness = result.get("data_completeness", 0.5)
        base_confidence += (data_completeness - 0.5) * 0.2
        
        # Adjust based on risk factor count
        contributing_factors = result.get("contributing_factors", [])
        if len(contributing_factors) >= 3:
            base_confidence += 0.1
        
        return min(0.9, max(0.1, base_confidence))
    
    async def calculate_risk_score(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score with multiple dimensions.
        
        Args:
            request: Analysis request
            
        Returns:
            Risk score with breakdown and contributing factors
        """
        # Calculate individual risk dimensions
        technical_risk = await self._calculate_technical_risk(request)
        business_risk = await self._calculate_business_risk(request)
        regulatory_risk = await self._calculate_regulatory_risk(request)
        temporal_risk = await self._calculate_temporal_risk(request)
        
        # Calculate weighted composite score
        composite_score = (
            technical_risk["score"] * self.risk_weights["technical"] +
            business_risk["score"] * self.risk_weights["business"] +
            regulatory_risk["score"] * self.risk_weights["regulatory"] +
            temporal_risk["score"] * self.risk_weights["temporal"]
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(composite_score)
        
        # Collect contributing factors
        contributing_factors = []
        contributing_factors.extend(technical_risk["factors"])
        contributing_factors.extend(business_risk["factors"])
        contributing_factors.extend(regulatory_risk["factors"])
        contributing_factors.extend(temporal_risk["factors"])
        
        # Calculate data completeness
        data_completeness = self._calculate_data_completeness(request)
        
        return {
            "composite_score": composite_score,
            "risk_level": risk_level,
            "technical_risk": technical_risk,
            "business_risk": business_risk,
            "regulatory_risk": regulatory_risk,
            "temporal_risk": temporal_risk,
            "contributing_factors": contributing_factors,
            "data_completeness": data_completeness,
            "risk_weights": self.risk_weights
        }
    
    async def get_risk_breakdown(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed risk breakdown with explanations.
        
        Args:
            risk_data: Risk calculation data
            
        Returns:
            Detailed risk breakdown
        """
        breakdown = {
            "overall_assessment": self._generate_overall_assessment(risk_data),
            "dimension_analysis": self._analyze_risk_dimensions(risk_data),
            "top_risk_factors": self._identify_top_risk_factors(risk_data),
            "mitigation_priorities": self._generate_mitigation_priorities(risk_data),
            "risk_trend": self._analyze_risk_trend(risk_data)
        }
        
        return breakdown
    
    async def _calculate_technical_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate technical risk based on detector performance and coverage."""
        factors = []
        risk_score = 0.0
        
        # Coverage gap risk
        coverage_risk = self._assess_coverage_risk(request)
        risk_score += coverage_risk["score"] * 0.4
        factors.extend(coverage_risk["factors"])
        
        # False positive risk
        fp_risk = self._assess_false_positive_risk(request)
        risk_score += fp_risk["score"] * 0.3
        factors.extend(fp_risk["factors"])
        
        # Detector error risk
        error_risk = self._assess_detector_error_risk(request)
        risk_score += error_risk["score"] * 0.3
        factors.extend(error_risk["factors"])
        
        return {
            "score": min(1.0, risk_score),
            "factors": factors,
            "category": "technical",
            "components": {
                "coverage_risk": coverage_risk["score"],
                "false_positive_risk": fp_risk["score"],
                "detector_error_risk": error_risk["score"]
            }
        }
    
    async def _calculate_business_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate business risk based on application context and impact."""
        factors = []
        risk_score = 0.0
        
        # Environment risk (prod > stage > dev)
        env_risk = self._assess_environment_risk(request.env)
        risk_score += env_risk["score"] * 0.4
        factors.extend(env_risk["factors"])
        
        # Application criticality risk
        app_risk = self._assess_application_risk(request.app)
        risk_score += app_risk["score"] * 0.3
        factors.extend(app_risk["factors"])
        
        # Route sensitivity risk
        route_risk = self._assess_route_risk(request.route)
        risk_score += route_risk["score"] * 0.3
        factors.extend(route_risk["factors"])
        
        return {
            "score": min(1.0, risk_score),
            "factors": factors,
            "category": "business",
            "components": {
                "environment_risk": env_risk["score"],
                "application_risk": app_risk["score"],
                "route_risk": route_risk["score"]
            }
        }
    
    async def _calculate_regulatory_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate regulatory compliance risk."""
        factors = []
        risk_score = 0.0
        
        # High severity incidents risk
        incident_risk = self._assess_incident_regulatory_risk(request.high_sev_hits)
        risk_score += incident_risk["score"] * 0.5
        factors.extend(incident_risk["factors"])
        
        # Detector compliance risk
        detector_risk = self._assess_detector_compliance_risk(request.required_detectors)
        risk_score += detector_risk["score"] * 0.3
        factors.extend(detector_risk["factors"])
        
        # Policy bundle risk
        policy_risk = self._assess_policy_compliance_risk(request.policy_bundle)
        risk_score += policy_risk["score"] * 0.2
        factors.extend(policy_risk["factors"])
        
        return {
            "score": min(1.0, risk_score),
            "factors": factors,
            "category": "regulatory",
            "components": {
                "incident_risk": incident_risk["score"],
                "detector_risk": detector_risk["score"],
                "policy_risk": policy_risk["score"]
            }
        }
    
    async def _calculate_temporal_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Calculate temporal risk based on time-sensitive factors."""
        factors = []
        risk_score = 0.0
        
        # Parse period to assess time sensitivity
        period_risk = self._assess_period_risk(request.period)
        risk_score += period_risk["score"]
        factors.extend(period_risk["factors"])
        
        return {
            "score": min(1.0, risk_score),
            "factors": factors,
            "category": "temporal",
            "components": {
                "period_risk": period_risk["score"]
            }
        }
    
    def _assess_coverage_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Assess risk from coverage gaps."""
        factors = []
        total_gap_risk = 0.0
        
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            
            if observed < required:
                gap_severity = (required - observed) / required
                detector_risk = gap_severity * self._get_detector_criticality(detector)
                total_gap_risk += detector_risk
                
                factors.append({
                    "type": "coverage_gap",
                    "detector": detector,
                    "gap_severity": gap_severity,
                    "risk_contribution": detector_risk,
                    "description": f"{detector} coverage gap: {gap_severity:.1%}"
                })
        
        # Normalize by number of detectors
        avg_risk = total_gap_risk / len(request.required_detectors) if request.required_detectors else 0.0
        
        return {
            "score": min(1.0, avg_risk),
            "factors": factors
        }
    
    def _assess_false_positive_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Assess risk from false positives."""
        factors = []
        total_fp_risk = 0.0
        
        for band in request.false_positive_bands:
            detector = band.get("detector", "unknown")
            fp_rate = band.get("false_positive_rate", 0.0)
            
            # False positives create operational risk
            fp_risk = fp_rate * 0.5  # Scale down as FPs are operational, not security risk
            total_fp_risk += fp_risk
            
            if fp_rate > 0.1:  # Only report significant false positive rates
                factors.append({
                    "type": "false_positive",
                    "detector": detector,
                    "fp_rate": fp_rate,
                    "risk_contribution": fp_risk,
                    "description": f"{detector} false positive rate: {fp_rate:.1%}"
                })
        
        # Normalize by number of bands
        avg_risk = total_fp_risk / len(request.false_positive_bands) if request.false_positive_bands else 0.0
        
        return {
            "score": min(1.0, avg_risk),
            "factors": factors
        }
    
    def _assess_detector_error_risk(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Assess risk from detector errors."""
        factors = []
        error_risk = 0.0
        
        for detector, error_info in request.detector_errors.items():
            error_count = error_info.get("error_count", 0)
            error_rate = error_info.get("error_rate", 0.0)
            
            # Detector errors create availability risk
            detector_error_risk = error_rate * self._get_detector_criticality(detector)
            error_risk += detector_error_risk
            
            if error_rate > 0.05:  # Only report significant error rates
                factors.append({
                    "type": "detector_error",
                    "detector": detector,
                    "error_rate": error_rate,
                    "error_count": error_count,
                    "risk_contribution": detector_error_risk,
                    "description": f"{detector} error rate: {error_rate:.1%}"
                })
        
        # Normalize by number of detectors with errors
        avg_risk = error_risk / len(request.detector_errors) if request.detector_errors else 0.0
        
        return {
            "score": min(1.0, avg_risk),
            "factors": factors
        }
    
    def _assess_environment_risk(self, env: str) -> Dict[str, Any]:
        """Assess risk based on environment."""
        env_risk_map = {
            "prod": 0.8,
            "stage": 0.4,
            "dev": 0.1
        }
        
        risk_score = env_risk_map.get(env, 0.5)
        factors = []
        
        if risk_score >= 0.7:
            factors.append({
                "type": "environment",
                "environment": env,
                "risk_contribution": risk_score,
                "description": f"Production environment increases risk exposure"
            })
        
        return {
            "score": risk_score,
            "factors": factors
        }
    
    def _assess_application_risk(self, app: str) -> Dict[str, Any]:
        """Assess risk based on application type."""
        # High-risk application patterns
        high_risk_apps = ["payment", "auth", "user-data", "financial", "healthcare"]
        medium_risk_apps = ["api", "web", "mobile", "admin"]
        
        risk_score = 0.3  # Default risk
        factors = []
        
        app_lower = app.lower()
        if any(pattern in app_lower for pattern in high_risk_apps):
            risk_score = 0.8
            factors.append({
                "type": "application",
                "application": app,
                "risk_contribution": risk_score,
                "description": f"High-risk application type: {app}"
            })
        elif any(pattern in app_lower for pattern in medium_risk_apps):
            risk_score = 0.5
            factors.append({
                "type": "application",
                "application": app,
                "risk_contribution": risk_score,
                "description": f"Medium-risk application type: {app}"
            })
        
        return {
            "score": risk_score,
            "factors": factors
        }
    
    def _assess_route_risk(self, route: str) -> Dict[str, Any]:
        """Assess risk based on route sensitivity."""
        # High-risk route patterns
        high_risk_routes = ["/api/v1/users", "/admin", "/payment", "/auth", "/data"]
        medium_risk_routes = ["/api", "/upload", "/download", "/profile"]
        
        risk_score = 0.2  # Default risk
        factors = []
        
        route_lower = route.lower()
        if any(pattern in route_lower for pattern in high_risk_routes):
            risk_score = 0.7
            factors.append({
                "type": "route",
                "route": route,
                "risk_contribution": risk_score,
                "description": f"High-risk route: {route}"
            })
        elif any(pattern in route_lower for pattern in medium_risk_routes):
            risk_score = 0.4
            factors.append({
                "type": "route",
                "route": route,
                "risk_contribution": risk_score,
                "description": f"Medium-risk route: {route}"
            })
        
        return {
            "score": risk_score,
            "factors": factors
        }
    
    def _assess_incident_regulatory_risk(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess regulatory risk from incidents."""
        factors = []
        incident_risk = 0.0
        
        for incident in incidents:
            severity = incident.get("severity", "medium")
            incident_type = incident.get("type", "unknown")
            
            # Map severity to risk
            severity_risk_map = {
                "critical": 0.9,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.2
            }
            
            base_risk = severity_risk_map.get(severity, 0.4)
            
            # Increase risk for compliance-related incidents
            compliance_types = ["pii_exposure", "data_breach", "compliance_violation"]
            if incident_type in compliance_types:
                base_risk *= 1.5
            
            incident_risk += base_risk
            
            if base_risk >= 0.6:
                factors.append({
                    "type": "regulatory_incident",
                    "severity": severity,
                    "incident_type": incident_type,
                    "risk_contribution": base_risk,
                    "description": f"{severity} {incident_type} incident"
                })
        
        # Normalize by incident count but cap at 1.0
        normalized_risk = min(1.0, incident_risk / max(1, len(incidents)))
        
        return {
            "score": normalized_risk,
            "factors": factors
        }
    
    def _assess_detector_compliance_risk(self, detectors: List[str]) -> Dict[str, Any]:
        """Assess compliance risk based on required detectors."""
        compliance_critical_detectors = [
            "presidio", "pii-detector", "gdpr-scanner", "hipaa-validator",
            "financial-data", "pci-scanner"
        ]
        
        critical_count = len([d for d in detectors if d in compliance_critical_detectors])
        risk_score = min(0.8, critical_count * 0.2)  # Each critical detector adds 20% risk
        
        factors = []
        if risk_score >= 0.4:
            factors.append({
                "type": "compliance_detectors",
                "critical_detector_count": critical_count,
                "risk_contribution": risk_score,
                "description": f"{critical_count} compliance-critical detectors required"
            })
        
        return {
            "score": risk_score,
            "factors": factors
        }
    
    def _assess_policy_compliance_risk(self, policy_bundle: str) -> Dict[str, Any]:
        """Assess risk based on policy bundle."""
        # High-risk policy bundles
        high_risk_policies = ["gdpr", "hipaa", "pci", "sox", "financial"]
        
        risk_score = 0.2  # Default risk
        factors = []
        
        policy_lower = policy_bundle.lower()
        if any(policy in policy_lower for policy in high_risk_policies):
            risk_score = 0.6
            factors.append({
                "type": "policy_compliance",
                "policy_bundle": policy_bundle,
                "risk_contribution": risk_score,
                "description": f"High-compliance policy bundle: {policy_bundle}"
            })
        
        return {
            "score": risk_score,
            "factors": factors
        }
    
    def _assess_period_risk(self, period: str) -> Dict[str, Any]:
        """Assess temporal risk based on analysis period."""
        # For now, return minimal temporal risk
        # In a real implementation, this would parse the period and assess
        # factors like analysis recency, period length, etc.
        
        return {
            "score": 0.1,
            "factors": [{
                "type": "temporal",
                "period": period,
                "risk_contribution": 0.1,
                "description": "Standard temporal risk assessment"
            }]
        }
    
    def _get_detector_criticality(self, detector: str) -> float:
        """Get criticality multiplier for detector."""
        critical_detectors = {
            "presidio": 1.0,
            "pii-detector": 1.0,
            "gdpr-scanner": 0.9,
            "hipaa-validator": 0.9,
            "financial-data": 0.8,
            "toxicity-detector": 0.6,
            "hate-speech": 0.6,
            "content-moderation": 0.5
        }
        
        return critical_detectors.get(detector, 0.4)
    
    def _determine_risk_level(self, composite_score: float) -> str:
        """Determine risk level from composite score."""
        if composite_score >= 0.8:
            return "critical"
        elif composite_score >= 0.6:
            return "high"
        elif composite_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_data_completeness(self, request: AnalysisRequest) -> float:
        """Calculate data completeness score."""
        completeness_factors = []
        
        # Check coverage data completeness
        if request.observed_coverage and request.required_coverage:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.0)
        
        # Check detector data completeness
        if request.required_detectors:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.0)
        
        # Check incident data
        if request.high_sev_hits is not None:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.5)
        
        # Check false positive data
        if request.false_positive_bands is not None:
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.5)
        
        return sum(completeness_factors) / len(completeness_factors)
    
    def _generate_overall_assessment(self, risk_data: Dict[str, Any]) -> str:
        """Generate overall risk assessment summary."""
        risk_level = risk_data["risk_level"]
        composite_score = risk_data["composite_score"]
        
        if risk_level == "critical":
            return f"Critical risk level detected (score: {composite_score:.2f}) - immediate attention required"
        elif risk_level == "high":
            return f"High risk level identified (score: {composite_score:.2f}) - urgent remediation needed"
        elif risk_level == "medium":
            return f"Medium risk level (score: {composite_score:.2f}) - monitor and plan remediation"
        else:
            return f"Low risk level (score: {composite_score:.2f}) - routine monitoring sufficient"
    
    def _analyze_risk_dimensions(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual risk dimensions."""
        return {
            "technical": {
                "score": risk_data["technical_risk"]["score"],
                "primary_concern": self._get_primary_technical_concern(risk_data["technical_risk"]),
                "weight": self.risk_weights["technical"]
            },
            "business": {
                "score": risk_data["business_risk"]["score"],
                "primary_concern": self._get_primary_business_concern(risk_data["business_risk"]),
                "weight": self.risk_weights["business"]
            },
            "regulatory": {
                "score": risk_data["regulatory_risk"]["score"],
                "primary_concern": self._get_primary_regulatory_concern(risk_data["regulatory_risk"]),
                "weight": self.risk_weights["regulatory"]
            },
            "temporal": {
                "score": risk_data["temporal_risk"]["score"],
                "primary_concern": "Standard temporal assessment",
                "weight": self.risk_weights["temporal"]
            }
        }
    
    def _identify_top_risk_factors(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify top contributing risk factors."""
        all_factors = risk_data["contributing_factors"]
        
        # Sort by risk contribution
        sorted_factors = sorted(
            all_factors,
            key=lambda x: x.get("risk_contribution", 0.0),
            reverse=True
        )
        
        return sorted_factors[:5]  # Top 5 risk factors
    
    def _generate_mitigation_priorities(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized mitigation recommendations."""
        priorities = []
        top_factors = self._identify_top_risk_factors(risk_data)
        
        for i, factor in enumerate(top_factors[:3]):  # Top 3 priorities
            priority = {
                "priority": i + 1,
                "factor_type": factor["type"],
                "description": factor["description"],
                "risk_reduction": factor["risk_contribution"],
                "effort_estimate": self._estimate_mitigation_effort(factor),
                "timeline": self._estimate_mitigation_timeline(factor)
            }
            priorities.append(priority)
        
        return priorities
    
    def _analyze_risk_trend(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk trend (simplified for this implementation)."""
        composite_score = risk_data["composite_score"]
        
        # In a real implementation, this would compare with historical data
        return {
            "trend": "stable",
            "current_score": composite_score,
            "trend_direction": "neutral",
            "confidence": 0.6
        }
    
    def _get_primary_technical_concern(self, technical_risk: Dict[str, Any]) -> str:
        """Get primary technical concern."""
        components = technical_risk["components"]
        max_component = max(components.items(), key=lambda x: x[1])
        return max_component[0].replace("_", " ").title()
    
    def _get_primary_business_concern(self, business_risk: Dict[str, Any]) -> str:
        """Get primary business concern."""
        components = business_risk["components"]
        max_component = max(components.items(), key=lambda x: x[1])
        return max_component[0].replace("_", " ").title()
    
    def _get_primary_regulatory_concern(self, regulatory_risk: Dict[str, Any]) -> str:
        """Get primary regulatory concern."""
        components = regulatory_risk["components"]
        max_component = max(components.items(), key=lambda x: x[1])
        return max_component[0].replace("_", " ").title()
    
    def _estimate_mitigation_effort(self, factor: Dict[str, Any]) -> str:
        """Estimate effort required for mitigation."""
        factor_type = factor["type"]
        
        effort_map = {
            "coverage_gap": "medium",
            "false_positive": "low",
            "detector_error": "high",
            "environment": "high",
            "application": "high",
            "route": "medium",
            "regulatory_incident": "high",
            "compliance_detectors": "medium",
            "policy_compliance": "high"
        }
        
        return effort_map.get(factor_type, "medium")
    
    def _estimate_mitigation_timeline(self, factor: Dict[str, Any]) -> str:
        """Estimate timeline for mitigation."""
        factor_type = factor["type"]
        risk_contribution = factor.get("risk_contribution", 0.0)
        
        if risk_contribution >= 0.7:
            return "immediate"
        elif risk_contribution >= 0.4:
            return "1-2 weeks"
        else:
            return "1 month"