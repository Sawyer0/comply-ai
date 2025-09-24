"""
Pattern Recognition Engine

Extracted from the monolithic template provider to provide specialized
pattern detection and analysis capabilities for security data.
"""

import logging
from typing import Any, Dict, List
from ..domain.entities import AnalysisRequest
from .interfaces import IPatternRecognitionEngine

logger = logging.getLogger(__name__)


class PatternRecognitionEngine(IPatternRecognitionEngine):
    """
    Specialized engine for detecting patterns in security data.
    
    This engine was extracted from the monolithic AnalysisTemplateProvider
    to provide focused pattern recognition capabilities including:
    - False positive pattern detection
    - Coverage gap pattern analysis
    - Incident pattern correlation
    - Statistical pattern strength assessment
    """
    
    def __init__(self):
        """Initialize the pattern recognition engine."""
        logger.info("Initialized Pattern Recognition Engine")
    
    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis.
        
        Args:
            request: Analysis request containing security data
            
        Returns:
            Pattern analysis results
        """
        patterns = await self.detect_patterns(request)
        strength = await self.classify_pattern_strength(patterns)
        
        return {
            "patterns": patterns,
            "pattern_strength": strength,
            "confidence": self.get_confidence(patterns),
            "analysis_type": "pattern_recognition"
        }
    
    def get_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for pattern analysis.
        
        Args:
            result: Pattern analysis result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not result:
            return 0.1
        
        base_confidence = 0.6
        
        # Adjust based on pattern strength
        pattern_strength = result.get("pattern_strength", "weak")
        if pattern_strength == "strong":
            base_confidence += 0.2
        elif pattern_strength == "moderate":
            base_confidence += 0.1
        
        # Adjust based on number of patterns detected
        patterns = result.get("patterns", {})
        if isinstance(patterns, dict):
            pattern_count = len(patterns.get("false_positive_patterns", []))
            if pattern_count >= 3:
                base_confidence += 0.1
        
        return min(0.9, base_confidence)
    
    async def detect_patterns(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Detect various patterns in security data.
        
        Args:
            request: Analysis request
            
        Returns:
            Detected patterns with metadata
        """
        patterns = {
            "false_positive_patterns": await self._detect_false_positive_patterns(request),
            "coverage_gap_patterns": await self._detect_coverage_gap_patterns(request),
            "incident_patterns": await self._detect_incident_patterns(request),
            "temporal_patterns": await self._detect_temporal_patterns(request)
        }
        
        return patterns
    
    async def classify_pattern_strength(self, patterns: Dict[str, Any]) -> str:
        """
        Classify the overall strength of detected patterns.
        
        Args:
            patterns: Detected patterns
            
        Returns:
            Pattern strength classification
        """
        if not patterns:
            return "weak"
        
        # Analyze false positive patterns
        fp_patterns = patterns.get("false_positive_patterns", [])
        fp_strength = self._classify_fp_pattern_strength(fp_patterns)
        
        # Analyze coverage gap patterns
        gap_patterns = patterns.get("coverage_gap_patterns", [])
        gap_strength = self._classify_gap_pattern_strength(gap_patterns)
        
        # Analyze incident patterns
        incident_patterns = patterns.get("incident_patterns", [])
        incident_strength = self._classify_incident_pattern_strength(incident_patterns)
        
        # Determine overall strength
        strengths = [fp_strength, gap_strength, incident_strength]
        strength_scores = {"strong": 3, "moderate": 2, "weak": 1}
        avg_score = sum(strength_scores[s] for s in strengths) / len(strengths)
        
        if avg_score >= 2.5:
            return "strong"
        elif avg_score >= 1.5:
            return "moderate"
        else:
            return "weak"
    
    async def _detect_false_positive_patterns(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Detect false positive patterns in detector data."""
        patterns = []
        
        for band in request.false_positive_bands:
            detector = band.get("detector", "unknown")
            fp_rate = band.get("false_positive_rate", 0.0)
            current_threshold = band.get("current_threshold", 0.5)
            
            if fp_rate > 0.05:  # 5% threshold for pattern detection
                pattern_strength = self._calculate_fp_pattern_strength(fp_rate)
                recommended_threshold = self._calculate_recommended_threshold(
                    current_threshold, fp_rate
                )
                expected_reduction = self._calculate_expected_reduction(fp_rate, pattern_strength)
                
                patterns.append({
                    "detector": detector,
                    "type": "false_positive",
                    "current_fp_rate": fp_rate,
                    "current_threshold": current_threshold,
                    "pattern_strength": pattern_strength,
                    "recommended_threshold": recommended_threshold,
                    "expected_reduction": expected_reduction,
                    "confidence": self._calculate_fp_confidence(fp_rate, pattern_strength)
                })
        
        return patterns
    
    async def _detect_coverage_gap_patterns(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Detect coverage gap patterns."""
        patterns = []
        
        for detector in request.required_detectors:
            observed = request.observed_coverage.get(detector, 0.0)
            required = request.required_coverage.get(detector, 0.0)
            
            if observed < required:
                gap_severity = (required - observed) / required
                if gap_severity > 0.1:  # 10% threshold for pattern detection
                    patterns.append({
                        "detector": detector,
                        "type": "coverage_gap",
                        "observed_coverage": observed,
                        "required_coverage": required,
                        "gap_severity": gap_severity,
                        "pattern_strength": self._classify_gap_severity(gap_severity),
                        "business_impact": self._assess_business_impact(detector, gap_severity),
                        "confidence": self._calculate_gap_confidence(gap_severity)
                    })
        
        return patterns
    
    async def _detect_incident_patterns(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Detect incident patterns."""
        patterns = []
        detector_incidents = {}
        
        # Group incidents by detector
        for hit in request.high_sev_hits:
            detector = hit.get("detector", "unknown")
            if detector not in detector_incidents:
                detector_incidents[detector] = []
            detector_incidents[detector].append(hit)
        
        # Analyze patterns per detector
        for detector, incidents in detector_incidents.items():
            if len(incidents) >= 2:  # Pattern requires multiple incidents
                severity_distribution = self._analyze_severity_distribution(incidents)
                incident_types = [hit.get("type", "unknown") for hit in incidents]
                
                patterns.append({
                    "detector": detector,
                    "type": "incident_cluster",
                    "incident_count": len(incidents),
                    "severity_distribution": severity_distribution,
                    "incident_types": incident_types,
                    "pattern_strength": self._classify_incident_cluster_strength(incidents),
                    "requires_escalation": self._requires_escalation(incidents),
                    "confidence": self._calculate_incident_confidence(incidents)
                })
        
        return patterns
    
    async def _detect_temporal_patterns(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Detect temporal patterns in the data."""
        patterns = []
        
        # Analyze temporal distribution of incidents
        if request.high_sev_hits:
            temporal_analysis = self._analyze_temporal_distribution(request.high_sev_hits)
            if temporal_analysis["has_pattern"]:
                patterns.append({
                    "type": "temporal_incident_pattern",
                    "pattern_type": temporal_analysis["pattern_type"],
                    "time_distribution": temporal_analysis["distribution"],
                    "pattern_strength": temporal_analysis["strength"],
                    "confidence": temporal_analysis["confidence"]
                })
        
        return patterns
    
    def _calculate_fp_pattern_strength(self, fp_rate: float) -> str:
        """Calculate false positive pattern strength."""
        if fp_rate > 0.2:
            return "strong"
        elif fp_rate > 0.1:
            return "moderate"
        else:
            return "weak"
    
    def _calculate_recommended_threshold(self, current: float, fp_rate: float) -> float:
        """Calculate recommended threshold adjustment."""
        if fp_rate > 0.2:
            return min(current + 0.2, 0.9)
        elif fp_rate > 0.1:
            return min(current + 0.1, 0.8)
        else:
            return min(current + 0.05, 0.7)
    
    def _calculate_expected_reduction(self, fp_rate: float, strength: str) -> float:
        """Calculate expected false positive reduction."""
        base_reduction = fp_rate * 0.5  # Conservative 50% reduction
        
        if strength == "strong":
            return min(fp_rate * 0.7, 0.8)
        elif strength == "moderate":
            return min(fp_rate * 0.5, 0.6)
        else:
            return min(fp_rate * 0.3, 0.4)
    
    def _calculate_fp_confidence(self, fp_rate: float, strength: str) -> float:
        """Calculate confidence for false positive pattern."""
        base_confidence = 0.6
        
        if strength == "strong":
            base_confidence += 0.2
        elif strength == "moderate":
            base_confidence += 0.1
        
        # Higher confidence for higher false positive rates (clearer signal)
        if fp_rate > 0.15:
            base_confidence += 0.1
        
        return min(0.9, base_confidence)
    
    def _classify_gap_severity(self, gap_severity: float) -> str:
        """Classify coverage gap severity."""
        if gap_severity >= 0.7:
            return "critical"
        elif gap_severity >= 0.4:
            return "high"
        elif gap_severity >= 0.2:
            return "medium"
        else:
            return "low"
    
    def _assess_business_impact(self, detector: str, gap_severity: float) -> str:
        """Assess business impact of coverage gap."""
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
    
    def _calculate_gap_confidence(self, gap_severity: float) -> float:
        """Calculate confidence for coverage gap pattern."""
        base_confidence = 0.7
        
        # Higher confidence for larger gaps (clearer signal)
        if gap_severity >= 0.5:
            base_confidence += 0.1
        
        return min(0.9, base_confidence)
    
    def _analyze_severity_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze severity distribution of incidents."""
        distribution = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for incident in incidents:
            severity = incident.get("severity", "medium")
            distribution[severity] = distribution.get(severity, 0) + 1
        
        return distribution
    
    def _classify_incident_cluster_strength(self, incidents: List[Dict[str, Any]]) -> str:
        """Classify strength of incident cluster pattern."""
        incident_count = len(incidents)
        critical_count = len([i for i in incidents if i.get("severity") == "critical"])
        
        if critical_count >= 2 or incident_count >= 5:
            return "strong"
        elif critical_count >= 1 or incident_count >= 3:
            return "moderate"
        else:
            return "weak"
    
    def _requires_escalation(self, incidents: List[Dict[str, Any]]) -> bool:
        """Determine if incident cluster requires escalation."""
        critical_count = len([i for i in incidents if i.get("severity") == "critical"])
        high_count = len([i for i in incidents if i.get("severity") == "high"])
        
        return critical_count > 0 or high_count >= 3
    
    def _calculate_incident_confidence(self, incidents: List[Dict[str, Any]]) -> float:
        """Calculate confidence for incident pattern."""
        base_confidence = 0.6
        
        # Higher confidence for more incidents
        if len(incidents) >= 5:
            base_confidence += 0.2
        elif len(incidents) >= 3:
            base_confidence += 0.1
        
        # Higher confidence for critical incidents
        critical_count = len([i for i in incidents if i.get("severity") == "critical"])
        if critical_count > 0:
            base_confidence += 0.1
        
        return min(0.9, base_confidence)
    
    def _analyze_temporal_distribution(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal distribution of incidents."""
        # Simplified temporal analysis - in a real implementation,
        # this would analyze timestamps and detect patterns
        incident_count = len(incidents)
        
        if incident_count >= 5:
            return {
                "has_pattern": True,
                "pattern_type": "burst",
                "distribution": {"concentrated": True},
                "strength": "strong",
                "confidence": 0.8
            }
        elif incident_count >= 3:
            return {
                "has_pattern": True,
                "pattern_type": "cluster",
                "distribution": {"moderate_concentration": True},
                "strength": "moderate",
                "confidence": 0.6
            }
        else:
            return {
                "has_pattern": False,
                "pattern_type": "scattered",
                "distribution": {"random": True},
                "strength": "weak",
                "confidence": 0.3
            }
    
    def _classify_fp_pattern_strength(self, patterns: List[Dict[str, Any]]) -> str:
        """Classify strength of false positive patterns."""
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
    
    def _classify_gap_pattern_strength(self, patterns: List[Dict[str, Any]]) -> str:
        """Classify strength of coverage gap patterns."""
        if not patterns:
            return "weak"
        
        critical_gaps = len([p for p in patterns if p.get("business_impact") == "critical"])
        high_gaps = len([p for p in patterns if p.get("business_impact") == "high"])
        
        if critical_gaps >= 2:
            return "strong"
        elif critical_gaps >= 1 or high_gaps >= 3:
            return "moderate"
        else:
            return "weak"
    
    def _classify_incident_pattern_strength(self, patterns: List[Dict[str, Any]]) -> str:
        """Classify strength of incident patterns."""
        if not patterns:
            return "weak"
        
        escalation_patterns = len([p for p in patterns if p.get("requires_escalation", False)])
        strong_patterns = len([p for p in patterns if p.get("pattern_strength") == "strong"])
        
        if escalation_patterns >= 1 or strong_patterns >= 2:
            return "strong"
        elif strong_patterns >= 1:
            return "moderate"
        else:
            return "weak"