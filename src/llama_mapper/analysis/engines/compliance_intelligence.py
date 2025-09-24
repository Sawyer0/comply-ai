"""
Compliance Intelligence Engine

Extracted from the monolithic template provider to provide specialized
compliance mapping and regulatory framework intelligence.
"""

import logging
from typing import Any, Dict, List
from ..domain.entities import AnalysisRequest
from .interfaces import IComplianceIntelligence

logger = logging.getLogger(__name__)


class ComplianceIntelligence(IComplianceIntelligence):
    """
    Specialized engine for compliance mapping and regulatory intelligence.
    
    This engine was extracted from the monolithic AnalysisTemplateProvider
    to provide focused compliance capabilities including:
    - Regulatory framework mapping (SOC 2, ISO 27001, HIPAA)
    - Compliance gap analysis
    - Policy generation for compliance enforcement
    - Risk assessment with regulatory context
    """
    
    def __init__(self):
        """Initialize the compliance intelligence engine."""
        self.framework_mappings = self._initialize_framework_mappings()
        self.compliance_detectors = self._initialize_compliance_detectors()
        logger.info("Initialized Compliance Intelligence Engine")
    
    async def analyze(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Perform comprehensive compliance analysis.
        
        Args:
            request: Analysis request containing security data
            
        Returns:
            Compliance analysis results
        """
        framework_mappings = await self.map_to_frameworks(request)
        compliance_policy = await self.generate_compliance_policy(framework_mappings)
        
        return {
            "framework_mappings": framework_mappings,
            "compliance_policy": compliance_policy,
            "confidence": self.get_confidence(framework_mappings),
            "analysis_type": "compliance_intelligence"
        }
    
    def get_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for compliance analysis.
        
        Args:
            result: Compliance analysis result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not result:
            return 0.1
        
        base_confidence = 0.8  # High confidence for rule-based compliance mapping
        
        # Adjust based on framework coverage
        frameworks = result.get("frameworks", {})
        if len(frameworks) >= 2:
            base_confidence += 0.1
        
        # Adjust based on compliance gaps
        gaps = result.get("compliance_gaps", [])
        if len(gaps) == 0:
            base_confidence += 0.1
        elif len(gaps) > 3:
            base_confidence -= 0.1
        
        return min(0.9, max(0.1, base_confidence))
    
    async def map_to_frameworks(self, request: AnalysisRequest) -> Dict[str, Any]:
        """
        Map security findings to regulatory frameworks.
        
        Args:
            request: Analysis request
            
        Returns:
            Framework mappings and compliance status
        """
        frameworks = {}
        
        # Map to SOC 2
        soc2_mapping = await self._map_to_soc2(request)
        frameworks["soc2"] = soc2_mapping
        
        # Map to ISO 27001
        iso27001_mapping = await self._map_to_iso27001(request)
        frameworks["iso27001"] = iso27001_mapping
        
        # Map to HIPAA (if applicable)
        hipaa_mapping = await self._map_to_hipaa(request)
        if hipaa_mapping["applicable"]:
            frameworks["hipaa"] = hipaa_mapping
        
        # Identify compliance gaps
        compliance_gaps = self._identify_compliance_gaps(frameworks, request)
        
        # Calculate overall compliance score
        compliance_score = self._calculate_compliance_score(frameworks)
        
        return {
            "frameworks": frameworks,
            "compliance_gaps": compliance_gaps,
            "compliance_score": compliance_score,
            "recommendations": self._generate_compliance_recommendations(frameworks, compliance_gaps)
        }
    
    async def generate_compliance_policy(self, mappings: Dict[str, Any]) -> str:
        """
        Generate OPA policy for compliance enforcement.
        
        Args:
            mappings: Compliance mappings
            
        Returns:
            OPA policy string
        """
        policy_parts = [
            "package compliance_enforcement",
            "",
            "# Advanced compliance enforcement policy",
            "# Generated from framework mappings and gap analysis",
            ""
        ]
        
        frameworks = mappings.get("frameworks", {})
        
        # Generate framework-specific policies
        if "soc2" in frameworks:
            policy_parts.extend(self._generate_soc2_policy(frameworks["soc2"]))
        
        if "iso27001" in frameworks:
            policy_parts.extend(self._generate_iso27001_policy(frameworks["iso27001"]))
        
        if "hipaa" in frameworks:
            policy_parts.extend(self._generate_hipaa_policy(frameworks["hipaa"]))
        
        # Generate gap enforcement policies
        compliance_gaps = mappings.get("compliance_gaps", [])
        if compliance_gaps:
            policy_parts.extend(self._generate_gap_enforcement_policy(compliance_gaps))
        
        return "\n".join(policy_parts)
    
    async def _map_to_soc2(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Map findings to SOC 2 controls."""
        soc2_controls = {
            "CC6.1": {"name": "Logical Access", "status": "compliant", "findings": []},
            "CC6.7": {"name": "Data Transmission", "status": "compliant", "findings": []},
            "CC7.1": {"name": "System Boundaries", "status": "compliant", "findings": []}
        }
        
        # Check coverage gaps against SOC 2 requirements
        for detector in request.required_detectors:
            if detector in self.compliance_detectors["soc2"]:
                observed = request.observed_coverage.get(detector, 0.0)
                required = request.required_coverage.get(detector, 0.0)
                
                if observed < required:
                    # Map to relevant controls
                    relevant_controls = self.compliance_detectors["soc2"][detector]
                    for control in relevant_controls:
                        if control in soc2_controls:
                            soc2_controls[control]["status"] = "non_compliant"
                            soc2_controls[control]["findings"].append({
                                "detector": detector,
                                "gap": required - observed,
                                "severity": "high" if (required - observed) > 0.3 else "medium"
                            })
        
        # Check high severity incidents
        for incident in request.high_sev_hits:
            incident_type = incident.get("type", "unknown")
            if incident_type in ["pii_exposure", "data_breach", "access_violation"]:
                # These typically map to CC6.1 (Logical Access)
                soc2_controls["CC6.1"]["status"] = "non_compliant"
                soc2_controls["CC6.1"]["findings"].append({
                    "incident_type": incident_type,
                    "severity": incident.get("severity", "medium"),
                    "detector": incident.get("detector", "unknown")
                })
        
        # Calculate compliance percentage
        compliant_controls = len([c for c in soc2_controls.values() if c["status"] == "compliant"])
        compliance_percentage = (compliant_controls / len(soc2_controls)) * 100
        
        return {
            "framework": "SOC 2 Type II",
            "controls": soc2_controls,
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage == 100 else "non_compliant",
            "priority_actions": self._generate_soc2_actions(soc2_controls)
        }
    
    async def _map_to_iso27001(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Map findings to ISO 27001 controls."""
        iso_controls = {
            "A.8.2.1": {"name": "Data Classification", "status": "compliant", "findings": []},
            "A.8.2.2": {"name": "Data Labeling", "status": "compliant", "findings": []},
            "A.13.2.1": {"name": "Information Transfer", "status": "compliant", "findings": []}
        }
        
        # Check coverage gaps against ISO 27001 requirements
        for detector in request.required_detectors:
            if detector in self.compliance_detectors["iso27001"]:
                observed = request.observed_coverage.get(detector, 0.0)
                required = request.required_coverage.get(detector, 0.0)
                
                if observed < required:
                    relevant_controls = self.compliance_detectors["iso27001"][detector]
                    for control in relevant_controls:
                        if control in iso_controls:
                            iso_controls[control]["status"] = "non_compliant"
                            iso_controls[control]["findings"].append({
                                "detector": detector,
                                "gap": required - observed,
                                "severity": "high" if (required - observed) > 0.3 else "medium"
                            })
        
        # Check incidents against ISO controls
        for incident in request.high_sev_hits:
            incident_type = incident.get("type", "unknown")
            if incident_type in ["data_classification_error", "labeling_violation"]:
                iso_controls["A.8.2.1"]["status"] = "non_compliant"
                iso_controls["A.8.2.1"]["findings"].append({
                    "incident_type": incident_type,
                    "severity": incident.get("severity", "medium"),
                    "detector": incident.get("detector", "unknown")
                })
        
        compliant_controls = len([c for c in iso_controls.values() if c["status"] == "compliant"])
        compliance_percentage = (compliant_controls / len(iso_controls)) * 100
        
        return {
            "framework": "ISO 27001:2022",
            "controls": iso_controls,
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage == 100 else "non_compliant",
            "priority_actions": self._generate_iso27001_actions(iso_controls)
        }
    
    async def _map_to_hipaa(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Map findings to HIPAA requirements."""
        # Check if HIPAA is applicable
        hipaa_detectors = ["hipaa-validator", "phi-detector", "healthcare-scanner"]
        is_applicable = any(detector in request.required_detectors for detector in hipaa_detectors)
        
        if not is_applicable:
            return {"applicable": False}
        
        hipaa_safeguards = {
            "administrative": {"status": "compliant", "findings": []},
            "physical": {"status": "compliant", "findings": []},
            "technical": {"status": "compliant", "findings": []}
        }
        
        # Check for PHI-related incidents
        for incident in request.high_sev_hits:
            incident_type = incident.get("type", "unknown")
            if incident_type in ["phi_exposure", "healthcare_breach", "patient_data_leak"]:
                # PHI incidents typically affect technical safeguards
                hipaa_safeguards["technical"]["status"] = "non_compliant"
                hipaa_safeguards["technical"]["findings"].append({
                    "incident_type": incident_type,
                    "severity": incident.get("severity", "medium"),
                    "detector": incident.get("detector", "unknown")
                })
        
        # Check HIPAA detector coverage
        for detector in hipaa_detectors:
            if detector in request.required_detectors:
                observed = request.observed_coverage.get(detector, 0.0)
                required = request.required_coverage.get(detector, 0.0)
                
                if observed < required:
                    hipaa_safeguards["technical"]["status"] = "non_compliant"
                    hipaa_safeguards["technical"]["findings"].append({
                        "detector": detector,
                        "gap": required - observed,
                        "severity": "critical"  # HIPAA gaps are always critical
                    })
        
        compliant_safeguards = len([s for s in hipaa_safeguards.values() if s["status"] == "compliant"])
        compliance_percentage = (compliant_safeguards / len(hipaa_safeguards)) * 100
        
        return {
            "applicable": True,
            "framework": "HIPAA",
            "safeguards": hipaa_safeguards,
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage == 100 else "non_compliant",
            "priority_actions": self._generate_hipaa_actions(hipaa_safeguards)
        }
    
    def _identify_compliance_gaps(self, frameworks: Dict[str, Any], request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Identify compliance gaps across frameworks."""
        gaps = []
        
        for framework_name, framework_data in frameworks.items():
            if framework_name == "hipaa" and not framework_data.get("applicable", True):
                continue
            
            if framework_data.get("status") == "non_compliant":
                # Identify specific gaps
                if "controls" in framework_data:
                    for control_id, control_data in framework_data["controls"].items():
                        if control_data["status"] == "non_compliant":
                            gaps.append({
                                "framework": framework_name,
                                "control": control_id,
                                "control_name": control_data["name"],
                                "findings_count": len(control_data["findings"]),
                                "severity": self._assess_gap_severity(control_data["findings"]),
                                "remediation_effort": self._estimate_remediation_effort(control_data["findings"])
                            })
                elif "safeguards" in framework_data:
                    for safeguard_name, safeguard_data in framework_data["safeguards"].items():
                        if safeguard_data["status"] == "non_compliant":
                            gaps.append({
                                "framework": framework_name,
                                "safeguard": safeguard_name,
                                "findings_count": len(safeguard_data["findings"]),
                                "severity": "critical",  # HIPAA gaps are always critical
                                "remediation_effort": "high"
                            })
        
        return gaps
    
    def _calculate_compliance_score(self, frameworks: Dict[str, Any]) -> float:
        """Calculate overall compliance score."""
        total_score = 0.0
        framework_count = 0
        
        for framework_name, framework_data in frameworks.items():
            if framework_name == "hipaa" and not framework_data.get("applicable", True):
                continue
            
            compliance_percentage = framework_data.get("compliance_percentage", 0.0)
            total_score += compliance_percentage
            framework_count += 1
        
        return total_score / framework_count if framework_count > 0 else 0.0
    
    def _generate_compliance_recommendations(self, frameworks: Dict[str, Any], gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate compliance recommendations."""
        recommendations = []
        
        # Sort gaps by severity and effort
        sorted_gaps = sorted(gaps, key=lambda x: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}[x["severity"]],
            {"low": 1, "medium": 2, "high": 3}[x["remediation_effort"]]
        ), reverse=True)
        
        for i, gap in enumerate(sorted_gaps[:5]):  # Top 5 recommendations
            recommendations.append({
                "priority": i + 1,
                "framework": gap["framework"],
                "control": gap.get("control", gap.get("safeguard", "unknown")),
                "action": self._generate_remediation_action(gap),
                "effort": gap["remediation_effort"],
                "timeline": self._estimate_remediation_timeline(gap),
                "business_impact": self._assess_business_impact(gap)
            })
        
        return recommendations
    
    def _generate_soc2_policy(self, soc2_data: Dict[str, Any]) -> List[str]:
        """Generate SOC 2 specific OPA policy."""
        policy_lines = [
            "# SOC 2 Type II Compliance Enforcement",
            ""
        ]
        
        for control_id, control_data in soc2_data["controls"].items():
            if control_data["status"] == "non_compliant":
                policy_lines.extend([
                    f"# {control_id}: {control_data['name']}",
                    f"soc2_{control_id.lower().replace('.', '_')}_violation[msg] {{",
                    f"    # Control {control_id} non-compliance detected",
                    f"    input.framework == \"soc2\"",
                    f"    input.control == \"{control_id}\"",
                    f"    count(input.findings) > 0",
                    f'    msg := "SOC 2 {control_id} ({control_data["name"]}) compliance violation detected"',
                    "}",
                    ""
                ])
        
        return policy_lines
    
    def _generate_iso27001_policy(self, iso_data: Dict[str, Any]) -> List[str]:
        """Generate ISO 27001 specific OPA policy."""
        policy_lines = [
            "# ISO 27001:2022 Compliance Enforcement",
            ""
        ]
        
        for control_id, control_data in iso_data["controls"].items():
            if control_data["status"] == "non_compliant":
                policy_lines.extend([
                    f"# {control_id}: {control_data['name']}",
                    f"iso27001_{control_id.lower().replace('.', '_')}_violation[msg] {{",
                    f"    input.framework == \"iso27001\"",
                    f"    input.control == \"{control_id}\"",
                    f"    count(input.findings) > 0",
                    f'    msg := "ISO 27001 {control_id} ({control_data["name"]}) compliance violation detected"',
                    "}",
                    ""
                ])
        
        return policy_lines
    
    def _generate_hipaa_policy(self, hipaa_data: Dict[str, Any]) -> List[str]:
        """Generate HIPAA specific OPA policy."""
        policy_lines = [
            "# HIPAA Compliance Enforcement",
            ""
        ]
        
        for safeguard_name, safeguard_data in hipaa_data["safeguards"].items():
            if safeguard_data["status"] == "non_compliant":
                policy_lines.extend([
                    f"# HIPAA {safeguard_name.title()} Safeguards",
                    f"hipaa_{safeguard_name}_violation[msg] {{",
                    f"    input.framework == \"hipaa\"",
                    f"    input.safeguard == \"{safeguard_name}\"",
                    f"    count(input.findings) > 0",
                    f'    msg := "HIPAA {safeguard_name} safeguard violation - immediate remediation required"',
                    "}",
                    ""
                ])
        
        return policy_lines
    
    def _generate_gap_enforcement_policy(self, gaps: List[Dict[str, Any]]) -> List[str]:
        """Generate policy for compliance gap enforcement."""
        policy_lines = [
            "# Compliance Gap Enforcement",
            "",
            "compliance_gap_violation[msg] {",
            "    gap := input.compliance_gaps[_]",
            "    gap.severity in [\"critical\", \"high\"]",
            '    msg := sprintf("Critical compliance gap in %v framework: %v", [gap.framework, gap.control])',
            "}",
            ""
        ]
        
        return policy_lines
    
    def _initialize_framework_mappings(self) -> Dict[str, Any]:
        """Initialize framework mappings."""
        return {
            "soc2": {
                "CC6.1": ["presidio", "pii-detector", "access-control"],
                "CC6.7": ["encryption-detector", "tls-validator"],
                "CC7.1": ["boundary-scanner", "network-monitor"]
            },
            "iso27001": {
                "A.8.2.1": ["data-classifier", "pii-detector"],
                "A.8.2.2": ["labeling-validator", "tag-scanner"],
                "A.13.2.1": ["transfer-monitor", "encryption-detector"]
            },
            "hipaa": {
                "technical": ["hipaa-validator", "phi-detector", "encryption-detector"],
                "administrative": ["access-control", "audit-logger"],
                "physical": ["physical-access-monitor"]
            }
        }
    
    def _initialize_compliance_detectors(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize compliance detector mappings."""
        return {
            "soc2": {
                "presidio": ["CC6.1"],
                "pii-detector": ["CC6.1"],
                "access-control": ["CC6.1"],
                "encryption-detector": ["CC6.7"],
                "tls-validator": ["CC6.7"],
                "boundary-scanner": ["CC7.1"],
                "network-monitor": ["CC7.1"]
            },
            "iso27001": {
                "data-classifier": ["A.8.2.1"],
                "pii-detector": ["A.8.2.1"],
                "labeling-validator": ["A.8.2.2"],
                "tag-scanner": ["A.8.2.2"],
                "transfer-monitor": ["A.13.2.1"],
                "encryption-detector": ["A.13.2.1"]
            }
        }
    
    def _generate_soc2_actions(self, controls: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate SOC 2 specific actions."""
        actions = []
        for control_id, control_data in controls.items():
            if control_data["status"] == "non_compliant":
                actions.append({
                    "control": control_id,
                    "action": f"Remediate {control_data['name']} control violations",
                    "findings_count": len(control_data["findings"]),
                    "priority": "high" if len(control_data["findings"]) > 2 else "medium"
                })
        return actions
    
    def _generate_iso27001_actions(self, controls: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ISO 27001 specific actions."""
        actions = []
        for control_id, control_data in controls.items():
            if control_data["status"] == "non_compliant":
                actions.append({
                    "control": control_id,
                    "action": f"Address {control_data['name']} control gaps",
                    "findings_count": len(control_data["findings"]),
                    "priority": "high" if len(control_data["findings"]) > 2 else "medium"
                })
        return actions
    
    def _generate_hipaa_actions(self, safeguards: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate HIPAA specific actions."""
        actions = []
        for safeguard_name, safeguard_data in safeguards.items():
            if safeguard_data["status"] == "non_compliant":
                actions.append({
                    "safeguard": safeguard_name,
                    "action": f"Implement {safeguard_name} safeguard controls",
                    "findings_count": len(safeguard_data["findings"]),
                    "priority": "critical"  # HIPAA is always critical
                })
        return actions
    
    def _assess_gap_severity(self, findings: List[Dict[str, Any]]) -> str:
        """Assess severity of compliance gap."""
        if not findings:
            return "low"
        
        critical_count = len([f for f in findings if f.get("severity") == "critical"])
        high_count = len([f for f in findings if f.get("severity") == "high"])
        
        if critical_count > 0:
            return "critical"
        elif high_count > 0:
            return "high"
        elif len(findings) > 2:
            return "medium"
        else:
            return "low"
    
    def _estimate_remediation_effort(self, findings: List[Dict[str, Any]]) -> str:
        """Estimate effort required for remediation."""
        if not findings:
            return "low"
        
        # More findings = more effort
        if len(findings) >= 5:
            return "high"
        elif len(findings) >= 2:
            return "medium"
        else:
            return "low"
    
    def _generate_remediation_action(self, gap: Dict[str, Any]) -> str:
        """Generate specific remediation action."""
        framework = gap["framework"]
        control = gap.get("control", gap.get("safeguard", "unknown"))
        
        if framework == "soc2":
            return f"Implement SOC 2 {control} control requirements"
        elif framework == "iso27001":
            return f"Address ISO 27001 {control} control gaps"
        elif framework == "hipaa":
            return f"Implement HIPAA {control} safeguard controls"
        else:
            return f"Address {framework} compliance gap in {control}"
    
    def _estimate_remediation_timeline(self, gap: Dict[str, Any]) -> str:
        """Estimate timeline for remediation."""
        severity = gap["severity"]
        effort = gap["remediation_effort"]
        
        if severity == "critical":
            return "immediate"
        elif severity == "high" and effort == "low":
            return "1 week"
        elif severity == "high":
            return "2-4 weeks"
        elif effort == "high":
            return "1-2 months"
        else:
            return "2-4 weeks"
    
    def _assess_business_impact(self, gap: Dict[str, Any]) -> str:
        """Assess business impact of compliance gap."""
        framework = gap["framework"]
        severity = gap["severity"]
        
        if framework == "hipaa":
            return "critical"  # HIPAA violations have severe penalties
        elif severity == "critical":
            return "high"
        elif severity == "high":
            return "medium"
        else:
            return "low"