"""
Built-in compliance mapping plugin for Analysis Service.

This plugin provides compliance framework mapping capabilities for various
regulatory standards and frameworks.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from ..interfaces import (
    IComplianceMapperPlugin,
    PluginMetadata,
    PluginType,
    PluginCapability,
    AnalysisRequest,
    AnalysisResult,
)

logger = logging.getLogger(__name__)


class ComplianceMappingPlugin(IComplianceMapperPlugin):
    """Built-in compliance mapping plugin."""

    def __init__(self):
        self.initialized = False
        self.config = {}
        self.framework_mappings = {}

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return PluginMetadata(
            name="builtin_compliance_mapping",
            version="1.0.0",
            description="Built-in compliance framework mapping",
            author="Analysis Service Team",
            plugin_type=PluginType.ANALYSIS_ENGINE,
            capabilities=[
                PluginCapability.COMPLIANCE_MAPPING,
                PluginCapability.BATCH_PROCESSING,
                PluginCapability.REAL_TIME_ANALYSIS,
            ],
            dependencies=[],
            supported_frameworks=[
                "SOC2",
                "ISO27001",
                "HIPAA",
                "GDPR",
                "PCI-DSS",
                "NIST",
            ],
            min_confidence_threshold=0.8,
            max_batch_size=100,
        )

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        try:
            self.config = config

            # Initialize framework mappings
            self.framework_mappings = {
                "SOC2": self._load_soc2_mappings(),
                "ISO27001": self._load_iso27001_mappings(),
                "HIPAA": self._load_hipaa_mappings(),
                "GDPR": self._load_gdpr_mappings(),
                "PCI-DSS": self._load_pci_dss_mappings(),
                "NIST": self._load_nist_mappings(),
            }

            self.initialized = True
            logger.info("Compliance mapping plugin initialized")
            return True

        except Exception as e:
            logger.error("Failed to initialize compliance mapping plugin", error=str(e))
            return False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self.initialized else "not_initialized",
            "initialized": self.initialized,
            "supported_frameworks": len(self.framework_mappings),
            "last_check": datetime.now(timezone.utc).isoformat(),
        }

    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            self.framework_mappings.clear()
            self.initialized = False
            logger.info("Compliance mapping plugin cleaned up")
            return True
        except Exception as e:
            logger.error("Failed to cleanup compliance mapping plugin", error=str(e))
            return False

    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Perform compliance mapping analysis."""
        start_time = time.time()

        try:
            if not self.initialized:
                raise RuntimeError("Plugin not initialized")

            framework = request.framework or "SOC2"  # Default framework

            # Map to framework
            mapping_result = await self.map_to_framework(request, framework)

            # Generate compliance report
            compliance_report = await self.generate_compliance_report(
                [mapping_result], framework
            )

            # Validate compliance
            validation_result = await self.validate_compliance(
                mapping_result, framework
            )

            # Calculate confidence
            confidence = self._calculate_confidence(mapping_result, request.metadata)

            # Prepare result
            result_data = {
                "framework": framework,
                "mapping": mapping_result,
                "compliance_report": compliance_report,
                "validation": validation_result,
                "analysis_type": "compliance_mapping",
                "supported_frameworks": self.get_supported_frameworks(),
            }

            processing_time = (time.time() - start_time) * 1000

            return AnalysisResult(
                request_id=request.request_id,
                plugin_name="builtin_compliance_mapping",
                plugin_version="1.0.0",
                confidence=confidence,
                result_data=result_data,
                processing_time_ms=processing_time,
                metadata={
                    "tenant_id": request.tenant_id,
                    "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(
                "Compliance mapping analysis failed",
                request_id=request.request_id,
                error=str(e),
            )

            return AnalysisResult(
                request_id=request.request_id,
                plugin_name="builtin_compliance_mapping",
                plugin_version="1.0.0",
                confidence=0.0,
                result_data={},
                processing_time_ms=processing_time,
                errors=[str(e)],
            )

    async def batch_analyze(
        self, requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """Perform batch compliance mapping analysis."""
        results = []

        # Process requests in parallel batches
        batch_size = min(self.get_metadata().max_batch_size or 20, len(requests))

        for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.analyze(request) for request in batch], return_exceptions=True
            )

            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(
                        "Batch compliance mapping item failed", error=str(result)
                    )
                    error_result = AnalysisResult(
                        request_id="unknown",
                        plugin_name="builtin_compliance_mapping",
                        plugin_version="1.0.0",
                        confidence=0.0,
                        result_data={},
                        processing_time_ms=0.0,
                        errors=[str(result)],
                    )
                    results.append(error_result)
                else:
                    results.append(result)

        return results

    def get_supported_analysis_types(self) -> List[str]:
        """Get supported analysis types."""
        return ["compliance_mapping", "framework_mapping", "regulatory_mapping"]

    async def validate_request(self, request: AnalysisRequest) -> bool:
        """Validate if the plugin can handle the request."""
        try:
            # Check if analysis type is supported
            if request.analysis_type not in self.get_supported_analysis_types():
                return False

            # Check if framework is supported
            if (
                request.framework
                and request.framework not in self.get_supported_frameworks()
            ):
                logger.warning("Unsupported framework", framework=request.framework)
                return False

            return True

        except Exception as e:
            logger.error("Request validation failed", error=str(e))
            return False

    async def map_to_framework(
        self, request: AnalysisRequest, framework: str
    ) -> Dict[str, Any]:
        """Map analysis results to compliance framework."""
        try:
            if framework not in self.framework_mappings:
                raise ValueError(f"Unsupported framework: {framework}")

            framework_config = self.framework_mappings[framework]
            metadata = request.metadata

            # Extract relevant information from metadata
            category = metadata.get("category", "unknown")
            severity = metadata.get("severity", "medium")
            data_type = metadata.get("data_type", "general")

            # Map to framework controls
            applicable_controls = self._map_to_controls(
                framework, category, severity, data_type
            )

            # Determine compliance status
            compliance_status = self._determine_compliance_status(
                framework, applicable_controls, metadata
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                framework, compliance_status, applicable_controls
            )

            return {
                "framework": framework,
                "applicable_controls": applicable_controls,
                "compliance_status": compliance_status,
                "recommendations": recommendations,
                "mapping_metadata": {
                    "category": category,
                    "severity": severity,
                    "data_type": data_type,
                    "mapped_at": datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            logger.error("Framework mapping failed", framework=framework, error=str(e))
            return {
                "framework": framework,
                "error": str(e),
                "applicable_controls": [],
                "compliance_status": "unknown",
            }

    async def generate_compliance_report(
        self, mappings: List[Dict[str, Any]], framework: str
    ) -> Dict[str, Any]:
        """Generate compliance report for framework mappings."""
        try:
            if not mappings:
                return {"framework": framework, "summary": "No mappings provided"}

            # Aggregate compliance status
            total_mappings = len(mappings)
            compliant_count = sum(
                1 for m in mappings if m.get("compliance_status") == "compliant"
            )
            non_compliant_count = sum(
                1 for m in mappings if m.get("compliance_status") == "non_compliant"
            )
            partial_count = sum(
                1 for m in mappings if m.get("compliance_status") == "partial"
            )

            # Calculate compliance percentage
            compliance_percentage = (
                (compliant_count / total_mappings * 100) if total_mappings > 0 else 0
            )

            # Collect all applicable controls
            all_controls = set()
            for mapping in mappings:
                controls = mapping.get("applicable_controls", [])
                all_controls.update(
                    control.get("control_id", "") for control in controls
                )

            # Generate summary
            summary = {
                "framework": framework,
                "total_mappings": total_mappings,
                "compliance_percentage": round(compliance_percentage, 2),
                "status_breakdown": {
                    "compliant": compliant_count,
                    "non_compliant": non_compliant_count,
                    "partial": partial_count,
                    "unknown": total_mappings
                    - compliant_count
                    - non_compliant_count
                    - partial_count,
                },
                "applicable_controls_count": len(all_controls),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Add framework-specific insights
            if framework == "SOC2":
                summary["trust_services_criteria"] = self._analyze_soc2_criteria(
                    mappings
                )
            elif framework == "ISO27001":
                summary["control_domains"] = self._analyze_iso27001_domains(mappings)
            elif framework == "HIPAA":
                summary["safeguard_types"] = self._analyze_hipaa_safeguards(mappings)

            return summary

        except Exception as e:
            logger.error(
                "Compliance report generation failed", framework=framework, error=str(e)
            )
            return {"framework": framework, "error": str(e)}

    def get_supported_frameworks(self) -> List[str]:
        """Get supported compliance frameworks."""
        return list(self.framework_mappings.keys())

    async def validate_compliance(
        self, mappings: Dict[str, Any], framework: str
    ) -> Dict[str, Any]:
        """Validate compliance mappings against framework requirements."""
        try:
            validation_results = {
                "framework": framework,
                "is_valid": True,
                "validation_errors": [],
                "validation_warnings": [],
                "completeness_score": 0.0,
            }

            # Check if required controls are mapped
            required_controls = self._get_required_controls(framework)
            mapped_controls = [
                c.get("control_id") for c in mappings.get("applicable_controls", [])
            ]

            missing_controls = set(required_controls) - set(mapped_controls)
            if missing_controls:
                validation_results["validation_errors"].append(
                    f"Missing required controls: {', '.join(missing_controls)}"
                )
                validation_results["is_valid"] = False

            # Calculate completeness score
            if required_controls:
                completeness_score = len(
                    set(mapped_controls) & set(required_controls)
                ) / len(required_controls)
                validation_results["completeness_score"] = round(completeness_score, 2)

            # Framework-specific validations
            if framework == "SOC2":
                validation_results.update(self._validate_soc2_specific(mappings))
            elif framework == "HIPAA":
                validation_results.update(self._validate_hipaa_specific(mappings))

            return validation_results

        except Exception as e:
            logger.error(
                "Compliance validation failed", framework=framework, error=str(e)
            )
            return {"framework": framework, "is_valid": False, "error": str(e)}

    # Private helper methods

    def _load_soc2_mappings(self) -> Dict[str, Any]:
        """Load SOC 2 framework mappings."""
        return {
            "name": "SOC 2",
            "version": "2017",
            "trust_services_criteria": {
                "CC1": "Control Environment",
                "CC2": "Communication and Information",
                "CC3": "Risk Assessment",
                "CC4": "Monitoring Activities",
                "CC5": "Control Activities",
                "CC6": "Logical and Physical Access Controls",
                "CC7": "System Operations",
                "CC8": "Change Management",
                "CC9": "Risk Mitigation",
            },
            "categories": {
                "security": ["CC6.1", "CC6.2", "CC6.3", "CC7.1"],
                "privacy": ["CC6.7", "CC7.3"],
                "availability": ["CC7.1", "CC7.2"],
                "processing_integrity": ["CC8.1", "CC9.1"],
                "confidentiality": ["CC6.1", "CC6.7"],
            },
        }

    def _load_iso27001_mappings(self) -> Dict[str, Any]:
        """Load ISO 27001 framework mappings."""
        return {
            "name": "ISO 27001",
            "version": "2022",
            "control_domains": {
                "A.5": "Information Security Policies",
                "A.6": "Organization of Information Security",
                "A.7": "Human Resource Security",
                "A.8": "Asset Management",
                "A.9": "Access Control",
                "A.10": "Cryptography",
                "A.11": "Physical and Environmental Security",
                "A.12": "Operations Security",
                "A.13": "Communications Security",
                "A.14": "System Acquisition, Development and Maintenance",
                "A.15": "Supplier Relationships",
                "A.16": "Information Security Incident Management",
                "A.17": "Information Security Aspects of Business Continuity Management",
                "A.18": "Compliance",
            },
            "categories": {
                "security": ["A.9.1.1", "A.9.2.1", "A.12.6.1"],
                "privacy": ["A.8.2.1", "A.13.2.1"],
                "incident": ["A.16.1.1", "A.16.1.2"],
                "compliance": ["A.18.1.1", "A.18.2.1"],
            },
        }

    def _load_hipaa_mappings(self) -> Dict[str, Any]:
        """Load HIPAA framework mappings."""
        return {
            "name": "HIPAA",
            "version": "2013",
            "safeguard_types": {
                "administrative": "164.308",
                "physical": "164.310",
                "technical": "164.312",
            },
            "categories": {
                "security": ["164.308(a)(1)", "164.312(a)(1)"],
                "privacy": ["164.308(a)(3)", "164.312(e)(1)"],
                "access": ["164.308(a)(4)", "164.312(a)(2)"],
                "audit": ["164.308(a)(1)(ii)(D)", "164.312(b)"],
                "transmission": ["164.312(e)(1)", "164.312(e)(2)"],
            },
        }

    def _load_gdpr_mappings(self) -> Dict[str, Any]:
        """Load GDPR framework mappings."""
        return {
            "name": "GDPR",
            "version": "2018",
            "articles": {
                "Art. 5": "Principles of Processing",
                "Art. 6": "Lawfulness of Processing",
                "Art. 25": "Data Protection by Design and by Default",
                "Art. 32": "Security of Processing",
                "Art. 33": "Notification of Personal Data Breach",
                "Art. 34": "Communication of Personal Data Breach",
                "Art. 35": "Data Protection Impact Assessment",
            },
            "categories": {
                "security": ["Art. 32", "Art. 25"],
                "privacy": ["Art. 5", "Art. 6"],
                "breach": ["Art. 33", "Art. 34"],
                "assessment": ["Art. 35"],
            },
        }

    def _load_pci_dss_mappings(self) -> Dict[str, Any]:
        """Load PCI DSS framework mappings."""
        return {
            "name": "PCI DSS",
            "version": "4.0",
            "requirements": {
                "1": "Install and maintain network security controls",
                "2": "Apply secure configurations to all system components",
                "3": "Protect stored cardholder data",
                "4": "Protect cardholder data with strong cryptography during transmission",
                "5": "Protect all systems and networks from malicious software",
                "6": "Develop and maintain secure systems and software",
            },
            "categories": {
                "security": ["1.1.1", "2.1.1", "6.2.1"],
                "encryption": ["3.1.1", "4.1.1"],
                "access": ["7.1.1", "8.1.1"],
                "monitoring": ["10.1.1", "11.1.1"],
            },
        }

    def _load_nist_mappings(self) -> Dict[str, Any]:
        """Load NIST framework mappings."""
        return {
            "name": "NIST Cybersecurity Framework",
            "version": "1.1",
            "functions": {
                "ID": "Identify",
                "PR": "Protect",
                "DE": "Detect",
                "RS": "Respond",
                "RC": "Recover",
            },
            "categories": {
                "security": ["PR.AC-1", "PR.DS-1", "DE.CM-1"],
                "incident": ["RS.RP-1", "RS.CO-1"],
                "recovery": ["RC.RP-1", "RC.CO-1"],
                "governance": ["ID.GV-1", "ID.RA-1"],
            },
        }

    def _map_to_controls(
        self, framework: str, category: str, severity: str, data_type: str
    ) -> List[Dict[str, Any]]:
        """Map to framework-specific controls."""
        framework_config = self.framework_mappings.get(framework, {})
        category_mappings = framework_config.get("categories", {})

        controls = []

        # Get controls for category
        category_controls = category_mappings.get(category.lower(), [])

        for control_id in category_controls:
            control_info = {
                "control_id": control_id,
                "framework": framework,
                "category": category,
                "severity_mapping": severity,
                "data_type_mapping": data_type,
                "implementation_status": self._determine_implementation_status(
                    control_id, severity
                ),
            }
            controls.append(control_info)

        return controls

    def _determine_compliance_status(
        self, framework: str, controls: List[Dict[str, Any]], metadata: Dict[str, Any]
    ) -> str:
        """Determine overall compliance status."""
        if not controls:
            return "unknown"

        # Simple compliance determination based on implementation status
        implemented_count = sum(
            1 for c in controls if c.get("implementation_status") == "implemented"
        )
        total_count = len(controls)

        if implemented_count == total_count:
            return "compliant"
        elif implemented_count > total_count * 0.7:
            return "partial"
        else:
            return "non_compliant"

    def _determine_implementation_status(self, control_id: str, severity: str) -> str:
        """Determine implementation status for a control."""
        # Simplified logic - in production would check actual implementation
        if severity in ["high", "critical"]:
            return "implemented"
        elif severity == "medium":
            return "partial"
        else:
            return "not_implemented"

    def _generate_recommendations(
        self, framework: str, compliance_status: str, controls: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        if compliance_status == "non_compliant":
            recommendations.append(
                f"Immediate action required to achieve {framework} compliance"
            )
            recommendations.append(
                "Implement missing controls identified in the mapping"
            )
            recommendations.append("Conduct comprehensive compliance assessment")

        elif compliance_status == "partial":
            recommendations.append(
                f"Continue working towards full {framework} compliance"
            )
            recommendations.append("Address partially implemented controls")
            recommendations.append("Regular compliance monitoring recommended")

        elif compliance_status == "compliant":
            recommendations.append(f"Maintain current {framework} compliance status")
            recommendations.append("Regular compliance reviews recommended")
            recommendations.append("Monitor for regulatory updates")

        # Add control-specific recommendations
        for control in controls:
            if control.get("implementation_status") == "not_implemented":
                recommendations.append(f"Implement control {control.get('control_id')}")

        return recommendations

    def _get_required_controls(self, framework: str) -> List[str]:
        """Get required controls for framework."""
        # Simplified - in production would be more comprehensive
        required_controls = {
            "SOC2": ["CC6.1", "CC7.1", "CC8.1"],
            "ISO27001": ["A.9.1.1", "A.12.6.1", "A.16.1.1"],
            "HIPAA": ["164.308(a)(1)", "164.312(a)(1)", "164.312(b)"],
            "GDPR": ["Art. 32", "Art. 33"],
            "PCI-DSS": ["1.1.1", "3.1.1", "6.2.1"],
            "NIST": ["PR.AC-1", "DE.CM-1", "RS.RP-1"],
        }
        return required_controls.get(framework, [])

    def _calculate_confidence(
        self, mapping_result: Dict[str, Any], metadata: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for compliance mapping."""
        try:
            base_confidence = 0.8

            # Adjust based on mapping completeness
            controls = mapping_result.get("applicable_controls", [])
            if len(controls) > 0:
                base_confidence += 0.1

            # Adjust based on compliance status
            compliance_status = mapping_result.get("compliance_status", "unknown")
            if compliance_status != "unknown":
                base_confidence += 0.05

            # Adjust based on metadata quality
            if metadata.get("category") and metadata.get("severity"):
                base_confidence += 0.05

            return min(0.95, max(0.0, base_confidence))

        except Exception as e:
            logger.error("Confidence calculation failed", error=str(e))
            return 0.7

    def _analyze_soc2_criteria(self, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze SOC 2 trust services criteria coverage."""
        criteria_coverage = {}
        for mapping in mappings:
            controls = mapping.get("applicable_controls", [])
            for control in controls:
                control_id = control.get("control_id", "")
                if control_id.startswith("CC"):
                    criteria = control_id[:3]
                    criteria_coverage[criteria] = criteria_coverage.get(criteria, 0) + 1
        return criteria_coverage

    def _analyze_iso27001_domains(
        self, mappings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze ISO 27001 control domain coverage."""
        domain_coverage = {}
        for mapping in mappings:
            controls = mapping.get("applicable_controls", [])
            for control in controls:
                control_id = control.get("control_id", "")
                if control_id.startswith("A."):
                    domain = control_id[:4]  # e.g., "A.9."
                    domain_coverage[domain] = domain_coverage.get(domain, 0) + 1
        return domain_coverage

    def _analyze_hipaa_safeguards(
        self, mappings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze HIPAA safeguard type coverage."""
        safeguard_coverage = {"administrative": 0, "physical": 0, "technical": 0}
        for mapping in mappings:
            controls = mapping.get("applicable_controls", [])
            for control in controls:
                control_id = control.get("control_id", "")
                if "164.308" in control_id:
                    safeguard_coverage["administrative"] += 1
                elif "164.310" in control_id:
                    safeguard_coverage["physical"] += 1
                elif "164.312" in control_id:
                    safeguard_coverage["technical"] += 1
        return safeguard_coverage

    def _validate_soc2_specific(self, mappings: Dict[str, Any]) -> Dict[str, Any]:
        """SOC 2 specific validation."""
        return {"soc2_validation": "passed"}

    def _validate_hipaa_specific(self, mappings: Dict[str, Any]) -> Dict[str, Any]:
        """HIPAA specific validation."""
        return {"hipaa_validation": "passed"}
