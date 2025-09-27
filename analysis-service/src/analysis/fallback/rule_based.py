"""
Rule-based fallback mechanisms for Analysis Service.

Provides deterministic analysis when ML models are unavailable.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

from ..shared_integration import get_shared_logger

logger = get_shared_logger(__name__)


@dataclass
class RuleResult:
    """Result from rule-based analysis."""

    rule_name: str
    matched: bool
    confidence: float
    value: Any
    metadata: Dict[str, Any]


class RuleBasedRiskAssessment:
    """Rule-based risk assessment fallback."""

    def __init__(self):
        self.logger = logger.bind(component="rule_based_risk")

        # Risk scoring rules
        self.risk_rules = {
            "high_sensitivity_data": {
                "patterns": [
                    "ssn",
                    "social security",
                    "credit card",
                    "passport",
                    "driver license",
                ],
                "risk_score": 0.9,
                "description": "High sensitivity personal data detected",
            },
            "financial_data": {
                "patterns": [
                    "bank account",
                    "routing number",
                    "iban",
                    "swift",
                    "account number",
                ],
                "risk_score": 0.85,
                "description": "Financial information detected",
            },
            "health_data": {
                "patterns": [
                    "medical record",
                    "diagnosis",
                    "prescription",
                    "patient",
                    "hipaa",
                ],
                "risk_score": 0.8,
                "description": "Health information detected",
            },
            "authentication_data": {
                "patterns": ["password", "api key", "token", "secret", "credential"],
                "risk_score": 0.75,
                "description": "Authentication data detected",
            },
            "personal_identifiers": {
                "patterns": [
                    "email",
                    "phone",
                    "address",
                    "name",
                    "dob",
                    "date of birth",
                ],
                "risk_score": 0.6,
                "description": "Personal identifiers detected",
            },
        }

        # Compliance framework mappings
        self.compliance_mappings = {
            "GDPR": {
                "personal_data": ["name", "email", "phone", "address", "ip address"],
                "sensitive_data": [
                    "health",
                    "biometric",
                    "genetic",
                    "political",
                    "religious",
                ],
                "requirements": ["consent", "data minimization", "right to erasure"],
            },
            "HIPAA": {
                "phi_identifiers": [
                    "name",
                    "address",
                    "dates",
                    "phone",
                    "email",
                    "ssn",
                    "medical record",
                ],
                "requirements": [
                    "access controls",
                    "audit logs",
                    "encryption",
                    "minimum necessary",
                ],
            },
            "SOC2": {
                "security_controls": [
                    "access management",
                    "logical access",
                    "system operations",
                ],
                "requirements": [
                    "monitoring",
                    "incident response",
                    "change management",
                ],
            },
        }

    async def assess_risk(
        self, findings: Dict[str, Any], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform rule-based risk assessment."""
        try:
            self.logger.info("Starting rule-based risk assessment")

            # Extract text content for analysis
            text_content = self._extract_text_content(findings)

            # Apply risk rules
            rule_results = []
            total_risk_score = 0.0
            max_risk_score = 0.0

            for rule_name, rule_config in self.risk_rules.items():
                result = self._apply_risk_rule(rule_name, rule_config, text_content)
                rule_results.append(result)

                if result.matched:
                    total_risk_score += result.confidence
                    max_risk_score = max(max_risk_score, result.confidence)

            # Calculate overall risk score
            if rule_results:
                # Use weighted average with emphasis on highest risk
                avg_risk = (
                    total_risk_score / len([r for r in rule_results if r.matched])
                    if any(r.matched for r in rule_results)
                    else 0.0
                )
                overall_risk = (max_risk_score * 0.7) + (avg_risk * 0.3)
            else:
                overall_risk = 0.1  # Minimal baseline risk

            # Determine risk level
            risk_level = self._determine_risk_level(overall_risk)

            # Generate recommendations
            recommendations = self._generate_risk_recommendations(
                rule_results, risk_level
            )

            return {
                "overall_risk_score": overall_risk,
                "risk_level": risk_level,
                "rule_results": [
                    {
                        "rule": r.rule_name,
                        "matched": r.matched,
                        "confidence": r.confidence,
                        "metadata": r.metadata,
                    }
                    for r in rule_results
                    if r.matched
                ],
                "recommendations": recommendations,
                "confidence": 0.7,  # Rule-based confidence
                "method": "rule_based",
                "processing_time_ms": 50,  # Fast rule-based processing
            }

        except Exception as e:
            self.logger.error("Rule-based risk assessment failed", error=str(e))
            return {"error": str(e), "method": "rule_based", "confidence": 0.0}

    def _extract_text_content(self, findings: Dict[str, Any]) -> str:
        """Extract text content from findings for analysis."""
        text_parts = []

        def extract_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj.lower())
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(findings)
        return " ".join(text_parts)

    def _apply_risk_rule(
        self, rule_name: str, rule_config: Dict[str, Any], text_content: str
    ) -> RuleResult:
        """Apply a single risk rule to text content."""
        patterns = rule_config["patterns"]
        risk_score = rule_config["risk_score"]

        matches = []
        for pattern in patterns:
            if pattern in text_content:
                matches.append(pattern)

        matched = len(matches) > 0
        confidence = risk_score if matched else 0.0

        return RuleResult(
            rule_name=rule_name,
            matched=matched,
            confidence=confidence,
            value=matches,
            metadata={
                "patterns_matched": matches,
                "total_patterns": len(patterns),
                "description": rule_config["description"],
            },
        )

    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score."""
        if risk_score >= 0.8:
            return "HIGH"
        elif risk_score >= 0.6:
            return "MEDIUM"
        elif risk_score >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_risk_recommendations(
        self, rule_results: List[RuleResult], risk_level: str
    ) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []

        # General recommendations based on risk level
        if risk_level == "HIGH":
            recommendations.extend(
                [
                    "Implement immediate access restrictions",
                    "Enable comprehensive audit logging",
                    "Consider data encryption at rest and in transit",
                    "Conduct security review within 24 hours",
                ]
            )
        elif risk_level == "MEDIUM":
            recommendations.extend(
                [
                    "Review access permissions",
                    "Enable monitoring and alerting",
                    "Consider additional security controls",
                ]
            )
        elif risk_level == "LOW":
            recommendations.extend(
                ["Monitor for unusual activity", "Review data handling procedures"]
            )

        # Specific recommendations based on matched rules
        for result in rule_results:
            if not result.matched:
                continue

            if result.rule_name == "high_sensitivity_data":
                recommendations.append("Implement data loss prevention (DLP) controls")
            elif result.rule_name == "financial_data":
                recommendations.append("Apply PCI DSS compliance measures")
            elif result.rule_name == "health_data":
                recommendations.append("Ensure HIPAA compliance measures")
            elif result.rule_name == "authentication_data":
                recommendations.append("Rotate credentials immediately")

        return list(set(recommendations))  # Remove duplicates


class RuleBasedPatternAnalysis:
    """Rule-based pattern analysis fallback."""

    def __init__(self):
        self.logger = logger.bind(component="rule_based_patterns")

        # Pattern detection rules
        self.pattern_rules = {
            "email_pattern": {
                "regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                "description": "Email address pattern",
            },
            "phone_pattern": {
                "regex": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
                "description": "Phone number pattern",
            },
            "ssn_pattern": {
                "regex": r"\b\d{3}-?\d{2}-?\d{4}\b",
                "description": "Social Security Number pattern",
            },
            "credit_card_pattern": {
                "regex": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
                "description": "Credit card number pattern",
            },
            "ip_address_pattern": {
                "regex": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
                "description": "IP address pattern",
            },
        }

    async def analyze_patterns(
        self, findings: Dict[str, Any], context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform rule-based pattern analysis."""
        try:
            self.logger.info("Starting rule-based pattern analysis")

            # Extract text content
            text_content = self._extract_text_content(findings)

            # Apply pattern rules
            detected_patterns = []

            for pattern_name, pattern_config in self.pattern_rules.items():
                matches = self._find_pattern_matches(
                    pattern_config["regex"], text_content
                )

                if matches:
                    detected_patterns.append(
                        {
                            "pattern_name": pattern_name,
                            "description": pattern_config["description"],
                            "matches": matches,
                            "count": len(matches),
                        }
                    )

            # Calculate confidence based on pattern diversity
            confidence = min(0.8, len(detected_patterns) * 0.15 + 0.2)

            return {
                "detected_patterns": detected_patterns,
                "pattern_count": len(detected_patterns),
                "total_matches": sum(p["count"] for p in detected_patterns),
                "confidence": confidence,
                "method": "rule_based",
                "processing_time_ms": 30,
            }

        except Exception as e:
            self.logger.error("Rule-based pattern analysis failed", error=str(e))
            return {"error": str(e), "method": "rule_based", "confidence": 0.0}

    def _extract_text_content(self, findings: Dict[str, Any]) -> str:
        """Extract text content from findings."""
        text_parts = []

        def extract_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(findings)
        return " ".join(text_parts)

    def _find_pattern_matches(self, regex_pattern: str, text: str) -> List[str]:
        """Find all matches for a regex pattern in text."""
        try:
            matches = re.findall(regex_pattern, text, re.IGNORECASE)
            return list(set(matches))  # Remove duplicates
        except re.error as e:
            self.logger.error("Regex error", pattern=regex_pattern, error=str(e))
            return []


class RuleBasedComplianceMapping:
    """Rule-based compliance mapping fallback."""

    def __init__(self):
        self.logger = logger.bind(component="rule_based_compliance")

        # Compliance framework rules
        self.framework_rules = {
            "GDPR": {
                "data_categories": {
                    "personal_data": ["name", "email", "phone", "address"],
                    "sensitive_data": ["health", "biometric", "political", "religious"],
                },
                "requirements": {
                    "consent": "Explicit consent required for processing",
                    "data_minimization": "Process only necessary data",
                    "right_to_erasure": "Provide data deletion capabilities",
                },
            },
            "HIPAA": {
                "data_categories": {
                    "phi": ["medical record", "patient", "diagnosis", "treatment"]
                },
                "requirements": {
                    "access_controls": "Implement role-based access controls",
                    "audit_logs": "Maintain comprehensive audit trails",
                    "encryption": "Encrypt PHI at rest and in transit",
                },
            },
            "SOC2": {
                "data_categories": {
                    "customer_data": ["customer", "client", "user data"]
                },
                "requirements": {
                    "monitoring": "Implement continuous monitoring",
                    "incident_response": "Maintain incident response procedures",
                    "change_management": "Control system changes",
                },
            },
        }

    async def map_to_framework(
        self, findings: Dict[str, Any], framework: str
    ) -> Dict[str, Any]:
        """Map findings to compliance framework."""
        try:
            self.logger.info(
                "Starting rule-based compliance mapping", framework=framework
            )

            if framework not in self.framework_rules:
                return {
                    "error": f"Framework '{framework}' not supported",
                    "supported_frameworks": list(self.framework_rules.keys()),
                }

            # Extract text content
            text_content = self._extract_text_content(findings).lower()

            framework_config = self.framework_rules[framework]

            # Check data categories
            detected_categories = []
            for category, keywords in framework_config["data_categories"].items():
                if any(keyword in text_content for keyword in keywords):
                    detected_categories.append(category)

            # Generate applicable requirements
            applicable_requirements = []
            if detected_categories:
                for req_name, req_desc in framework_config["requirements"].items():
                    applicable_requirements.append(
                        {
                            "requirement": req_name,
                            "description": req_desc,
                            "applicable": True,
                        }
                    )

            # Calculate compliance score
            compliance_score = 0.7 if detected_categories else 0.9

            return {
                "framework": framework,
                "detected_categories": detected_categories,
                "applicable_requirements": applicable_requirements,
                "compliance_score": compliance_score,
                "recommendations": self._generate_compliance_recommendations(
                    framework, detected_categories
                ),
                "method": "rule_based",
                "confidence": 0.75,
            }

        except Exception as e:
            self.logger.error("Rule-based compliance mapping failed", error=str(e))
            return {"error": str(e), "method": "rule_based", "confidence": 0.0}

    def _extract_text_content(self, findings: Dict[str, Any]) -> str:
        """Extract text content from findings."""
        text_parts = []

        def extract_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)

        extract_recursive(findings)
        return " ".join(text_parts)

    def _generate_compliance_recommendations(
        self, framework: str, detected_categories: List[str]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []

        if framework == "GDPR" and detected_categories:
            recommendations.extend(
                [
                    "Implement data subject consent mechanisms",
                    "Provide data portability features",
                    "Establish data retention policies",
                    "Conduct privacy impact assessments",
                ]
            )

        elif framework == "HIPAA" and detected_categories:
            recommendations.extend(
                [
                    "Implement HIPAA-compliant access controls",
                    "Establish business associate agreements",
                    "Conduct regular security risk assessments",
                    "Implement breach notification procedures",
                ]
            )

        elif framework == "SOC2" and detected_categories:
            recommendations.extend(
                [
                    "Implement security monitoring controls",
                    "Establish change management procedures",
                    "Conduct regular security assessments",
                    "Maintain incident response capabilities",
                ]
            )

        return recommendations
