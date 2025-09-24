"""
Business Relevance Assessor for pattern impact evaluation.

This module implements sophisticated business impact assessment logic
to determine the business relevance of detected security patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ...domain import (
    Pattern,
    PatternType,
    BusinessRelevance,
    SecurityData,
    ComplianceFramework,
)

logger = logging.getLogger(__name__)


class BusinessRelevanceAssessor:
    """
    Assesses the business relevance and impact of detected security patterns.

    Uses business context, compliance requirements, and risk factors
    to determine how relevant patterns are to business operations.
    """

    def __init__(self, config: Dict[str, any] = None):
        self.config = config or {}
        self.business_context = self._load_business_context()
        self.compliance_weights = self._load_compliance_weights()
        self.detector_criticality = self._load_detector_criticality()
        self.impact_factors = self._load_impact_factors()

    def assess_business_relevance(
        self, pattern: Pattern, context_data: SecurityData
    ) -> BusinessRelevance:
        """
        Assess the business relevance of a security pattern.

        Args:
            pattern: The pattern to assess
            context_data: Additional security data for context

        Returns:
            BusinessRelevance classification based on business impact analysis
        """
        try:
            # Calculate business impact score
            impact_score = self._calculate_business_impact_score(pattern, context_data)

            # Assess compliance implications
            compliance_impact = self._assess_compliance_impact(pattern, context_data)

            # Evaluate operational impact
            operational_impact = self._assess_operational_impact(pattern, context_data)

            # Consider financial implications
            financial_impact = self._assess_financial_impact(pattern, context_data)

            # Assess reputational risk
            reputational_impact = self._assess_reputational_impact(
                pattern, context_data
            )

            # Calculate composite relevance score
            relevance_score = self._calculate_composite_relevance_score(
                impact_score,
                compliance_impact,
                operational_impact,
                financial_impact,
                reputational_impact,
            )

            # Classify business relevance
            relevance_classification = self._classify_business_relevance(
                relevance_score
            )

            # Add detailed assessment to pattern evidence
            self._add_relevance_evidence(
                pattern,
                impact_score,
                compliance_impact,
                operational_impact,
                financial_impact,
                reputational_impact,
                relevance_score,
            )

            logger.info(
                "Business relevance assessed",
                pattern_id=pattern.pattern_id,
                relevance=relevance_classification.value,
                relevance_score=relevance_score,
            )

            return relevance_classification

        except Exception as e:
            logger.error(
                "Business relevance assessment failed",
                error=str(e),
                pattern_id=pattern.pattern_id,
            )
            return BusinessRelevance.LOW

    def assess_multiple_patterns_relevance(
        self, patterns: List[Pattern], context_data: SecurityData
    ) -> Dict[str, BusinessRelevance]:
        """
        Assess business relevance for multiple patterns with cross-pattern analysis.

        Args:
            patterns: List of patterns to assess
            context_data: Security data for context

        Returns:
            Dictionary mapping pattern IDs to their business relevance
        """
        relevance_results = {}
        pattern_impacts = {}

        # Assess individual patterns
        for pattern in patterns:
            relevance = self.assess_business_relevance(pattern, context_data)
            relevance_results[pattern.pattern_id] = relevance

            # Extract impact score for cross-pattern analysis
            for evidence in pattern.supporting_evidence:
                if (
                    isinstance(evidence, dict)
                    and "business_relevance_analysis" in evidence
                ):
                    pattern_impacts[pattern.pattern_id] = evidence[
                        "business_relevance_analysis"
                    ]["composite_score"]
                    break

        # Perform cross-pattern business impact analysis
        self._perform_cross_pattern_business_analysis(
            patterns, pattern_impacts, relevance_results
        )

        return relevance_results

    def _calculate_business_impact_score(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Calculate base business impact score."""
        try:
            # Start with pattern confidence and significance
            base_score = (pattern.confidence + pattern.statistical_significance) / 2

            # Adjust based on affected detectors' criticality
            detector_factor = self._calculate_detector_criticality_factor(
                pattern.affected_detectors
            )

            # Adjust based on pattern type business impact
            pattern_type_factor = self._calculate_pattern_type_factor(
                pattern.pattern_type
            )

            # Adjust based on pattern scope and duration
            scope_factor = self._calculate_scope_factor(pattern, context_data)

            # Adjust based on timing (business hours, weekends, etc.)
            timing_factor = self._calculate_timing_factor(pattern)

            # Calculate composite impact score
            impact_score = (
                base_score
                * detector_factor
                * pattern_type_factor
                * scope_factor
                * timing_factor
            )

            return min(1.0, max(0.0, impact_score))

        except Exception as e:
            logger.error("Business impact score calculation failed", error=str(e))
            return 0.0

    def _assess_compliance_impact(
        self, pattern: Pattern, context_data: SecurityData
    ) -> Dict[str, any]:
        """Assess compliance-related impact of the pattern."""
        compliance_impact = {
            "frameworks_affected": [],
            "severity_by_framework": {},
            "compliance_risk_score": 0.0,
            "regulatory_deadlines": [],
            "audit_implications": [],
        }

        try:
            # Check which compliance frameworks are affected
            for detector in pattern.affected_detectors:
                frameworks = self._get_detector_compliance_frameworks(detector)
                compliance_impact["frameworks_affected"].extend(frameworks)

            # Remove duplicates
            compliance_impact["frameworks_affected"] = list(
                set(compliance_impact["frameworks_affected"])
            )

            # Assess severity for each framework
            for framework in compliance_impact["frameworks_affected"]:
                severity = self._assess_framework_severity(pattern, framework)
                compliance_impact["severity_by_framework"][framework] = severity

            # Calculate overall compliance risk score
            if compliance_impact["frameworks_affected"]:
                framework_scores = list(
                    compliance_impact["severity_by_framework"].values()
                )
                compliance_impact["compliance_risk_score"] = sum(
                    framework_scores
                ) / len(framework_scores)

            # Identify regulatory deadlines that might be affected
            compliance_impact["regulatory_deadlines"] = (
                self._identify_affected_deadlines(
                    compliance_impact["frameworks_affected"]
                )
            )

            # Assess audit implications
            compliance_impact["audit_implications"] = self._assess_audit_implications(
                pattern, compliance_impact["frameworks_affected"]
            )

        except Exception as e:
            logger.error("Compliance impact assessment failed", error=str(e))

        return compliance_impact

    def _assess_operational_impact(
        self, pattern: Pattern, context_data: SecurityData
    ) -> Dict[str, any]:
        """Assess operational impact of the pattern."""
        operational_impact = {
            "affected_systems": [],
            "service_disruption_risk": 0.0,
            "resource_requirements": {},
            "response_urgency": "low",
            "escalation_needed": False,
        }

        try:
            # Identify affected systems
            operational_impact["affected_systems"] = self._identify_affected_systems(
                pattern, context_data
            )

            # Assess service disruption risk
            operational_impact["service_disruption_risk"] = (
                self._calculate_service_disruption_risk(
                    pattern, operational_impact["affected_systems"]
                )
            )

            # Estimate resource requirements for response
            operational_impact["resource_requirements"] = (
                self._estimate_response_resources(pattern)
            )

            # Determine response urgency
            operational_impact["response_urgency"] = self._determine_response_urgency(
                pattern, operational_impact["service_disruption_risk"]
            )

            # Check if escalation is needed
            operational_impact["escalation_needed"] = self._check_escalation_needed(
                pattern, operational_impact
            )

        except Exception as e:
            logger.error("Operational impact assessment failed", error=str(e))

        return operational_impact

    def _assess_financial_impact(
        self, pattern: Pattern, context_data: SecurityData
    ) -> Dict[str, any]:
        """Assess financial impact of the pattern."""
        financial_impact = {
            "potential_loss_range": {"min": 0, "max": 0, "expected": 0},
            "cost_categories": {},
            "revenue_impact": 0.0,
            "compliance_penalties": 0.0,
            "response_costs": 0.0,
            "business_disruption_costs": 0.0,
        }

        try:
            # Calculate potential revenue impact
            financial_impact["revenue_impact"] = self._calculate_revenue_impact(
                pattern, context_data
            )

            # Estimate compliance penalties
            financial_impact["compliance_penalties"] = (
                self._estimate_compliance_penalties(pattern)
            )

            # Estimate response costs
            financial_impact["response_costs"] = self._estimate_response_costs(pattern)

            # Estimate business disruption costs
            financial_impact["business_disruption_costs"] = (
                self._estimate_disruption_costs(pattern, context_data)
            )

            # Calculate total potential loss range
            total_expected = (
                financial_impact["revenue_impact"]
                + financial_impact["compliance_penalties"]
                + financial_impact["response_costs"]
                + financial_impact["business_disruption_costs"]
            )

            financial_impact["potential_loss_range"] = {
                "min": total_expected * 0.5,
                "max": total_expected * 2.0,
                "expected": total_expected,
            }

            # Categorize costs
            financial_impact["cost_categories"] = {
                "direct_costs": financial_impact["response_costs"],
                "indirect_costs": financial_impact["business_disruption_costs"],
                "regulatory_costs": financial_impact["compliance_penalties"],
                "opportunity_costs": financial_impact["revenue_impact"],
            }

        except Exception as e:
            logger.error("Financial impact assessment failed", error=str(e))

        return financial_impact

    def _assess_reputational_impact(
        self, pattern: Pattern, context_data: SecurityData
    ) -> Dict[str, any]:
        """Assess reputational impact of the pattern."""
        reputational_impact = {
            "public_exposure_risk": 0.0,
            "customer_trust_impact": 0.0,
            "media_attention_likelihood": 0.0,
            "stakeholder_confidence_impact": 0.0,
            "brand_damage_potential": 0.0,
            "recovery_timeline": "unknown",
        }

        try:
            # Assess public exposure risk
            reputational_impact["public_exposure_risk"] = (
                self._calculate_public_exposure_risk(pattern)
            )

            # Assess customer trust impact
            reputational_impact["customer_trust_impact"] = (
                self._calculate_customer_trust_impact(pattern, context_data)
            )

            # Assess media attention likelihood
            reputational_impact["media_attention_likelihood"] = (
                self._calculate_media_attention_likelihood(pattern)
            )

            # Assess stakeholder confidence impact
            reputational_impact["stakeholder_confidence_impact"] = (
                self._calculate_stakeholder_impact(pattern)
            )

            # Calculate overall brand damage potential
            reputational_impact["brand_damage_potential"] = (
                reputational_impact["public_exposure_risk"]
                + reputational_impact["customer_trust_impact"]
                + reputational_impact["media_attention_likelihood"]
                + reputational_impact["stakeholder_confidence_impact"]
            ) / 4

            # Estimate recovery timeline
            reputational_impact["recovery_timeline"] = (
                self._estimate_reputation_recovery_timeline(
                    reputational_impact["brand_damage_potential"]
                )
            )

        except Exception as e:
            logger.error("Reputational impact assessment failed", error=str(e))

        return reputational_impact

    def _calculate_detector_criticality_factor(
        self, affected_detectors: List[str]
    ) -> float:
        """Calculate factor based on criticality of affected detectors."""
        if not affected_detectors:
            return 0.5

        criticality_scores = []
        for detector in affected_detectors:
            criticality = self.detector_criticality.get(detector.lower(), 0.5)
            criticality_scores.append(criticality)

        # Use maximum criticality (most critical detector drives the factor)
        return max(criticality_scores)

    def _calculate_pattern_type_factor(self, pattern_type: PatternType) -> float:
        """Calculate business impact factor based on pattern type."""
        type_factors = {
            PatternType.ANOMALY: 1.2,  # Anomalies often indicate serious issues
            PatternType.CORRELATION: 1.1,  # Correlations suggest systemic issues
            PatternType.TEMPORAL: 1.0,  # Temporal patterns are concerning but predictable
            PatternType.FREQUENCY: 0.9,  # Frequency patterns are often operational
        }

        return type_factors.get(pattern_type, 1.0)

    def _calculate_scope_factor(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Calculate factor based on pattern scope."""
        # More detectors affected = higher business impact
        detector_count = len(pattern.affected_detectors)

        if detector_count >= 5:
            return 1.3
        elif detector_count >= 3:
            return 1.1
        elif detector_count >= 2:
            return 1.0
        else:
            return 0.8

    def _calculate_timing_factor(self, pattern: Pattern) -> float:
        """Calculate factor based on when pattern occurred."""
        # Patterns during business hours have higher impact
        start_time = pattern.time_range.start

        # Simple heuristic: weekday business hours (9 AM - 5 PM) have higher impact
        if start_time.weekday() < 5:  # Monday = 0, Friday = 4
            if 9 <= start_time.hour <= 17:
                return 1.2  # Business hours
            else:
                return 1.0  # After hours
        else:
            return 0.8  # Weekend

    def _calculate_composite_relevance_score(
        self,
        impact_score: float,
        compliance_impact: Dict[str, any],
        operational_impact: Dict[str, any],
        financial_impact: Dict[str, any],
        reputational_impact: Dict[str, any],
    ) -> float:
        """Calculate composite business relevance score."""
        try:
            # Weight different impact types
            weights = {
                "base_impact": 0.2,
                "compliance": 0.25,
                "operational": 0.25,
                "financial": 0.2,
                "reputational": 0.1,
            }

            # Normalize impact scores
            compliance_score = compliance_impact.get("compliance_risk_score", 0.0)
            operational_score = operational_impact.get("service_disruption_risk", 0.0)
            financial_score = min(
                1.0,
                financial_impact.get("potential_loss_range", {}).get("expected", 0)
                / 100000,
            )
            reputational_score = reputational_impact.get("brand_damage_potential", 0.0)

            # Calculate weighted composite score
            composite_score = (
                weights["base_impact"] * impact_score
                + weights["compliance"] * compliance_score
                + weights["operational"] * operational_score
                + weights["financial"] * financial_score
                + weights["reputational"] * reputational_score
            )

            return min(1.0, max(0.0, composite_score))

        except Exception as e:
            logger.error("Composite relevance score calculation failed", error=str(e))
            return 0.0

    def _classify_business_relevance(self, relevance_score: float) -> BusinessRelevance:
        """Classify business relevance based on composite score."""
        if relevance_score >= 0.8:
            return BusinessRelevance.CRITICAL
        elif relevance_score >= 0.6:
            return BusinessRelevance.HIGH
        elif relevance_score >= 0.4:
            return BusinessRelevance.MEDIUM
        else:
            return BusinessRelevance.LOW

    def _add_relevance_evidence(
        self,
        pattern: Pattern,
        impact_score: float,
        compliance_impact: Dict[str, any],
        operational_impact: Dict[str, any],
        financial_impact: Dict[str, any],
        reputational_impact: Dict[str, any],
        relevance_score: float,
    ):
        """Add detailed business relevance analysis to pattern evidence."""
        relevance_evidence = {
            "business_relevance_analysis": {
                "composite_score": relevance_score,
                "base_impact_score": impact_score,
                "compliance_impact": compliance_impact,
                "operational_impact": operational_impact,
                "financial_impact": financial_impact,
                "reputational_impact": reputational_impact,
                "assessment_timestamp": datetime.utcnow().isoformat(),
                "assessment_method": "comprehensive_business_impact_analysis",
            }
        }

        pattern.supporting_evidence.append(relevance_evidence)

    # Helper methods for specific assessments
    def _get_detector_compliance_frameworks(self, detector: str) -> List[str]:
        """Get compliance frameworks associated with a detector."""
        detector_frameworks = {
            "presidio": ["GDPR", "HIPAA", "PCI_DSS"],
            "pii-detector": ["GDPR", "HIPAA", "SOC2"],
            "gdpr-scanner": ["GDPR"],
            "hipaa-detector": ["HIPAA"],
            "financial-detector": ["PCI_DSS", "SOC2"],
            "deberta-toxicity": ["SOC2"],
        }

        return detector_frameworks.get(detector.lower(), ["SOC2"])  # Default to SOC2

    def _assess_framework_severity(self, pattern: Pattern, framework: str) -> float:
        """Assess severity for a specific compliance framework."""
        framework_weights = self.compliance_weights.get(framework, 0.5)
        return pattern.confidence * framework_weights

    def _identify_affected_deadlines(
        self, frameworks: List[str]
    ) -> List[Dict[str, any]]:
        """Identify regulatory deadlines that might be affected."""
        # This would typically come from a compliance calendar
        return [
            {
                "framework": "GDPR",
                "deadline": "2024-05-25",
                "type": "annual_assessment",
            },
            {
                "framework": "SOC2",
                "deadline": "2024-12-31",
                "type": "audit_preparation",
            },
        ]

    def _assess_audit_implications(
        self, pattern: Pattern, frameworks: List[str]
    ) -> List[str]:
        """Assess audit implications."""
        implications = []

        if "GDPR" in frameworks:
            implications.append("data_protection_audit_finding")
        if "SOC2" in frameworks:
            implications.append("security_control_deficiency")
        if "HIPAA" in frameworks:
            implications.append("phi_protection_gap")

        return implications

    def _load_business_context(self) -> Dict[str, any]:
        """Load business context configuration."""
        return self.config.get(
            "business_context",
            {
                "industry": "technology",
                "company_size": "medium",
                "risk_tolerance": "medium",
                "compliance_requirements": ["SOC2", "GDPR"],
            },
        )

    def _load_compliance_weights(self) -> Dict[str, float]:
        """Load compliance framework weights."""
        return self.config.get(
            "compliance_weights",
            {"SOC2": 0.8, "GDPR": 0.9, "HIPAA": 0.85, "PCI_DSS": 0.9, "ISO27001": 0.7},
        )

    def _load_detector_criticality(self) -> Dict[str, float]:
        """Load detector criticality mappings."""
        return self.config.get(
            "detector_criticality",
            {
                "presidio": 0.9,
                "pii-detector": 0.9,
                "gdpr-scanner": 0.85,
                "hipaa-detector": 0.85,
                "financial-detector": 0.8,
                "deberta-toxicity": 0.6,
                "custom-detector": 0.5,
            },
        )

    def _load_impact_factors(self) -> Dict[str, any]:
        """Load impact calculation factors."""
        return self.config.get(
            "impact_factors",
            {
                "revenue_per_hour": 10000,
                "compliance_penalty_base": 50000,
                "response_cost_per_hour": 500,
                "reputation_recovery_months": 6,
            },
        )

    # Placeholder methods for complex calculations (would be implemented based on business requirements)
    def _identify_affected_systems(
        self, pattern: Pattern, context_data: SecurityData
    ) -> List[str]:
        """Identify systems affected by the pattern."""
        return ["web_application", "database", "api_gateway"]  # Placeholder

    def _calculate_service_disruption_risk(
        self, pattern: Pattern, affected_systems: List[str]
    ) -> float:
        """Calculate service disruption risk."""
        return min(1.0, len(affected_systems) * 0.2)  # Placeholder

    def _estimate_response_resources(self, pattern: Pattern) -> Dict[str, any]:
        """Estimate resources needed for response."""
        return {"hours": 8, "personnel": 2, "cost": 4000}  # Placeholder

    def _determine_response_urgency(
        self, pattern: Pattern, disruption_risk: float
    ) -> str:
        """Determine response urgency."""
        if disruption_risk > 0.7:
            return "critical"
        elif disruption_risk > 0.4:
            return "high"
        else:
            return "medium"

    def _check_escalation_needed(
        self, pattern: Pattern, operational_impact: Dict[str, any]
    ) -> bool:
        """Check if escalation is needed."""
        return operational_impact.get("service_disruption_risk", 0) > 0.6

    def _calculate_revenue_impact(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Calculate potential revenue impact."""
        return 5000.0  # Placeholder

    def _estimate_compliance_penalties(self, pattern: Pattern) -> float:
        """Estimate potential compliance penalties."""
        return 25000.0  # Placeholder

    def _estimate_response_costs(self, pattern: Pattern) -> float:
        """Estimate response costs."""
        return 4000.0  # Placeholder

    def _estimate_disruption_costs(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Estimate business disruption costs."""
        return 10000.0  # Placeholder

    def _calculate_public_exposure_risk(self, pattern: Pattern) -> float:
        """Calculate public exposure risk."""
        return 0.3  # Placeholder

    def _calculate_customer_trust_impact(
        self, pattern: Pattern, context_data: SecurityData
    ) -> float:
        """Calculate customer trust impact."""
        return 0.4  # Placeholder

    def _calculate_media_attention_likelihood(self, pattern: Pattern) -> float:
        """Calculate media attention likelihood."""
        return 0.2  # Placeholder

    def _calculate_stakeholder_impact(self, pattern: Pattern) -> float:
        """Calculate stakeholder confidence impact."""
        return 0.3  # Placeholder

    def _estimate_reputation_recovery_timeline(
        self, brand_damage_potential: float
    ) -> str:
        """Estimate reputation recovery timeline."""
        if brand_damage_potential > 0.7:
            return "12+ months"
        elif brand_damage_potential > 0.4:
            return "6-12 months"
        else:
            return "3-6 months"

    def _perform_cross_pattern_business_analysis(
        self,
        patterns: List[Pattern],
        pattern_impacts: Dict[str, float],
        relevance_results: Dict[str, BusinessRelevance],
    ):
        """Perform cross-pattern business impact analysis."""
        # This would analyze how multiple patterns together affect business
        # For now, just add metadata about pattern interactions

        high_impact_patterns = [
            pid
            for pid, relevance in relevance_results.items()
            if relevance in [BusinessRelevance.HIGH, BusinessRelevance.CRITICAL]
        ]

        if len(high_impact_patterns) > 1:
            for pattern in patterns:
                if pattern.pattern_id in high_impact_patterns:
                    cross_analysis = {
                        "cross_pattern_business_analysis": {
                            "multiple_high_impact_patterns": True,
                            "total_high_impact_count": len(high_impact_patterns),
                            "compound_risk_factor": min(
                                2.0, len(high_impact_patterns) * 0.3
                            ),
                            "business_continuity_risk": "elevated",
                        }
                    }
                    pattern.supporting_evidence.append(cross_analysis)
