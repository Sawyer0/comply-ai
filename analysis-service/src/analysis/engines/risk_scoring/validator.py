"""
Risk scoring domain validator.

This module provides validation specifically for risk scoring domain objects
and calculations, following SRP by focusing only on risk scoring validation.
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

from ...schemas.domain_models import (
    SecurityFinding,
    RiskScore,
    RiskLevel,
    BusinessImpact,
)

logger = logging.getLogger(__name__)


@dataclass
class RiskValidationResult:
    """Result of risk scoring validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    corrected_data: Optional[any] = None


class RiskScoringValidator:
    """
    Validator for risk scoring domain objects.

    Focuses specifically on risk scoring domain validation following SRP.
    Does not duplicate general security or privacy validation.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.min_confidence = self.config.get("min_confidence", 0.0)
        self.max_findings_per_request = self.config.get(
            "max_findings_per_request", 1000
        )

    def validate_security_findings(
        self, findings: List[SecurityFinding]
    ) -> RiskValidationResult:
        """
        Validate security findings for risk scoring.

        Args:
            findings: List of security findings to validate

        Returns:
            Validation result for risk scoring purposes
        """
        errors = []
        warnings = []

        try:
            # Check if findings list is empty
            if not findings:
                warnings.append("No security findings provided for risk scoring")
                return RiskValidationResult(
                    is_valid=True,  # Empty is valid, just low risk
                    errors=errors,
                    warnings=warnings,
                )

            # Check findings count limit
            if len(findings) > self.max_findings_per_request:
                errors.append(
                    f"Too many findings: {len(findings)}, max allowed: {self.max_findings_per_request}"
                )

            # Validate each finding for risk scoring requirements
            for i, finding in enumerate(findings):
                finding_errors = self._validate_single_finding(finding, i)
                errors.extend(finding_errors)

            # Check for risk scoring specific requirements
            severity_distribution = self._analyze_severity_distribution(findings)
            if severity_distribution.get("critical", 0) > len(findings) * 0.8:
                warnings.append(
                    "High proportion of critical findings - verify accuracy"
                )

            return RiskValidationResult(
                is_valid=len(errors) == 0, errors=errors, warnings=warnings
            )

        except Exception as e:
            logger.error("Risk scoring validation failed: %s", e)
            return RiskValidationResult(
                is_valid=False, errors=[f"Validation error: {e}"], warnings=[]
            )

    def _validate_single_finding(
        self, finding: SecurityFinding, index: int
    ) -> List[str]:
        """Validate a single finding for risk scoring purposes."""
        errors = []

        # Validate required fields for risk scoring
        if not finding.finding_id:
            errors.append(
                f"Finding {index}: missing finding_id required for risk scoring"
            )

        if not finding.detector_id:
            errors.append(
                f"Finding {index}: missing detector_id required for risk scoring"
            )

        # Validate severity for risk calculation
        if not isinstance(finding.severity, RiskLevel):
            errors.append(f"Finding {index}: invalid severity type for risk scoring")

        # Validate confidence for risk weighting
        if not (0.0 <= finding.confidence <= 1.0):
            errors.append(
                f"Finding {index}: confidence must be 0.0-1.0 for risk scoring, got {finding.confidence}"
            )

        if finding.confidence < self.min_confidence:
            errors.append(
                f"Finding {index}: confidence {finding.confidence} below minimum {self.min_confidence}"
            )

        # Validate category for risk categorization
        if not finding.category or finding.category.strip() == "":
            errors.append(f"Finding {index}: category required for risk categorization")

        return errors

    def _analyze_severity_distribution(self, findings: List[SecurityFinding]) -> dict:
        """Analyze severity distribution for validation purposes."""
        distribution = {}

        for finding in findings:
            severity = (
                finding.severity.value
                if hasattr(finding.severity, "value")
                else str(finding.severity)
            )
            distribution[severity] = distribution.get(severity, 0) + 1

        return distribution

    def validate_risk_score(self, risk_score: RiskScore) -> RiskValidationResult:
        """
        Validate a calculated risk score.

        Args:
            risk_score: Risk score to validate

        Returns:
            Validation result for the risk score
        """
        errors = []
        warnings = []

        try:
            # Validate composite score
            if not (0.0 <= risk_score.composite_score <= 1.0):
                errors.append(
                    f"Composite score must be 0.0-1.0, got {risk_score.composite_score}"
                )

            # Validate confidence
            if not (0.0 <= risk_score.confidence <= 1.0):
                errors.append(
                    f"Risk confidence must be 0.0-1.0, got {risk_score.confidence}"
                )

            # Validate risk level consistency
            expected_level = self._determine_expected_risk_level(
                risk_score.composite_score
            )
            if risk_score.risk_level != expected_level:
                warnings.append(
                    f"Risk level {risk_score.risk_level.value} may be inconsistent with score {risk_score.composite_score}"
                )

            # Validate breakdown components
            if risk_score.breakdown:
                breakdown_errors = self._validate_risk_breakdown(risk_score.breakdown)
                errors.extend(breakdown_errors)

            return RiskValidationResult(
                is_valid=len(errors) == 0, errors=errors, warnings=warnings
            )

        except Exception as e:
            logger.error("Risk score validation failed: %s", e)
            return RiskValidationResult(
                is_valid=False,
                errors=[f"Risk score validation error: {e}"],
                warnings=[],
            )

    def _validate_risk_breakdown(self, breakdown) -> List[str]:
        """Validate risk breakdown components."""
        errors = []

        # Validate individual risk components
        risk_components = [
            ("technical_risk", breakdown.technical_risk),
            ("business_risk", breakdown.business_risk),
            ("regulatory_risk", breakdown.regulatory_risk),
            ("temporal_risk", breakdown.temporal_risk),
        ]

        for component_name, value in risk_components:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{component_name} must be 0.0-1.0, got {value}")

        # Validate contributing factors
        if breakdown.contributing_factors:
            for i, factor in enumerate(breakdown.contributing_factors):
                if not (0.0 <= factor.weight <= 1.0):
                    errors.append(
                        f"Factor {i} weight must be 0.0-1.0, got {factor.weight}"
                    )
                if not (0.0 <= factor.value <= 1.0):
                    errors.append(
                        f"Factor {i} value must be 0.0-1.0, got {factor.value}"
                    )
                if not (0.0 <= factor.contribution <= 1.0):
                    errors.append(
                        f"Factor {i} contribution must be 0.0-1.0, got {factor.contribution}"
                    )

        return errors

    def _determine_expected_risk_level(self, composite_score: float) -> RiskLevel:
        """Determine expected risk level from composite score."""
        if composite_score >= 0.9:
            return RiskLevel.CRITICAL
        elif composite_score >= 0.7:
            return RiskLevel.HIGH
        elif composite_score >= 0.4:
            return RiskLevel.MEDIUM
        elif composite_score >= 0.1:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFORMATIONAL

    def validate_business_impact(
        self, business_impact: BusinessImpact
    ) -> RiskValidationResult:
        """
        Validate business impact assessment.

        Args:
            business_impact: Business impact to validate

        Returns:
            Validation result for business impact
        """
        errors = []
        warnings = []

        try:
            # Validate total risk value
            if business_impact.total_risk_value < 0:
                errors.append(
                    f"Total risk value cannot be negative: {business_impact.total_risk_value}"
                )

            # Validate required impact categories
            required_categories = [
                "financial_impact",
                "operational_impact",
                "reputational_impact",
                "compliance_impact",
            ]
            for category in required_categories:
                impact_data = getattr(business_impact, category, {})
                if not impact_data:
                    warnings.append(f"Missing {category} assessment")

            # Validate confidence intervals if present
            if business_impact.confidence_interval:
                if (
                    "lower_bound" in business_impact.confidence_interval
                    and "upper_bound" in business_impact.confidence_interval
                ):
                    lower = business_impact.confidence_interval["lower_bound"]
                    upper = business_impact.confidence_interval["upper_bound"]
                    if lower > upper:
                        errors.append(
                            "Confidence interval lower bound cannot exceed upper bound"
                        )

            return RiskValidationResult(
                is_valid=len(errors) == 0, errors=errors, warnings=warnings
            )

        except Exception as e:
            logger.error("Business impact validation failed: %s", e)
            return RiskValidationResult(
                is_valid=False,
                errors=[f"Business impact validation error: {e}"],
                warnings=[],
            )
