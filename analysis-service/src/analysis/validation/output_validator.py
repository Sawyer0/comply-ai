"""
Output validation for Analysis Service.

Validates analysis outputs against expected schemas and business rules.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from ..shared_integration import get_shared_logger

    logger = get_shared_logger(__name__)
except ImportError:
    # Fallback to standard logging if shared integration is not available
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class OutputValidationResult:
    """Result of output validation."""

    is_valid: bool
    validated_output: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = None


class OutputValidator:
    """Validates analysis service outputs."""

    def __init__(self):
        self.logger = logger.bind(component="output_validator")

        # Define valid values for different fields
        self.valid_risk_levels = {"low", "medium", "high", "critical"}
        self.valid_categories = {
            "pii",
            "security",
            "compliance",
            "data_quality",
            "privacy",
            "financial",
            "operational",
            "technical",
        }
        self.valid_frameworks = {
            "SOC2",
            "ISO27001",
            "HIPAA",
            "GDPR",
            "PCI-DSS",
            "NIST",
            "CIS",
            "OWASP",
            "General",
        }

    def validate_canonical_result(
        self, result: Dict[str, Any]
    ) -> OutputValidationResult:
        """
        Validate canonical taxonomy result.

        Args:
            result: Canonical result to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        validated_output = result.copy()

        try:
            # Validate required fields
            required_fields = ["category", "subcategory", "confidence", "risk_level"]
            for field in required_fields:
                if field not in result:
                    errors.append(f"Missing required field: {field}")

            # Validate category
            category = result.get("category")
            if category:
                if not isinstance(category, str):
                    errors.append("category must be a string")
                elif category not in self.valid_categories:
                    warnings.append(f"Unknown category: {category}")

            # Validate subcategory
            subcategory = result.get("subcategory")
            if subcategory and not isinstance(subcategory, str):
                errors.append("subcategory must be a string")

            # Validate confidence
            confidence = result.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    errors.append("confidence must be a number")
                elif not (0 <= confidence <= 1):
                    errors.append("confidence must be between 0 and 1")
                elif confidence < 0.5:
                    warnings.append("Low confidence score detected")

            # Validate risk_level
            risk_level = result.get("risk_level")
            if risk_level:
                if not isinstance(risk_level, str):
                    errors.append("risk_level must be a string")
                elif risk_level not in self.valid_risk_levels:
                    errors.append(f"Invalid risk_level: {risk_level}")

            # Validate tags if present
            tags = result.get("tags", [])
            if tags and not isinstance(tags, list):
                errors.append("tags must be a list")
            elif tags and not all(isinstance(tag, str) for tag in tags):
                errors.append("all tags must be strings")

            # Validate metadata if present
            metadata = result.get("metadata", {})
            if metadata and not isinstance(metadata, dict):
                errors.append("metadata must be a dictionary")

            return OutputValidationResult(
                is_valid=len(errors) == 0,
                validated_output=validated_output,
                errors=errors,
                warnings=warnings,
                confidence_score=confidence,
                metadata={"validation_type": "canonical_result"},
            )

        except Exception as e:
            self.logger.error("Canonical result validation failed", error=str(e))
            return OutputValidationResult(
                is_valid=False,
                validated_output=result,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def validate_risk_score(self, risk_score: Dict[str, Any]) -> OutputValidationResult:
        """
        Validate risk score output.

        Args:
            risk_score: Risk score to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        validated_output = risk_score.copy()

        try:
            # Validate required fields
            required_fields = [
                "overall_risk_score",
                "technical_risk",
                "business_risk",
                "regulatory_risk",
                "temporal_risk",
            ]

            for field in required_fields:
                if field not in risk_score:
                    errors.append(f"Missing required field: {field}")
                else:
                    score = risk_score[field]
                    if not isinstance(score, (int, float)):
                        errors.append(f"{field} must be a number")
                    elif not (0 <= score <= 1):
                        errors.append(f"{field} must be between 0 and 1")

            # Validate risk_factors if present
            risk_factors = risk_score.get("risk_factors", [])
            if risk_factors:
                if not isinstance(risk_factors, list):
                    errors.append("risk_factors must be a list")
                else:
                    for i, factor in enumerate(risk_factors):
                        if not isinstance(factor, dict):
                            errors.append(f"risk_factors[{i}] must be a dictionary")
                        elif "name" not in factor or "weight" not in factor:
                            errors.append(f"risk_factors[{i}] missing name or weight")

            # Validate mitigation_recommendations if present
            recommendations = risk_score.get("mitigation_recommendations", [])
            if recommendations:
                if not isinstance(recommendations, list):
                    errors.append("mitigation_recommendations must be a list")
                elif not all(isinstance(rec, str) for rec in recommendations):
                    errors.append("all mitigation_recommendations must be strings")

            # Check for consistency
            overall_score = risk_score.get("overall_risk_score")
            if overall_score is not None and len(errors) == 0:
                # Calculate expected overall score
                component_scores = [
                    risk_score.get("technical_risk", 0),
                    risk_score.get("business_risk", 0),
                    risk_score.get("regulatory_risk", 0),
                    risk_score.get("temporal_risk", 0),
                ]
                expected_overall = sum(component_scores) / len(component_scores)

                if abs(overall_score - expected_overall) > 0.1:
                    warnings.append(
                        "Overall risk score may be inconsistent with components"
                    )

            return OutputValidationResult(
                is_valid=len(errors) == 0,
                validated_output=validated_output,
                errors=errors,
                warnings=warnings,
                confidence_score=overall_score,
                metadata={"validation_type": "risk_score"},
            )

        except Exception as e:
            self.logger.error("Risk score validation failed", error=str(e))
            return OutputValidationResult(
                is_valid=False,
                validated_output=risk_score,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def validate_compliance_mapping(
        self, mapping: Dict[str, Any]
    ) -> OutputValidationResult:
        """
        Validate compliance mapping output.

        Args:
            mapping: Compliance mapping to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        validated_output = mapping.copy()

        try:
            # Validate required fields
            required_fields = ["framework", "mappings", "compliance_score"]
            for field in required_fields:
                if field not in mapping:
                    errors.append(f"Missing required field: {field}")

            # Validate framework
            framework = mapping.get("framework")
            if framework:
                if not isinstance(framework, str):
                    errors.append("framework must be a string")
                elif framework not in self.valid_frameworks:
                    warnings.append(f"Unknown framework: {framework}")

            # Validate mappings
            mappings = mapping.get("mappings")
            if mappings:
                if not isinstance(mappings, (dict, list)):
                    errors.append("mappings must be a dictionary or list")
                elif isinstance(mappings, dict) and not mappings:
                    warnings.append("mappings dictionary is empty")
                elif isinstance(mappings, list) and not mappings:
                    warnings.append("mappings list is empty")

            # Validate compliance_score
            compliance_score = mapping.get("compliance_score")
            if compliance_score is not None:
                if not isinstance(compliance_score, (int, float)):
                    errors.append("compliance_score must be a number")
                elif not (0 <= compliance_score <= 1):
                    errors.append("compliance_score must be between 0 and 1")
                elif compliance_score < 0.7:
                    warnings.append("Low compliance score detected")

            # Validate gaps if present
            gaps = mapping.get("gaps", [])
            if gaps and not isinstance(gaps, list):
                errors.append("gaps must be a list")

            # Validate recommendations if present
            recommendations = mapping.get("recommendations", [])
            if recommendations:
                if not isinstance(recommendations, list):
                    errors.append("recommendations must be a list")
                elif not all(isinstance(rec, str) for rec in recommendations):
                    errors.append("all recommendations must be strings")

            return OutputValidationResult(
                is_valid=len(errors) == 0,
                validated_output=validated_output,
                errors=errors,
                warnings=warnings,
                confidence_score=compliance_score,
                metadata={"validation_type": "compliance_mapping"},
            )

        except Exception as e:
            self.logger.error("Compliance mapping validation failed", error=str(e))
            return OutputValidationResult(
                is_valid=False,
                validated_output=mapping,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def validate_pattern_analysis(
        self, analysis: Dict[str, Any]
    ) -> OutputValidationResult:
        """
        Validate pattern analysis output.

        Args:
            analysis: Pattern analysis to validate

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        validated_output = analysis.copy()

        try:
            # Validate required fields
            required_fields = ["confidence"]
            for field in required_fields:
                if field not in analysis:
                    errors.append(f"Missing required field: {field}")

            # Validate confidence
            confidence = analysis.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    errors.append("confidence must be a number")
                elif not (0 <= confidence <= 1):
                    errors.append("confidence must be between 0 and 1")

            # Validate pattern types
            pattern_fields = [
                "temporal_patterns",
                "frequency_patterns",
                "correlation_patterns",
                "anomaly_patterns",
            ]

            for field in pattern_fields:
                patterns = analysis.get(field, [])
                if patterns and not isinstance(patterns, list):
                    errors.append(f"{field} must be a list")

            # Check if at least one pattern type has data
            has_patterns = any(analysis.get(field) for field in pattern_fields)
            if not has_patterns:
                warnings.append("No patterns detected in analysis")

            return OutputValidationResult(
                is_valid=len(errors) == 0,
                validated_output=validated_output,
                errors=errors,
                warnings=warnings,
                confidence_score=confidence,
                metadata={"validation_type": "pattern_analysis"},
            )

        except Exception as e:
            self.logger.error("Pattern analysis validation failed", error=str(e))
            return OutputValidationResult(
                is_valid=False,
                validated_output=analysis,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def validate_output_consistency(self, outputs: List[Dict[str, Any]]) -> List[str]:
        """
        Validate consistency across multiple outputs.

        Args:
            outputs: List of outputs to check for consistency

        Returns:
            List of consistency warnings
        """
        warnings = []

        if len(outputs) < 2:
            return warnings

        try:
            # Check confidence score consistency
            confidences = [
                output.get("confidence")
                for output in outputs
                if output.get("confidence") is not None
            ]

            if len(confidences) > 1:
                confidence_range = max(confidences) - min(confidences)
                if confidence_range > 0.3:
                    warnings.append("Large confidence score variation detected")

            # Check risk level consistency
            risk_levels = [
                output.get("risk_level")
                for output in outputs
                if output.get("risk_level")
            ]

            if len(set(risk_levels)) > 1:
                warnings.append("Inconsistent risk levels across outputs")

        except Exception as e:
            self.logger.warning("Consistency validation failed", error=str(e))

        return warnings
