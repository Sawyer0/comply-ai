"""
Input validation for Analysis Service.

Validates and sanitizes inputs for risk assessment and compliance analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

try:
    from ..shared_integration import get_shared_logger

    logger = get_shared_logger(__name__)
except ImportError:
    # Fallback to standard logging if shared integration is not available
    import logging

    logger = logging.getLogger(__name__)


@dataclass
class InputValidationResult:
    """Result of input validation."""

    is_valid: bool
    sanitized_input: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class InputValidator:
    """Validates inputs for analysis service operations."""

    def __init__(self):
        self.logger = logger.bind(component="input_validator")

        # Define required fields for different analysis types
        self.required_fields = {
            "risk_assessment": ["findings", "context"],
            "pattern_analysis": ["data_points", "time_range"],
            "compliance_mapping": ["findings", "framework"],
            "rag_analysis": ["query", "context"],
        }

        # Define valid frameworks
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

        # Define valid risk levels
        self.valid_risk_levels = {"low", "medium", "high", "critical"}

        # Define valid analysis types
        self.valid_analysis_types = {
            "risk_assessment",
            "pattern_analysis",
            "compliance_mapping",
            "rag_analysis",
            "statistical_analysis",
        }

    def validate_analysis_request(
        self, request_data: Dict[str, Any]
    ) -> InputValidationResult:
        """
        Validate an analysis request.

        Args:
            request_data: Analysis request data

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        sanitized_input = request_data.copy()

        try:
            # Validate tenant_id
            if not request_data.get("tenant_id"):
                errors.append("tenant_id is required")
            elif not isinstance(request_data["tenant_id"], str):
                errors.append("tenant_id must be a string")
            elif len(request_data["tenant_id"]) > 100:
                errors.append("tenant_id too long (max 100 characters)")

            # Validate analysis_types
            analysis_types = request_data.get("analysis_types", [])
            if not analysis_types:
                errors.append("analysis_types is required and cannot be empty")
            elif not isinstance(analysis_types, list):
                errors.append("analysis_types must be a list")
            else:
                invalid_types = [
                    t for t in analysis_types if t not in self.valid_analysis_types
                ]
                if invalid_types:
                    errors.append(f"Invalid analysis types: {invalid_types}")

            # Validate frameworks if provided
            frameworks = request_data.get("frameworks", [])
            if frameworks:
                if not isinstance(frameworks, list):
                    errors.append("frameworks must be a list")
                else:
                    invalid_frameworks = [
                        f for f in frameworks if f not in self.valid_frameworks
                    ]
                    if invalid_frameworks:
                        warnings.append(f"Unknown frameworks: {invalid_frameworks}")

            # Validate orchestration_response_id
            orchestration_response_id = request_data.get("orchestration_response_id")
            if orchestration_response_id and not isinstance(
                orchestration_response_id, str
            ):
                errors.append("orchestration_response_id must be a string")

            # Validate include_recommendations
            include_recommendations = request_data.get("include_recommendations", True)
            if not isinstance(include_recommendations, bool):
                sanitized_input["include_recommendations"] = bool(
                    include_recommendations
                )
                warnings.append("include_recommendations converted to boolean")

            # Validate correlation_id if provided
            correlation_id = request_data.get("correlation_id")
            if correlation_id and not isinstance(correlation_id, str):
                errors.append("correlation_id must be a string")

            # Type-specific validation
            for analysis_type in analysis_types:
                if analysis_type in self.required_fields:
                    type_errors = self._validate_analysis_type_data(
                        analysis_type, request_data
                    )
                    errors.extend(type_errors)

            return InputValidationResult(
                is_valid=len(errors) == 0,
                sanitized_input=sanitized_input,
                errors=errors,
                warnings=warnings,
                metadata={"validation_timestamp": self._get_timestamp()},
            )

        except Exception as e:
            self.logger.error("Input validation failed", error=str(e))
            return InputValidationResult(
                is_valid=False,
                sanitized_input=request_data,
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )

    def validate_risk_assessment_input(
        self, input_data: Dict[str, Any]
    ) -> InputValidationResult:
        """
        Validate risk assessment specific input.

        Args:
            input_data: Risk assessment input data

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        sanitized_input = input_data.copy()

        # Validate findings
        findings = input_data.get("findings")
        if not findings:
            errors.append("findings is required for risk assessment")
        elif not isinstance(findings, (dict, list)):
            errors.append("findings must be a dictionary or list")

        # Validate context
        context = input_data.get("context")
        if context and not isinstance(context, str):
            errors.append("context must be a string")
        elif context and len(context) > 10000:
            warnings.append("context is very long, may affect performance")

        # Validate framework
        framework = input_data.get("framework")
        if framework:
            if not isinstance(framework, str):
                errors.append("framework must be a string")
            elif framework not in self.valid_frameworks:
                warnings.append(f"Unknown framework: {framework}")

        # Validate existing risk level if provided
        existing_risk_level = input_data.get("existing_risk_level")
        if existing_risk_level:
            if existing_risk_level not in self.valid_risk_levels:
                errors.append(f"Invalid risk level: {existing_risk_level}")

        return InputValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized_input,
            errors=errors,
            warnings=warnings,
            metadata={"validation_type": "risk_assessment"},
        )

    def validate_pattern_analysis_input(
        self, input_data: Dict[str, Any]
    ) -> InputValidationResult:
        """
        Validate pattern analysis input.

        Args:
            input_data: Pattern analysis input data

        Returns:
            Validation result
        """
        errors = []
        warnings = []
        sanitized_input = input_data.copy()

        # Validate data_points
        data_points = input_data.get("data_points")
        if not data_points:
            errors.append("data_points is required for pattern analysis")
        elif not isinstance(data_points, list):
            errors.append("data_points must be a list")
        elif len(data_points) < 2:
            errors.append("data_points must contain at least 2 points")

        # Validate time_range
        time_range = input_data.get("time_range")
        if time_range:
            if not isinstance(time_range, dict):
                errors.append("time_range must be a dictionary")
            else:
                if "start" not in time_range or "end" not in time_range:
                    errors.append("time_range must contain 'start' and 'end'")

        # Validate pattern_types if provided
        pattern_types = input_data.get("pattern_types", [])
        if pattern_types and not isinstance(pattern_types, list):
            errors.append("pattern_types must be a list")

        return InputValidationResult(
            is_valid=len(errors) == 0,
            sanitized_input=sanitized_input,
            errors=errors,
            warnings=warnings,
            metadata={"validation_type": "pattern_analysis"},
        )

    def _validate_analysis_type_data(
        self, analysis_type: str, request_data: Dict[str, Any]
    ) -> List[str]:
        """Validate data specific to analysis type."""
        errors = []

        required_fields = self.required_fields.get(analysis_type, [])

        for field in required_fields:
            if field not in request_data:
                errors.append(f"{field} is required for {analysis_type}")

        return errors

    def sanitize_string_input(self, text: str, max_length: int = 10000) -> str:
        """
        Sanitize string input.

        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length

        Returns:
            Sanitized text
        """
        if not isinstance(text, str):
            return str(text)

        # Remove null bytes
        text = text.replace("\x00", "")

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]

        # Basic HTML escaping
        text = text.replace("<", "&lt;").replace(">", "&gt;")

        return text.strip()

    def validate_json_structure(
        self, data: Any, required_keys: Set[str] = None, optional_keys: Set[str] = None
    ) -> List[str]:
        """
        Validate JSON structure.

        Args:
            data: Data to validate
            required_keys: Set of required keys
            optional_keys: Set of optional keys

        Returns:
            List of validation errors
        """
        errors = []

        if not isinstance(data, dict):
            errors.append("Data must be a dictionary")
            return errors

        if required_keys:
            missing_keys = required_keys - set(data.keys())
            if missing_keys:
                errors.append(f"Missing required keys: {missing_keys}")

        if required_keys and optional_keys:
            allowed_keys = required_keys | optional_keys
            extra_keys = set(data.keys()) - allowed_keys
            if extra_keys:
                errors.append(f"Unexpected keys: {extra_keys}")

        return errors

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime

        return datetime.utcnow().isoformat()
