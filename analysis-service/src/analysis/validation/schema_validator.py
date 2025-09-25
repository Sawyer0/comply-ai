"""
Schema validation for Analysis Service.

Validates data against predefined schemas for different analysis types.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    from ..shared_integration import get_shared_logger

    logger = get_shared_logger(__name__)
except ImportError:
    # Fallback to standard logging if shared integration is not available
    import logging

    logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Supported analysis types."""

    RISK_ASSESSMENT = "risk_assessment"
    PATTERN_ANALYSIS = "pattern_analysis"
    COMPLIANCE_MAPPING = "compliance_mapping"
    RAG_ANALYSIS = "rag_analysis"
    STATISTICAL_ANALYSIS = "statistical_analysis"


@dataclass
class SchemaValidationConfig:
    """Configuration for schema validation."""

    strict_mode: bool = True
    allow_additional_properties: bool = False
    validate_formats: bool = True
    max_string_length: int = 10000
    max_array_length: int = 1000


class SchemaValidator:
    """Base schema validator."""

    def __init__(self, config: SchemaValidationConfig = None):
        self.config = config or SchemaValidationConfig()
        self.logger = logger.bind(component="schema_validator")

    def validate_against_schema(
        self, data: Any, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate data against a JSON schema.

        Args:
            data: Data to validate
            schema: JSON schema to validate against

        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []

        try:
            # Basic type validation
            expected_type = schema.get("type")
            if expected_type:
                type_valid = self._validate_type(data, expected_type)
                if not type_valid:
                    errors.append(
                        f"Expected type {expected_type}, got {type(data).__name__}"
                    )
                    return {"valid": False, "errors": errors, "warnings": warnings}

            # Validate properties for objects
            if expected_type == "object" and isinstance(data, dict):
                prop_validation = self._validate_object_properties(data, schema)
                errors.extend(prop_validation["errors"])
                warnings.extend(prop_validation["warnings"])

            # Validate array items
            elif expected_type == "array" and isinstance(data, list):
                array_validation = self._validate_array_items(data, schema)
                errors.extend(array_validation["errors"])
                warnings.extend(array_validation["warnings"])

            # Validate string constraints
            elif expected_type == "string" and isinstance(data, str):
                string_validation = self._validate_string_constraints(data, schema)
                errors.extend(string_validation["errors"])
                warnings.extend(string_validation["warnings"])

            # Validate number constraints
            elif expected_type in ["number", "integer"] and isinstance(
                data, (int, float)
            ):
                number_validation = self._validate_number_constraints(data, schema)
                errors.extend(number_validation["errors"])
                warnings.extend(number_validation["warnings"])

            return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

        except Exception as e:
            self.logger.error("Schema validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Schema validation error: {str(e)}"],
                "warnings": [],
            }

    def _validate_type(self, data: Any, expected_type: str) -> bool:
        """Validate data type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return False

        return isinstance(data, expected_python_type)

    def _validate_object_properties(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate object properties."""
        errors = []
        warnings = []

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_properties = schema.get("additionalProperties", True)

        # Check required properties
        for prop in required:
            if prop not in data:
                errors.append(f"Missing required property: {prop}")

        # Validate existing properties
        for prop, value in data.items():
            if prop in properties:
                prop_schema = properties[prop]
                prop_validation = self.validate_against_schema(value, prop_schema)

                for error in prop_validation["errors"]:
                    errors.append(f"Property '{prop}': {error}")

                for warning in prop_validation["warnings"]:
                    warnings.append(f"Property '{prop}': {warning}")

            elif not additional_properties and self.config.strict_mode:
                errors.append(f"Additional property not allowed: {prop}")
            elif not additional_properties:
                warnings.append(f"Unexpected property: {prop}")

        return {"errors": errors, "warnings": warnings}

    def _validate_array_items(
        self, data: List[Any], schema: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate array items."""
        errors = []
        warnings = []

        # Check array length constraints
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")

        if min_items is not None and len(data) < min_items:
            errors.append(f"Array too short: {len(data)} < {min_items}")

        if max_items is not None and len(data) > max_items:
            errors.append(f"Array too long: {len(data)} > {max_items}")

        # Validate items schema
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_validation = self.validate_against_schema(item, items_schema)

                for error in item_validation["errors"]:
                    errors.append(f"Item {i}: {error}")

                for warning in item_validation["warnings"]:
                    warnings.append(f"Item {i}: {warning}")

        return {"errors": errors, "warnings": warnings}

    def _validate_string_constraints(
        self, data: str, schema: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate string constraints."""
        errors = []
        warnings = []

        # Length constraints
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")

        if min_length is not None and len(data) < min_length:
            errors.append(f"String too short: {len(data)} < {min_length}")

        if max_length is not None and len(data) > max_length:
            errors.append(f"String too long: {len(data)} > {max_length}")

        # Pattern validation
        pattern = schema.get("pattern")
        if pattern and self.config.validate_formats:
            import re

            if not re.match(pattern, data):
                errors.append(f"String does not match pattern: {pattern}")

        # Format validation
        format_type = schema.get("format")
        if format_type and self.config.validate_formats:
            format_validation = self._validate_string_format(data, format_type)
            errors.extend(format_validation["errors"])
            warnings.extend(format_validation["warnings"])

        return {"errors": errors, "warnings": warnings}

    def _validate_number_constraints(
        self, data: Union[int, float], schema: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Validate number constraints."""
        errors = []
        warnings = []

        # Range constraints
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")

        if minimum is not None and data < minimum:
            errors.append(f"Number too small: {data} < {minimum}")

        if maximum is not None and data > maximum:
            errors.append(f"Number too large: {data} > {maximum}")

        if exclusive_minimum is not None and data <= exclusive_minimum:
            errors.append(f"Number must be greater than: {exclusive_minimum}")

        if exclusive_maximum is not None and data >= exclusive_maximum:
            errors.append(f"Number must be less than: {exclusive_maximum}")

        # Multiple of constraint
        multiple_of = schema.get("multipleOf")
        if multiple_of is not None and data % multiple_of != 0:
            errors.append(f"Number is not a multiple of: {multiple_of}")

        return {"errors": errors, "warnings": warnings}

    def _validate_string_format(
        self, data: str, format_type: str
    ) -> Dict[str, List[str]]:
        """Validate string format."""
        errors = []
        warnings = []

        if format_type == "email":
            import re

            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            if not re.match(email_pattern, data):
                errors.append("Invalid email format")

        elif format_type == "uuid":
            import re

            uuid_pattern = (
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            )
            if not re.match(uuid_pattern, data, re.IGNORECASE):
                errors.append("Invalid UUID format")

        elif format_type == "date-time":
            try:
                from datetime import datetime

                datetime.fromisoformat(data.replace("Z", "+00:00"))
            except ValueError:
                errors.append("Invalid date-time format")

        elif format_type == "uri":
            import re

            uri_pattern = r"^https?://[^\s/$.?#].[^\s]*$"
            if not re.match(uri_pattern, data):
                errors.append("Invalid URI format")

        return {"errors": errors, "warnings": warnings}


class AnalysisSchemaValidator(SchemaValidator):
    """Schema validator specifically for analysis service data."""

    def __init__(self, config: SchemaValidationConfig = None):
        super().__init__(config)
        self.schemas = self._load_analysis_schemas()

    def _load_analysis_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined schemas for analysis types."""
        return {
            "risk_assessment_input": {
                "type": "object",
                "required": ["findings", "context"],
                "properties": {
                    "findings": {
                        "type": ["object", "array"],
                        "description": "Security findings to assess",
                    },
                    "context": {
                        "type": "string",
                        "maxLength": 10000,
                        "description": "Context for risk assessment",
                    },
                    "framework": {
                        "type": "string",
                        "enum": ["SOC2", "ISO27001", "HIPAA", "GDPR", "General"],
                    },
                },
            },
            "risk_assessment_output": {
                "type": "object",
                "required": ["overall_risk_score", "risk_factors"],
                "properties": {
                    "overall_risk_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "technical_risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "business_risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "regulatory_risk": {"type": "number", "minimum": 0, "maximum": 1},
                    "risk_factors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "weight", "value"],
                            "properties": {
                                "name": {"type": "string"},
                                "weight": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "value": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                },
            },
            "canonical_result": {
                "type": "object",
                "required": ["category", "subcategory", "confidence", "risk_level"],
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": [
                            "pii",
                            "security",
                            "compliance",
                            "data_quality",
                            "privacy",
                        ],
                    },
                    "subcategory": {"type": "string", "maxLength": 100},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
        }

    def validate_analysis_input(
        self, data: Dict[str, Any], analysis_type: AnalysisType
    ) -> Dict[str, Any]:
        """
        Validate analysis input data.

        Args:
            data: Input data to validate
            analysis_type: Type of analysis

        Returns:
            Validation result
        """
        schema_key = f"{analysis_type.value}_input"
        schema = self.schemas.get(schema_key)

        if not schema:
            return {
                "valid": False,
                "errors": [f"No schema found for {analysis_type.value}"],
                "warnings": [],
            }

        return self.validate_against_schema(data, schema)

    def validate_analysis_output(
        self, data: Dict[str, Any], analysis_type: AnalysisType
    ) -> Dict[str, Any]:
        """
        Validate analysis output data.

        Args:
            data: Output data to validate
            analysis_type: Type of analysis

        Returns:
            Validation result
        """
        schema_key = f"{analysis_type.value}_output"
        schema = self.schemas.get(schema_key)

        if not schema:
            return {
                "valid": False,
                "errors": [f"No output schema found for {analysis_type.value}"],
                "warnings": [],
            }

        return self.validate_against_schema(data, schema)


def validate_analysis_output(
    data: Dict[str, Any], analysis_type: str, config: SchemaValidationConfig = None
) -> Dict[str, Any]:
    """
    Convenience function to validate analysis output.

    Args:
        data: Data to validate
        analysis_type: Type of analysis
        config: Optional validation configuration

    Returns:
        Validation result
    """
    try:
        analysis_enum = AnalysisType(analysis_type)
        validator = AnalysisSchemaValidator(config)
        return validator.validate_analysis_output(data, analysis_enum)
    except ValueError:
        return {
            "valid": False,
            "errors": [f"Unknown analysis type: {analysis_type}"],
            "warnings": [],
        }
