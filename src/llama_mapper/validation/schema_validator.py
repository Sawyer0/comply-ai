"""
Schema-First Output Validation for Compliance AI Models

Strict JSON schema validation with template forcing for reliable outputs.
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import jsonschema
from jsonschema import ValidationError, validate


class ModelType(Enum):
    """Model types for schema validation."""

    MAPPER = "mapper"
    ANALYST = "analyst"


@dataclass
class SchemaValidationConfig:
    """Configuration for schema validation."""

    max_reason_length: int = 120  # Max 120 chars for reason field
    max_remediation_length: int = 200  # Max 200 chars for remediation
    max_opa_diff_length: int = 100  # Max 100 chars for OPA diff
    strict_mode: bool = True  # Fail on any validation error
    allow_retry: bool = True  # Allow one retry with schema reminder


class MapperSchemaValidator:
    """Schema validator for Llama-3-8B Mapper outputs."""

    def __init__(self, config: SchemaValidationConfig):
        self.config = config
        self.schema = self._get_mapper_schema()

    def _get_mapper_schema(self) -> Dict[str, Any]:
        """Get the strict schema for Mapper outputs."""
        return {
            "type": "object",
            "required": ["taxonomy", "scores", "confidence"],
            "properties": {
                "taxonomy": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": r"^[A-Z]+\.[A-Za-z]+(\.[A-Za-z]+)*$",
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
                "scores": {
                    "type": "object",
                    "patternProperties": {
                        r"^[A-Z]+\.[A-Za-z]+(\.[A-Za-z]+)*$": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        }
                    },
                    "minProperties": 1,
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 5,
                },
                "reasoning_text": {"type": "string", "maxLength": 200},
                "provenance": {
                    "type": "object",
                    "properties": {
                        "detector": {"type": "string"},
                        "source": {"type": "string"},
                    },
                },
                "notes": {"type": "string", "maxLength": 100},
            },
            "additionalProperties": False,
        }

    def validate(self, output: Union[str, Dict]) -> Dict[str, Any]:
        """Validate Mapper output against schema."""
        try:
            # Parse JSON if string
            if isinstance(output, str):
                parsed_output = json.loads(output)
            else:
                parsed_output = output

            # Validate against schema
            validate(instance=parsed_output, schema=self.schema)

            # Additional business logic validation
            self._validate_business_logic(parsed_output)

            return {
                "valid": True,
                "errors": [],
                "warnings": [],
                "parsed_output": parsed_output,
            }

        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"JSON decode error: {str(e)}"],
                "warnings": [],
                "parsed_output": None,
            }
        except ValidationError as e:
            return {
                "valid": False,
                "errors": [f"Schema validation error: {str(e)}"],
                "warnings": [],
                "parsed_output": None,
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "parsed_output": None,
            }

    def _validate_business_logic(self, output: Dict[str, Any]):
        """Validate business logic constraints."""
        # Check that scores match taxonomy
        taxonomy = output.get("taxonomy", [])
        scores = output.get("scores", {})

        for tax_item in taxonomy:
            if tax_item not in scores:
                raise ValueError(f"Taxonomy item {tax_item} missing from scores")

        # Check confidence is reasonable
        confidence = output.get("confidence", 0)
        if confidence < 0.5:
            raise ValueError(f"Confidence {confidence} too low for production use")

        # Check score consistency
        for tax_item, score in scores.items():
            if tax_item in taxonomy and score < 0.7:
                raise ValueError(
                    f"Score {score} for {tax_item} too low for production use"
                )


class AnalystSchemaValidator:
    """Schema validator for Phi-3 Analyst outputs."""

    def __init__(self, config: SchemaValidationConfig):
        self.config = config
        self.schema = self._get_analyst_schema()

    def _get_analyst_schema(self) -> Dict[str, Any]:
        """Get the strict schema for Analyst outputs."""
        return {
            "type": "object",
            "required": ["analysis_type", "risk_level", "recommendations"],
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": [
                        "GDPR_Compliance_Analysis",
                        "Legal_Compliance_Analysis",
                        "Policy_Compliance_Analysis",
                        "Stakeholder_Engagement_Analysis",
                        "Multi_Framework_Compliance_Analysis",
                        "policy_violation",
                        "privacy_risk",
                        "security_risk",
                        "compliance_gap",
                    ],
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "recommendations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
                "reason": {
                    "type": "string",
                    "maxLength": self.config.max_reason_length,
                },
                "remediation": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
                "opa_diff": {
                    "type": "object",
                    "properties": {
                        "before": {
                            "type": "string",
                            "maxLength": self.config.max_opa_diff_length,
                        },
                        "after": {
                            "type": "string",
                            "maxLength": self.config.max_opa_diff_length,
                        },
                    },
                },
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "affected_data": {"type": "string"},
                "applicable_frameworks": {"type": "array", "items": {"type": "string"}},
                "compliance_gaps": {"type": "array", "items": {"type": "string"}},
                "unified_strategy": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        }

    def validate(self, output: Union[str, Dict]) -> Dict[str, Any]:
        """Validate Analyst output against schema."""
        try:
            # Parse JSON if string
            if isinstance(output, str):
                parsed_output = json.loads(output)
            else:
                parsed_output = output

            # Validate against schema
            validate(instance=parsed_output, schema=self.schema)

            # Additional business logic validation
            self._validate_business_logic(parsed_output)

            return {
                "valid": True,
                "errors": [],
                "warnings": [],
                "parsed_output": parsed_output,
            }

        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"JSON decode error: {str(e)}"],
                "warnings": [],
                "parsed_output": None,
            }
        except ValidationError as e:
            return {
                "valid": False,
                "errors": [f"Schema validation error: {str(e)}"],
                "warnings": [],
                "parsed_output": None,
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "parsed_output": None,
            }

    def _validate_business_logic(self, output: Dict[str, Any]):
        """Validate business logic constraints."""
        # Check reason length
        reason = output.get("reason", "")
        if len(reason) > self.config.max_reason_length:
            raise ValueError(
                f"Reason too long: {len(reason)} > {self.config.max_reason_length}"
            )

        # Check remediation length
        remediation = output.get("remediation", [])
        for item in remediation:
            if len(item) > self.config.max_remediation_length:
                raise ValueError(
                    f"Remediation item too long: {len(item)} > {self.config.max_remediation_length}"
                )

        # Check OPA diff length
        opa_diff = output.get("opa_diff", {})
        if opa_diff:
            for key in ["before", "after"]:
                if (
                    key in opa_diff
                    and len(opa_diff[key]) > self.config.max_opa_diff_length
                ):
                    raise ValueError(
                        f"OPA diff {key} too long: {len(opa_diff[key])} > {self.config.max_opa_diff_length}"
                    )


class SchemaValidatorFactory:
    """Factory for creating schema validators."""

    @staticmethod
    def create_validator(
        model_type: ModelType, config: SchemaValidationConfig
    ) -> Union[MapperSchemaValidator, AnalystSchemaValidator]:
        """Create appropriate validator for model type."""
        if model_type == ModelType.MAPPER:
            return MapperSchemaValidator(config)
        elif model_type == ModelType.ANALYST:
            return AnalystSchemaValidator(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class RejectSamplingHandler:
    """Handles reject sampling for schema validation failures."""

    def __init__(self, config: SchemaValidationConfig):
        self.config = config
        self.max_retries = 1 if config.allow_retry else 0

    def handle_validation_failure(
        self,
        model_type: ModelType,
        original_output: str,
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle validation failure with retry logic."""
        if not validation_result["valid"] and self.max_retries > 0:
            # Create retry prompt with schema reminder
            retry_prompt = self._create_retry_prompt(
                model_type, validation_result["errors"]
            )

            return {
                "should_retry": True,
                "retry_prompt": retry_prompt,
                "original_errors": validation_result["errors"],
                "retry_count": 1,
            }

        return {
            "should_retry": False,
            "retry_prompt": None,
            "original_errors": validation_result["errors"],
            "retry_count": 0,
        }

    def _create_retry_prompt(self, model_type: ModelType, errors: List[str]) -> str:
        """Create retry prompt with schema reminder."""
        if model_type == ModelType.MAPPER:
            return f"""JSON-only response required. Fix these errors: {', '.join(errors)}

Required format:
{{
  "taxonomy": ["PII.Contact.Email"],
  "scores": {{"PII.Contact.Email": 0.95}},
  "confidence": 0.95
}}"""
        else:  # ANALYST
            return f"""JSON-only response required. Fix these errors: {', '.join(errors)}

Required format:
{{
  "analysis_type": "privacy_risk",
  "risk_level": "medium", 
  "recommendations": ["Implement log sanitization"],
  "reason": "PII detected in logs (max 120 chars)"
}}"""


def validate_model_output(
    model_type: ModelType,
    output: Union[str, Dict],
    config: Optional[SchemaValidationConfig] = None,
) -> Dict[str, Any]:
    """Convenience function for validating model outputs."""
    if config is None:
        config = SchemaValidationConfig()

    validator = SchemaValidatorFactory.create_validator(model_type, config)
    validation_result = validator.validate(output)

    # Handle reject sampling if validation failed
    if not validation_result["valid"]:
        handler = RejectSamplingHandler(config)
        retry_info = handler.handle_validation_failure(
            model_type, str(output), validation_result
        )
        validation_result.update(retry_info)

    return validation_result


# Example usage and testing
if __name__ == "__main__":
    # Test Mapper validation
    mapper_output = {
        "taxonomy": ["PII.Contact.Email"],
        "scores": {"PII.Contact.Email": 0.95},
        "confidence": 0.95,
    }

    mapper_result = validate_model_output(ModelType.MAPPER, mapper_output)
    print(f"Mapper validation: {mapper_result['valid']}")

    # Test Analyst validation
    analyst_output = {
        "analysis_type": "privacy_risk",
        "risk_level": "medium",
        "recommendations": ["Implement log sanitization"],
        "reason": "PII detected in logs",
    }

    analyst_result = validate_model_output(ModelType.ANALYST, analyst_output)
    print(f"Analyst validation: {analyst_result['valid']}")
