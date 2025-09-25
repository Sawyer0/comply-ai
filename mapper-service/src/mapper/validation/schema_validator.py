"""
Schema validation for model outputs.

Single responsibility: JSON schema validation for model responses.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import jsonschema
from jsonschema import ValidationError, validate

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Model types for schema validation."""

    MAPPER = "mapper"


@dataclass
class SchemaValidationConfig:
    """Configuration for schema validation."""

    max_reason_length: int = 120
    strict_mode: bool = True
    allow_retry: bool = True


class SchemaValidator:
    """
    Base schema validator.

    Single responsibility: Schema validation logic.
    """

    def __init__(self, schema: Dict[str, Any], config: SchemaValidationConfig):
        self.schema = schema
        self.config = config

    def validate(self, output: Union[str, Dict]) -> Dict[str, Any]:
        """Validate output against schema."""
        try:
            # Parse JSON if string
            if isinstance(output, str):
                parsed_output = json.loads(output)
            else:
                parsed_output = output

            # Validate against schema
            validate(instance=parsed_output, schema=self.schema)

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


class MapperSchemaValidator(SchemaValidator):
    """
    Schema validator for Mapper outputs.

    Single responsibility: Mapper-specific schema validation.
    """

    def __init__(self, config: SchemaValidationConfig):
        schema = self._get_mapper_schema()
        super().__init__(schema, config)

    def _get_mapper_schema(self) -> Dict[str, Any]:
        """Get the schema for Mapper outputs."""
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
                "notes": {"type": "string", "maxLength": 200},
            },
            "additionalProperties": False,
        }


def validate_model_output(
    model_type: ModelType,
    output: Union[str, Dict],
    config: Optional[SchemaValidationConfig] = None,
) -> Dict[str, Any]:
    """
    Convenience function for validating model outputs.

    Single responsibility: Factory function for validation.
    """
    if config is None:
        config = SchemaValidationConfig()

    if model_type == ModelType.MAPPER:
        validator = MapperSchemaValidator(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return validator.validate(output)
