"""Schema validation utilities for enhanced validation."""

import json
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path
import logging

from pydantic import BaseModel, ValidationError as PydanticValidationError
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError

from ..exceptions.base import ValidationError
from ..utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class ValidationContext:
    """Context for validation operations."""

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        strict_mode: bool = True,
        allow_extra_fields: bool = False,
    ):
        self.tenant_id = tenant_id
        self.correlation_id = correlation_id or get_correlation_id()
        self.strict_mode = strict_mode
        self.allow_extra_fields = allow_extra_fields


class SchemaValidator:
    """Enhanced schema validator with multiple validation backends."""

    def __init__(self, schema_dir: Optional[Path] = None):
        self.schema_dir = schema_dir
        self._json_schemas: Dict[str, Dict[str, Any]] = {}
        self._pydantic_models: Dict[str, Type[BaseModel]] = {}

        if schema_dir and schema_dir.exists():
            self._load_json_schemas()

    def _load_json_schemas(self):
        """Load JSON schemas from schema directory."""
        if not self.schema_dir:
            return

        for schema_file in self.schema_dir.glob("*.json"):
            try:
                with open(schema_file, "r") as f:
                    schema = json.load(f)
                    schema_name = schema_file.stem
                    self._json_schemas[schema_name] = schema
                    logger.debug("Loaded JSON schema: %s", schema_name)
            except Exception as e:
                logger.error("Failed to load schema %s: %s", schema_file, e)

    def register_pydantic_model(self, name: str, model: Type[BaseModel]):
        """Register a Pydantic model for validation."""
        self._pydantic_models[name] = model
        logger.debug("Registered Pydantic model: %s", name)

    def validate_with_pydantic(
        self,
        data: Union[Dict[str, Any], BaseModel],
        model: Type[BaseModel],
        context: Optional[ValidationContext] = None,
    ) -> BaseModel:
        """Validate data using Pydantic model."""
        context = context or ValidationContext()

        try:
            if isinstance(data, dict):
                # Configure model based on context
                if not context.allow_extra_fields:
                    # Ensure model forbids extra fields
                    if hasattr(model.Config, "extra"):
                        original_extra = model.Config.extra
                    else:
                        original_extra = None

                    model.Config.extra = "forbid"

                    try:
                        validated = model(**data)
                    finally:
                        # Restore original config
                        if original_extra is not None:
                            model.Config.extra = original_extra
                        else:
                            delattr(model.Config, "extra")
                else:
                    validated = model(**data)

                return validated
            elif isinstance(data, model):
                return data
            else:
                raise ValidationError(
                    f"Data must be dict or {model.__name__} instance",
                    correlation_id=context.correlation_id,
                )

        except PydanticValidationError as e:
            logger.warning(
                "Pydantic validation failed for %s",
                model.__name__,
                extra={
                    "model": model.__name__,
                    "errors": e.errors(),
                    "correlation_id": context.correlation_id,
                    "tenant_id": context.tenant_id,
                },
            )
            raise ValidationError(
                f"Validation failed for {model.__name__}",
                error_code="PYDANTIC_VALIDATION_ERROR",
                details={"model": model.__name__, "errors": e.errors()},
                correlation_id=context.correlation_id,
            ) from e

    def validate_with_json_schema(
        self,
        data: Dict[str, Any],
        schema_name: str,
        context: Optional[ValidationContext] = None,
    ) -> Dict[str, Any]:
        """Validate data using JSON schema."""
        context = context or ValidationContext()

        if schema_name not in self._json_schemas:
            raise ValidationError(
                f"JSON schema '{schema_name}' not found",
                error_code="SCHEMA_NOT_FOUND",
                correlation_id=context.correlation_id,
            )

        schema = self._json_schemas[schema_name]

        try:
            validate(instance=data, schema=schema)
            return data
        except JsonSchemaValidationError as e:
            logger.warning(
                "JSON schema validation failed for %s",
                schema_name,
                extra={
                    "schema": schema_name,
                    "error": str(e),
                    "correlation_id": context.correlation_id,
                    "tenant_id": context.tenant_id,
                },
            )
            raise ValidationError(
                f"JSON schema validation failed for {schema_name}",
                error_code="JSON_SCHEMA_VALIDATION_ERROR",
                details={
                    "schema": schema_name,
                    "error": str(e),
                    "path": list(e.path) if e.path else [],
                },
                correlation_id=context.correlation_id,
            ) from e

    def validate_request(
        self,
        data: Union[Dict[str, Any], BaseModel],
        model_name: str,
        context: Optional[ValidationContext] = None,
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Validate request data using registered model or schema."""
        context = context or ValidationContext()

        # Try Pydantic model first
        if model_name in self._pydantic_models:
            model = self._pydantic_models[model_name]
            return self.validate_with_pydantic(data, model, context)

        # Fall back to JSON schema
        if model_name in self._json_schemas and isinstance(data, dict):
            return self.validate_with_json_schema(data, model_name, context)

        raise ValidationError(
            f"No validator found for '{model_name}'",
            error_code="VALIDATOR_NOT_FOUND",
            correlation_id=context.correlation_id,
        )

    def validate_response(
        self,
        data: Union[Dict[str, Any], BaseModel],
        model_name: str,
        context: Optional[ValidationContext] = None,
    ) -> Union[BaseModel, Dict[str, Any]]:
        """Validate response data using registered model or schema."""
        # Response validation is the same as request validation
        return self.validate_request(data, model_name, context)

    def validate_list(
        self,
        data: List[Union[Dict[str, Any], BaseModel]],
        model_name: str,
        context: Optional[ValidationContext] = None,
    ) -> List[Union[BaseModel, Dict[str, Any]]]:
        """Validate a list of items."""
        context = context or ValidationContext()
        validated_items = []
        errors = []

        for i, item in enumerate(data):
            try:
                validated_item = self.validate_request(item, model_name, context)
                validated_items.append(validated_item)
            except ValidationError as e:
                errors.append({"index": i, "error": e.to_dict()})

        if errors:
            raise ValidationError(
                f"List validation failed for {model_name}",
                error_code="LIST_VALIDATION_ERROR",
                details={
                    "model": model_name,
                    "total_items": len(data),
                    "failed_items": len(errors),
                    "errors": errors,
                },
                correlation_id=context.correlation_id,
            )

        return validated_items

    def get_schema_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a registered schema or model."""
        info = {"name": model_name, "type": None, "available": False}

        if model_name in self._pydantic_models:
            model = self._pydantic_models[model_name]
            info.update(
                {"type": "pydantic", "available": True, "schema": model.schema()}
            )
        elif model_name in self._json_schemas:
            info.update(
                {
                    "type": "json_schema",
                    "available": True,
                    "schema": self._json_schemas[model_name],
                }
            )

        return info

    def list_available_schemas(self) -> List[str]:
        """List all available schema names."""
        return list(set(self._pydantic_models.keys()) | set(self._json_schemas.keys()))


# Global validator instance
default_validator = SchemaValidator()


def validate_with_schema(
    data: Union[Dict[str, Any], BaseModel],
    schema_name: str,
    context: Optional[ValidationContext] = None,
) -> Union[BaseModel, Dict[str, Any]]:
    """Validate data using the default schema validator."""
    return default_validator.validate_request(data, schema_name, context)
