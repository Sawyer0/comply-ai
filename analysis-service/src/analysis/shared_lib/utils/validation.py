"""Validation utilities for request/response models."""

from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError as PydanticValidationError
import logging

from ..exceptions.base import ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def validate_request(data: Dict[str, Any], model_class: Type[T]) -> T:
    """Validate request data against a Pydantic model."""
    try:
        return model_class(**data)
    except PydanticValidationError as e:
        logger.warning(
            "Request validation failed",
            extra={
                "model": model_class.__name__,
                "errors": e.errors(),
                "data_keys": list(data.keys()) if data else [],
            },
        )
        raise ValidationError(
            message=f"Request validation failed for {model_class.__name__}",
            error_code="VALIDATION_ERROR",
            details={"model": model_class.__name__, "errors": e.errors()},
        ) from e


def validate_response(data: Dict[str, Any], model_class: Type[T]) -> T:
    """Validate response data against a Pydantic model."""
    try:
        return model_class(**data)
    except PydanticValidationError as e:
        logger.error(
            "Response validation failed",
            extra={
                "model": model_class.__name__,
                "errors": e.errors(),
                "data_keys": list(data.keys()) if data else [],
            },
        )
        raise ValidationError(
            message=f"Response validation failed for {model_class.__name__}",
            error_code="RESPONSE_VALIDATION_ERROR",
            details={"model": model_class.__name__, "errors": e.errors()},
        ) from e


def validate_list(data: List[Dict[str, Any]], model_class: Type[T]) -> List[T]:
    """Validate a list of data against a Pydantic model."""
    validated_items = []
    errors = []

    for i, item in enumerate(data):
        try:
            validated_items.append(model_class(**item))
        except PydanticValidationError as e:
            errors.append({"index": i, "errors": e.errors()})

    if errors:
        logger.warning(
            "List validation failed",
            extra={
                "model": model_class.__name__,
                "total_items": len(data),
                "failed_items": len(errors),
                "errors": errors,
            },
        )
        raise ValidationError(
            message=f"List validation failed for {model_class.__name__}",
            error_code="LIST_VALIDATION_ERROR",
            details={
                "model": model_class.__name__,
                "total_items": len(data),
                "failed_items": len(errors),
                "errors": errors,
            },
        )

    return validated_items


def validate_optional(
    data: Optional[Dict[str, Any]], model_class: Type[T]
) -> Optional[T]:
    """Validate optional data against a Pydantic model."""
    if data is None:
        return None
    return validate_request(data, model_class)


def extract_validation_errors(e: PydanticValidationError) -> List[Dict[str, Any]]:
    """Extract validation errors in a standardized format."""
    errors = []
    for error in e.errors():
        errors.append(
            {
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input"),
            }
        )
    return errors
