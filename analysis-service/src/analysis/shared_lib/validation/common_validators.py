"""Common validation functions following SRP.

This module provides ONLY reusable validation functions to avoid duplication.
Single Responsibility: Provide common validation logic for Pydantic models.
"""

from typing import Any, List
from pydantic import validator


def validate_non_empty_string(cls, v: Any) -> str:
    """Validate that a field is a non-empty string."""
    if not v or not isinstance(v, str) or not v.strip():
        raise ValueError("Field is required and must be a non-empty string")
    return v.strip()


def validate_non_empty_list(cls, v: Any) -> List[Any]:
    """Validate that a field is a non-empty list."""
    if not v or not isinstance(v, list):
        raise ValueError("Field is required and must be a non-empty list")
    return v


def validate_unique_list(cls, v: List[Any]) -> List[Any]:
    """Remove duplicates from a list while preserving order."""
    if not v:
        return v

    seen = set()
    unique_items = []
    for item in v:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def validate_confidence_score(cls, v: Any) -> float:
    """Validate confidence score is between 0.0 and 1.0."""
    if not isinstance(v, (int, float)):
        raise ValueError("Confidence score must be a number")
    if not 0.0 <= v <= 1.0:
        raise ValueError("Confidence score must be between 0.0 and 1.0")
    return float(v)


def validate_positive_number(cls, v: Any) -> float:
    """Validate that a number is positive."""
    if not isinstance(v, (int, float)):
        raise ValueError("Field must be a number")
    if v < 0:
        raise ValueError("Field must be positive")
    return float(v)


def validate_percentage(cls, v: Any) -> int:
    """Validate percentage is between 0 and 100."""
    if not isinstance(v, int):
        raise ValueError("Percentage must be an integer")
    if not 0 <= v <= 100:
        raise ValueError("Percentage must be between 0 and 100")
    return v


# Export validation functions
__all__ = [
    "validate_non_empty_string",
    "validate_non_empty_list",
    "validate_unique_list",
    "validate_confidence_score",
    "validate_positive_number",
    "validate_percentage",
]
