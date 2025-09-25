"""
Response validation for mapping outputs.

Single responsibility: Validate mapping responses against business rules and schemas.
"""

import logging
from typing import Any, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ResponseValidator:
    """
    Validates mapping responses against business rules.

    Single responsibility: Response validation only.
    """

    def __init__(self, min_confidence: float = 0.0):
        """
        Initialize response validator.

        Args:
            min_confidence: Minimum acceptable confidence score
        """
        self.min_confidence = min_confidence

    def validate(self, response: "MappingResponse") -> ValidationResult:
        """
        Validate a mapping response.

        Args:
            response: Mapping response to validate

        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        errors = []
        warnings = []

        # Validate taxonomy format
        taxonomy_errors = self._validate_taxonomy(response.taxonomy)
        errors.extend(taxonomy_errors)

        # Validate scores consistency
        score_errors = self._validate_scores(response.taxonomy, response.scores)
        errors.extend(score_errors)

        # Validate confidence
        confidence_errors = self._validate_confidence(response.confidence)
        errors.extend(confidence_errors)

        # Check for warnings
        confidence_warnings = self._check_confidence_warnings(response.confidence)
        warnings.extend(confidence_warnings)

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def _validate_taxonomy(self, taxonomy: List[str]) -> List[str]:
        """Validate taxonomy format."""
        errors = []

        if not taxonomy:
            errors.append("Taxonomy cannot be empty")
            return errors

        for label in taxonomy:
            if not isinstance(label, str):
                errors.append(f"Taxonomy label must be string: {label}")
                continue

            # Check format: CATEGORY.SUBCATEGORY[.Type]
            parts = label.split(".")
            if len(parts) < 2:
                errors.append(f"Invalid taxonomy format: {label}")
                continue

            # Check category is uppercase
            if not parts[0].isupper():
                errors.append(f"Category must be uppercase: {parts[0]}")

        return errors

    def _validate_scores(
        self, taxonomy: List[str], scores: Dict[str, float]
    ) -> List[str]:
        """Validate scores consistency with taxonomy."""
        errors = []

        # Check all taxonomy items have scores
        for label in taxonomy:
            if label not in scores:
                errors.append(f"Missing score for taxonomy label: {label}")

        # Check all scores are valid
        for label, score in scores.items():
            if not isinstance(score, (int, float)):
                errors.append(f"Score must be numeric: {label}={score}")
                continue

            if not 0.0 <= score <= 1.0:
                errors.append(f"Score must be between 0.0 and 1.0: {label}={score}")

        return errors

    def _validate_confidence(self, confidence: float) -> List[str]:
        """Validate confidence score."""
        errors = []

        if not isinstance(confidence, (int, float)):
            errors.append(f"Confidence must be numeric: {confidence}")
            return errors

        if not 0.0 <= confidence <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0: {confidence}")

        if confidence < self.min_confidence:
            errors.append(
                f"Confidence below minimum threshold: {confidence} < {self.min_confidence}"
            )

        return errors

    def _check_confidence_warnings(self, confidence: float) -> List[str]:
        """Check for confidence-related warnings."""
        warnings = []

        if confidence < 0.5:
            warnings.append("Low confidence score may indicate uncertain prediction")

        return warnings
