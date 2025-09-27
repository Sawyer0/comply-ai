"""
Response validation for mapping outputs.

Single responsibility: Validate mapping responses against business rules and schemas.
"""

import logging
from typing import List
from dataclasses import dataclass

from shared.interfaces.mapper import MappingResponse

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

    def validate(self, response: MappingResponse) -> ValidationResult:
        """
        Validate a mapping response.

        Args:
            response: Mapping response to validate

        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        errors = []
        warnings = []

        # Validate overall confidence
        confidence_errors = self._validate_confidence(response.overall_confidence)
        errors.extend(confidence_errors)

        # Validate each mapping result
        for i, mapping_result in enumerate(response.mapping_results):
            result_errors = self._validate_mapping_result(mapping_result, i)
            errors.extend(result_errors)

            result_warnings = self._check_confidence_warnings(mapping_result.confidence)
            warnings.extend([f"Result {i}: {w}" for w in result_warnings])

        # Validate cost metrics
        cost_errors = self._validate_cost_metrics(response.cost_metrics)
        errors.extend(cost_errors)

        # Validate model metrics
        model_errors = self._validate_model_metrics(response.model_metrics)
        errors.extend(model_errors)

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def _validate_mapping_result(self, mapping_result, index: int) -> List[str]:
        """Validate an individual mapping result."""
        from shared.interfaces.mapper import MappingResult

        errors = []
        prefix = f"Result {index}:"

        # Validate canonical result
        canonical_result = mapping_result.canonical_result
        if not canonical_result.canonical_labels:
            errors.append(f"{prefix} Canonical labels cannot be empty")

        for label in canonical_result.canonical_labels:
            if not isinstance(label, str):
                errors.append(f"{prefix} Canonical label must be string: {label}")
                continue

            # Check format: CATEGORY.SUBCATEGORY[.Type]
            parts = label.split(".")
            if len(parts) < 2:
                errors.append(f"{prefix} Invalid canonical format: {label}")
                continue

            # Check category is uppercase
            if not parts[0].isupper():
                errors.append(f"{prefix} Category must be uppercase: {parts[0]}")

        # Validate confidence scores consistency
        for label in canonical_result.canonical_labels:
            if label not in canonical_result.confidence_scores:
                errors.append(f"{prefix} Missing confidence score for label: {label}")

        for label, score in canonical_result.confidence_scores.items():
            if not isinstance(score, (int, float)):
                errors.append(
                    f"{prefix} Confidence score must be numeric: {label}={score}"
                )
                continue

            if not 0.0 <= score <= 1.0:
                errors.append(
                    f"{prefix} Confidence score must be between 0.0 and 1.0: {label}={score}"
                )

        # Validate framework mappings
        for i, framework_mapping in enumerate(mapping_result.framework_mappings):
            if not framework_mapping.framework:
                errors.append(f"{prefix} Framework mapping {i} must have framework")
            if not framework_mapping.control_id:
                errors.append(f"{prefix} Framework mapping {i} must have control_id")
            if not 0.0 <= framework_mapping.confidence <= 1.0:
                errors.append(
                    f"{prefix} Framework mapping {i} confidence must be between 0.0 and 1.0"
                )

        # Validate overall confidence
        if not 0.0 <= mapping_result.confidence <= 1.0:
            errors.append(f"{prefix} Overall confidence must be between 0.0 and 1.0")

        return errors

    def _validate_cost_metrics(self, cost_metrics) -> List[str]:
        """Validate cost metrics."""
        errors = []

        if cost_metrics.tokens_processed < 0:
            errors.append("Tokens processed cannot be negative")
        if cost_metrics.inference_cost < 0:
            errors.append("Inference cost cannot be negative")
        if cost_metrics.storage_cost < 0:
            errors.append("Storage cost cannot be negative")
        if cost_metrics.total_cost < 0:
            errors.append("Total cost cannot be negative")
        if cost_metrics.cost_per_request < 0:
            errors.append("Cost per request cannot be negative")

        return errors

    def _validate_model_metrics(self, model_metrics) -> List[str]:
        """Validate model metrics."""
        errors = []

        if not model_metrics.model_name:
            errors.append("Model name cannot be empty")
        if not model_metrics.model_version:
            errors.append("Model version cannot be empty")
        if model_metrics.inference_time_ms < 0:
            errors.append("Inference time cannot be negative")
        if not 0.0 <= model_metrics.gpu_utilization <= 1.0:
            errors.append("GPU utilization must be between 0.0 and 1.0")
        if model_metrics.memory_usage_mb < 0:
            errors.append("Memory usage cannot be negative")
        if model_metrics.batch_size < 1:
            errors.append("Batch size must be at least 1")

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
