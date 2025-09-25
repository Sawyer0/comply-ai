"""
Output validation for mapping responses.

Single responsibility: Validate mapping outputs against schemas and business rules.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..schemas.models import MappingResponse

logger = logging.getLogger(__name__)


@dataclass
class OutputValidationResult:
    """Result of output validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence_score: float = 0.0
    should_retry: bool = False
    retry_prompt: Optional[str] = None


class OutputValidator:
    """
    Validates mapping response outputs.

    Single responsibility: Output validation and quality assessment.
    """

    def __init__(self, min_confidence: float = 0.5, strict_mode: bool = True):
        """
        Initialize output validator.

        Args:
            min_confidence: Minimum acceptable confidence score
            strict_mode: Whether to enforce strict validation rules
        """
        self.min_confidence = min_confidence
        self.strict_mode = strict_mode

    def validate_response(
        self, response: Union[str, Dict, MappingResponse]
    ) -> OutputValidationResult:
        """
        Validate a mapping response.

        Args:
            response: Mapping response to validate

        Returns:
            OutputValidationResult: Validation result
        """
        errors = []
        warnings = []

        # Parse response if it's a string
        if isinstance(response, str):
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError as e:
                return OutputValidationResult(
                    is_valid=False,
                    errors=[f"Invalid JSON: {str(e)}"],
                    warnings=[],
                    should_retry=True,
                    retry_prompt=self._create_json_retry_prompt(response),
                )
        elif isinstance(response, dict):
            parsed_response = response
        else:
            # Assume it's a MappingResponse object
            parsed_response = {
                "taxonomy": response.taxonomy,
                "scores": response.scores,
                "confidence": response.confidence,
                "notes": response.notes,
            }

        # Validate required fields
        required_errors = self._validate_required_fields(parsed_response)
        errors.extend(required_errors)

        # Validate taxonomy format
        taxonomy_errors = self._validate_taxonomy(parsed_response.get("taxonomy", []))
        errors.extend(taxonomy_errors)

        # Validate scores
        scores_errors, scores_warnings = self._validate_scores(
            parsed_response.get("taxonomy", []), parsed_response.get("scores", {})
        )
        errors.extend(scores_errors)
        warnings.extend(scores_warnings)

        # Validate confidence
        confidence_errors, confidence_warnings = self._validate_confidence(
            parsed_response.get("confidence", 0.0)
        )
        errors.extend(confidence_errors)
        warnings.extend(confidence_warnings)

        # Validate notes if present
        if "notes" in parsed_response:
            notes_errors = self._validate_notes(parsed_response["notes"])
            errors.extend(notes_errors)

        # Calculate overall confidence score
        confidence_score = self._calculate_confidence_score(parsed_response)

        # Determine if retry is needed
        should_retry = len(errors) > 0 and self._should_retry(errors)
        retry_prompt = None
        if should_retry:
            retry_prompt = self._create_retry_prompt(errors, parsed_response)

        return OutputValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score,
            should_retry=should_retry,
            retry_prompt=retry_prompt,
        )

    def _validate_required_fields(self, response: Dict[str, Any]) -> List[str]:
        """Validate required fields are present."""
        errors = []
        required_fields = ["taxonomy", "scores", "confidence"]

        for field in required_fields:
            if field not in response:
                errors.append(f"Missing required field: {field}")

        return errors

    def _validate_taxonomy(self, taxonomy: List[str]) -> List[str]:
        """Validate taxonomy format and content."""
        errors = []

        if not isinstance(taxonomy, list):
            errors.append("Taxonomy must be a list")
            return errors

        if not taxonomy:
            errors.append("Taxonomy cannot be empty")
            return errors

        if len(taxonomy) > 10:
            errors.append("Too many taxonomy labels (max 10)")

        for label in taxonomy:
            if not isinstance(label, str):
                errors.append(f"Taxonomy label must be string: {label}")
                continue

            # Check format: CATEGORY.SUBCATEGORY[.Type]
            parts = label.split(".")
            if len(parts) < 2:
                errors.append(
                    f"Invalid taxonomy format (need CATEGORY.SUBCATEGORY): {label}"
                )
                continue

            # Check category is uppercase
            if not parts[0].isupper():
                errors.append(f"Category must be uppercase: {parts[0]} in {label}")

            # Check for valid characters
            for part in parts:
                if not part.replace("_", "").replace("-", "").isalnum():
                    errors.append(
                        f"Invalid characters in taxonomy part: {part} in {label}"
                    )

        return errors

    def _validate_scores(
        self, taxonomy: List[str], scores: Dict[str, float]
    ) -> tuple[List[str], List[str]]:
        """Validate scores consistency and format."""
        errors = []
        warnings = []

        if not isinstance(scores, dict):
            errors.append("Scores must be a dictionary")
            return errors, warnings

        if not scores:
            errors.append("Scores cannot be empty")
            return errors, warnings

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

            # Warning for low scores
            if score < 0.3:
                warnings.append(f"Very low score for {label}: {score}")

        # Check for extra scores not in taxonomy
        for label in scores:
            if label not in taxonomy:
                warnings.append(f"Score provided for label not in taxonomy: {label}")

        return errors, warnings

    def _validate_confidence(self, confidence: float) -> tuple[List[str], List[str]]:
        """Validate confidence score."""
        errors = []
        warnings = []

        if not isinstance(confidence, (int, float)):
            errors.append(f"Confidence must be numeric: {confidence}")
            return errors, warnings

        if not 0.0 <= confidence <= 1.0:
            errors.append(f"Confidence must be between 0.0 and 1.0: {confidence}")

        if confidence < self.min_confidence:
            if self.strict_mode:
                errors.append(
                    f"Confidence below minimum threshold: {confidence} < {self.min_confidence}"
                )
            else:
                warnings.append(f"Low confidence score: {confidence}")

        return errors, warnings

    def _validate_notes(self, notes: str) -> List[str]:
        """Validate notes field."""
        errors = []

        if not isinstance(notes, str):
            errors.append("Notes must be a string")
            return errors

        if len(notes) > 500:
            errors.append("Notes too long (max 500 characters)")

        return errors

    def _calculate_confidence_score(self, response: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the response."""
        base_confidence = response.get("confidence", 0.0)

        # Adjust based on taxonomy quality
        taxonomy = response.get("taxonomy", [])
        if len(taxonomy) == 1:
            # Single clear prediction is good
            confidence_adjustment = 0.1
        elif len(taxonomy) > 5:
            # Too many predictions reduces confidence
            confidence_adjustment = -0.1
        else:
            confidence_adjustment = 0.0

        # Adjust based on score consistency
        scores = response.get("scores", {})
        if scores:
            score_values = list(scores.values())
            max_score = max(score_values)
            min_score = min(score_values)
            score_range = max_score - min_score

            if score_range > 0.5:
                # High variance in scores reduces confidence
                confidence_adjustment -= 0.05

        final_confidence = max(0.0, min(1.0, base_confidence + confidence_adjustment))
        return final_confidence

    def _should_retry(self, errors: List[str]) -> bool:
        """Determine if the response should be retried."""
        # Retry for format errors but not for business logic errors
        retry_keywords = ["Invalid JSON", "Missing required field", "must be"]

        for error in errors:
            for keyword in retry_keywords:
                if keyword in error:
                    return True

        return False

    def _create_retry_prompt(self, errors: List[str], response: Dict[str, Any]) -> str:
        """Create retry prompt with specific error corrections."""
        error_summary = "; ".join(errors[:3])  # Limit to first 3 errors

        return f"""The previous response had validation errors: {error_summary}

Please provide a corrected JSON response with this exact format:
{{
  "taxonomy": ["CATEGORY.SUBCATEGORY.Type"],
  "scores": {{"CATEGORY.SUBCATEGORY.Type": 0.95}},
  "confidence": 0.95,
  "notes": "Brief explanation"
}}

Requirements:
- taxonomy: List of strings in CATEGORY.SUBCATEGORY format (CATEGORY must be uppercase)
- scores: Dictionary with same keys as taxonomy, values between 0.0-1.0
- confidence: Number between 0.0-1.0
- notes: Optional string under 500 characters"""

    def _create_json_retry_prompt(self, invalid_response: str) -> str:
        """Create retry prompt for JSON parsing errors."""
        return f"""The response was not valid JSON. Please provide a valid JSON response with this format:

{{
  "taxonomy": ["CATEGORY.SUBCATEGORY"],
  "scores": {{"CATEGORY.SUBCATEGORY": 0.95}},
  "confidence": 0.95,
  "notes": "Brief explanation"
}}

Make sure to:
- Use double quotes for strings
- No trailing commas
- Proper JSON syntax"""
