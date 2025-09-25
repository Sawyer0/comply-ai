"""
Input validation for mapping requests.

Single responsibility: Validate and sanitize input data before processing.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..schemas.models import MappingRequest

logger = logging.getLogger(__name__)


@dataclass
class InputValidationResult:
    """Result of input validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[Dict[str, Any]] = None


class InputValidator:
    """
    Validates and sanitizes mapping request inputs.

    Single responsibility: Input validation and sanitization.
    """

    def __init__(self, max_input_length: int = 10000):
        """
        Initialize input validator.

        Args:
            max_input_length: Maximum allowed input length
        """
        self.max_input_length = max_input_length

        # Dangerous patterns to detect
        self.dangerous_patterns = [
            r"<script[^>]*>.*?</script>",  # XSS
            r"javascript:",
            r"on\w+\s*=",
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",  # SQL injection
            r"(--|#|/\*|\*/)",
            r"\.\./",  # Path traversal
            r"\.\.\\",
        ]

    def validate_request(self, request: MappingRequest) -> InputValidationResult:
        """
        Validate a mapping request.

        Args:
            request: Mapping request to validate

        Returns:
            InputValidationResult: Validation result
        """
        errors = []
        warnings = []

        # Validate detector name
        detector_errors = self._validate_detector(request.detector)
        errors.extend(detector_errors)

        # Validate output content
        output_errors, output_warnings = self._validate_output(request.output)
        errors.extend(output_errors)
        warnings.extend(output_warnings)

        # Validate metadata
        if request.metadata:
            metadata_errors = self._validate_metadata(request.metadata)
            errors.extend(metadata_errors)

        # Validate tenant ID
        if request.tenant_id:
            tenant_errors = self._validate_tenant_id(request.tenant_id)
            errors.extend(tenant_errors)

        # Validate framework
        if request.framework:
            framework_errors = self._validate_framework(request.framework)
            errors.extend(framework_errors)

        # Create sanitized input if validation passes
        sanitized_input = None
        if not errors:
            sanitized_input = {
                "detector": self._sanitize_string(request.detector),
                "output": self._sanitize_string(request.output),
                "metadata": (
                    self._sanitize_metadata(request.metadata)
                    if request.metadata
                    else None
                ),
                "tenant_id": (
                    self._sanitize_string(request.tenant_id)
                    if request.tenant_id
                    else None
                ),
                "framework": (
                    self._sanitize_string(request.framework)
                    if request.framework
                    else None
                ),
                "confidence_threshold": request.confidence_threshold,
            }

        return InputValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized_input,
        )

    def _validate_detector(self, detector: str) -> List[str]:
        """Validate detector name."""
        errors = []

        if not detector or not detector.strip():
            errors.append("Detector name cannot be empty")
            return errors

        # Check length
        if len(detector) > 100:
            errors.append("Detector name too long (max 100 characters)")

        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r"^[a-zA-Z0-9_-]+$", detector):
            errors.append("Detector name contains invalid characters")

        return errors

    def _validate_output(self, output: str) -> tuple[List[str], List[str]]:
        """Validate output content."""
        errors = []
        warnings = []

        if not output or not output.strip():
            errors.append("Output cannot be empty")
            return errors, warnings

        # Check length
        if len(output) > self.max_input_length:
            errors.append(f"Output too long (max {self.max_input_length} characters)")

        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                warnings.append(f"Potentially dangerous pattern detected: {pattern}")

        # Check for null bytes
        if "\x00" in output:
            errors.append("Null bytes not allowed in output")

        return errors, warnings

    def _validate_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Validate metadata dictionary."""
        errors = []

        # Check metadata size
        if len(str(metadata)) > 5000:
            errors.append("Metadata too large (max 5000 characters when serialized)")

        # Check for dangerous keys or values
        for key, value in metadata.items():
            if not isinstance(key, str):
                errors.append("Metadata keys must be strings")
                continue

            if len(key) > 100:
                errors.append(f"Metadata key too long: {key}")

            # Check for dangerous patterns in string values
            if isinstance(value, str):
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(
                            f"Dangerous pattern in metadata value for key {key}"
                        )

        return errors

    def _validate_tenant_id(self, tenant_id: str) -> List[str]:
        """Validate tenant ID."""
        errors = []

        # Check format (UUID-like or alphanumeric)
        if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", tenant_id):
            errors.append("Invalid tenant ID format")

        return errors

    def _validate_framework(self, framework: str) -> List[str]:
        """Validate framework name."""
        errors = []

        # Check against known frameworks
        valid_frameworks = ["GDPR", "HIPAA", "SOC2", "ISO27001", "CCPA", "PCI-DSS"]
        if framework not in valid_frameworks:
            errors.append(f"Unknown framework: {framework}")

        return errors

    def _sanitize_string(self, text: str) -> str:
        """Sanitize string input."""
        if not text:
            return text

        # Remove null bytes
        text = text.replace("\x00", "")

        # Trim whitespace
        text = text.strip()

        # Limit length
        if len(text) > self.max_input_length:
            text = text[: self.max_input_length]

        return text

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata dictionary."""
        sanitized = {}

        for key, value in metadata.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key))

            # Sanitize value
            if isinstance(value, str):
                clean_value = self._sanitize_string(value)
            elif isinstance(value, (int, float, bool)):
                clean_value = value
            elif isinstance(value, dict):
                clean_value = self._sanitize_metadata(value)
            elif isinstance(value, list):
                clean_value = [
                    self._sanitize_string(str(item)) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                clean_value = str(value)

            sanitized[clean_key] = clean_value

        return sanitized
