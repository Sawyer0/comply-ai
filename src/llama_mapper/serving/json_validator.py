"""
JSON schema validator for model outputs.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from jsonschema import (  # type: ignore[import-not-found,import-untyped]
    ValidationError,
    validate,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..api.models import MappingResponse


class JSONValidator:
    """Validates model outputs against the pillars-detectors schema."""

    def __init__(self, schema_path: str = "pillars-detectors/schema.json"):
        """
        Initialize the JSON validator.

        Args:
            schema_path: Path to the JSON schema file
        """
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "retry_attempts": 0,
        }

    def _load_schema(self) -> Dict[str, Any]:
        """Load the JSON schema from file."""
        try:
            with open(self.schema_path, "r") as f:
                schema = json.load(f)
            logger.info("Loaded JSON schema from %s", self.schema_path)
            return cast(Dict[str, Any], schema)
        except FileNotFoundError:
            logger.error("Schema file not found: %s", self.schema_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in schema file: %s", e)
            raise

    def validate(self, model_output: str) -> Tuple[bool, Optional[List[str]]]:
        """
        Validate model output against the JSON schema.

        Args:
            model_output: Raw model output string

        Returns:
            Tuple[bool, Optional[List[str]]]: (is_valid, validation_errors)
        """
        self.validation_stats["total_validations"] += 1

        try:
            # First, try to parse as JSON
            parsed_output = json.loads(model_output)

            # Validate against schema (tolerate optional version_info in model output)
            validate_dict = parsed_output
            if isinstance(parsed_output, dict) and "version_info" in parsed_output:
                try:
                    # Work on a shallow copy to avoid mutating caller data
                    validate_dict = dict(parsed_output)
                    validate_dict.pop("version_info", None)
                except Exception:
                    validate_dict = parsed_output
            validate(instance=validate_dict, schema=self.schema)

            # Additional custom validations
            custom_errors = self._custom_validations(validate_dict)
            if custom_errors:
                self.validation_stats["failed_validations"] += 1
                return False, custom_errors

            self.validation_stats["successful_validations"] += 1
            return True, None

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            logger.warning("JSON parsing failed: %s", error_msg)
            self.validation_stats["failed_validations"] += 1
            return False, [error_msg]

        except ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            logger.warning("Schema validation failed: %s", error_msg)
            self.validation_stats["failed_validations"] += 1
            return False, [error_msg]

        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            logger.error("Unexpected validation error: %s", error_msg)
            self.validation_stats["failed_validations"] += 1
            return False, [error_msg]

    def _custom_validations(self, parsed_output: Dict[str, Any]) -> List[str]:
        """
        Perform additional custom validations beyond the JSON schema.

        Args:
            parsed_output: Parsed JSON output

        Returns:
            List[str]: List of validation errors
        """
        errors = []

        # Validate taxonomy label format
        taxonomy = parsed_output.get("taxonomy", [])
        for label in taxonomy:
            if not self._validate_taxonomy_format(label):
                errors.append(f"Invalid taxonomy label format: {label}")

        # Validate scores consistency
        scores = parsed_output.get("scores", {})
        for label in taxonomy:
            if label not in scores:
                errors.append(f"Missing score for taxonomy label: {label}")

        # Check for extra scores not in taxonomy
        for label in scores:
            if label not in taxonomy:
                errors.append(f"Score provided for label not in taxonomy: {label}")

        # Validate score ranges (additional check beyond schema)
        for label, score in scores.items():
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                errors.append(f"Invalid score for {label}: {score} (must be 0.0-1.0)")

        # Validate confidence range
        confidence = parsed_output.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or not (
                0.0 <= confidence <= 1.0
            ):
                errors.append(f"Invalid confidence: {confidence} (must be 0.0-1.0)")

        # Validate notes length
        notes = parsed_output.get("notes")
        if notes is not None and len(notes) > 500:
            errors.append(f"Notes too long: {len(notes)} characters (max 500)")

        return errors

    def _validate_taxonomy_format(self, label: str) -> bool:
        """
        Validate taxonomy label format.

        Args:
            label: Taxonomy label to validate

        Returns:
            bool: True if format is valid
        """
        pattern = r"^[A-Z][A-Z0-9_]*(\.[A-Za-z0-9_]+)*$"
        return bool(re.match(pattern, label))

    def parse_output(self, model_output: str) -> MappingResponse:
        """
        Parse and convert model output to MappingResponse.

        Args:
            model_output: Validated model output string

        Returns:
            MappingResponse: Parsed response object

        Raises:
            ValueError: If output cannot be parsed
        """
        # Local import to avoid circular dependency
        from ..api.models import MappingResponse, PolicyContext, Provenance

        try:
            parsed = json.loads(model_output)

            # Extract provenance if present
            provenance = None
            if "provenance" in parsed:
                prov_data = parsed["provenance"]
                provenance = Provenance(
                    vendor=prov_data.get("vendor"),
                    detector=prov_data.get("detector"),
                    detector_version=prov_data.get("detector_version"),
                    raw_ref=prov_data.get("raw_ref"),
                    route=prov_data.get("route"),
                    model=prov_data.get("model"),
                    tenant_id=prov_data.get("tenant_id"),
                    ts=(
                        datetime.fromisoformat(prov_data["ts"])
                        if prov_data.get("ts")
                        else None
                    ),
                )

            # Extract policy context if present
            policy_context = None
            if "policy_context" in parsed:
                ctx_data = parsed["policy_context"]
                policy_context = PolicyContext(
                    expected_detectors=ctx_data.get("expected_detectors"),
                    environment=ctx_data.get("environment"),
                )

            return MappingResponse(
                taxonomy=parsed["taxonomy"],
                scores=parsed["scores"],
                confidence=parsed["confidence"],
                notes=parsed.get("notes"),
                provenance=provenance,
                policy_context=policy_context,
                version_info=None,
            )

        except Exception as e:
            logger.error("Failed to parse model output: %s", str(e))
            raise ValueError(f"Cannot parse model output: {str(e)}")

    def validate_with_retry(
        self, model_output: str, max_retries: int = 2
    ) -> Tuple[bool, Optional[List[str]], int]:
        """
        Validate with retry logic for handling generation parameter adjustments.

        Args:
            model_output: Model output to validate
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple[bool, Optional[List[str]], int]: (is_valid, errors, retry_count)
        """
        retry_count = 0

        while retry_count <= max_retries:
            is_valid, errors = self.validate(model_output)

            if is_valid:
                return True, None, retry_count

            if retry_count < max_retries:
                self.validation_stats["retry_attempts"] += 1
                logger.info(
                    "Validation failed, retry %s/%s", retry_count + 1, max_retries
                )
                retry_count += 1
                # In a real implementation, this would trigger model regeneration
                # with adjusted parameters (e.g., lower temperature)
                continue
            else:
                break

        return False, errors, retry_count

    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics for monitoring.

        Returns:
            Dict[str, Any]: Validation statistics
        """
        stats: Dict[str, Any] = dict(self.validation_stats)

        if stats["total_validations"] > 0:
            stats["success_rate"] = (
                stats["successful_validations"] / stats["total_validations"]
            )
            stats["failure_rate"] = (
                stats["failed_validations"] / stats["total_validations"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        return stats

    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "retry_attempts": 0,
        }

    def extract_json_from_text(self, text: str) -> Optional[str]:
        """
        Extract JSON from model output that might contain extra text.

        Args:
            text: Raw model output text

        Returns:
            Optional[str]: Extracted JSON string or None
        """
        # Try to find JSON block in the text
        json_patterns = [
            r"\{.*\}",  # Simple JSON object
            r"```json\s*(\{.*\})\s*```",  # JSON in code block
            r"```\s*(\{.*\})\s*```",  # JSON in generic code block
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                first = matches[0]
                if isinstance(first, tuple):
                    candidate: Any = first[0]
                else:
                    candidate = first
                json_str: str = cast(str, candidate)
                try:
                    # Validate it's actually JSON
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue

        # If no JSON found, try the whole text
        try:
            json.loads(text.strip())
            return text.strip()
        except json.JSONDecodeError:
            return None

    def create_fallback_response(
        self, detector: str, _original_output: str, error_message: str
    ) -> MappingResponse:
        """
        Create a fallback response when validation fails completely.

        Args:
            detector: Detector name
            original_output: Original detector output
            error_message: Validation error message

        Returns:
            MappingResponse: Fallback response
        """
        # Local import to avoid circular dependency
        from ..api.models import MappingResponse, Provenance

        return MappingResponse(
            taxonomy=["OTHER.ModelError"],
            scores={"OTHER.ModelError": 0.0},
            confidence=0.0,
            notes=f"Validation failed for {detector}: {error_message}",
            provenance=Provenance(detector=detector, raw_ref=None),
            version_info=None,
        )
