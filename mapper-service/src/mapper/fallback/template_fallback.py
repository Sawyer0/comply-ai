"""
Template-based fallback mapping system.

Single responsibility: Provide template-based responses for low confidence or failed mappings.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from ..schemas.models import MappingResponse, Provenance

logger = logging.getLogger(__name__)


@dataclass
class FallbackTrigger:
    """Conditions that trigger template fallback."""

    low_confidence: bool = False
    validation_failed: bool = False
    model_error: bool = False
    timeout: bool = False
    unknown_detector: bool = False


class TemplateFallback:
    """
    Template-based fallback for mapping responses.

    Single responsibility: Generate template-based fallback responses.
    """

    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize template fallback.

        Args:
            confidence_threshold: Confidence threshold for triggering fallback
        """
        self.confidence_threshold = confidence_threshold
        self.usage_stats: Dict[str, int] = {}

    def should_use_fallback(
        self,
        response: Optional[MappingResponse] = None,
        confidence: Optional[float] = None,
        validation_errors: Optional[List[str]] = None,
        error_type: Optional[str] = None,
    ) -> FallbackTrigger:
        """
        Determine if template fallback should be used.

        Args:
            response: Original mapping response (if any)
            confidence: Confidence score
            validation_errors: List of validation errors
            error_type: Type of error that occurred

        Returns:
            FallbackTrigger: Conditions that triggered fallback
        """
        triggers = FallbackTrigger()

        # Low confidence trigger
        if confidence is not None and confidence < self.confidence_threshold:
            triggers.low_confidence = True
        elif response and response.confidence < self.confidence_threshold:
            triggers.low_confidence = True

        # Validation failure trigger
        if validation_errors and len(validation_errors) > 0:
            triggers.validation_failed = True

        # Model error trigger
        if error_type in ["model_error", "generation_failed", "parsing_error"]:
            triggers.model_error = True

        # Timeout trigger
        if error_type == "timeout":
            triggers.timeout = True

        # Unknown detector trigger
        if error_type == "unknown_detector":
            triggers.unknown_detector = True

        return triggers

    def generate_fallback_response(
        self,
        detector: str,
        output: str,
        triggers: FallbackTrigger,
        context: Optional[Dict[str, Any]] = None,
    ) -> MappingResponse:
        """
        Generate template-based fallback response.

        Args:
            detector: Detector name
            output: Original detector output
            triggers: Fallback triggers
            context: Additional context information

        Returns:
            MappingResponse: Template-based response
        """
        # Track usage
        self._track_usage(detector, triggers)

        # Determine fallback type based on triggers
        if triggers.unknown_detector:
            return self._generate_unknown_detector_response(detector, output)
        elif triggers.model_error or triggers.timeout:
            return self._generate_error_response(detector, output, triggers)
        elif triggers.validation_failed:
            return self._generate_validation_failed_response(detector, output)
        elif triggers.low_confidence:
            return self._generate_low_confidence_response(detector, output, context)
        else:
            return self._generate_generic_fallback_response(detector, output)

    def _generate_unknown_detector_response(
        self, detector: str, output: str
    ) -> MappingResponse:
        """Generate response for unknown detector."""
        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.1},
            confidence=0.1,
            notes=f"Unknown detector '{detector}' - cannot classify output: {output[:50]}...",
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _generate_error_response(
        self, detector: str, output: str, triggers: FallbackTrigger
    ) -> MappingResponse:
        """Generate response for model errors."""
        error_type = "timeout" if triggers.timeout else "model_error"

        return MappingResponse(
            taxonomy=["OTHER.Error"],
            scores={"OTHER.Error": 0.0},
            confidence=0.0,
            notes=f"Mapping failed due to {error_type} for detector {detector}",
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _generate_validation_failed_response(
        self, detector: str, output: str
    ) -> MappingResponse:
        """Generate response for validation failures."""
        return MappingResponse(
            taxonomy=["OTHER.ValidationFailed"],
            scores={"OTHER.ValidationFailed": 0.2},
            confidence=0.2,
            notes=f"Model output failed validation for detector {detector} - using conservative fallback",
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _generate_low_confidence_response(
        self, detector: str, output: str, context: Optional[Dict[str, Any]] = None
    ) -> MappingResponse:
        """Generate response for low confidence predictions."""

        # Try to infer category from detector name or output
        inferred_category = self._infer_category_from_context(detector, output, context)

        if inferred_category:
            taxonomy = [inferred_category]
            scores = {inferred_category: 0.4}
            confidence = 0.4
            notes = f"Low confidence prediction for {detector} - inferred category: {inferred_category}"
        else:
            taxonomy = ["OTHER.LowConfidence"]
            scores = {"OTHER.LowConfidence": 0.3}
            confidence = 0.3
            notes = f"Low confidence prediction for {detector} - requires manual review"

        return MappingResponse(
            taxonomy=taxonomy,
            scores=scores,
            confidence=confidence,
            notes=notes,
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _generate_generic_fallback_response(
        self, detector: str, output: str
    ) -> MappingResponse:
        """Generate generic fallback response."""
        return MappingResponse(
            taxonomy=["OTHER.RequiresReview"],
            scores={"OTHER.RequiresReview": 0.5},
            confidence=0.5,
            notes=f"Generic fallback for detector {detector} - manual review recommended",
            provenance=Provenance(detector=detector, raw_ref=output),
            version_info=None,
        )

    def _infer_category_from_context(
        self, detector: str, output: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Infer category from detector name, output, or context.

        Args:
            detector: Detector name
            output: Detector output
            context: Additional context

        Returns:
            Optional[str]: Inferred category or None
        """
        # Detector name patterns
        detector_patterns = {
            "presidio": "PII.Identity.Name",  # Default for Presidio
            "pii": "PII.Identity.Name",
            "email": "PII.Contact.Email",
            "phone": "PII.Contact.Phone",
            "credit": "PII.Financial.CreditCard",
            "ssn": "PII.Identity.SSN",
            "toxic": "CONTENT.Harmful.Toxic",
            "security": "SECURITY.Access.Unauthorized",
            "password": "SECURITY.Credentials.Password",
            "api": "SECURITY.Credentials.APIKey",
        }

        # Check detector name
        detector_lower = detector.lower()
        for pattern, category in detector_patterns.items():
            if pattern in detector_lower:
                return category

        # Check output content
        output_lower = output.lower()
        output_patterns = {
            "email": "PII.Contact.Email",
            "phone": "PII.Contact.Phone",
            "name": "PII.Identity.Name",
            "address": "PII.Contact.Address",
            "credit": "PII.Financial.CreditCard",
            "ssn": "PII.Identity.SSN",
            "password": "SECURITY.Credentials.Password",
            "token": "SECURITY.Credentials.Token",
            "key": "SECURITY.Credentials.APIKey",
            "toxic": "CONTENT.Harmful.Toxic",
            "hate": "CONTENT.Harmful.Hate",
        }

        for pattern, category in output_patterns.items():
            if pattern in output_lower:
                return category

        # Check context if available
        if context:
            metadata = context.get("metadata", {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    key_lower = str(key).lower()
                    value_lower = str(value).lower()

                    for pattern, category in output_patterns.items():
                        if pattern in key_lower or pattern in value_lower:
                            return category

        return None

    def _track_usage(self, detector: str, triggers: FallbackTrigger) -> None:
        """Track fallback usage for analytics."""
        trigger_types = []
        if triggers.low_confidence:
            trigger_types.append("low_confidence")
        if triggers.validation_failed:
            trigger_types.append("validation_failed")
        if triggers.model_error:
            trigger_types.append("model_error")
        if triggers.timeout:
            trigger_types.append("timeout")
        if triggers.unknown_detector:
            trigger_types.append("unknown_detector")

        for trigger_type in trigger_types:
            key = f"{detector}:{trigger_type}"
            self.usage_stats[key] = self.usage_stats.get(key, 0) + 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total_usage = sum(self.usage_stats.values())

        by_detector = {}
        by_trigger = {}

        for key, count in self.usage_stats.items():
            if ":" in key:
                detector, trigger = key.split(":", 1)
                by_detector[detector] = by_detector.get(detector, 0) + count
                by_trigger[trigger] = by_trigger.get(trigger, 0) + count

        return {
            "total_usage": total_usage,
            "by_detector": by_detector,
            "by_trigger": by_trigger,
            "detailed_stats": self.usage_stats,
        }
