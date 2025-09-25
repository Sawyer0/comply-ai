"""
Fallback coordinator for managing different fallback strategies.

Single responsibility: Coordinate between different fallback mechanisms.
"""

import logging
from typing import Any, Dict, List, Optional

from .rule_based_fallback import RuleBasedFallback
from .template_fallback import TemplateFallback, FallbackTrigger
from ..schemas.models import MappingResponse

logger = logging.getLogger(__name__)


class FallbackCoordinator:
    """
    Coordinates different fallback strategies.

    Single responsibility: Orchestrate fallback selection and execution.
    """

    def __init__(
        self,
        detector_configs_path: str = "config/detectors",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize fallback coordinator.

        Args:
            detector_configs_path: Path to detector configurations
            confidence_threshold: Confidence threshold for fallbacks
        """
        self.rule_based_fallback = RuleBasedFallback(detector_configs_path)
        self.template_fallback = TemplateFallback(confidence_threshold)
        self.confidence_threshold = confidence_threshold

    def execute_fallback(
        self,
        detector: str,
        output: str,
        reason: str = "unknown",
        original_response: Optional[MappingResponse] = None,
        validation_errors: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MappingResponse:
        """
        Execute appropriate fallback strategy.

        Args:
            detector: Detector name
            output: Original detector output
            reason: Reason for fallback
            original_response: Original mapping response (if any)
            validation_errors: Validation errors (if any)
            context: Additional context

        Returns:
            MappingResponse: Fallback response
        """
        logger.info("Executing fallback for detector %s, reason: %s", detector, reason)

        # Determine fallback strategy
        strategy = self._select_fallback_strategy(
            detector, reason, original_response, validation_errors
        )

        if strategy == "rule_based":
            return self._execute_rule_based_fallback(detector, output, reason)
        elif strategy == "template":
            return self._execute_template_fallback(
                detector, output, reason, original_response, validation_errors, context
            )
        else:
            # Hybrid approach - try rule-based first, then template
            return self._execute_hybrid_fallback(
                detector, output, reason, original_response, validation_errors, context
            )

    def _select_fallback_strategy(
        self,
        detector: str,
        reason: str,
        original_response: Optional[MappingResponse],
        validation_errors: Optional[List[str]],
    ) -> str:
        """
        Select appropriate fallback strategy.

        Args:
            detector: Detector name
            reason: Reason for fallback
            original_response: Original response
            validation_errors: Validation errors

        Returns:
            str: Selected strategy ("rule_based", "template", or "hybrid")
        """
        # Use rule-based for known detectors with mapping rules
        if detector in self.rule_based_fallback.get_supported_detectors():
            if reason in ["low_confidence", "model_unavailable"]:
                return "rule_based"

        # Use template for validation failures or model errors
        if reason in ["validation_failed", "model_error", "timeout"]:
            return "template"

        # Use template for unknown detectors
        if reason == "unknown_detector":
            return "template"

        # Default to hybrid approach
        return "hybrid"

    def _execute_rule_based_fallback(
        self, detector: str, output: str, reason: str
    ) -> MappingResponse:
        """Execute rule-based fallback."""
        logger.debug("Using rule-based fallback for %s", detector)
        return self.rule_based_fallback.map(detector, output, reason)

    def _execute_template_fallback(
        self,
        detector: str,
        output: str,
        reason: str,
        original_response: Optional[MappingResponse],
        validation_errors: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> MappingResponse:
        """Execute template-based fallback."""
        logger.debug("Using template fallback for %s", detector)

        # Determine triggers
        triggers = self.template_fallback.should_use_fallback(
            response=original_response,
            validation_errors=validation_errors,
            error_type=reason,
        )

        return self.template_fallback.generate_fallback_response(
            detector, output, triggers, context
        )

    def _execute_hybrid_fallback(
        self,
        detector: str,
        output: str,
        reason: str,
        original_response: Optional[MappingResponse],
        validation_errors: Optional[List[str]],
        context: Optional[Dict[str, Any]],
    ) -> MappingResponse:
        """Execute hybrid fallback (try rule-based first, then template)."""
        logger.debug("Using hybrid fallback for %s", detector)

        # Try rule-based first
        if detector in self.rule_based_fallback.get_supported_detectors():
            rule_response = self.rule_based_fallback.map(detector, output, reason)

            # If rule-based gives a good result, use it
            if (
                rule_response.confidence >= 0.5
                and "Unknown" not in rule_response.taxonomy[0]
            ):
                logger.debug("Rule-based fallback successful for %s", detector)
                return rule_response

        # Fall back to template-based
        logger.debug(
            "Rule-based fallback insufficient, using template for %s", detector
        )
        return self._execute_template_fallback(
            detector, output, reason, original_response, validation_errors, context
        )

    def get_supported_detectors(self) -> List[str]:
        """Get list of detectors supported by rule-based fallback."""
        return self.rule_based_fallback.get_supported_detectors()

    def add_detector_mapping(self, detector: str, mappings: Dict[str, str]) -> None:
        """Add detector mappings to rule-based fallback."""
        self.rule_based_fallback.add_detector_mapping(detector, mappings)

    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get comprehensive fallback usage statistics."""
        rule_stats = self.rule_based_fallback.get_usage_stats()
        template_stats = self.template_fallback.get_usage_stats()

        return {
            "rule_based": rule_stats,
            "template": template_stats,
            "total_rule_usage": rule_stats["total_usage"],
            "total_template_usage": template_stats["total_usage"],
            "total_fallback_usage": rule_stats["total_usage"]
            + template_stats["total_usage"],
        }

    def reload_configurations(self) -> None:
        """Reload all fallback configurations."""
        logger.info("Reloading fallback configurations")
        self.rule_based_fallback.reload_mappings()
