"""
Validation pipeline for comprehensive input/output validation.

Single responsibility: Orchestrate the complete validation process.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from .input_validator import InputValidator, InputValidationResult
from .output_validator import OutputValidator, OutputValidationResult
from .schema_validator import validate_model_output, ModelType, SchemaValidationConfig
from ..schemas.models import MappingRequest, MappingResponse
from ..fallback.fallback_coordinator import FallbackCoordinator

logger = logging.getLogger(__name__)


@dataclass
class ValidationPipelineResult:
    """Result of the complete validation pipeline."""

    input_valid: bool
    output_valid: bool
    overall_valid: bool
    input_errors: List[str]
    output_errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[Dict[str, Any]] = None
    should_retry: bool = False
    retry_prompt: Optional[str] = None
    requires_fallback: bool = False
    fallback_reason: Optional[str] = None


class ValidationPipeline:
    """
    Comprehensive validation pipeline.

    Single responsibility: Coordinate all validation steps.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        strict_mode: bool = True,
        enable_retry: bool = True,
        detector_configs_path: str = "config/detectors",
    ):
        """
        Initialize validation pipeline.

        Args:
            confidence_threshold: Minimum confidence threshold
            strict_mode: Whether to enforce strict validation
            enable_retry: Whether to enable retry on validation failures
            detector_configs_path: Path to detector configurations
        """
        self.input_validator = InputValidator()
        self.output_validator = OutputValidator(
            min_confidence=confidence_threshold, strict_mode=strict_mode
        )
        self.schema_config = SchemaValidationConfig(
            strict_mode=strict_mode, allow_retry=enable_retry
        )
        self.fallback_coordinator = FallbackCoordinator(
            detector_configs_path=detector_configs_path,
            confidence_threshold=confidence_threshold,
        )
        self.confidence_threshold = confidence_threshold

    def validate_input(self, request: MappingRequest) -> InputValidationResult:
        """
        Validate input request.

        Args:
            request: Mapping request to validate

        Returns:
            InputValidationResult: Input validation result
        """
        return self.input_validator.validate_request(request)

    def validate_output(
        self, response: Union[str, Dict, MappingResponse], detector: str = "unknown"
    ) -> ValidationPipelineResult:
        """
        Validate output response with comprehensive checks.

        Args:
            response: Response to validate
            detector: Detector name for context

        Returns:
            ValidationPipelineResult: Complete validation result
        """
        # Step 1: Basic output validation
        output_result = self.output_validator.validate_response(response)

        # Step 2: Schema validation (if output is valid enough)
        schema_errors = []
        if output_result.is_valid or not output_result.should_retry:
            try:
                # Convert response to dict for schema validation
                if isinstance(response, str):
                    import json

                    response_dict = json.loads(response)
                elif isinstance(response, MappingResponse):
                    response_dict = {
                        "taxonomy": response.taxonomy,
                        "scores": response.scores,
                        "confidence": response.confidence,
                        "notes": response.notes,
                    }
                else:
                    response_dict = response

                schema_result = validate_model_output(
                    ModelType.MAPPER, response_dict, self.schema_config
                )

                if not schema_result["valid"]:
                    schema_errors = schema_result["errors"]

            except Exception as e:
                schema_errors = [f"Schema validation error: {str(e)}"]

        # Combine all errors
        all_errors = output_result.errors + schema_errors
        all_warnings = output_result.warnings

        # Determine if fallback is needed
        requires_fallback = False
        fallback_reason = None

        if len(all_errors) > 0:
            requires_fallback = True
            fallback_reason = "validation_failed"
        elif output_result.confidence_score < self.confidence_threshold:
            requires_fallback = True
            fallback_reason = "low_confidence"

        # Determine retry logic
        should_retry = output_result.should_retry and not requires_fallback
        retry_prompt = output_result.retry_prompt if should_retry else None

        return ValidationPipelineResult(
            input_valid=True,  # Input validation done separately
            output_valid=len(all_errors) == 0,
            overall_valid=len(all_errors) == 0,
            input_errors=[],
            output_errors=all_errors,
            warnings=all_warnings,
            should_retry=should_retry,
            retry_prompt=retry_prompt,
            requires_fallback=requires_fallback,
            fallback_reason=fallback_reason,
        )

    def validate_complete_flow(
        self,
        request: MappingRequest,
        response: Optional[Union[str, Dict, MappingResponse]] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> ValidationPipelineResult:
        """
        Validate complete mapping flow (input + output).

        Args:
            request: Original mapping request
            response: Mapping response (if any)
            error_context: Error context if mapping failed

        Returns:
            ValidationPipelineResult: Complete validation result
        """
        # Validate input
        input_result = self.validate_input(request)

        if not input_result.is_valid:
            return ValidationPipelineResult(
                input_valid=False,
                output_valid=False,
                overall_valid=False,
                input_errors=input_result.errors,
                output_errors=[],
                warnings=input_result.warnings,
                sanitized_input=input_result.sanitized_input,
                requires_fallback=True,
                fallback_reason="invalid_input",
            )

        # If no response, it means mapping failed
        if response is None:
            error_type = (
                error_context.get("error_type", "unknown")
                if error_context
                else "unknown"
            )
            return ValidationPipelineResult(
                input_valid=True,
                output_valid=False,
                overall_valid=False,
                input_errors=[],
                output_errors=[f"Mapping failed: {error_type}"],
                warnings=input_result.warnings,
                sanitized_input=input_result.sanitized_input,
                requires_fallback=True,
                fallback_reason=error_type,
            )

        # Validate output
        output_result = self.validate_output(response, request.detector)

        # Combine results
        return ValidationPipelineResult(
            input_valid=input_result.is_valid,
            output_valid=output_result.output_valid,
            overall_valid=input_result.is_valid and output_result.output_valid,
            input_errors=input_result.errors,
            output_errors=output_result.output_errors,
            warnings=input_result.warnings + output_result.warnings,
            sanitized_input=input_result.sanitized_input,
            should_retry=output_result.should_retry,
            retry_prompt=output_result.retry_prompt,
            requires_fallback=output_result.requires_fallback,
            fallback_reason=output_result.fallback_reason,
        )

    def execute_fallback_if_needed(
        self,
        request: MappingRequest,
        validation_result: ValidationPipelineResult,
        original_response: Optional[MappingResponse] = None,
    ) -> Optional[MappingResponse]:
        """
        Execute fallback if validation indicates it's needed.

        Args:
            request: Original mapping request
            validation_result: Validation result
            original_response: Original response (if any)

        Returns:
            Optional[MappingResponse]: Fallback response or None
        """
        if not validation_result.requires_fallback:
            return None

        logger.info(
            "Executing fallback for request",
            detector=request.detector,
            reason=validation_result.fallback_reason,
        )

        return self.fallback_coordinator.execute_fallback(
            detector=request.detector,
            output=request.output,
            reason=validation_result.fallback_reason or "unknown",
            original_response=original_response,
            validation_errors=validation_result.output_errors,
            context={
                "metadata": request.metadata,
                "tenant_id": request.tenant_id,
                "framework": request.framework,
            },
        )

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation and fallback statistics."""
        fallback_stats = self.fallback_coordinator.get_fallback_stats()

        return {
            "fallback_stats": fallback_stats,
            "supported_detectors": self.fallback_coordinator.get_supported_detectors(),
            "confidence_threshold": self.confidence_threshold,
        }
