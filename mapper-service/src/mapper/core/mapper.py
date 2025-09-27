"""
Core mapping functionality for the Mapper Service.

This module consolidates the core mapping logic from the original llama-mapper
implementation, providing unified mapping from detector outputs to canonical
taxonomy with framework adaptation.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import shared components first
from ..shared_integration import (
    get_shared_logger,
    get_shared_metrics,
    get_shared_database,
    CircuitBreaker,
    validate_confidence_score,
    validate_non_empty_string,
    BaseServiceException,
    ValidationError,
    ServiceUnavailableError,
    set_correlation_id,
    get_correlation_id,
    track_request_metrics,
)

# Import service-specific components
from ..serving.model_server import ModelServer, create_model_server, GenerationConfig
from ..serving.fallback_mapper import FallbackMapper
from ..validation.validation_pipeline import ValidationPipeline
from ..taxonomy.framework_adapter import FrameworkAdapter
from ..schemas.models import MappingRequest, MappingResponse, Provenance, VersionInfo
from ..config.settings import MapperSettings
from ..resilience import (
    ComprehensiveResilienceManager,
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    FallbackConfig,
)

# Use shared logger
logger = get_shared_logger(__name__)


@dataclass
class MappingContext:
    """Context for mapping operations."""

    detector: str
    output: str
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    framework: Optional[str] = None
    confidence_threshold: float = 0.7


class CoreMapper:
    """
    Core mapping engine that consolidates all mapping functionality.

    Handles:
    - Model serving with vLLM, TGI, and CPU fallback backends
    - Taxonomy mapping and framework adaptation
    - Response validation and fallback mechanisms
    - Confidence-based routing
    """

    def __init__(self, settings: MapperSettings):
        self.settings = settings
        self.model_server: Optional[ModelServer] = None
        self.is_initialized = False

        # Group related components to reduce attribute count
        self.components = {
            "fallback_mapper": FallbackMapper(settings.detector_configs_path),
            "validation_pipeline": ValidationPipeline(
                confidence_threshold=settings.confidence_threshold,
                detector_configs_path=settings.detector_configs_path,
            ),
            "framework_adapter": FrameworkAdapter(settings.frameworks_path),
            "resilience_manager": ComprehensiveResilienceManager(),
        }

        self._setup_resilience_patterns()

    async def initialize(self) -> None:
        """Initialize the mapper with model server."""
        if self.is_initialized:
            return

        try:
            # Initialize model server based on configuration
            self.model_server = create_model_server(
                backend=self.settings.model_backend,
                model_path=self.settings.model_path,
                generation_config=GenerationConfig(
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    max_new_tokens=self.settings.max_new_tokens,
                ),
                **self.settings.backend_kwargs,
            )

            await self.model_server.load_model()
            self.is_initialized = True

            logger.info(
                "Core mapper initialized successfully",
                backend=self.settings.model_backend,
                model_path=self.settings.model_path,
            )

        except (ImportError, RuntimeError, ValueError) as e:
            logger.error("Failed to initialize core mapper", error=str(e))
            # Continue without model server - will use fallback only
            self.model_server = None
            self.is_initialized = True

    @track_request_metrics
    async def map_detector_output(self, request: MappingRequest) -> MappingResponse:
        """
        Map detector output to canonical taxonomy.

        Args:
            request: Mapping request with detector output

        Returns:
            MappingResponse: Mapped response with taxonomy and scores
        """
        # Set correlation ID for request tracking
        if hasattr(request, 'correlation_id') and request.correlation_id:
            set_correlation_id(request.correlation_id)
        
        # Validate inputs using shared validators
        try:
            validate_non_empty_string(request.detector, "detector")
            validate_non_empty_string(request.output, "output")
            if request.confidence_threshold is not None:
                validate_confidence_score(request.confidence_threshold)
        except ValidationError as e:
            logger.error("Input validation failed", error=str(e))
            raise ValidationError(f"Invalid mapping request: {str(e)}")
        
        if not self.is_initialized:
            await self.initialize()

        context = MappingContext(
            detector=request.detector,
            output=request.output,
            metadata=request.metadata,
            tenant_id=request.tenant_id,
            framework=request.framework,
            confidence_threshold=request.confidence_threshold
            or self.settings.confidence_threshold,
        )

        try:
            # Try model-based mapping first
            if self.model_server and await self.model_server.health_check():
                response = await self._model_based_mapping(context)

                # Validate response using comprehensive pipeline
                validation_result = self.components[
                    "validation_pipeline"
                ].validate_output(response, context.detector)

                if validation_result.overall_valid:
                    logger.info(
                        "Model-based mapping successful",
                        detector=context.detector,
                        confidence=getattr(validation_result, "confidence_score", 0.0),
                        correlation_id=get_correlation_id(),
                    )
                    return response

                if validation_result.requires_fallback:
                    logger.warning(
                        "Model mapping requires fallback",
                        detector=context.detector,
                        reason=validation_result.fallback_reason,
                        errors=validation_result.output_errors,
                        correlation_id=get_correlation_id(),
                    )
                    # Execute fallback through validation pipeline
                    fallback_response = self.components[
                        "validation_pipeline"
                    ].execute_fallback_if_needed(request, validation_result, response)
                    if fallback_response:
                        return fallback_response

            # Fall back to rule-based mapping
            logger.info("Using fallback mapping", detector=context.detector, correlation_id=get_correlation_id())
            return self._fallback_mapping(context)

        except ValidationError as e:
            logger.error(
                "Validation error in mapping, using fallback",
                detector=context.detector,
                error=str(e),
                correlation_id=get_correlation_id(),
            )
            return self._fallback_mapping(context)
        except ServiceUnavailableError as e:
            logger.error(
                "Service unavailable during mapping, using fallback", 
                detector=context.detector,
                error=str(e),
                correlation_id=get_correlation_id(),
            )
            return self._fallback_mapping(context)
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            logger.error(
                "Mapping failed with unexpected error, using fallback",
                detector=context.detector,
                error=str(e),
                error_type=type(e).__name__,
                correlation_id=get_correlation_id(),
            )
            return self._fallback_mapping(context)

    async def _model_based_mapping(self, context: MappingContext) -> MappingResponse:
        """Perform model-based mapping using the loaded model server."""
        if not self.model_server:
            raise RuntimeError("Model server not available")

        # Generate mapping using model with resilience patterns
        async def _model_call():
            return await self.model_server.generate_mapping(
                detector=context.detector,
                output=context.output,
                metadata=context.metadata,
            )

        # Execute with circuit breaker, retry, and bulkhead protection
        try:
            # Use circuit breaker and retry manager from resilience stack
            circuit_breaker = self.model_resilience.get("circuit_breaker")
            retry_manager = self.model_resilience.get("retry_manager")

            if circuit_breaker and retry_manager:
                raw_output = await retry_manager.execute_with_retry(
                    lambda: circuit_breaker.call(_model_call)
                )
            else:
                # Fallback to direct call if resilience not available
                raw_output = await _model_call()
        except Exception:
            # Fallback to rule-based mapping on any error
            logger.warning(
                "Model call failed, using fallback mapping", detector=context.detector
            )
            return self._fallback_mapping(context)

        # Parse and validate the model output
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse model output as JSON", error=str(e))
            raise ValueError(f"Invalid JSON output from model: {e}") from e

        # Convert to MappingResponse
        response = MappingResponse(
            taxonomy=parsed_output.get("taxonomy", []),
            scores=parsed_output.get("scores", {}),
            confidence=parsed_output.get("confidence", 0.0),
            notes=parsed_output.get("notes", ""),
            provenance=Provenance(detector=context.detector, raw_ref=context.output),
            version_info=VersionInfo(
                model_version=self.settings.model_version,
                taxonomy_version=self.settings.taxonomy_version,
                timestamp=datetime.utcnow(),
            ),
        )

        # Apply framework adaptation if requested
        if context.framework:
            response = await self._adapt_to_framework(response, context.framework)

        return response

    def _fallback_mapping(self, context: MappingContext) -> MappingResponse:
        """Perform rule-based fallback mapping."""
        return self.components["fallback_mapper"].map(
            detector=context.detector,
            output=context.output,
            reason="model_unavailable_or_low_confidence",
        )

    async def _adapt_to_framework(
        self, response: MappingResponse, framework: str
    ) -> MappingResponse:
        """
        Adapt canonical taxonomy to specific compliance framework.

        Args:
            response: Original mapping response
            framework: Target framework (e.g., "GDPR", "HIPAA", "SOC2")

        Returns:
            MappingResponse: Framework-adapted response
        """
        return self.components["framework_adapter"].adapt_to_framework(
            response, framework
        )

    async def batch_map(self, requests: List[MappingRequest]) -> List[MappingResponse]:
        """
        Process multiple mapping requests in batch.

        Args:
            requests: List of mapping requests

        Returns:
            List[MappingResponse]: List of mapping responses
        """
        if not self.is_initialized:
            await self.initialize()

        # Process requests concurrently
        tasks = [self.map_detector_output(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(
                    "Batch mapping failed for request",
                    request_index=i,
                    error=str(response),
                )
                # Create error response
                error_response = MappingResponse(
                    taxonomy=["OTHER.Error"],
                    scores={"OTHER.Error": 0.0},
                    confidence=0.0,
                    notes=f"Mapping failed: {str(response)}",
                    provenance=Provenance(
                        detector=requests[i].detector, raw_ref=requests[i].output
                    ),
                    version_info=None,
                )
                results.append(error_response)
            else:
                results.append(response)

        return results

    async def get_supported_detectors(self) -> List[str]:
        """Get list of supported detector types."""
        return self.components["fallback_mapper"].get_supported_detectors()

    async def get_supported_frameworks(self) -> List[str]:
        """Get list of supported compliance frameworks."""
        return self.components["framework_adapter"].get_supported_frameworks()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the mapper components.

        Returns:
            Dict[str, Any]: Health status
        """
        health_status = {
            "status": "healthy",
            "components": {
                "core_mapper": True,
                "fallback_mapper": True,
                "response_validator": True,
            },
            "model_server": False,
            "supported_detectors": len(await self.get_supported_detectors()),
            "supported_frameworks": len(await self.get_supported_frameworks()),
        }

        # Check model server health
        if self.model_server:
            try:
                model_healthy = await self.model_server.health_check()
                health_status["components"]["model_server"] = model_healthy
                health_status["model_server"] = model_healthy
            except (RuntimeError, ConnectionError, TimeoutError) as e:
                logger.warning("Model server health check failed", error=str(e))
                health_status["components"]["model_server"] = False
                health_status["model_server"] = False

        # Overall status
        all_critical_healthy = (
            health_status["components"]["core_mapper"]
            and health_status["components"]["fallback_mapper"]
        )

        health_status["status"] = "healthy" if all_critical_healthy else "degraded"

        return health_status

    def _setup_resilience_patterns(self) -> None:
        """Setup resilience patterns for core operations."""

        # Circuit breaker for model server calls
        model_server_cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=(ConnectionError, TimeoutError, RuntimeError),
        )

        # Retry configuration for model inference
        model_retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_enabled=True,
        )

        # Bulkhead for model operations
        model_bulkhead_config = BulkheadConfig(
            max_concurrent_calls=10, queue_size=50, timeout=30.0
        )

        # Fallback configuration
        model_fallback_config = FallbackConfig(
            fallback_timeout=5.0, enable_fallback=True
        )

        # Create resilience stack for model operations
        self.model_resilience = self.components[
            "resilience_manager"
        ].create_resilience_stack(
            name="model_server",
            circuit_breaker_config=model_server_cb_config,
            retry_config=model_retry_config,
            bulkhead_config=model_bulkhead_config,
            fallback_config=model_fallback_config,
        )

        logger.info("Resilience patterns configured for core mapper")

    async def shutdown(self) -> None:
        """Shutdown the mapper and cleanup resources."""
        if self.model_server:
            try:
                if hasattr(self.model_server, "close"):
                    await self.model_server.close()
            except (RuntimeError, ConnectionError) as e:
                logger.error("Error during model server shutdown", error=str(e))

        logger.info("Core mapper shutdown complete")
