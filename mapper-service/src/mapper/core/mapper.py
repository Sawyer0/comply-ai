"""
Core mapping functionality for the Mapper Service.

This module consolidates the core mapping logic from the original llama-mapper
implementation, providing unified mapping from detector outputs to canonical
taxonomy with framework adaptation.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Import shared components first
from ..shared_integration import (
    get_shared_logger,
    get_correlation_id,
)

# Import service-specific components
from ..serving.fallback_mapper import FallbackMapper
from ..validation.validation_pipeline import ValidationPipeline
from ..taxonomy.framework_adapter import FrameworkAdapter
from ..schemas.models import MappingRequest, MappingResponse, Provenance, VersionInfo
from ..config.settings import MapperSettings
from ..shared_lib.interfaces.mapper import (
    MappingRequest as CanonicalMappingRequest,
    MappingResponse as CanonicalMappingResponse,
    MappingResult,
    ComplianceMapping,
    ValidationResult,
    CostMetrics,
    ModelMetrics,
    ComplianceStatus,
)
from ..shared_lib.interfaces.analysis import (
    CanonicalTaxonomyResult,
    RiskLevel as CanonicalRiskLevel,
)
from .ports import CanonicalTaxonomyPort, FrameworkMappingPort, ModelInferencePort
from .domain import DetectorMappingCommand, MappingContext as DomainMappingContext

# Use shared logger
logger = get_shared_logger(__name__)


class CoreMapper:
    """
    Core mapping engine that consolidates all mapping functionality.

    Handles:
    - Model serving with vLLM, TGI, and CPU fallback backends
    - Taxonomy mapping and framework adaptation
    - Response validation and fallback mechanisms
    - Confidence-based routing
    """
    def __init__(
        self,
        settings: MapperSettings,
        canonical_taxonomy_port: CanonicalTaxonomyPort,
        framework_mapping_port: FrameworkMappingPort,
        model_inference_port: ModelInferencePort,
    ):
        self.settings = settings
        self.is_initialized = False

        self._canonical_taxonomy = canonical_taxonomy_port
        self._framework_mapping = framework_mapping_port
        self._model_inference = model_inference_port

        # Group related components to reduce attribute count
        self.components = {
            "fallback_mapper": FallbackMapper(settings.detector_configs_path),
            "validation_pipeline": ValidationPipeline(
                confidence_threshold=settings.confidence_threshold,
                detector_configs_path=settings.detector_configs_path,
            ),
            "framework_adapter": FrameworkAdapter(settings.frameworks_path),
        }

    async def initialize(self) -> None:
        """Initialize the mapper with model server."""
        if self.is_initialized:
            return

        try:
            await self._model_inference.initialize()
            self.is_initialized = True

            logger.info(
                "Core mapper initialized successfully",
                backend=self.settings.model_backend,
                model_path=self.settings.model_path,
            )

        except (ImportError, RuntimeError, ValueError) as e:
            logger.error("Failed to initialize core mapper", error=str(e))
            # Continue without model server - will use fallback only
            self.is_initialized = True

    async def map_detector_output(self, request: MappingRequest) -> MappingResponse:
        """Map raw detector output to canonical taxonomy using model + fallback."""

        if not self.is_initialized:
            await self.initialize()

        command = DetectorMappingCommand.from_api_request(request, self.settings)
        context = DomainMappingContext.from_command(command)

        validation_pipeline: ValidationPipeline = self.components["validation_pipeline"]

        model_response: Optional[MappingResponse] = None
        error_context: Optional[Dict[str, Any]] = None

        if await self._model_inference.is_available():
            try:
                model_response = await self._model_based_mapping(context)
            except Exception as e:  # pragma: no cover - defensive
                logger.error(
                    "Model-based mapping failed",
                    error=str(e),
                    detector=context.detector,
                )
                error_context = {"error_type": "model_error", "details": str(e)}
        else:
            error_context = {"error_type": "model_unavailable"}

        validation_result = validation_pipeline.validate_complete_flow(
            request=request,
            response=model_response,
            error_context=error_context,
        )

        fallback_response = validation_pipeline.execute_fallback_if_needed(
            request=request,
            validation_result=validation_result,
            original_response=model_response,
        )

        final_response = fallback_response or model_response

        if final_response is None:
            raise RuntimeError("Mapping failed: no response from model or fallback")

        return final_response

    async def map_canonical(self, request: CanonicalMappingRequest) -> CanonicalMappingResponse:
        """Map canonical taxonomy results to compliance frameworks.

        This path operates on shared MappingRequest / MappingResponse models and
        uses the centralized framework_mapping_registry to adapt canonical
        taxonomy labels (category.subcategory[.type]) to framework-specific
        controls.
        """

        start_time = datetime.utcnow()

        # Ensure mapper is initialized for telemetry and shared components
        if not self.is_initialized:
            await self.initialize()

        analysis_response = request.analysis_response
        canonical_results: List[CanonicalTaxonomyResult] = (
            analysis_response.canonical_results or []
        )

        mapping_results: List[MappingResult] = []

        # Precompute canonical labels in canonical taxonomy format
        def _canonical_label(result: CanonicalTaxonomyResult) -> str:
            """Build a canonical taxonomy label aligned with canonical_taxonomy.

            Prefer category.subcategory, with basic normalization to match the
            canonical taxonomy's label set when possible. Fall back to the
            raw category/subcategory combination if no exact match is found.
            """

            raw_category = str(result.category)
            raw_subcategory = str(result.subcategory) if result.subcategory else None

            # First try the raw combination
            if raw_subcategory:
                candidate = f"{raw_category}.{raw_subcategory}"
            else:
                candidate = raw_category

            if self._canonical_taxonomy.is_valid_label(candidate):
                return candidate

            # Try a normalized form (e.g. PII.Contact, SECURITY.Access)
            norm_category = raw_category.upper()
            norm_subcategory = (
                raw_subcategory.title() if raw_subcategory is not None else None
            )
            if norm_subcategory:
                candidate2 = f"{norm_category}.{norm_subcategory}"
            else:
                candidate2 = norm_category

            if self._canonical_taxonomy.is_valid_label(candidate2):
                return candidate2

            # Fallback to the original candidate if nothing matches exactly
            return candidate

        for canonical_result in canonical_results:
            label = _canonical_label(canonical_result)

            framework_mappings: List[ComplianceMapping] = []

            for framework in request.target_frameworks:
                try:
                    mapped = self._framework_mapping.map_to_framework([label], framework)
                    framework_label = mapped.get(label, label)
                except Exception as e:  # pragma: no cover - defensive
                    logger.warning(
                        "Framework mapping failed, using canonical label",
                        framework=framework,
                        canonical_label=label,
                        error=str(e),
                    )
                    framework_label = label

                compliance_status = _compliance_status_from_risk(
                    canonical_result.risk_level
                )

                framework_mappings.append(
                    ComplianceMapping(
                        framework=framework,
                        control_id=framework_label,
                        control_name=framework_label,
                        requirement=f"Mapped from canonical label {label}",
                        evidence_type="canonical_taxonomy",
                        compliance_status=compliance_status,
                        confidence=canonical_result.confidence,
                        metadata={
                            "canonical_label": label,
                            "framework_label": framework_label,
                        },
                    )
                )

            # Derive mapping-level confidence and validation
            confidence = (
                max((m.confidence for m in framework_mappings), default=0.0)
                if framework_mappings
                else canonical_result.confidence
            )

            validation_result = ValidationResult(
                is_valid=True,
                schema_compliance=True,
                confidence_threshold_met=True,
                validation_errors=[],
                validation_warnings=[],
            )

            mapping_results.append(
                MappingResult(
                    canonical_result=canonical_result,
                    framework_mappings=framework_mappings,
                    confidence=confidence,
                    validation_result=validation_result,
                    fallback_used=False,
                )
            )

        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        overall_confidence = max(
            (mr.confidence for mr in mapping_results), default=0.0
        )

        cost_metrics = CostMetrics(
            tokens_processed=0,
            inference_cost=0.0,
            storage_cost=0.0,
            total_cost=0.0,
            cost_per_request=0.0,
        )

        model_metrics = ModelMetrics(
            model_name="canonical-framework-mapper",
            model_version="1.0.0",
            inference_time_ms=processing_time_ms,
            gpu_utilization=0.0,
            memory_usage_mb=0,
            batch_size=1,
        )

        response = CanonicalMappingResponse(
            request_id=request.correlation_id or str(uuid.uuid4()),
            success=True,
            processing_time_ms=processing_time_ms,
            correlation_id=request.correlation_id or get_correlation_id(),
            mapping_results=mapping_results,
            overall_confidence=overall_confidence,
            cost_metrics=cost_metrics,
            model_metrics=model_metrics,
            recommendations=[],
        )

        return response

    async def _model_based_mapping(self, context: DomainMappingContext) -> MappingResponse:
        try:
            raw_output = await self._model_inference.generate_mapping(
                detector=context.detector,
                output=context.output,
                metadata=context.metadata,
            )
        except Exception:
            logger.warning(
                "Model call failed, using fallback mapping", detector=context.detector
            )
            return self._fallback_mapping(context)

        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse model output as JSON", error=str(e))
            raise ValueError(f"Invalid JSON output from model: {e}") from e

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

        if context.framework:
            response = await self._adapt_to_framework(response, context.framework)

        return response

    def _fallback_mapping(self, context: DomainMappingContext) -> MappingResponse:
        return self.components["fallback_mapper"].map(
            detector=context.detector,
            output=context.output,
            reason="model_unavailable_or_low_confidence",
        )

    async def _adapt_to_framework(
        self, response: MappingResponse, framework: str
    ) -> MappingResponse:
        return self.components["framework_adapter"].adapt_to_framework(
            response, framework
        )

    async def batch_map(self, requests: List[MappingRequest]) -> List[MappingResponse]:
        if not self.is_initialized:
            await self.initialize()

        tasks = [self._safe_map_request(request) for request in requests]
        responses = await asyncio.gather(*tasks)

        return list(responses)

    async def _safe_map_request(self, request: MappingRequest) -> MappingResponse:
        """Safely map a single request, converting errors into error responses."""
        try:
            return await self.map_detector_output(request)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Batch mapping failed for request",
                detector=request.detector,
                error=str(exc),
            )
            return MappingResponse(
                taxonomy=["OTHER.Error"],
                scores={"OTHER.Error": 0.0},
                confidence=0.0,
                notes=f"Mapping failed: {str(exc)}",
                provenance=Provenance(
                    detector=request.detector,
                    raw_ref=request.output,
                ),
                version_info=None,
            )

    async def get_supported_detectors(self) -> List[str]:
        return self.components["fallback_mapper"].get_supported_detectors()

    async def get_supported_frameworks(self) -> List[str]:
        return self.components["framework_adapter"].get_supported_frameworks()

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the mapper components."""

        health_status: Dict[str, Any] = {
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

        try:
            model_healthy = await self._model_inference.health_check()
            health_status["components"]["model_server"] = model_healthy
            health_status["model_server"] = model_healthy
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Model server health check failed", error=str(e))
            health_status["components"]["model_server"] = False
            health_status["model_server"] = False

        all_critical_healthy = (
            health_status["components"]["core_mapper"]
            and health_status["components"]["fallback_mapper"]
        )

        health_status["status"] = "healthy" if all_critical_healthy else "degraded"

        return health_status

    async def shutdown(self) -> None:
        """Shutdown the mapper and cleanup resources."""
        await self._model_inference.shutdown()
        logger.info("Core mapper shutdown complete")


def _compliance_status_from_risk(
    risk_level: CanonicalRiskLevel,
) -> ComplianceStatus:
    """Map canonical risk level to a coarse compliance status.

    This provides a simple, deterministic mapping for initial framework
    mappings. HIGH/CRITICAL risk is treated as NON_COMPLIANT, while LOW
    risk is treated as COMPLIANT. MEDIUM risk is conservative and also
    treated as NON_COMPLIANT by default.
    """

    if risk_level in (CanonicalRiskLevel.CRITICAL, CanonicalRiskLevel.HIGH):
        return ComplianceStatus.NON_COMPLIANT
    if risk_level == CanonicalRiskLevel.MEDIUM:
        return ComplianceStatus.NON_COMPLIANT
    return ComplianceStatus.COMPLIANT
