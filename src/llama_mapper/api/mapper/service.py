"""Mapping core logic extracted from the FastAPI wiring."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...security.redaction import SENSITIVE_KEYS, redact_dict
from ...storage.manager import StorageRecord
from ..models import DetectorRequest, MappingResponse, Provenance, VersionInfo

if TYPE_CHECKING:  # pragma: no cover
    from .app import MapperAPI

logger = logging.getLogger(__name__)


@dataclass
class MappingContext:
    """Context information for mapping operations."""

    request_id: str
    mapping_method: str
    fallback_reason: Optional[str]


class MappingService:
    """Encapsulates mapping logic so the FastAPI layer stays thin."""

    def __init__(self, mapper: "MapperAPI") -> None:
        self._mapper = mapper

    def is_raw_content_like(self, obj: Any, depth: int = 0) -> bool:
        """Heuristic to detect raw customer content in payloads."""
        if obj is None or depth > 6:
            return False
        suspicious_keys = {
            "text",
            "content",
            "raw",
            "body",
            "document",
            "html",
            "markdown",
            "message",
            "attachment",
            "blob",
        }
        try:
            if isinstance(obj, str):
                if len(obj) >= 2048:
                    return True
                if obj.count("\n") >= 3:
                    return True
                if re.fullmatch(r"[A-Za-z0-9+/=]{256,}", obj or ""):
                    return True
                return False
            if isinstance(obj, dict):
                lowered_sensitive = {s.lower() for s in SENSITIVE_KEYS}
                for key, value in obj.items():
                    key_lower = str(key).lower()
                    if key_lower in suspicious_keys or key_lower in lowered_sensitive:
                        if isinstance(value, str) and (
                            len(value) >= 512 or value.count("\n") >= 2
                        ):
                            return True
                    if self.is_raw_content_like(value, depth + 1):
                        return True
                return False
            if isinstance(obj, (list, tuple)):
                return any(self.is_raw_content_like(item, depth + 1) for item in obj)
        except (TypeError, AttributeError, RecursionError) as _:
            # Content validation failed - this can happen with complex nested structures
            # Return False to be conservative (assume content is safe)
            return False
        return False

    async def map_single_request(
        self, request: DetectorRequest, request_id: str
    ) -> MappingResponse:
        """Map a single detector request to canonical taxonomy.

        Args:
            request: The detector request containing raw output
            request_id: Unique identifier for the request

        Returns:
            MappingResponse with canonical taxonomy labels and scores
        """
        provenance = Provenance(
            detector=request.detector,
            tenant_id=request.tenant_id,
            ts=datetime.now(timezone.utc),
            raw_ref=None,
        )

        mapping_method = "model"
        fallback_reason: Optional[str] = None

        if self._should_use_fallback_for_rules_only_mode():
            return await self._handle_rules_only_fallback(
                request, provenance, request_id
            )

        try:
            model_output = await self._generate_model_mapping_with_timeout(request)
        except (RuntimeError, ConnectionError, OSError, asyncio.TimeoutError) as exc:
            # Model failed, use fallback mapping
            logger.warning("Model failed for detector %s: %s", request.detector, exc)
            self._mapper.metrics_collector.record_fallback_usage(
                request.detector, "model_failure"
            )
            mapping_method = "fallback"
            fallback_reason = "model_failure"
            fallback_result = self._mapper.fallback_mapper.map(
                request.detector, request.output, reason=fallback_reason
            )
            self._mapper.version_manager.apply_to_provenance(provenance)
            provenance.mapping_method = mapping_method
            # Set model_version from version manager
            version_info = self._mapper.version_manager.get_version_info_dict()
            provenance.model_version = version_info.get("model", "unknown")
            fallback_result.provenance = provenance
            fallback_result.notes = (
                self._mapper.version_manager.annotate_notes_with_versions(
                    f"Rule-based mapping due to model failure: {exc}"
                )
            )
            vi = self._mapper.version_manager.get_version_info_dict()
            # Ensure all version info values are strings
            safe_vi = {
                "taxonomy": str(vi.get("taxonomy", "unknown")),
                "frameworks": str(vi.get("frameworks", "unknown")),
                "model": str(vi.get("model", "unknown")),
            }
            try:
                fallback_result.version_info = VersionInfo(**safe_vi)
            except (TypeError, ValueError, KeyError) as _:
                # Version info parsing failed - continuing without version info
                pass
            await self._persist_mapping_result(
                request,
                fallback_result,
                provenance,
                MappingContext(request_id, mapping_method, fallback_reason),
            )
            return fallback_result

        validation_result = self._validate_model_output(model_output, request)
        is_valid, parsed_output, confidence_threshold = validation_result

        # Check if confidence is above threshold
        if is_valid and parsed_output:
            confidence = getattr(parsed_output, "confidence", 0.0)
            if confidence >= confidence_threshold:
                return await self._process_valid_model_output(
                    model_output, request, provenance, request_id
                )

        # Model output is invalid, use fallback mapping
        logger.warning(
            "Model confidence %s below threshold %s",
            getattr(parsed_output, "confidence", 0.0) if parsed_output else 0.0,
            confidence_threshold,
        )
        self._mapper.metrics_collector.record_fallback_usage(
            request.detector, "low_confidence"
        )
        mapping_method = "fallback"
        fallback_reason = "low_confidence"
        fallback_result = self._mapper.fallback_mapper.map(
            request.detector, request.output, reason=fallback_reason
        )
        self._mapper.version_manager.apply_to_provenance(provenance)
        provenance.mapping_method = mapping_method
        # Set model_version from version manager
        version_info = self._mapper.version_manager.get_version_info_dict()
        provenance.model_version = version_info.get("model", "unknown")
        fallback_result.provenance = provenance
        fallback_result.notes = (
            self._mapper.version_manager.annotate_notes_with_versions(
                "Rule-based mapping due to low confidence"
            )
        )
        vi = self._mapper.version_manager.get_version_info_dict()
        # Ensure all version info values are strings
        safe_vi = {
            "taxonomy": str(vi.get("taxonomy", "unknown")),
            "frameworks": str(vi.get("frameworks", "unknown")),
            "model": str(vi.get("model", "unknown")),
        }
        try:
            fallback_result.version_info = VersionInfo(**safe_vi)
        except (TypeError, ValueError, KeyError) as _:
            # Version info parsing failed - continuing without version info
            pass
        await self._persist_mapping_result(
            request,
            fallback_result,
            provenance,
            MappingContext(request_id, mapping_method, fallback_reason),
        )
        return fallback_result

    def _is_model_output_valid(
        self, model_output: str, request: DetectorRequest
    ) -> bool:
        """Check if model output is valid and meets confidence threshold."""
        is_valid, _, _ = self._validate_model_output(model_output, request)
        return is_valid

    async def _process_valid_model_output(
        self,
        model_output: str,
        request: DetectorRequest,
        provenance: Provenance,
        request_id: str,
    ) -> MappingResponse:
        """Process valid model output and return mapping response."""
        parsed_output = self._mapper.json_validator.parse_output(model_output)
        self._mapper.metrics_collector.record_model_success(request.detector)
        self._mapper.version_manager.apply_to_provenance(provenance)

        if not isinstance(parsed_output, MappingResponse):
            try:
                vi = self._mapper.version_manager.get_version_info_dict()
                # Ensure all version info values are strings
                safe_vi = {
                    "taxonomy": str(vi.get("taxonomy", "unknown")),
                    "frameworks": str(vi.get("frameworks", "unknown")),
                    "model": str(vi.get("model", "unknown")),
                }
                parsed_output = MappingResponse(
                    taxonomy=list(getattr(parsed_output, "taxonomy", []) or []),
                    scores=dict(getattr(parsed_output, "scores", {}) or {}),
                    confidence=float(getattr(parsed_output, "confidence", 0.0) or 0.0),
                    notes=getattr(parsed_output, "notes", None),
                    provenance=None,
                    policy_context=getattr(parsed_output, "policy_context", None),
                    version_info=VersionInfo(**safe_vi),
                )
            except (TypeError, ValueError, AttributeError) as exc:
                # pragma: no cover - defensive path
                raise RuntimeError("parsed_output_invalid") from exc

        provenance.mapping_method = "model"
        # Set model_version from version manager
        version_info = self._mapper.version_manager.get_version_info_dict()
        provenance.model_version = version_info.get("model", "unknown")
        parsed_output.provenance = provenance
        parsed_output.notes = self._mapper.version_manager.annotate_notes_with_versions(
            parsed_output.notes or ""
        )
        await self._persist_mapping_result(
            request,
            parsed_output,
            provenance,
            MappingContext(request_id, "model", None),
        )
        return parsed_output

    async def _handle_invalid_model_output(
        self,
        model_output: str,
        request: DetectorRequest,
        provenance: Provenance,
        context: MappingContext,
    ) -> MappingResponse:
        """Handle invalid model output with fallback mapping."""
        is_valid, validation_errors = self._mapper.json_validator.validate(model_output)
        self._mapper.metrics_collector.record_schema_validation(
            request.detector, is_valid
        )

        if not is_valid:
            logger.warning("Schema validation failed: %s", validation_errors)
            self._mapper.metrics_collector.record_fallback_usage(
                request.detector, "schema_validation_failed"
            )
            context.mapping_method = "fallback"
            context.fallback_reason = "schema_validation_failed"
        else:
            logger.info("Using fallback mapping for detector %s", request.detector)
            context.mapping_method = "fallback"
            context.fallback_reason = "model_error"

        fallback_result = self._mapper.fallback_mapper.map(
            request.detector, request.output, reason=context.fallback_reason
        )
        self._mapper.version_manager.apply_to_provenance(provenance)
        fallback_result.provenance = provenance
        fallback_result.notes = (
            self._mapper.version_manager.annotate_notes_with_versions(
                "Generated using rule-based fallback mapping"
            )
        )
        vi = self._mapper.version_manager.get_version_info_dict()
        # Ensure all version info values are strings
        safe_vi = {
            "taxonomy": str(vi.get("taxonomy", "unknown")),
            "frameworks": str(vi.get("frameworks", "unknown")),
            "model": str(vi.get("model", "unknown")),
        }
        try:
            fallback_result.version_info = VersionInfo(**safe_vi)
        except (TypeError, ValueError, KeyError) as _:
            # Version info parsing failed - continuing without version info
            pass
        fallback_metric = (
            context.fallback_reason.split("_", maxsplit=1)[0]
            if context.fallback_reason
            else "unknown"
        )
        self._mapper.metrics_collector.record_fallback_usage(
            request.detector, fallback_metric
        )
        await self._persist_mapping_result(
            request,
            fallback_result,
            provenance,
            context,
        )
        return fallback_result

    async def _persist_mapping_result(
        self,
        request: DetectorRequest,
        response_obj: MappingResponse,
        provenance: Provenance,
        context: MappingContext,
    ) -> None:
        # Group related parameters to reduce function complexity
        metadata = {
            "mapping_method": context.mapping_method,
            "fallback_reason": context.fallback_reason,
            "request_id": context.request_id,
        }
        taxonomy_hit = (
            response_obj.taxonomy[0] if response_obj.taxonomy else "OTHER.Unknown"
        )
        version_info = self._mapper.version_manager.get_version_info_dict()
        model_version = version_info.get("model", "unknown")

        metadata: Dict[str, Any] = {
            "detector": request.detector,
            "confidence": response_obj.confidence,
            "taxonomy_hit": taxonomy_hit,
            "mapping_method": context.mapping_method,
        }
        if context.fallback_reason:
            metadata["fallback_reason"] = context.fallback_reason
        if request.metadata:
            try:
                metadata["request_metadata"] = redact_dict(request.metadata)
            except (TypeError, AttributeError, KeyError) as _:
                # Request metadata redaction failed - using empty dict to avoid data exposure
                metadata["request_metadata"] = {}
        if getattr(provenance, "model", None):
            metadata["model_provenance"] = provenance.model

        storage_manager = self._mapper.storage_manager
        if storage_manager:
            try:
                source_payload: Dict[str, Any] = {
                    "detector": request.detector,
                    "output": request.output,
                }
                if request.metadata:
                    try:
                        source_payload["metadata"] = redact_dict(request.metadata)
                    except (TypeError, AttributeError, KeyError) as _:
                        # Request metadata redaction failed - using empty dict
                        # to avoid data exposure
                        source_payload["metadata"] = {}
                record = StorageRecord(
                    id=context.request_id,
                    source_data=json.dumps(
                        source_payload, ensure_ascii=False, default=str
                    ),
                    mapped_data=response_obj.model_dump_json(),
                    model_version=model_version,
                    timestamp=datetime.now(timezone.utc),
                    metadata=dict(metadata),
                    tenant_id=request.tenant_id or "unknown",
                )
                await storage_manager.store_record(record)
            except (ConnectionError, TimeoutError, AttributeError, OSError) as _:
                logger.warning(
                    "Failed to persist mapping record to storage", exc_info=True
                )

        try:
            self._mapper.audit_trail_manager.create_audit_record(
                tenant_id=request.tenant_id or "unknown",
                detector_type=request.detector,
                taxonomy_hit=taxonomy_hit,
                confidence_score=response_obj.confidence,
                model_version=model_version,
                mapping_method=context.mapping_method,
                metadata=dict(metadata),
            )
            self._mapper.audit_trail_manager.create_lineage_record(
                detector_name=request.detector,
                detector_version=str(
                    (request.metadata or {}).get("detector_version", "unknown")
                ),
                original_label=request.output,
                canonical_label=taxonomy_hit,
                confidence_score=response_obj.confidence,
                mapping_method=context.mapping_method,
                model_version=model_version,
                tenant_id=request.tenant_id,
            )
        except (ConnectionError, TimeoutError, AttributeError, OSError) as _:
            logger.warning(
                "Failed to update audit trail for request %s",
                context.request_id,
                exc_info=True,
            )

    def create_error_response(
        self, detector: str, error_message: str
    ) -> MappingResponse:
        """Create an error response for failed mapping operations.

        Args:
            detector: The detector that failed
            error_message: Description of the error

        Returns:
            MappingResponse with error information
        """
        version_info = self._mapper.version_manager.get_version_info_dict()
        # Ensure all version info values are strings
        safe_version_info = {
            "taxonomy": str(version_info.get("taxonomy", "unknown")),
            "frameworks": str(version_info.get("frameworks", "unknown")),
            "model": str(version_info.get("model", "unknown")),
        }
        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.0},
            confidence=0.0,
            notes=f"Mapping failed: {error_message}",
            provenance=Provenance(detector=detector, raw_ref=None),
            version_info=VersionInfo(**safe_version_info),
        )

    def _should_use_fallback_for_rules_only_mode(self) -> bool:
        """Check if kill-switch is active and fallback should be used."""
        try:
            runtime_mode = getattr(
                self._mapper.config_manager.serving, "mode", "hybrid"
            )
        except (AttributeError, TypeError) as _:
            # Configuration retrieval failed, using hybrid mode as default
            runtime_mode = "hybrid"
        return runtime_mode == "rules_only"

    def _validate_model_output(
        self, model_output: str, request: DetectorRequest
    ) -> tuple[bool, Optional[Any], float]:
        """Validate model output and return validation result, parsed output, and confidence threshold."""
        is_valid, _ = self._mapper.json_validator.validate(model_output)
        self._mapper.metrics_collector.record_schema_validation(
            request.detector, is_valid
        )

        if not is_valid:
            return False, None, 0.0

        parsed_output = self._mapper.json_validator.parse_output(model_output)
        self._mapper.metrics_collector.record_confidence_score(
            request.detector, getattr(parsed_output, "confidence", 0.0)
        )
        confidence_threshold = self._mapper.config_manager.confidence.threshold
        return True, parsed_output, confidence_threshold

    async def _generate_model_mapping_with_timeout(
        self, request: DetectorRequest
    ) -> str:
        """Generate model mapping with proper timeout handling."""
        try:
            timeout_ms = getattr(
                self._mapper.config_manager.serving, "mapper_timeout_ms", 500
            )
            if not isinstance(timeout_ms, (int, float)):
                timeout_ms = 500
        except (AttributeError, TypeError) as _:
            # Configuration retrieval failed, using default timeout
            timeout_ms = 500
        return await asyncio.wait_for(
            self._mapper.model_server.generate_mapping(
                detector=request.detector,
                output=request.output,
                metadata=request.metadata,
            ),
            timeout=float(timeout_ms) / 1000.0,
        )

    async def _handle_rules_only_fallback(
        self, request: DetectorRequest, provenance: Provenance, request_id: str
    ) -> MappingResponse:
        """Handle fallback mapping when rules-only mode is active."""
        logger.warning("Kill-switch active: forcing rule-only mapping")
        mapping_method = "fallback"
        fallback_reason = "kill_switch"
        fallback_result = self._mapper.fallback_mapper.map(
            request.detector, request.output, reason=fallback_reason
        )
        self._mapper.version_manager.apply_to_provenance(provenance)
        fallback_result.provenance = provenance
        fallback_result.notes = (
            self._mapper.version_manager.annotate_notes_with_versions(
                "Kill-switch active: rule-based mapping enforced"
            )
        )
        self._mapper.metrics_collector.record_fallback_usage(
            request.detector, "kill_switch"
        )
        await self._persist_mapping_result(
            request,
            fallback_result,
            provenance,
            MappingContext(request_id, mapping_method, fallback_reason),
        )
        return fallback_result
