"""
Data Transfer Objects (DTOs) for the Analysis Module.

This module contains DTOs that are used for data transfer between
layers and external interfaces.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from ..domain.entities import (
    AnalysisErrorResponse,
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    VersionInfo,
)


class AnalysisRequestDTO(BaseModel):
    """DTO for analysis request data transfer."""

    period: str
    tenant: str
    app: str
    route: str
    required_detectors: List[str]
    observed_coverage: Dict[str, float]
    required_coverage: Dict[str, float]
    detector_errors: Dict[str, Dict[str, Any]]
    high_sev_hits: List[Dict[str, Any]]
    false_positive_bands: List[Dict[str, Any]]
    policy_bundle: str
    env: Literal["dev", "stage", "prod"]

    def to_domain_entity(self) -> AnalysisRequest:
        """Convert DTO to domain entity."""
        return AnalysisRequest(
            period=self.period,
            tenant=self.tenant,
            app=self.app,
            route=self.route,
            required_detectors=self.required_detectors,
            observed_coverage=self.observed_coverage,
            required_coverage=self.required_coverage,
            detector_errors=self.detector_errors,
            high_sev_hits=self.high_sev_hits,
            false_positive_bands=self.false_positive_bands,
            policy_bundle=self.policy_bundle,
            env=self.env,
        )


class AnalysisResponseDTO(BaseModel):
    """DTO for analysis response data transfer."""

    reason: str
    remediation: str
    opa_diff: str
    confidence: float
    confidence_cutoff_used: float
    evidence_refs: List[str]
    notes: str
    version_info: VersionInfo
    request_id: str
    timestamp: datetime
    processing_time_ms: int

    @classmethod
    def from_domain_entity(cls, entity: AnalysisResponse) -> "AnalysisResponseDTO":
        """Create DTO from domain entity."""
        return cls(
            reason=entity.reason,
            remediation=entity.remediation,
            opa_diff=entity.opa_diff,
            confidence=entity.confidence,
            confidence_cutoff_used=entity.confidence_cutoff_used,
            evidence_refs=entity.evidence_refs,
            notes=entity.notes,
            version_info=entity.version_info,
            request_id=entity.request_id,
            timestamp=entity.timestamp,
            processing_time_ms=entity.processing_time_ms,
        )


class AnalysisErrorResponseDTO(BaseModel):
    """DTO for analysis error response data transfer."""

    error_type: Literal[
        "validation_error", "processing_error", "timeout_error", "confidence_fallback"
    ]
    message: str
    request_id: str
    timestamp: datetime
    fallback_used: bool
    mode: Literal["error", "fallback"]
    template_response: Optional[AnalysisResponseDTO] = None

    @classmethod
    def from_domain_entity(
        cls, entity: AnalysisErrorResponse
    ) -> "AnalysisErrorResponseDTO":
        """Create DTO from domain entity."""
        template_response = None
        if entity.template_response:
            template_response = AnalysisResponseDTO.from_domain_entity(
                entity.template_response
            )

        return cls(
            error_type=entity.error_type,
            message=entity.message,
            request_id=entity.request_id,
            timestamp=entity.timestamp,
            fallback_used=entity.fallback_used,
            mode=entity.mode,
            template_response=template_response,
        )


class BatchAnalysisRequestDTO(BaseModel):
    """DTO for batch analysis request data transfer."""

    requests: List[AnalysisRequestDTO]

    def to_domain_entity(self) -> BatchAnalysisRequest:
        """Convert DTO to domain entity."""
        return BatchAnalysisRequest(
            requests=[req.to_domain_entity() for req in self.requests]
        )


class BatchAnalysisResponseDTO(BaseModel):
    """DTO for batch analysis response data transfer."""

    responses: List[Union[AnalysisResponseDTO, AnalysisErrorResponseDTO]]
    batch_id: str
    idempotency_key: str
    total_processing_time_ms: int
    success_count: int
    error_count: int

    @classmethod
    def from_domain_entity(
        cls, entity: BatchAnalysisResponse
    ) -> "BatchAnalysisResponseDTO":
        """Create DTO from domain entity."""
        responses = []
        for response in entity.responses:
            if isinstance(response, AnalysisResponse):
                responses.append(AnalysisResponseDTO.from_domain_entity(response))
            elif isinstance(response, AnalysisErrorResponse):
                responses.append(AnalysisErrorResponseDTO.from_domain_entity(response))

        return cls(
            responses=responses,
            batch_id=entity.batch_id,
            idempotency_key=entity.idempotency_key,
            total_processing_time_ms=entity.total_processing_time_ms,
            success_count=entity.success_count,
            error_count=entity.error_count,
        )


class AnalysisMetricsDTO(BaseModel):
    """DTO for analysis metrics data transfer."""

    schema_valid_rate: float
    template_fallback_rate: float
    opa_compile_success_rate: float
    average_confidence: float
    processing_time_p95: float
    error_rate: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QualityEvaluationDTO(BaseModel):
    """DTO for quality evaluation data transfer."""

    total_examples: int
    schema_valid_rate: float
    rubric_score: float
    opa_compile_success_rate: float
    evidence_accuracy: float
    individual_rubric_scores: List[float]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthCheckDTO(BaseModel):
    """DTO for health check data transfer."""

    status: Literal["healthy", "unhealthy", "degraded"]
    service: str
    version: str
    timestamp: datetime
    checks: Dict[str, Any] = Field(default_factory=dict)
