"""
Domain entities for the Analysis Module.

This module contains the core domain entities that represent
the business concepts and rules of the analysis domain.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class AnalysisType(Enum):
    """Types of analysis that can be performed."""
    COVERAGE_GAP = "coverage_gap"
    FALSE_POSITIVE_TUNING = "false_positive_tuning"
    INCIDENT_SUMMARY = "incident_summary"
    INSUFFICIENT_DATA = "insufficient_data"


class VersionInfo(BaseModel):
    """Version information for auditability."""
    taxonomy: str = Field(description="Taxonomy version")
    frameworks: str = Field(description="Compliance frameworks version")
    analyst_model: str = Field(description="Analysis model version")


class EvidenceReference(BaseModel):
    """Structured evidence reference for analysis responses."""
    field_name: str = Field(description="Name of the referenced field")
    value: Any = Field(description="Value of the referenced field")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this evidence")
    source: str = Field(description="Source of the evidence (e.g., 'detector_output', 'coverage_metrics')")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When this evidence was collected")


class AnalysisRequest(BaseModel):
    """Domain entity for analysis request."""
    period: str = Field(
        pattern=r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$',
        description="RFC3339 interval start/end"
    )
    tenant: str = Field(min_length=1, max_length=64, description="Tenant identifier")
    app: str = Field(min_length=1, max_length=64, description="Application identifier")
    route: str = Field(min_length=1, max_length=256, description="Route identifier")
    required_detectors: List[str] = Field(
        min_length=1, 
        max_length=20, 
        description="List of required detector names"
    )
    observed_coverage: Dict[str, float] = Field(
        description="Observed coverage per detector (0.0-1.0)"
    )
    required_coverage: Dict[str, float] = Field(
        description="Required coverage per detector (0.0-1.0)"
    )
    detector_errors: Dict[str, Dict[str, Any]] = Field(
        description="Detector error information"
    )
    high_sev_hits: List[Dict[str, Any]] = Field(
        max_length=100,
        description="High severity hits with taxonomy and scores"
    )
    false_positive_bands: List[Dict[str, Any]] = Field(
        max_length=50,
        description="False positive analysis bands"
    )
    policy_bundle: str = Field(
        pattern=r'^[a-zA-Z0-9\-\.]+$',
        description="Policy bundle identifier"
    )
    env: Literal["dev", "stage", "prod"] = Field(description="Environment")

    @field_validator('observed_coverage', 'required_coverage')
    @classmethod
    def validate_coverage_values(cls, v):
        """Validate coverage values are between 0.0 and 1.0."""
        for key, value in v.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Coverage value {value} for {key} must be between 0.0 and 1.0")
        return v

    @field_validator('observed_coverage', 'required_coverage')
    @classmethod
    def validate_detector_keys_match(cls, v, info):
        """Validate that coverage keys match required_detectors."""
        if info.data and 'required_detectors' in info.data:
            required_detectors = info.data['required_detectors']
            for key in v.keys():
                if key not in required_detectors:
                    raise ValueError(f"Detector {key} not in required_detectors list")
        return v


class AnalysisResponse(BaseModel):
    """Domain entity for analysis response."""
    reason: str = Field(max_length=120, description="Concise explanation of the issue")
    remediation: str = Field(max_length=120, description="Suggested remediation")
    opa_diff: str = Field(max_length=2000, description="OPA/Rego policy diff or empty")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    confidence_cutoff_used: float = Field(description="Confidence cutoff applied")
    evidence_refs: List[str] = Field(
        min_length=1,
        description="References to source metrics used"
    )
    notes: str = Field(max_length=500, description="Additional notes")
    version_info: VersionInfo = Field(description="Version information")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")


class BatchAnalysisRequest(BaseModel):
    """Domain entity for batch analysis request."""
    requests: List[AnalysisRequest] = Field(
        min_length=1,
        max_length=100,
        description="List of analysis requests"
    )


class AnalysisErrorResponse(BaseModel):
    """Domain entity for analysis error response."""
    error_type: Literal[
        "validation_error", 
        "processing_error", 
        "timeout_error", 
        "confidence_fallback"
    ] = Field(description="Type of error")
    message: str = Field(description="Error message")
    request_id: str = Field(default_factory=lambda: str(uuid4()), description="Request ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")
    fallback_used: bool = Field(description="Whether fallback was used")
    mode: Literal["error", "fallback"] = Field(description="Response mode")
    template_response: Optional[AnalysisResponse] = Field(
        default=None,
        description="Template response if fallback was used"
    )


class BatchAnalysisResponse(BaseModel):
    """Domain entity for batch analysis response."""
    responses: List[Union[AnalysisResponse, AnalysisErrorResponse]] = Field(
        description="List of analysis responses or errors"
    )
    batch_id: str = Field(default_factory=lambda: str(uuid4()), description="Batch ID")
    idempotency_key: str = Field(description="Idempotency key")
    total_processing_time_ms: int = Field(ge=0, description="Total processing time")
    success_count: int = Field(ge=0, description="Number of successful analyses")
    error_count: int = Field(ge=0, description="Number of failed analyses")


class IdempotencyCache(BaseModel):
    """Domain entity for idempotency cache entry."""
    key: str = Field(description="Cache key")
    response: BatchAnalysisResponse = Field(description="Cached response")
    created_at: datetime = Field(description="Creation timestamp")
    expires_at: datetime = Field(description="Expiration timestamp")


class QualityMetrics(BaseModel):
    """Domain entity for quality metrics."""
    total_examples: int = Field(ge=0, description="Total number of examples evaluated")
    schema_valid_rate: float = Field(ge=0.0, le=1.0, description="Schema validation success rate")
    rubric_score: float = Field(ge=0.0, le=5.0, description="Average rubric score")
    opa_compile_success_rate: float = Field(ge=0.0, le=1.0, description="OPA compilation success rate")
    evidence_accuracy: float = Field(ge=0.0, le=1.0, description="Evidence reference accuracy")
    individual_rubric_scores: List[float] = Field(description="Individual rubric scores")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Evaluation timestamp")


class HealthStatus(BaseModel):
    """Domain entity for health status."""
    status: Literal["healthy", "unhealthy", "degraded"] = Field(description="Overall health status")
    service: str = Field(description="Service name")
    version: str = Field(description="Service version")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Health check timestamp")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Individual health checks")