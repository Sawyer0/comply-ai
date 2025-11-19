"""Orchestration service interfaces with enhanced validation following SRP."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, model_validator

from .base import BaseRequest, BaseResponse
from .common import ProcessingMode, HealthStatus, Severity, JobStatus, RiskLevel
from ..validation.common_validators import (
    validate_non_empty_string,
    validate_unique_list,
    validate_confidence_score,
    validate_positive_number,
)


class DetectorStatus(str, Enum):
    """Detector status enumeration specific to orchestration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


class OrchestrationRequest(BaseRequest):
    """Request for detector orchestration."""

    content: str = Field(
        description="Content to analyze (will be hashed for privacy)",
        max_length=10000,
        min_length=1,
    )
    detector_types: List[str] = Field(
        description="List of detector types to execute"
    )
    policy_bundle: Optional[str] = Field(None, description="Policy bundle to apply")
    processing_mode: ProcessingMode = Field(
        ProcessingMode.STANDARD, description="Processing mode for orchestration"
    )
    max_detectors: Optional[int] = Field(
        None, description="Maximum number of detectors to execute", ge=1, le=10
    )
    timeout_seconds: Optional[int] = Field(
        None, description="Request timeout in seconds", ge=1, le=300
    )

    @validator("content")
    def validate_content(cls, v):
        """Validate that content is not empty."""
        return validate_non_empty_string(cls, v)

    @validator("detector_types")
    def validate_detector_types(cls, v):
        """Validate detector types and remove duplicates."""
        if not v:
            raise ValueError("detector_types cannot be empty")
        if len(v) > 20:
            raise ValueError("detector_types cannot contain more than 20 items")
        return validate_unique_list(cls, v)


class DetectorResult(BaseModel):
    """Result from a single detector."""

    detector_id: str = Field(description="Detector identifier")
    detector_type: str = Field(description="Type of detector")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
    category: str = Field(description="Detection category")
    subcategory: Optional[str] = Field(None, description="Detection subcategory")
    severity: Severity = Field(description="Severity level")
    findings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed findings"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds", ge=0
    )

    @validator("detector_id")
    def validate_detector_id(cls, v):
        """Validate detector ID is a non-empty string."""
        return validate_non_empty_string(cls, v)

    @validator("category")
    def validate_category(cls, v):
        """Validate category is a non-empty string."""
        return validate_non_empty_string(cls, v)


class AggregationSummary(BaseModel):
    """Summary of detector aggregation."""

    total_detectors: int = Field(description="Total number of detectors", ge=0)
    successful_detectors: int = Field(
        description="Number of successful detectors", ge=0
    )
    failed_detectors: int = Field(description="Number of failed detectors", ge=0)
    average_confidence: Optional[float] = Field(
        None, description="Average confidence score", ge=0.0, le=1.0
    )

    @model_validator(mode='after')
    def validate_detector_counts(cls, values):
        """Validate that detector counts are consistent."""
        total = values.total_detectors
        successful = values.successful_detectors
        failed = values.failed_detectors

        if successful + failed != total:
            raise ValueError(
                "successful_detectors + failed_detectors must equal total_detectors"
            )

        return values


class PolicyViolation(BaseModel):
    """Policy violation details."""

    policy_id: str = Field(description="Policy identifier")
    violation_type: str = Field(description="Type of violation")
    message: str = Field(description="Violation message")
    severity: Severity = Field(description="Violation severity")

    @validator("policy_id", "violation_type", "message")
    def validate_required_strings(cls, v):
        """Validate that required string fields are non-empty."""
        return validate_non_empty_string(cls, v)


class RiskSummary(BaseModel):
    """Risk summary for an orchestration request."""

    level: RiskLevel = Field(description="Overall risk level")
    score: float = Field(description="Overall risk score", ge=0.0, le=1.0)
    rules_evaluation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Rule evaluation details used to derive the risk score",
    )
    model_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model features used when computing the risk score",
    )


class OrchestrationResponse(BaseResponse):
    """Response from detector orchestration."""

    detector_results: List[DetectorResult] = Field(
        description="Results from individual detectors"
    )
    aggregation_summary: AggregationSummary = Field(
        description="Summary of aggregation"
    )
    coverage_achieved: Optional[float] = Field(
        None, description="Coverage achieved", ge=0.0, le=1.0
    )
    policy_violations: List[PolicyViolation] = Field(
        default_factory=list, description="Policy violations detected"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations for improvement"
    )
    canonical_outputs: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Canonical detector outputs for this request, serialized form of "
            "CanonicalDetectorOutputs"
        ),
    )
    risk_summary: Optional[RiskSummary] = Field(
        None,
        description="Optional risk summary computed by the orchestration risk scorer",
    )

    def canonical_results_dict(self) -> List[Dict[str, Any]]:
        """Extract canonical taxonomy results from canonical_outputs if present.

        The canonical_outputs field is the serialized form of CanonicalDetectorOutputs.
        This helper flattens that structure into a simple list of canonical
        taxonomy result dicts (one per detector output), suitable for passing to
        analysis or mapper services that expect canonical_results.
        """

        if not self.canonical_outputs:
            return []

        outputs = self.canonical_outputs.get("outputs") or []
        canonical_results: List[Dict[str, Any]] = []

        for output in outputs:
            if not isinstance(output, dict):
                continue
            canonical = output.get("canonical_result")
            if isinstance(canonical, dict):
                canonical_results.append(canonical)

        return canonical_results


class DetectorInfo(BaseModel):
    """Information about a detector."""

    detector_id: str = Field(description="Detector identifier")
    detector_type: str = Field(description="Type of detector")
    status: str = Field(description="Detector status")
    endpoint_url: str = Field(description="Detector endpoint URL")
    supported_content_types: List[str] = Field(
        default_factory=list, description="Supported content types"
    )
    timeout_ms: int = Field(description="Timeout in milliseconds", ge=0)
    max_retries: int = Field(description="Maximum retries", ge=0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator("detector_id", "detector_type", "status", "endpoint_url")
    def validate_required_strings(cls, v):
        """Validate that required string fields are non-empty."""
        return validate_non_empty_string(cls, v)


class DetectorRegistration(BaseModel):
    """Detector registration request."""

    detector_id: Optional[str] = Field(None, description="Detector identifier")
    detector_type: str = Field(description="Type of detector")
    endpoint_url: str = Field(description="Detector endpoint URL")
    timeout_ms: Optional[int] = Field(None, description="Timeout in milliseconds", ge=0)
    max_retries: Optional[int] = Field(None, description="Maximum retries", ge=0)
    supported_content_types: Optional[List[str]] = Field(
        None, description="Supported content types"
    )

    @validator("detector_type", "endpoint_url")
    def validate_required_strings(cls, v):
        """Validate that required string fields are non-empty."""
        return validate_non_empty_string(cls, v)


class DetectorHealthStatus(BaseModel):
    """Detector health status."""

    detector_id: str = Field(description="Detector identifier")
    status: HealthStatus = Field(description="Health status")
    last_check: datetime = Field(description="Last health check timestamp")
    response_time_ms: Optional[int] = Field(
        None, description="Response time in milliseconds", ge=0
    )
    error_rate: Optional[float] = Field(None, description="Error rate", ge=0.0, le=1.0)
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")

    @validator("detector_id")
    def validate_detector_id(cls, v):
        """Validate detector ID is a non-empty string."""
        return validate_non_empty_string(cls, v)


class PolicyValidationRequest(BaseModel):
    """Request for policy validation."""

    policy_bundle: str = Field(description="Policy bundle identifier")
    detector_results: List[DetectorResult] = Field(
        description="Detector results to validate"
    )

    @validator("policy_bundle")
    def validate_policy_bundle(cls, v):
        """Validate policy bundle is a non-empty string."""
        return validate_non_empty_string(cls, v)


class PolicyValidationResponse(BaseModel):
    """Response from policy validation."""

    is_valid: bool = Field(description="Whether validation passed")
    violations: List[PolicyViolation] = Field(
        default_factory=list, description="Policy violations"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    policy_version: str = Field(description="Policy version used")

    @validator("policy_version")
    def validate_policy_version(cls, v):
        """Validate policy version is a non-empty string."""
        return validate_non_empty_string(cls, v)


class AsyncJobRequest(BaseModel):
    """Request for async job submission."""

    job_type: str = Field(description="Type of job")
    job_data: Dict[str, Any] = Field(description="Job data")
    priority: int = Field(100, description="Job priority", ge=0, le=1000)

    @validator("job_type")
    def validate_job_type(cls, v):
        """Validate job type is a non-empty string."""
        return validate_non_empty_string(cls, v)


class AsyncJobResponse(BaseModel):
    """Response from async job submission."""

    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Job status")
    created_at: datetime = Field(description="Job creation timestamp")

    @validator("job_id")
    def validate_job_id(cls, v):
        """Validate job ID is a non-empty string."""
        return validate_non_empty_string(cls, v)


class AsyncJobStatus(BaseModel):
    """Async job status."""

    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Job status")
    created_at: datetime = Field(description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Job completion timestamp"
    )
    progress: Optional[int] = Field(
        None, description="Job progress percentage", ge=0, le=100
    )
    result_data: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    @validator("job_id")
    def validate_job_id(cls, v):
        """Validate job ID is a non-empty string."""
        return validate_non_empty_string(cls, v)


class BatchOrchestrationRequest(BaseModel):
    """Request for batch orchestration."""

    requests: List[OrchestrationRequest] = Field(
        description="List of orchestration requests", min_items=1, max_items=100
    )
    async_processing: bool = Field(
        default=False, description="Whether to process asynchronously"
    )
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    priority: int = Field(100, description="Batch priority", ge=0, le=1000)

    @validator("requests")
    def validate_requests(cls, v):
        """Validate that requests list is not empty."""
        if not v:
            raise ValueError("requests cannot be empty")
        return v


class BatchOrchestrationResponse(BaseModel):
    """Response from batch orchestration."""

    job_id: Optional[str] = Field(None, description="Job ID for async processing")
    status: str = Field(description="Batch processing status")
    total_requests: int = Field(description="Total number of requests", ge=0)
    completed_requests: int = Field(description="Number of completed requests", ge=0)
    failed_requests: int = Field(
        default=0, description="Number of failed requests", ge=0
    )
    results: List[OrchestrationResponse] = Field(
        default_factory=list, description="Orchestration results"
    )
    batch_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Batch processing metadata"
    )

    @model_validator(mode='after')
    def validate_request_counts(cls, values):
        """Validate that request counts are consistent."""
        total = values.total_requests
        completed = values.completed_requests
        failed = values.failed_requests

        if completed + failed > total:
            raise ValueError(
                "completed_requests + failed_requests cannot exceed total_requests"
            )

        return values
