"""Orchestration service interfaces with enhanced validation following SRP."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator

from .base import BaseRequest, BaseResponse
from .common import ProcessingMode, HealthStatus, Severity, JobStatus
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
        description="List of detector types to execute", min_items=1, max_items=20
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
        if not v or not v.strip():
            raise ValueError("content cannot be empty")
        return v.strip()

    @validator("detector_types")
    def validate_detector_types(cls, v):
        if not v:
            raise ValueError("detector_types cannot be empty")
        # Remove duplicates while preserving order
        seen = set()
        unique_types = []
        for detector_type in v:
            if detector_type not in seen:
                seen.add(detector_type)
                unique_types.append(detector_type)
        return unique_types


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
        if not v or not isinstance(v, str):
            raise ValueError("detector_id is required and must be a string")
        return v

    @validator("category")
    def validate_category(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("category is required and must be a string")
        return v


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

    @root_validator
    def validate_detector_counts(cls, values):
        total = values.get("total_detectors", 0)
        successful = values.get("successful_detectors", 0)
        failed = values.get("failed_detectors", 0)

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
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


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


class DetectorInfo(BaseModel):
    """Information about a detector."""

    id: str = Field(description="Detector identifier")
    detector_type: str = Field(description="Type of detector")
    detector_name: str = Field(description="Human-readable detector name")
    endpoint_url: Optional[str] = Field(None, description="Detector endpoint URL")
    status: DetectorStatus = Field(description="Detector status")
    version: str = Field(description="Detector version")
    capabilities: List[str] = Field(
        default_factory=list, description="Detector capabilities"
    )
    configuration: Optional[Dict[str, Any]] = Field(
        None, description="Detector configuration"
    )
    health_status: HealthStatus = Field(description="Health status")
    response_time_ms: Optional[int] = Field(
        None, description="Average response time in milliseconds", ge=0
    )
    error_rate: Optional[float] = Field(None, description="Error rate", ge=0.0, le=1.0)

    @validator("id", "detector_type", "detector_name", "version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class DetectorRegistration(BaseModel):
    """Detector registration request."""

    detector_type: str = Field(description="Type of detector")
    detector_name: str = Field(description="Human-readable detector name")
    endpoint_url: str = Field(description="Detector endpoint URL")
    health_check_url: Optional[str] = Field(None, description="Health check URL")
    version: str = Field(description="Detector version")
    capabilities: List[str] = Field(
        default_factory=list, description="Detector capabilities"
    )
    configuration: Optional[Dict[str, Any]] = Field(
        None, description="Detector configuration"
    )

    @validator("detector_type", "detector_name", "endpoint_url", "version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


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
        if not v or not isinstance(v, str):
            raise ValueError("detector_id is required and must be a string")
        return v


class PolicyValidationRequest(BaseModel):
    """Request for policy validation."""

    policy_bundle: str = Field(description="Policy bundle identifier")
    detector_results: List[DetectorResult] = Field(
        description="Detector results to validate"
    )

    @validator("policy_bundle")
    def validate_policy_bundle(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("policy_bundle is required and must be a string")
        return v


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
        if not v or not isinstance(v, str):
            raise ValueError("policy_version is required and must be a string")
        return v


class AsyncJobRequest(BaseModel):
    """Request for async job submission."""

    job_type: str = Field(description="Type of job")
    job_data: Dict[str, Any] = Field(description="Job data")
    priority: int = Field(100, description="Job priority", ge=0, le=1000)

    @validator("job_type")
    def validate_job_type(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("job_type is required and must be a string")
        return v


class AsyncJobResponse(BaseModel):
    """Response from async job submission."""

    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Job status")
    created_at: datetime = Field(description="Job creation timestamp")

    @validator("job_id")
    def validate_job_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("job_id is required and must be a string")
        return v


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
        if not v or not isinstance(v, str):
            raise ValueError("job_id is required and must be a string")
        return v
