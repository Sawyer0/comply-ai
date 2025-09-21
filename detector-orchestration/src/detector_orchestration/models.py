from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    CODE = "code"


class ProcessingMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class DetectorStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class OrchestrationRequest(BaseModel):
    """Request model for detector orchestration operations."""

    content: str = Field(..., max_length=50000, description="Content to analyze")
    content_type: ContentType = Field(..., description="Type of content")
    tenant_id: str = Field(..., min_length=1, max_length=64)
    policy_bundle: str = Field(..., description="Policy bundle identifier")
    environment: Literal["dev", "stage", "prod"] = "dev"
    processing_mode: ProcessingMode = Field(default=ProcessingMode.SYNC)
    priority: Priority = Field(default=Priority.NORMAL)
    metadata: Optional[Dict[str, Any]] = None
    required_detectors: Optional[List[str]] = None
    excluded_detectors: Optional[List[str]] = None


class DetectorResult(BaseModel):
    """Result from a single detector execution."""

    detector: str
    status: DetectorStatus
    output: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int = 0
    confidence: Optional[float] = None


class MapperPayload(BaseModel):
    """Payload for mapper service containing detector results."""

    detector: str
    output: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tenant_id: str


class RoutingDecision(BaseModel):
    """Decision made by the routing engine for detector selection."""

    selected_detectors: List[str]
    routing_reason: str
    policy_applied: str
    coverage_requirements: Dict[str, Any] = Field(default_factory=dict)
    health_status: Dict[str, bool] = Field(default_factory=dict)


class OrchestrationResponse(BaseModel):
    """Complete response from detector orchestration including all results and metadata."""

    request_id: str
    job_id: Optional[str] = None
    processing_mode: ProcessingMode
    detector_results: List[DetectorResult]
    aggregated_payload: Optional[MapperPayload] = None
    mapping_result: Optional["MappingResponse"] = None
    total_processing_time_ms: int
    detectors_attempted: int
    detectors_succeeded: int
    detectors_failed: int
    coverage_achieved: float
    routing_decision: RoutingDecision
    fallback_used: bool
    timestamp: datetime
    error_code: Optional[str] = None
    idempotency_key: Optional[str] = None


class Provenance(BaseModel):
    """Provenance information for detector results and mappings."""

    vendor: Optional[str] = None
    detector: Optional[str] = None
    detector_version: Optional[str] = None
    raw_ref: Optional[str] = None
    route: Optional[str] = None
    model: Optional[str] = None
    tenant_id: Optional[str] = None
    ts: Optional[datetime] = None


class PolicyContext(BaseModel):
    """Policy context for detector expectations."""

    expected_detectors: Optional[List[str]] = None
    environment: Optional[str] = Field(None, pattern="^(dev|stage|prod)$")


class MappingResponse(BaseModel):
    """Response from the mapper service containing taxonomy mapping results."""

    taxonomy: List[str]
    scores: Dict[str, float]
    confidence: float
    notes: Optional[str] = None
    provenance: Optional[Provenance] = None
    policy_context: Optional[PolicyContext] = None


class DetectorCapabilities(BaseModel):
    """Capabilities and characteristics of a detector."""

    supported_content_types: List[ContentType]
    max_content_length: int = 50000
    average_processing_time_ms: int | None = None
    confidence_calibrated: bool | None = None
    batch_supported: bool | None = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatusResponse(BaseModel):
    """Response containing the status of an asynchronous orchestration job."""

    job_id: str
    status: JobStatus
    progress: float
    result: Optional[OrchestrationResponse] = None
    error: Optional[str] = None


class ErrorBody(BaseModel):
    """Canonical error body (Sec 8)."""

    error_code: str
    message: Optional[str] = None
    request_id: Optional[str] = None
    retryable: bool = False


class RoutingPlan(BaseModel):
    """Detailed routing plan for detector orchestration."""

    primary_detectors: List[str]
    secondary_detectors: List[str] = Field(default_factory=list)
    parallel_groups: List[List[str]] = Field(default_factory=list)
    sequential_dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    timeout_config: Dict[str, int] = Field(default_factory=dict)
    retry_config: Dict[str, int] = Field(default_factory=dict)
    coverage_method: str = "required_set"
    weights: Dict[str, float] = Field(default_factory=dict)
    required_taxonomy_categories: List[str] = Field(default_factory=list)
