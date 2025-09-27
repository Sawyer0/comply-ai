"""
Analysis request and result schemas for the analysis service.

This module defines the core data structures used for analysis requests
and results throughout the analysis service. Uses shared interfaces.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator

# Import shared interfaces for consistency
from ...shared_integration import (
    CanonicalTaxonomyResult,
    QualityMetrics,
    PatternAnalysisResult,
    RiskScoringResult,
    ComplianceMappingResult,
    RAGInsights,
    ValidationError,
)


@dataclass
class AnalysisRequest:
    """
    Analysis request containing data to be analyzed.

    This represents a request for analysis containing security findings,
    detector outputs, and metadata for comprehensive risk assessment.
    """

    # Core request data
    request_id: str
    content_hash: str  # Hash of content for privacy
    timestamp: datetime = field(default_factory=datetime.now)

    # Security findings
    high_sev_hits: List[Dict[str, Any]] = field(default_factory=list)
    medium_sev_hits: List[Dict[str, Any]] = field(default_factory=list)
    low_sev_hits: List[Dict[str, Any]] = field(default_factory=list)

    # Detector information
    detector_errors: Dict[str, Any] = field(default_factory=dict)
    observed_coverage: Dict[str, float] = field(default_factory=dict)
    required_coverage: Dict[str, float] = field(default_factory=dict)

    # Context and metadata
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Analysis configuration
    analysis_types: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)


class AnalysisResult(BaseModel):
    """
    Analysis result containing findings and assessments.

    This represents the output of analysis operations including
    risk scores, compliance assessments, and evidence.
    Uses shared interfaces for consistency.
    """

    # Core result data
    analysis_type: str = Field(description="Type of analysis performed")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now, description="Analysis timestamp")

    # Analysis outputs using shared interfaces
    canonical_results: List[CanonicalTaxonomyResult] = Field(
        default_factory=list, description="Canonical taxonomy results"
    )
    quality_metrics: Optional[QualityMetrics] = Field(
        None, description="Quality metrics"
    )
    pattern_analysis: Optional[PatternAnalysisResult] = Field(
        None, description="Pattern analysis results"
    )
    risk_scores: Optional[RiskScoringResult] = Field(
        None, description="Risk scoring results"
    )
    compliance_mappings: Optional[List[ComplianceMappingResult]] = Field(
        None, description="Compliance mapping results"
    )
    rag_insights: Optional[RAGInsights] = Field(
        None, description="RAG-enhanced insights"
    )

    # Supporting data
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Supporting evidence")
    recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Recommendations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Quality metrics
    processing_time: Optional[float] = Field(None, description="Processing time in seconds", ge=0)
    data_quality_score: Optional[float] = Field(None, description="Data quality score", ge=0.0, le=1.0)

    @validator("confidence", "data_quality_score")
    def validate_confidence_scores(cls, v):
        """Validate confidence and quality scores are in valid range."""
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValidationError("Score must be between 0.0 and 1.0")
        return v

    @validator("analysis_type")
    def validate_analysis_type(cls, v):
        """Validate analysis type is not empty."""
        if not v or not v.strip():
            raise ValidationError("Analysis type cannot be empty")
        return v.strip()


@dataclass
class BatchAnalysisRequest:
    """Request for batch analysis of multiple items."""

    batch_id: str
    requests: List[AnalysisRequest]
    batch_config: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # low, normal, high, urgent


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis operations."""

    batch_id: str
    results: List[AnalysisResult]
    batch_summary: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchTask:
    """Individual task within a batch."""

    task_id: str
    request: AnalysisRequest
    status: str = "pending"
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BatchJobStatus:
    """Batch job status enumeration."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class BatchJob:
    """Batch job containing multiple tasks."""

    job_id: str
    tasks: List[BatchTask]
    status: str = BatchJobStatus.QUEUED
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_tasks: int = 0
    completed_tasks: int = 0
    progress: float = 0.0
    error: Optional[str] = None


@dataclass
class TrainingRequest:
    """Request for model training."""

    training_data: List[Dict[str, Any]]
    validation_data: Optional[List[Dict[str, Any]]] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result of model training."""

    training_id: str
    model_path: str
    metrics: Dict[str, Any]
    quality_score: float
    config: Dict[str, Any]
    completed_at: datetime
