"""
Analysis request and result schemas for the analysis service.

This module defines the core data structures used for analysis requests
and results throughout the analysis service.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


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


@dataclass
class AnalysisResult:
    """
    Analysis result containing findings and assessments.

    This represents the output of analysis operations including
    risk scores, compliance assessments, and evidence.
    """

    # Core result data
    analysis_type: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

    # Analysis outputs
    risk_score: Optional[Dict[str, Any]] = None
    compliance_assessment: Optional[Dict[str, Any]] = None
    patterns: Optional[List[Dict[str, Any]]] = None

    # Supporting data
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Quality metrics
    processing_time: Optional[float] = None
    data_quality_score: Optional[float] = None


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
