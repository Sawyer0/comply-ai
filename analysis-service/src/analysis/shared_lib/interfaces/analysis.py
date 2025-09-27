"""Analysis service interfaces with enhanced validation."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator, root_validator
from enum import Enum

from .base import BaseRequest, BaseResponse
from .orchestration import DetectorResult, OrchestrationResponse


class AnalysisType(str, Enum):
    """Analysis types."""

    PATTERN = "pattern"
    RISK = "risk"
    COMPLIANCE = "compliance"
    RAG = "rag"


class RiskLevel(str, Enum):
    """Risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class ModelType(str, Enum):
    """ML model types."""

    PHI3 = "phi3"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class ModelStatus(str, Enum):
    """Model status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    """Alert severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvaluationStatus(str, Enum):
    """Evaluation status."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisRequest(BaseRequest):
    """Request for comprehensive analysis."""

    orchestration_response: OrchestrationResponse = Field(
        description="Response from orchestration service"
    )
    analysis_types: List[AnalysisType] = Field(
        description="Types of analysis to perform", min_items=1
    )
    frameworks: Optional[List[str]] = Field(
        None, description="Compliance frameworks to map to"
    )
    include_recommendations: bool = Field(
        True, description="Whether to include recommendations"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for analysis"
    )

    @validator("analysis_types")
    def validate_analysis_types(cls, v):
        if not v:
            raise ValueError("analysis_types cannot be empty")
        # Remove duplicates while preserving order
        seen = set()
        unique_types = []
        for analysis_type in v:
            if analysis_type not in seen:
                seen.add(analysis_type)
                unique_types.append(analysis_type)
        return unique_types


class CanonicalTaxonomyResult(BaseModel):
    """Result from canonical taxonomy mapping."""

    category: str = Field(description="Primary category")
    subcategory: str = Field(description="Subcategory")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
    risk_level: RiskLevel = Field(description="Risk level")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator("category", "subcategory")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class PatternAnalysisRequest(BaseModel):
    """Request for pattern analysis."""

    detector_results: List[DetectorResult] = Field(
        description="Detector results to analyze"
    )
    time_window_hours: int = Field(
        24, description="Time window for analysis in hours", ge=1, le=168
    )


class PatternAnalysisResult(BaseModel):
    """Result from pattern analysis."""

    temporal_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Temporal patterns detected"
    )
    frequency_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Frequency patterns detected"
    )
    correlation_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Correlation patterns detected"
    )
    anomaly_patterns: List[Dict[str, Any]] = Field(
        default_factory=list, description="Anomaly patterns detected"
    )
    confidence: float = Field(
        description="Overall confidence in pattern analysis", ge=0.0, le=1.0
    )


class RiskScoringRequest(BaseModel):
    """Request for risk scoring."""

    canonical_results: List[CanonicalTaxonomyResult] = Field(
        description="Canonical taxonomy results"
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for risk scoring"
    )


class RiskScoringResult(BaseModel):
    """Result from risk scoring."""

    overall_risk_score: float = Field(description="Overall risk score", ge=0.0, le=1.0)
    technical_risk: float = Field(description="Technical risk score", ge=0.0, le=1.0)
    business_risk: float = Field(description="Business risk score", ge=0.0, le=1.0)
    regulatory_risk: float = Field(description="Regulatory risk score", ge=0.0, le=1.0)
    temporal_risk: float = Field(description="Temporal risk score", ge=0.0, le=1.0)
    risk_factors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Individual risk factors"
    )
    mitigation_recommendations: List[str] = Field(
        default_factory=list, description="Risk mitigation recommendations"
    )


class ComplianceMappingRequest(BaseModel):
    """Request for compliance mapping."""

    canonical_results: List[CanonicalTaxonomyResult] = Field(
        description="Canonical taxonomy results"
    )
    frameworks: List[str] = Field(
        description="Compliance frameworks to map to", min_items=1
    )

    @validator("frameworks")
    def validate_frameworks(cls, v):
        if not v:
            raise ValueError("frameworks cannot be empty")
        return v


class ComplianceMappingResult(BaseModel):
    """Result from compliance mapping."""

    framework: str = Field(description="Compliance framework")
    mappings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Framework mappings"
    )
    compliance_score: float = Field(description="Compliance score", ge=0.0, le=1.0)
    gaps: List[Dict[str, Any]] = Field(
        default_factory=list, description="Compliance gaps identified"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Compliance recommendations"
    )

    @validator("framework")
    def validate_framework(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("framework is required and must be a string")
        return v


class RAGQueryRequest(BaseModel):
    """Request for RAG-enhanced query."""

    query_text: str = Field(description="Query text", max_length=1000, min_length=1)
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_results: int = Field(5, description="Maximum number of results", ge=1, le=10)

    @validator("query_text")
    def validate_query_text(cls, v):
        if not v or not v.strip():
            raise ValueError("query_text cannot be empty")
        return v.strip()


class RAGInsights(BaseModel):
    """RAG-enhanced insights."""

    relevant_regulations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Relevant regulations"
    )
    compliance_guidance: List[str] = Field(
        default_factory=list, description="Compliance guidance"
    )
    risk_context: List[str] = Field(
        default_factory=list, description="Risk context information"
    )
    remediation_steps: List[str] = Field(
        default_factory=list, description="Remediation steps"
    )
    confidence: float = Field(description="Confidence in insights", ge=0.0, le=1.0)
    retrieved_documents: List[str] = Field(
        default_factory=list, description="Retrieved document IDs"
    )


class QualityMetrics(BaseModel):
    """Quality metrics for analysis."""

    accuracy_score: float = Field(description="Accuracy score", ge=0.0, le=1.0)
    confidence_distribution: Dict[str, Any] = Field(
        description="Confidence score distribution"
    )
    processing_time_ms: float = Field(
        description="Processing time in milliseconds", ge=0
    )
    model_version: str = Field(description="Model version used")
    fallback_used: bool = Field(description="Whether fallback was used")

    @validator("model_version")
    def validate_model_version(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("model_version is required and must be a string")
        return v


class AnalysisResponse(BaseResponse):
    """Response from comprehensive analysis."""

    canonical_results: List[CanonicalTaxonomyResult] = Field(
        description="Canonical taxonomy results"
    )
    quality_metrics: QualityMetrics = Field(description="Quality metrics")
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


class QualityMetric(BaseModel):
    """Individual quality metric."""

    metric_type: str = Field(description="Type of metric")
    metric_name: str = Field(description="Metric name")
    metric_value: float = Field(description="Metric value")
    model_version: Optional[str] = Field(None, description="Model version")
    evaluation_date: date = Field(description="Evaluation date")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator("metric_type", "metric_name")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class QualityAlert(BaseModel):
    """Quality alert."""

    id: Optional[str] = Field(None, description="Alert ID")
    alert_type: str = Field(description="Type of alert")
    severity: AlertSeverity = Field(description="Alert severity")
    message: str = Field(description="Alert message")
    metric_name: Optional[str] = Field(None, description="Related metric name")
    current_value: Optional[float] = Field(None, description="Current metric value")
    threshold_value: Optional[float] = Field(None, description="Threshold value")
    status: AlertStatus = Field(description="Alert status")
    created_at: datetime = Field(description="Alert creation timestamp")
    acknowledged_at: Optional[datetime] = Field(
        None, description="Acknowledgment timestamp"
    )
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")

    @validator("alert_type", "message")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class MLModel(BaseModel):
    """ML model information."""

    id: Optional[str] = Field(None, description="Model ID")
    model_name: str = Field(description="Model name")
    model_version: str = Field(description="Model version")
    model_type: ModelType = Field(description="Model type")
    model_path: Optional[str] = Field(None, description="Model path")
    configuration: Optional[Dict[str, Any]] = Field(
        None, description="Model configuration"
    )
    performance_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Performance metrics"
    )
    status: ModelStatus = Field(description="Model status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    @validator("model_name", "model_version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class WeeklyEvaluation(BaseModel):
    """Weekly evaluation results."""

    id: Optional[str] = Field(None, description="Evaluation ID")
    evaluation_week: date = Field(description="Evaluation week")
    model_version: str = Field(description="Model version")
    accuracy_score: Optional[float] = Field(
        None, description="Accuracy score", ge=0.0, le=1.0
    )
    precision_score: Optional[float] = Field(
        None, description="Precision score", ge=0.0, le=1.0
    )
    recall_score: Optional[float] = Field(
        None, description="Recall score", ge=0.0, le=1.0
    )
    f1_score: Optional[float] = Field(None, description="F1 score", ge=0.0, le=1.0)
    confidence_distribution: Optional[Dict[str, Any]] = Field(
        None, description="Confidence distribution"
    )
    performance_trends: Optional[Dict[str, Any]] = Field(
        None, description="Performance trends"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    status: EvaluationStatus = Field(description="Evaluation status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    @validator("model_version")
    def validate_model_version(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("model_version is required and must be a string")
        return v
