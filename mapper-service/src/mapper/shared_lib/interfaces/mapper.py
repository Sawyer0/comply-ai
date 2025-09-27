"""Mapper service interfaces with enhanced validation."""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
from pydantic import BaseModel, Field, validator, model_validator
from enum import Enum

from .base import BaseRequest, BaseResponse
from .analysis import AnalysisResponse, CanonicalTaxonomyResult


class MappingMode(str, Enum):
    """Mapping modes."""

    STANDARD = "standard"
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"


class ValidationType(str, Enum):
    """Validation types."""

    INPUT = "input"
    OUTPUT = "output"
    FRAMEWORK = "framework"
    TAXONOMY = "taxonomy"


class ModelType(str, Enum):
    """Model types."""

    BASE = "base"
    LORA = "lora"
    MERGED = "merged"
    QUANTIZED = "quantized"


class ModelStatus(str, Enum):
    """Model status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DeploymentStatus(str, Enum):
    """Deployment status."""

    NOT_DEPLOYED = "not_deployed"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class DeploymentType(str, Enum):
    """Deployment types."""

    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class TrainingType(str, Enum):
    """Training types."""

    LORA = "lora"
    FULL_FINETUNE = "full_finetune"
    QLORA = "qlora"


class JobStatus(str, Enum):
    """Job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentStatus(str, Enum):
    """Experiment status."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ComplianceStatus(str, Enum):
    """Compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class CostBudgetStatus(str, Enum):
    """Cost budget status."""

    OK = "ok"
    WARNING = "warning"
    EXCEEDED = "exceeded"


class MappingRequest(BaseRequest):
    """Request for core mapping functionality."""

    analysis_response: AnalysisResponse = Field(
        description="Response from analysis service"
    )
    target_frameworks: List[str] = Field(
        description="Target compliance frameworks", min_items=1, max_items=10
    )
    mapping_mode: MappingMode = Field(MappingMode.STANDARD, description="Mapping mode")
    include_validation: bool = Field(True, description="Whether to include validation")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context for mapping"
    )

    @validator("target_frameworks")
    def validate_target_frameworks(cls, v):
        if not v:
            raise ValueError("target_frameworks cannot be empty")
        # Remove duplicates while preserving order
        seen = set()
        unique_frameworks = []
        for framework in v:
            if framework not in seen:
                seen.add(framework)
                unique_frameworks.append(framework)
        return unique_frameworks


class ComplianceMapping(BaseModel):
    """Compliance mapping result."""

    framework: str = Field(description="Compliance framework")
    control_id: str = Field(description="Control identifier")
    control_name: str = Field(description="Control name")
    requirement: str = Field(description="Requirement description")
    evidence_type: str = Field(description="Type of evidence")
    compliance_status: ComplianceStatus = Field(description="Compliance status")
    confidence: float = Field(description="Confidence score", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @validator(
        "framework", "control_id", "control_name", "requirement", "evidence_type"
    )
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class ValidationResult(BaseModel):
    """Validation result."""

    is_valid: bool = Field(description="Whether validation passed")
    schema_compliance: bool = Field(description="Schema compliance status")
    confidence_threshold_met: bool = Field(
        description="Whether confidence threshold was met"
    )
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors"
    )
    validation_warnings: List[str] = Field(
        default_factory=list, description="Validation warnings"
    )


class MappingResult(BaseModel):
    """Individual mapping result."""

    canonical_result: CanonicalTaxonomyResult = Field(
        description="Canonical taxonomy result"
    )
    framework_mappings: List[ComplianceMapping] = Field(
        description="Framework mappings"
    )
    confidence: float = Field(description="Overall confidence", ge=0.0, le=1.0)
    validation_result: ValidationResult = Field(description="Validation result")
    fallback_used: bool = Field(description="Whether fallback was used")


class CostMetrics(BaseModel):
    """Cost metrics."""

    tokens_processed: int = Field(description="Number of tokens processed", ge=0)
    inference_cost: float = Field(description="Inference cost", ge=0)
    storage_cost: float = Field(description="Storage cost", ge=0)
    total_cost: float = Field(description="Total cost", ge=0)
    cost_per_request: float = Field(description="Cost per request", ge=0)


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    model_name: str = Field(description="Model name")
    model_version: str = Field(description="Model version")
    inference_time_ms: float = Field(description="Inference time in milliseconds", ge=0)
    gpu_utilization: float = Field(description="GPU utilization", ge=0.0, le=1.0)
    memory_usage_mb: int = Field(description="Memory usage in MB", ge=0)
    batch_size: int = Field(description="Batch size", ge=1)

    @validator("model_name", "model_version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class MappingResponse(BaseResponse):
    """Response from core mapping functionality."""

    mapping_results: List[MappingResult] = Field(description="Mapping results")
    overall_confidence: float = Field(
        description="Overall confidence score", ge=0.0, le=1.0
    )
    cost_metrics: CostMetrics = Field(description="Cost metrics")
    model_metrics: ModelMetrics = Field(description="Model metrics")
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )


class ValidationRequest(BaseModel):
    """Request for validation."""

    data: Dict[str, Any] = Field(description="Data to validate")
    validation_type: ValidationType = Field(description="Type of validation")
    schema_name: Optional[str] = Field(None, description="Schema name for validation")

    @validator("data")
    def validate_data(cls, v):
        if not v:
            raise ValueError("data cannot be empty")
        return v


class BatchMappingRequest(BaseModel):
    """Request for batch mapping operations."""

    requests: List[MappingRequest] = Field(
        description="List of mapping requests", min_items=1, max_items=100
    )

    @validator("requests")
    def validate_requests(cls, v):
        if not v:
            raise ValueError("requests cannot be empty")
        return v


class BatchMappingResponse(BaseModel):
    """Response from batch mapping operations."""

    batch_id: str = Field(description="Batch identifier")
    total_requests: int = Field(description="Total number of requests", ge=0)
    successful_requests: int = Field(description="Number of successful requests", ge=0)
    failed_requests: int = Field(description="Number of failed requests", ge=0)
    results: List[MappingResponse] = Field(description="Individual results")
    processing_time_ms: Optional[float] = Field(
        None, description="Total processing time in milliseconds", ge=0
    )
    total_cost: Optional[float] = Field(None, description="Total cost", ge=0)

    @validator("batch_id")
    def validate_batch_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("batch_id is required and must be a string")
        return v

    @model_validator(mode='after')
    def validate_request_counts(cls, values):
        total = values.total_requests
        successful = values.successful_requests
        failed = values.failed_requests

        if successful + failed != total:
            raise ValueError(
                "successful_requests + failed_requests must equal total_requests"
            )

        return values


class ModelVersion(BaseModel):
    """Model version information."""

    id: Optional[str] = Field(None, description="Model ID")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    model_type: ModelType = Field(description="Model type")
    model_path: Optional[str] = Field(None, description="Model path")
    parent_model_id: Optional[str] = Field(None, description="Parent model ID")
    training_job_id: Optional[str] = Field(None, description="Training job ID")
    configuration: Optional[Dict[str, Any]] = Field(
        None, description="Model configuration"
    )
    performance_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Performance metrics"
    )
    validation_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Validation metrics"
    )
    status: ModelStatus = Field(description="Model status")
    deployment_status: DeploymentStatus = Field(description="Deployment status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")

    @validator("model_name", "version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class ModelDeploymentRequest(BaseModel):
    """Request for model deployment."""

    deployment_type: DeploymentType = Field(description="Type of deployment")
    traffic_percentage: Optional[int] = Field(
        None, description="Traffic percentage for canary deployment", ge=0, le=100
    )
    rollback_model_id: Optional[str] = Field(
        None, description="Model ID to rollback to"
    )


class ModelDeploymentResponse(BaseModel):
    """Response from model deployment."""

    deployment_id: str = Field(description="Deployment identifier")
    status: str = Field(description="Deployment status")
    estimated_completion_time: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )

    @validator("deployment_id", "status")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class TrainingJobRequest(BaseModel):
    """Request for training job submission."""

    job_name: str = Field(description="Job name")
    base_model: str = Field(description="Base model")
    training_type: TrainingType = Field(description="Type of training")
    training_data_path: str = Field(description="Training data path")
    validation_data_path: Optional[str] = Field(
        None, description="Validation data path"
    )
    configuration: Dict[str, Any] = Field(description="Training configuration")

    @validator("job_name", "base_model", "training_data_path")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v

    @validator("configuration")
    def validate_configuration(cls, v):
        if not v:
            raise ValueError("configuration cannot be empty")
        return v


class TrainingJobResponse(BaseModel):
    """Response from training job submission."""

    job_id: str = Field(description="Job identifier")
    status: JobStatus = Field(description="Job status")
    estimated_completion_time: Optional[datetime] = Field(
        None, description="Estimated completion time"
    )

    @validator("job_id")
    def validate_job_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("job_id is required and must be a string")
        return v


class TrainingJob(BaseModel):
    """Training job information."""

    id: Optional[str] = Field(None, description="Job ID")
    job_name: str = Field(description="Job name")
    base_model: str = Field(description="Base model")
    training_type: TrainingType = Field(description="Training type")
    training_data_path: str = Field(description="Training data path")
    validation_data_path: Optional[str] = Field(
        None, description="Validation data path"
    )
    output_model_path: Optional[str] = Field(None, description="Output model path")
    status: JobStatus = Field(description="Job status")
    configuration: Optional[Dict[str, Any]] = Field(
        None, description="Training configuration"
    )
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    training_time_minutes: Optional[int] = Field(
        None, description="Training time in minutes", ge=0
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")

    @validator("job_name", "base_model", "training_data_path")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class ExperimentRequest(BaseModel):
    """Request for A/B testing experiment."""

    experiment_name: str = Field(description="Experiment name")
    model_a_id: str = Field(description="Model A identifier")
    model_b_id: str = Field(description="Model B identifier")
    traffic_split_percentage: int = Field(
        50, description="Traffic split percentage for model A", ge=0, le=100
    )
    success_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Success metrics"
    )

    @validator("experiment_name", "model_a_id", "model_b_id")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class DeploymentExperiment(BaseModel):
    """Deployment experiment information."""

    id: Optional[str] = Field(None, description="Experiment ID")
    experiment_name: str = Field(description="Experiment name")
    model_a_id: str = Field(description="Model A identifier")
    model_b_id: str = Field(description="Model B identifier")
    traffic_split_percentage: int = Field(
        description="Traffic split percentage", ge=0, le=100
    )
    status: ExperimentStatus = Field(description="Experiment status")
    success_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Success metrics"
    )
    current_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Current metrics"
    )
    statistical_significance: Optional[float] = Field(
        None, description="Statistical significance", ge=0.0, le=1.0
    )
    winner_model_id: Optional[str] = Field(None, description="Winner model ID")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    ended_at: Optional[datetime] = Field(None, description="End timestamp")

    @validator("experiment_name", "model_a_id", "model_b_id")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v


class Taxonomy(BaseModel):
    """Taxonomy information."""

    id: Optional[str] = Field(None, description="Taxonomy ID")
    taxonomy_name: str = Field(description="Taxonomy name")
    version: str = Field(description="Taxonomy version")
    taxonomy_data: Dict[str, Any] = Field(description="Taxonomy data")
    schema_version: str = Field(description="Schema version")
    is_active: Optional[bool] = Field(None, description="Whether taxonomy is active")
    backward_compatible: Optional[bool] = Field(
        None, description="Backward compatibility"
    )
    migration_notes: Optional[str] = Field(None, description="Migration notes")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    activated_at: Optional[datetime] = Field(None, description="Activation timestamp")

    @validator("taxonomy_name", "version", "schema_version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v

    @validator("taxonomy_data")
    def validate_taxonomy_data(cls, v):
        if not v:
            raise ValueError("taxonomy_data cannot be empty")
        return v


class TaxonomyRequest(BaseModel):
    """Request for taxonomy creation."""

    taxonomy_name: str = Field(description="Taxonomy name")
    version: str = Field(description="Taxonomy version")
    taxonomy_data: Dict[str, Any] = Field(description="Taxonomy data")
    schema_version: str = Field(description="Schema version")
    backward_compatible: bool = Field(True, description="Backward compatibility")
    migration_notes: Optional[str] = Field(None, description="Migration notes")

    @validator("taxonomy_name", "version", "schema_version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v

    @validator("taxonomy_data")
    def validate_taxonomy_data(cls, v):
        if not v:
            raise ValueError("taxonomy_data cannot be empty")
        return v


class FrameworkConfig(BaseModel):
    """Framework configuration."""

    id: Optional[str] = Field(None, description="Framework config ID")
    framework_name: str = Field(description="Framework name")
    framework_version: str = Field(description="Framework version")
    mapping_rules: Dict[str, Any] = Field(description="Mapping rules")
    validation_schema: Optional[Dict[str, Any]] = Field(
        None, description="Validation schema"
    )
    is_active: Optional[bool] = Field(None, description="Whether config is active")
    priority: Optional[int] = Field(None, description="Priority", ge=0)
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    @validator("framework_name", "framework_version")
    def validate_required_strings(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field is required and must be a non-empty string")
        return v

    @validator("mapping_rules")
    def validate_mapping_rules(cls, v):
        if not v:
            raise ValueError("mapping_rules cannot be empty")
        return v


class CostMetric(BaseModel):
    """Individual cost metric."""

    tokens_processed: int = Field(description="Tokens processed", ge=0)
    inference_cost: float = Field(description="Inference cost", ge=0)
    storage_cost: Optional[float] = Field(None, description="Storage cost", ge=0)
    total_cost: float = Field(description="Total cost", ge=0)
    cost_per_request: Optional[float] = Field(
        None, description="Cost per request", ge=0
    )
    billing_period: date = Field(description="Billing period")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")


class FeatureFlag(BaseModel):
    """Feature flag configuration."""

    id: Optional[str] = Field(None, description="Feature flag ID")
    flag_name: str = Field(description="Flag name")
    is_enabled: bool = Field(description="Whether flag is enabled")
    rollout_percentage: int = Field(description="Rollout percentage", ge=0, le=100)
    conditions: Optional[Dict[str, Any]] = Field(None, description="Conditions")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Update timestamp")

    @validator("flag_name")
    def validate_flag_name(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("flag_name is required and must be a string")
        return v
