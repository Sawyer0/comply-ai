"""Request models for microservice communication."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid


class BaseServiceRequest(BaseModel):
    """Base request model for all services."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = Field(..., min_length=1, max_length=100)
    correlation_id: Optional[str] = Field(default=None)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator("correlation_id")
    def validate_correlation_id(cls, v):
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError("correlation_id must be a valid UUID")
        return v


class OrchestrationRequest(BaseServiceRequest):
    """Request for detector orchestration."""

    content: str = Field(..., min_length=1, max_length=10000)
    detector_types: Optional[List[str]] = Field(default=None, min_items=1)
    policy_bundle: Optional[str] = Field(default=None)
    processing_mode: str = Field(default="standard")
    max_detectors: Optional[int] = Field(default=None, ge=1, le=10)

    @validator("processing_mode")
    def validate_processing_mode(cls, v):
        allowed_modes = ["standard", "fast", "thorough"]
        if v not in allowed_modes:
            raise ValueError(f"processing_mode must be one of {allowed_modes}")
        return v

    @validator("content")
    def validate_content_privacy(cls, v):
        # Basic privacy validation - no obvious PII patterns
        import re

        # Check for obvious email patterns
        if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", v):
            raise ValueError("Content appears to contain email addresses")

        # Check for obvious phone patterns
        if re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", v):
            raise ValueError("Content appears to contain phone numbers")

        return v


class AnalysisRequest(BaseServiceRequest):
    """Request for analysis service."""

    orchestration_response: Dict[str, Any] = Field(...)
    analysis_types: List[str] = Field(..., min_items=1)
    frameworks: Optional[List[str]] = Field(default=None)
    include_recommendations: bool = Field(default=True)

    @validator("analysis_types")
    def validate_analysis_types(cls, v):
        allowed_types = ["pattern", "risk", "compliance", "rag"]
        for analysis_type in v:
            if analysis_type not in allowed_types:
                raise ValueError(f"analysis_type must be one of {allowed_types}")
        return v

    @validator("frameworks")
    def validate_frameworks(cls, v):
        if v is not None:
            allowed_frameworks = [
                "SOC2",
                "GDPR",
                "HIPAA",
                "ISO27001",
                "NIST",
                "PCI-DSS",
            ]
            for framework in v:
                if framework not in allowed_frameworks:
                    raise ValueError(f"framework must be one of {allowed_frameworks}")
        return v


class MappingRequest(BaseServiceRequest):
    """Request for mapping service."""

    analysis_response: Dict[str, Any] = Field(...)
    target_frameworks: List[str] = Field(..., min_items=1)
    mapping_mode: str = Field(default="standard")
    include_validation: bool = Field(default=True)

    @validator("mapping_mode")
    def validate_mapping_mode(cls, v):
        allowed_modes = ["standard", "fast", "comprehensive"]
        if v not in allowed_modes:
            raise ValueError(f"mapping_mode must be one of {allowed_modes}")
        return v

    @validator("target_frameworks")
    def validate_target_frameworks(cls, v):
        allowed_frameworks = ["SOC2", "GDPR", "HIPAA", "ISO27001", "NIST", "PCI-DSS"]
        for framework in v:
            if framework not in allowed_frameworks:
                raise ValueError(f"framework must be one of {allowed_frameworks}")
        return v


class PolicyValidationRequest(BaseModel):
    """Request for policy validation."""

    policy_bundle: str = Field(..., min_length=1)
    detector_results: List[Dict[str, Any]] = Field(..., min_items=1)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("detector_results")
    def validate_detector_results(cls, v):
        required_fields = [
            "detector_id",
            "detector_type",
            "confidence",
            "category",
            "severity",
        ]
        for result in v:
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"detector_result missing required field: {field}")

            # Validate confidence score
            confidence = result.get("confidence")
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                raise ValueError("confidence must be a number between 0 and 1")

        return v


class AsyncJobRequest(BaseModel):
    """Request for async job submission."""

    job_type: str = Field(..., min_length=1)
    job_data: Dict[str, Any] = Field(...)
    priority: int = Field(default=100, ge=0)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("job_type")
    def validate_job_type(cls, v):
        allowed_types = [
            "detector_health_check",
            "policy_update",
            "bulk_analysis",
            "model_training",
            "data_migration",
            "cleanup",
        ]
        if v not in allowed_types:
            raise ValueError(f"job_type must be one of {allowed_types}")
        return v


class PatternAnalysisRequest(BaseModel):
    """Request for pattern analysis."""

    detector_results: List[Dict[str, Any]] = Field(..., min_items=1)
    time_window_hours: int = Field(default=24, ge=1, le=168)
    tenant_id: str = Field(..., min_length=1, max_length=100)


class RiskScoringRequest(BaseModel):
    """Request for risk scoring."""

    canonical_results: List[Dict[str, Any]] = Field(..., min_items=1)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tenant_id: str = Field(..., min_length=1, max_length=100)


class ComplianceMappingRequest(BaseModel):
    """Request for compliance mapping."""

    canonical_results: List[Dict[str, Any]] = Field(..., min_items=1)
    frameworks: List[str] = Field(..., min_items=1)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("frameworks")
    def validate_frameworks(cls, v):
        allowed_frameworks = ["SOC2", "GDPR", "HIPAA", "ISO27001", "NIST", "PCI-DSS"]
        for framework in v:
            if framework not in allowed_frameworks:
                raise ValueError(f"framework must be one of {allowed_frameworks}")
        return v


class RAGQueryRequest(BaseModel):
    """Request for RAG query."""

    query_text: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_results: int = Field(default=5, ge=1, le=10)
    tenant_id: str = Field(..., min_length=1, max_length=100)


class ValidationRequest(BaseModel):
    """Request for validation."""

    data: Dict[str, Any] = Field(...)
    validation_type: str = Field(...)
    schema_name: Optional[str] = Field(default=None)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("validation_type")
    def validate_validation_type(cls, v):
        allowed_types = ["input", "output", "framework", "taxonomy"]
        if v not in allowed_types:
            raise ValueError(f"validation_type must be one of {allowed_types}")
        return v


class BatchMappingRequest(BaseModel):
    """Request for batch mapping."""

    requests: List[MappingRequest] = Field(..., min_items=1, max_items=100)

    @validator("requests")
    def validate_batch_size(cls, v):
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        return v


class TrainingJobRequest(BaseModel):
    """Request for training job."""

    job_name: str = Field(..., min_length=1, max_length=100)
    base_model: str = Field(..., min_length=1)
    training_type: str = Field(...)
    training_data_path: str = Field(..., min_length=1)
    validation_data_path: Optional[str] = Field(default=None)
    configuration: Dict[str, Any] = Field(...)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("training_type")
    def validate_training_type(cls, v):
        allowed_types = ["lora", "full_finetune", "qlora"]
        if v not in allowed_types:
            raise ValueError(f"training_type must be one of {allowed_types}")
        return v

    @validator("configuration")
    def validate_configuration(cls, v):
        required_fields = ["learning_rate", "batch_size", "max_steps"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"configuration missing required field: {field}")

        # Validate learning rate
        lr = v.get("learning_rate")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("learning_rate must be a positive number")

        # Validate batch size
        batch_size = v.get("batch_size")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        # Validate max steps
        max_steps = v.get("max_steps")
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        return v


class ModelDeploymentRequest(BaseModel):
    """Request for model deployment."""

    deployment_type: str = Field(...)
    traffic_percentage: Optional[int] = Field(default=None, ge=0, le=100)
    rollback_model_id: Optional[str] = Field(default=None)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("deployment_type")
    def validate_deployment_type(cls, v):
        allowed_types = ["staging", "canary", "production", "rollback"]
        if v not in allowed_types:
            raise ValueError(f"deployment_type must be one of {allowed_types}")
        return v

    @validator("rollback_model_id")
    def validate_rollback_model_id(cls, v):
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError("rollback_model_id must be a valid UUID")
        return v


class ExperimentRequest(BaseModel):
    """Request for A/B testing experiment."""

    experiment_name: str = Field(..., min_length=1, max_length=100)
    model_a_id: str = Field(...)
    model_b_id: str = Field(...)
    traffic_split_percentage: int = Field(default=50, ge=0, le=100)
    success_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("model_a_id", "model_b_id")
    def validate_model_ids(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("model_id must be a valid UUID")
        return v

    @validator("model_a_id", "model_b_id")
    def validate_different_models(cls, v, values):
        if "model_a_id" in values and v == values["model_a_id"]:
            raise ValueError("model_a_id and model_b_id must be different")
        return v


class TaxonomyRequest(BaseModel):
    """Request for taxonomy creation."""

    taxonomy_name: str = Field(..., min_length=1, max_length=100)
    version: str = Field(..., min_length=1, max_length=50)
    taxonomy_data: Dict[str, Any] = Field(...)
    schema_version: str = Field(..., min_length=1, max_length=50)
    backward_compatible: bool = Field(default=True)
    migration_notes: Optional[str] = Field(default=None)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("taxonomy_data")
    def validate_taxonomy_data(cls, v):
        required_fields = ["categories"]
        for field in required_fields:
            if field not in v:
                raise ValueError(f"taxonomy_data missing required field: {field}")

        # Validate categories structure
        categories = v.get("categories", {})
        if not isinstance(categories, dict):
            raise ValueError("categories must be a dictionary")

        for category_name, category_data in categories.items():
            if not isinstance(category_data, dict):
                raise ValueError(f"category {category_name} must be a dictionary")

            if "subcategories" not in category_data:
                raise ValueError(f"category {category_name} missing subcategories")

        return v


class DetectorRegistrationRequest(BaseModel):
    """Request for detector registration."""

    detector_type: str = Field(..., min_length=1, max_length=50)
    detector_name: str = Field(..., min_length=1, max_length=100)
    endpoint_url: str = Field(..., min_length=1, max_length=500)
    health_check_url: Optional[str] = Field(default=None, max_length=500)
    version: str = Field(..., min_length=1, max_length=50)
    capabilities: Optional[List[str]] = Field(default_factory=list)
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tenant_id: str = Field(..., min_length=1, max_length=100)

    @validator("endpoint_url", "health_check_url")
    def validate_urls(cls, v):
        if v is not None:
            import re

            url_pattern = re.compile(
                r"^https?://"  # http:// or https://
                r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
                r"localhost|"  # localhost...
                r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
                r"(?::\d+)?"  # optional port
                r"(?:/?|[/?]\S+)$",
                re.IGNORECASE,
            )

            if not url_pattern.match(v):
                raise ValueError("Invalid URL format")
        return v
