"""Shared interfaces for microservice communication following SRP."""

# Base interfaces
from .base import (
    BaseRequest,
    BaseResponse,
    BaseService,
    HealthResponse,
    ApiResponse,
    PaginatedResponse,
)

# Common types (shared across services to avoid duplication)
from .common import *

# Service-specific interfaces
from .orchestration import *
from .analysis import *
from .mapper import *
from .detector_output import *

__all__ = [
    "BaseRequest",
    "BaseResponse",
    "BaseService",
    "HealthResponse",
    "ApiResponse",
    "PaginatedResponse",
    # Common types
    "ProcessingMode",
    "HealthStatus",
    "Severity",
    "RiskLevel",
    "JobStatus",
    "ModelStatus",
    "ComplianceStatus",
    "AlertStatus",
    "AlertSeverity",
    # Orchestration interfaces
    "OrchestrationRequest",
    "OrchestrationResponse",
    "DetectorResult",
    "DetectorInfo",
    "DetectorHealthStatus",
    "PolicyValidationRequest",
    "PolicyValidationResponse",
    "AsyncJobRequest",
    "AsyncJobResponse",
    "AsyncJobStatus",
    # Canonical detector outputs
    "CanonicalDetectorEntity",
    "CanonicalDetectorOutput",
    "CanonicalDetectorOutputs",
    # Analysis interfaces
    "AnalysisRequest",
    "AnalysisResponse",
    "CanonicalTaxonomyResult",
    "PatternAnalysisResult",
    "RiskScoringResult",
    "ComplianceMappingResult",
    "RAGInsights",
    "QualityMetrics",
    # Mapper interfaces
    "MappingRequest",
    "MappingResponse",
    "MappingResult",
    "ComplianceMapping",
    "ValidationRequest",
    "ValidationResult",
    "BatchMappingRequest",
    "BatchMappingResponse",
]
