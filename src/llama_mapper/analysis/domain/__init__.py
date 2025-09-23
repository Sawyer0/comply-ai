"""
Domain layer for the Analysis Module.

This module contains the core domain entities, services, and interfaces
that define the business logic and contracts for the analysis module.
"""

from .entities import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AnalysisErrorResponse,
    AnalysisType,
    VersionInfo,
    QualityMetrics,
    HealthStatus,
    IdempotencyCache,
)
from .interfaces import (
    IModelServer,
    IValidator,
    ITemplateProvider,
    IOPAGenerator,
    ISecurityValidator,
    IIdempotencyManager,
    IQualityEvaluator,
)
from .services import (
    AnalysisService,
    BatchAnalysisService,
    ValidationService,
    QualityService,
    HealthService,
)

__all__ = [
    # Entities
    "AnalysisRequest",
    "AnalysisResponse", 
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "AnalysisErrorResponse",
    "AnalysisType",
    "VersionInfo",
    "QualityMetrics",
    "HealthStatus",
    "IdempotencyCache",
    
    # Interfaces
    "IModelServer",
    "IValidator",
    "ITemplateProvider", 
    "IOPAGenerator",
    "ISecurityValidator",
    "IIdempotencyManager",
    "IQualityEvaluator",
    
    # Services
    "AnalysisService",
    "BatchAnalysisService",
    "ValidationService",
    "QualityService",
    "HealthService",
]