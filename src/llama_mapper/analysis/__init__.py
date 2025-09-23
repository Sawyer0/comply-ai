"""
Analysis Module for Comply-AI Platform

This module provides automated analysis of structured security metrics to generate
concise explanations, remediations, and policy recommendations using Phi-3 Mini.

The module follows a domain-driven design architecture with clear separation of concerns:
- Domain Layer: Core business logic, entities, and domain services
- Application Layer: Use cases, application services, and DTOs
- Infrastructure Layer: External service implementations and technical concerns
- Configuration Layer: Settings management and dependency injection

Key Components:
- AnalysisService: Core domain service for analysis operations
- AnalysisApplicationService: Application service for orchestration
- AnalysisModuleFactory: Factory for dependency injection
- AnalysisConfig: Configuration management

The module follows the service contracts defined in .kiro/specs/service-contracts.md
and provides deterministic, auditable analysis outputs for compliance reporting.
"""

# Domain layer exports
from .domain.entities import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AnalysisErrorResponse,
    VersionInfo,
    AnalysisType,
    IdempotencyCache,
    QualityMetrics,
    HealthStatus,
)

from .domain.services import (
    AnalysisService,
    BatchAnalysisService,
    ValidationService,
)

from .domain.interfaces import (
    IModelServer,
    IValidator,
    ITemplateProvider,
    IOPAGenerator,
    ISecurityValidator,
    IIdempotencyManager,
    IQualityEvaluator,
)

# Application layer exports
from .application.services import (
    AnalysisApplicationService,
    BatchAnalysisApplicationService,
)

from .application.use_cases import (
    AnalyzeMetricsUseCase,
    BatchAnalyzeMetricsUseCase,
    QualityEvaluationUseCase,
    CacheManagementUseCase,
    HealthCheckUseCase,
)

from .application.dto import (
    AnalysisRequestDTO,
    AnalysisResponseDTO,
    BatchAnalysisRequestDTO,
    BatchAnalysisResponseDTO,
    AnalysisErrorResponseDTO,
    AnalysisMetricsDTO,
    QualityEvaluationDTO,
    HealthCheckDTO,
)

# Infrastructure layer exports
from .infrastructure.model_server import Phi3AnalysisModelServer
from .infrastructure.validator import AnalysisValidator
from .infrastructure.template_provider import AnalysisTemplateProvider
from .infrastructure.opa_generator import OPAPolicyGenerator
from .infrastructure.security import AnalysisSecurityValidator
from .infrastructure.idempotency import IdempotencyManager
from .infrastructure.quality_evaluator import QualityEvaluator

# Security layer exports
from .security.waf import (
    AttackType, ViolationSeverity, WAFViolation,
    IWAFRule, IWAFRuleEngine, IWAFMiddleware, IWAFMetricsCollector,
    WAFRuleEngine, WAFRule, WAFMiddleware, WAFFactory
)

# Resilience layer exports
from .resilience import (
    CircuitState, RetryStrategy,
    ICircuitBreaker, IRetryManager, IResilienceManager,
    IResilienceMetricsCollector,
    CircuitBreaker, CircuitBreakerOpenException, RetryManager,
    RetryConfig, CircuitBreakerConfig,
    ResilienceFactory, ResilienceManager
)

# Configuration layer exports
from .config.settings import AnalysisSettings, AnalysisConfig
from .config.factory import AnalysisModuleFactory

# Quality monitoring and alerting exports
from .quality import (
    QualityMetricType, AlertSeverity, AlertStatus, DegradationType,
    QualityMetric, QualityThreshold, DegradationDetection, Alert,
    IQualityMonitor, IQualityDetector, IAlertHandler, IAlertManager,
    IQualityAlertingSystem, QualityMonitor, QualityDegradationDetector,
    AlertManager, LoggingAlertHandler, EmailAlertHandler, SlackAlertHandler,
    WebhookAlertHandler, CompositeAlertHandler, QualityAlertingSystem,
    QualityAlertingConfig, QualityAlertingSettings
)

# API factory
from .api.factory import create_analysis_app

__all__ = [
    # Domain entities
    "AnalysisRequest",
    "AnalysisResponse", 
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "AnalysisErrorResponse",
    "VersionInfo",
    "AnalysisType",
    "IdempotencyCache",
    "QualityMetrics",
    "HealthStatus",
    
    # Domain services
    "AnalysisService",
    "BatchAnalysisService",
    "ValidationService",
    
    # Domain interfaces
    "IModelServer",
    "IValidator",
    "ITemplateProvider",
    "IOPAGenerator",
    "ISecurityValidator",
    "IIdempotencyManager",
    "IQualityEvaluator",
    
    # Application services
    "AnalysisApplicationService",
    "BatchAnalysisApplicationService",
    
    # Application use cases
    "AnalyzeMetricsUseCase",
    "BatchAnalyzeMetricsUseCase",
    "QualityEvaluationUseCase",
    "CacheManagementUseCase",
    "HealthCheckUseCase",
    
    # Application DTOs
    "AnalysisRequestDTO",
    "AnalysisResponseDTO",
    "BatchAnalysisRequestDTO",
    "BatchAnalysisResponseDTO",
    "AnalysisErrorResponseDTO",
    "AnalysisMetricsDTO",
    "QualityEvaluationDTO",
    "HealthCheckDTO",
    
           # Infrastructure implementations
           "Phi3AnalysisModelServer",
           "AnalysisValidator",
           "AnalysisTemplateProvider",
           "OPAPolicyGenerator",
           "AnalysisSecurityValidator",
           "IdempotencyManager",
           "QualityEvaluator",

           # Security layer
           "AttackType",
           "ViolationSeverity",
           "WAFViolation",
           "IWAFRule",
           "IWAFRuleEngine",
           "IWAFMiddleware",
           "IWAFMetricsCollector",
           "WAFRuleEngine",
           "WAFRule",
           "WAFMiddleware",
           "WAFFactory",

           # Resilience layer
           "CircuitState",
           "RetryStrategy",
           "ICircuitBreaker",
           "IRetryManager",
           "IResilienceManager",
           "IResilienceMetricsCollector",
           "CircuitBreaker",
           "CircuitBreakerOpenException",
           "RetryManager",
           "RetryConfig",
           "CircuitBreakerConfig",
           "ResilienceFactory",
           "ResilienceManager",
    
    # Configuration
    "AnalysisSettings",
    "AnalysisConfig",
    "AnalysisModuleFactory",

    # Quality monitoring and alerting
    "QualityMetricType",
    "AlertSeverity",
    "AlertStatus", 
    "DegradationType",
    "QualityMetric",
    "QualityThreshold",
    "DegradationDetection",
    "Alert",
    "IQualityMonitor",
    "IQualityDetector",
    "IAlertHandler",
    "IAlertManager",
    "IQualityAlertingSystem",
    "QualityMonitor",
    "QualityDegradationDetector",
    "AlertManager",
    "LoggingAlertHandler",
    "EmailAlertHandler",
    "SlackAlertHandler",
    "WebhookAlertHandler",
    "CompositeAlertHandler",
    "QualityAlertingSystem",
    "QualityAlertingConfig",
    "QualityAlertingSettings",
    
    # Legacy API
    "create_analysis_app",
]
