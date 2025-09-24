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

# API factory
from .api.factory import create_analysis_app
from .application.dto import (
    AnalysisErrorResponseDTO,
    AnalysisMetricsDTO,
    AnalysisRequestDTO,
    AnalysisResponseDTO,
    BatchAnalysisRequestDTO,
    BatchAnalysisResponseDTO,
    HealthCheckDTO,
    QualityEvaluationDTO,
)

# Application layer exports
from .application.services import (
    AnalysisApplicationService,
    BatchAnalysisApplicationService,
)
from .application.use_cases import (
    AnalyzeMetricsUseCase,
    BatchAnalyzeMetricsUseCase,
    CacheManagementUseCase,
    HealthCheckUseCase,
    QualityEvaluationUseCase,
)
from .config.factory import AnalysisModuleFactory

# Configuration layer exports
from .config.settings import AnalysisConfig, AnalysisSettings

# Domain layer exports
from .domain.entities import (
    AnalysisErrorResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisType,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    HealthStatus,
    IdempotencyCache,
    QualityMetrics,
    VersionInfo,
)
from .domain.interfaces import (
    IIdempotencyManager,
    IModelServer,
    IOPAGenerator,
    IQualityEvaluator,
    ISecurityValidator,
    ITemplateProvider,
    IValidator,
)
from .domain.services import (
    AnalysisService,
    BatchAnalysisService,
    ValidationService,
)
from .infrastructure.idempotency import IdempotencyManager

# Infrastructure layer exports
from .infrastructure.model_server import Phi3AnalysisModelServer
from .infrastructure.opa_generator import OPAPolicyGenerator
from .infrastructure.quality_evaluator import QualityEvaluator
from .infrastructure.security import AnalysisSecurityValidator
from .infrastructure.template_provider import AnalysisTemplateProvider
from .infrastructure.refactored_template_provider import RefactoredTemplateProvider, EnhancedTemplateProvider
from .infrastructure.validator import AnalysisValidator

# Specialized engines exports
from .engines import (
    IAnalysisEngine,
    IPatternRecognitionEngine,
    IRiskScoringEngine,
    IComplianceIntelligence,
    ITemplateOrchestrator,
    PatternRecognitionEngine,
    RiskScoringEngine,
    ComplianceIntelligence,
    TemplateOrchestrator,
)

# Quality monitoring and alerting exports
from .quality import (
    Alert,
    AlertManager,
    AlertSeverity,
    AlertStatus,
    CompositeAlertHandler,
    DegradationDetection,
    DegradationType,
    EmailAlertHandler,
    IAlertHandler,
    IAlertManager,
    IQualityAlertingSystem,
    IQualityDetector,
    IQualityMonitor,
    LoggingAlertHandler,
    QualityAlertingConfig,
    QualityAlertingSettings,
    QualityAlertingSystem,
    QualityDegradationDetector,
    QualityMetric,
    QualityMetricType,
    QualityMonitor,
    QualityThreshold,
    SlackAlertHandler,
    WebhookAlertHandler,
)

# Resilience layer exports
from .resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitState,
    ICircuitBreaker,
    IResilienceManager,
    IResilienceMetricsCollector,
    IRetryManager,
    ResilienceFactory,
    ResilienceManager,
    RetryConfig,
    RetryManager,
    RetryStrategy,
)

# Security layer exports
from .security.waf import (
    AttackType,
    IWAFMetricsCollector,
    IWAFMiddleware,
    IWAFRule,
    IWAFRuleEngine,
    ViolationSeverity,
    WAFFactory,
    WAFMiddleware,
    WAFRule,
    WAFRuleEngine,
    WAFViolation,
)

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
    "RefactoredTemplateProvider",
    "EnhancedTemplateProvider",
    "OPAPolicyGenerator",
    "AnalysisSecurityValidator",
    "IdempotencyManager",
    "QualityEvaluator",
    # Specialized engines
    "IAnalysisEngine",
    "IPatternRecognitionEngine",
    "IRiskScoringEngine", 
    "IComplianceIntelligence",
    "ITemplateOrchestrator",
    "PatternRecognitionEngine",
    "RiskScoringEngine",
    "ComplianceIntelligence", 
    "TemplateOrchestrator",
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
