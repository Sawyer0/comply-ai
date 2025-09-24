"""
Domain layer for the Analysis Module.

This module contains the core domain entities, services, and interfaces
that define the business logic and contracts for the analysis module.
"""

from .entities import (
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
from .interfaces import (
    IIdempotencyManager,
    IModelServer,
    IOPAGenerator,
    IQualityEvaluator,
    ISecurityValidator,
    ITemplateProvider,
    IValidator,
)
from .services import (
    AnalysisService,
    BatchAnalysisService,
    HealthService,
    QualityService,
    ValidationService,
)

# Enhanced analysis interfaces and models
from .analysis_interfaces import (
    IAnalysisEngine,
    IPatternRecognitionEngine,
    IRiskScoringEngine,
    IComplianceIntelligenceEngine,
    IThresholdOptimizationEngine,
    IIncidentCorrelationEngine,
    IPredictiveAnalyticsEngine,
    IReportGenerationEngine,
    ITemplateOrchestrator,
)
from .analysis_models import (
    AnalysisConfiguration,
    AnalysisResult,
    AnalysisStrategy,
    AttackPattern,
    BusinessImpact,
    BusinessRelevance,
    ComplianceFramework,
    ComplianceGap,
    ComplianceMapping,
    IncidentCorrelation,
    IntegratedAnalysisResult,
    Pattern,
    PatternCorrelation,
    PatternStrength,
    PatternType,
    RemediationAction,
    RemediationPlan,
    RiskBreakdown,
    RiskFactor,
    RiskLevel,
    RiskScore,
    SecurityData,
    SecurityFinding,
    SecurityIncident,
    ThresholdPerformance,
    ThresholdRecommendation,
    TimeRange,
)
from .base_classes import (
    BaseAnalysisEngine,
    BusinessContextEngine,
    ConfigurableEngine,
    StatisticalAnalysisEngine,
)

__all__ = [
    # Original entities
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
    # Original interfaces
    "IModelServer",
    "IValidator",
    "ITemplateProvider",
    "IOPAGenerator",
    "ISecurityValidator",
    "IIdempotencyManager",
    "IQualityEvaluator",
    # Original services
    "AnalysisService",
    "BatchAnalysisService",
    "ValidationService",
    "QualityService",
    "HealthService",
    # Enhanced analysis interfaces
    "IAnalysisEngine",
    "IPatternRecognitionEngine",
    "IRiskScoringEngine",
    "IComplianceIntelligenceEngine",
    "IThresholdOptimizationEngine",
    "IIncidentCorrelationEngine",
    "IPredictiveAnalyticsEngine",
    "IReportGenerationEngine",
    "ITemplateOrchestrator",
    # Enhanced analysis models
    "AnalysisConfiguration",
    "AnalysisResult",
    "AnalysisStrategy",
    "AttackPattern",
    "BusinessImpact",
    "BusinessRelevance",
    "ComplianceFramework",
    "ComplianceGap",
    "ComplianceMapping",
    "IncidentCorrelation",
    "IntegratedAnalysisResult",
    "Pattern",
    "PatternCorrelation",
    "PatternStrength",
    "PatternType",
    "RemediationAction",
    "RemediationPlan",
    "RiskBreakdown",
    "RiskFactor",
    "RiskLevel",
    "RiskScore",
    "SecurityData",
    "SecurityFinding",
    "SecurityIncident",
    "ThresholdPerformance",
    "ThresholdRecommendation",
    "TimeRange",
    # Base classes
    "BaseAnalysisEngine",
    "BusinessContextEngine",
    "ConfigurableEngine",
    "StatisticalAnalysisEngine",
]
