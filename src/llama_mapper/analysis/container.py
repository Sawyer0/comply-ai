"""
Dependency injection container for the Analysis Module.

This module provides a container for managing dependencies and
wiring together the different layers of the analysis module.
"""

import logging
from typing import Optional

from ..config.settings import Settings
from .application.services import AnalysisApplicationService
from .application.use_cases import (
    AnalyzeMetricsUseCase,
    BatchAnalyzeMetricsUseCase,
    CacheManagementUseCase,
    HealthCheckUseCase,
    QualityEvaluationUseCase,
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
    HealthService,
    QualityService,
    ValidationService,
)
from .infrastructure import (
    AnalysisSecurityValidator,
    AnalysisTemplateProvider,
    AnalysisValidator,
    MemoryIdempotencyManager,
    OPAPolicyGenerator,
    Phi3AnalysisModelServer,
    QualityEvaluator,
)

logger = logging.getLogger(__name__)


class AnalysisContainer:
    """
    Dependency injection container for the Analysis Module.

    Manages the creation and wiring of all dependencies
    for the analysis module components.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the analysis container.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings()
        self._initialized = False

        # Infrastructure components
        self._model_server: Optional[IModelServer] = None
        self._validator: Optional[IValidator] = None
        self._template_provider: Optional[ITemplateProvider] = None
        self._opa_generator: Optional[IOPAGenerator] = None
        self._security_validator: Optional[ISecurityValidator] = None
        self._idempotency_manager: Optional[IIdempotencyManager] = None
        self._quality_evaluator: Optional[IQualityEvaluator] = None

        # Domain services
        self._analysis_service: Optional[AnalysisService] = None
        self._batch_analysis_service: Optional[BatchAnalysisService] = None
        self._validation_service: Optional[ValidationService] = None
        self._quality_service: Optional[QualityService] = None
        self._health_service: Optional[HealthService] = None

        # Use cases
        self._analyze_metrics_use_case: Optional[AnalyzeMetricsUseCase] = None
        self._batch_analyze_metrics_use_case: Optional[BatchAnalyzeMetricsUseCase] = (
            None
        )
        self._quality_evaluation_use_case: Optional[QualityEvaluationUseCase] = None
        self._cache_management_use_case: Optional[CacheManagementUseCase] = None
        self._health_check_use_case: Optional[HealthCheckUseCase] = None

        # Application service
        self._analysis_application_service: Optional[AnalysisApplicationService] = None

        logger.info("Initialized Analysis Container")

    def initialize(self) -> None:
        """Initialize all dependencies."""
        if self._initialized:
            return

        logger.info("Initializing Analysis Module dependencies...")

        # Initialize infrastructure components
        self._initialize_infrastructure()

        # Initialize domain services
        self._initialize_domain_services()

        # Initialize use cases
        self._initialize_use_cases()

        # Initialize application service
        self._initialize_application_service()

        self._initialized = True
        logger.info("Analysis Module dependencies initialized successfully")

    def _initialize_infrastructure(self) -> None:
        """Initialize infrastructure components."""
        # Model server
        self._model_server = Phi3AnalysisModelServer(
            model_path=self.settings.analysis.analysis_model_path,
            temperature=self.settings.analysis.analysis_temperature,
            confidence_cutoff=self.settings.analysis.analysis_confidence_cutoff,
            settings=self.settings,
        )

        # Validator
        self._validator = AnalysisValidator()

        # Template provider
        self._template_provider = AnalysisTemplateProvider()

        # OPA generator
        self._opa_generator = OPAPolicyGenerator()

        # Security validator
        self._security_validator = AnalysisSecurityValidator()

        # Idempotency manager
        self._idempotency_manager = MemoryIdempotencyManager(
            ttl_hours=self.settings.analysis.idempotency_cache_ttl_hours,
            max_items=self.settings.analysis.cache_max_items,
        )

        # Quality evaluator
        self._quality_evaluator = QualityEvaluator(
            golden_dataset_path=self.settings.analysis.golden_dataset_path
        )

        logger.info("Infrastructure components initialized")

    def _initialize_domain_services(self) -> None:
        """Initialize domain services."""
        # Analysis service
        self._analysis_service = AnalysisService(
            model_server=self._model_server,
            validator=self._validator,
            template_provider=self._template_provider,
            security_validator=self._security_validator,
        )

        # Batch analysis service
        self._batch_analysis_service = BatchAnalysisService(
            analysis_service=self._analysis_service,
            idempotency_manager=self._idempotency_manager,
            max_concurrent=self.settings.analysis.max_concurrent_requests,
            request_timeout=self.settings.analysis.request_timeout_seconds,
        )

        # Validation service
        self._validation_service = ValidationService(
            validator=self._validator,
            security_validator=self._security_validator,
            opa_generator=self._opa_generator,
        )

        # Quality service
        self._quality_service = QualityService(
            quality_evaluator=self._quality_evaluator
        )

        # Health service
        self._health_service = HealthService(
            model_server=self._model_server,
            idempotency_manager=self._idempotency_manager,
            quality_evaluator=self._quality_evaluator,
        )

        logger.info("Domain services initialized")

    def _initialize_use_cases(self) -> None:
        """Initialize use cases."""
        # Analyze metrics use case
        self._analyze_metrics_use_case = AnalyzeMetricsUseCase(
            analysis_service=self._analysis_service,
            validation_service=self._validation_service,
        )

        # Batch analyze metrics use case
        self._batch_analyze_metrics_use_case = BatchAnalyzeMetricsUseCase(
            batch_analysis_service=self._batch_analysis_service,
            validation_service=self._validation_service,
        )

        # Quality evaluation use case
        self._quality_evaluation_use_case = QualityEvaluationUseCase(
            quality_evaluator=self._quality_evaluator
        )

        # Cache management use case
        self._cache_management_use_case = CacheManagementUseCase(
            idempotency_manager=self._idempotency_manager
        )

        # Health check use case
        self._health_check_use_case = HealthCheckUseCase(
            model_server=self._model_server,
            validator=self._validator,
            opa_generator=self._opa_generator,
        )

        logger.info("Use cases initialized")

    def _initialize_application_service(self) -> None:
        """Initialize application service."""
        self._analysis_application_service = AnalysisApplicationService(
            analyze_metrics_use_case=self._analyze_metrics_use_case,
            batch_analyze_metrics_use_case=self._batch_analyze_metrics_use_case,
            quality_evaluation_use_case=self._quality_evaluation_use_case,
            cache_management_use_case=self._cache_management_use_case,
            health_check_use_case=self._health_check_use_case,
        )

        logger.info("Application service initialized")

    # Property getters for dependency injection
    @property
    def model_server(self) -> IModelServer:
        """Get model server instance."""
        if not self._initialized:
            self.initialize()
        return self._model_server

    @property
    def validator(self) -> IValidator:
        """Get validator instance."""
        if not self._initialized:
            self.initialize()
        return self._validator

    @property
    def template_provider(self) -> ITemplateProvider:
        """Get template provider instance."""
        if not self._initialized:
            self.initialize()
        return self._template_provider

    @property
    def opa_generator(self) -> IOPAGenerator:
        """Get OPA generator instance."""
        if not self._initialized:
            self.initialize()
        return self._opa_generator

    @property
    def security_validator(self) -> ISecurityValidator:
        """Get security validator instance."""
        if not self._initialized:
            self.initialize()
        return self._security_validator

    @property
    def idempotency_manager(self) -> IIdempotencyManager:
        """Get idempotency manager instance."""
        if not self._initialized:
            self.initialize()
        return self._idempotency_manager

    @property
    def quality_evaluator(self) -> IQualityEvaluator:
        """Get quality evaluator instance."""
        if not self._initialized:
            self.initialize()
        return self._quality_evaluator

    @property
    def analysis_service(self) -> AnalysisService:
        """Get analysis service instance."""
        if not self._initialized:
            self.initialize()
        return self._analysis_service

    @property
    def batch_analysis_service(self) -> BatchAnalysisService:
        """Get batch analysis service instance."""
        if not self._initialized:
            self.initialize()
        return self._batch_analysis_service

    @property
    def validation_service(self) -> ValidationService:
        """Get validation service instance."""
        if not self._initialized:
            self.initialize()
        return self._validation_service

    @property
    def quality_service(self) -> QualityService:
        """Get quality service instance."""
        if not self._initialized:
            self.initialize()
        return self._quality_service

    @property
    def health_service(self) -> HealthService:
        """Get health service instance."""
        if not self._initialized:
            self.initialize()
        return self._health_service

    @property
    def analyze_metrics_use_case(self) -> AnalyzeMetricsUseCase:
        """Get analyze metrics use case instance."""
        if not self._initialized:
            self.initialize()
        return self._analyze_metrics_use_case

    @property
    def batch_analyze_metrics_use_case(self) -> BatchAnalyzeMetricsUseCase:
        """Get batch analyze metrics use case instance."""
        if not self._initialized:
            self.initialize()
        return self._batch_analyze_metrics_use_case

    @property
    def quality_evaluation_use_case(self) -> QualityEvaluationUseCase:
        """Get quality evaluation use case instance."""
        if not self._initialized:
            self.initialize()
        return self._quality_evaluation_use_case

    @property
    def cache_management_use_case(self) -> CacheManagementUseCase:
        """Get cache management use case instance."""
        if not self._initialized:
            self.initialize()
        return self._cache_management_use_case

    @property
    def health_check_use_case(self) -> HealthCheckUseCase:
        """Get health check use case instance."""
        if not self._initialized:
            self.initialize()
        return self._health_check_use_case

    @property
    def analysis_application_service(self) -> AnalysisApplicationService:
        """Get analysis application service instance."""
        if not self._initialized:
            self.initialize()
        return self._analysis_application_service


# Global container instance
_container: Optional[AnalysisContainer] = None


def get_container(settings: Optional[Settings] = None) -> AnalysisContainer:
    """
    Get the global analysis container instance.

    Args:
        settings: Application settings

    Returns:
        Analysis container instance
    """
    global _container

    if _container is None:
        _container = AnalysisContainer(settings)
        _container.initialize()

    return _container


def reset_container() -> None:
    """Reset the global container instance."""
    global _container
    _container = None
