"""
Factory for creating analysis module components.

This module provides a factory pattern for creating and configuring
analysis module components with proper dependency injection.
"""

import logging
from typing import Any, Dict, Optional

from ..application.services import (
    AnalysisApplicationService,
    BatchAnalysisApplicationService,
)
from ..application.use_cases import (
    AnalyzeMetricsUseCase,
    BatchAnalyzeMetricsUseCase,
    CacheManagementUseCase,
    HealthCheckUseCase,
    QualityEvaluationUseCase,
)
from ..domain.entities import VersionInfo
from ..domain.interfaces import (
    IIdempotencyManager,
    IModelServer,
    IOPAGenerator,
    IQualityEvaluator,
    ISecurityValidator,
    ITemplateProvider,
    IValidator,
)
from ..domain.services import (
    AnalysisService,
    BatchAnalysisService,
    HealthService,
    QualityService,
    ValidationService,
)
from ..infrastructure.idempotency import MemoryIdempotencyManager
from ..infrastructure.model_server import Phi3AnalysisModelServer
from ..infrastructure.opa_generator import OPAPolicyGenerator
from ..infrastructure.quality_evaluator import QualityEvaluator
from ..infrastructure.security import AnalysisSecurityValidator
from ..infrastructure.template_provider import AnalysisTemplateProvider
from ..infrastructure.validator import AnalysisValidator
from .settings import AnalysisConfig

logger = logging.getLogger(__name__)


class AnalysisModuleFactory:
    """
    Factory for creating analysis module components.

    Provides a centralized way to create and configure all analysis
    module components with proper dependency injection.
    """

    def __init__(self, config: AnalysisConfig):
        """
        Initialize the factory with configuration.

        Args:
            config: Analysis module configuration
        """
        self.config = config
        self._components: Dict[str, Any] = {}

    def create_model_server(self) -> IModelServer:
        """Create model server implementation."""
        if "model_server" not in self._components:
            self._components["model_server"] = Phi3AnalysisModelServer(
                model_path=self.config.analysis_model_path,
                temperature=self.config.analysis_temperature,
                confidence_cutoff=self.config.analysis_confidence_cutoff,
                settings=self.config.base_settings,
            )
        return self._components["model_server"]

    def create_validator(self) -> IValidator:
        """Create validator implementation."""
        if "validator" not in self._components:
            self._components["validator"] = AnalysisValidator()
        return self._components["validator"]

    def create_template_provider(self) -> ITemplateProvider:
        """Create template provider implementation."""
        if "template_provider" not in self._components:
            self._components["template_provider"] = AnalysisTemplateProvider()
        return self._components["template_provider"]

    def create_opa_generator(self) -> IOPAGenerator:
        """Create OPA generator implementation."""
        if "opa_generator" not in self._components:
            self._components["opa_generator"] = OPAPolicyGenerator()
        return self._components["opa_generator"]

    def create_security_validator(self) -> ISecurityValidator:
        """Create security validator implementation."""
        if "security_validator" not in self._components:
            self._components["security_validator"] = AnalysisSecurityValidator()
        return self._components["security_validator"]

    def create_idempotency_manager(self) -> IIdempotencyManager:
        """Create idempotency manager implementation."""
        if "idempotency_manager" not in self._components:
            self._components["idempotency_manager"] = MemoryIdempotencyManager(
                ttl_hours=self.config.idempotency_cache_ttl_hours,
                max_items=self.config.cache_max_items,
            )
        return self._components["idempotency_manager"]

    def create_quality_evaluator(self) -> IQualityEvaluator:
        """Create quality evaluator implementation."""
        if "quality_evaluator" not in self._components:
            self._components["quality_evaluator"] = QualityEvaluator(
                golden_dataset_path=self.config.golden_dataset_path
            )
        return self._components["quality_evaluator"]

    def create_version_info(self) -> VersionInfo:
        """Create version information."""
        if "version_info" not in self._components:
            self._components["version_info"] = VersionInfo(
                taxonomy="v1.0.0",
                frameworks="v1.0.0",
                analyst_model="phi3-mini-3.8b-v1.0",
            )
        return self._components["version_info"]

    def create_analysis_service(self) -> AnalysisService:
        """Create analysis service."""
        if "analysis_service" not in self._components:
            self._components["analysis_service"] = AnalysisService(
                model_server=self.create_model_server(),
                validator=self.create_validator(),
                template_provider=self.create_template_provider(),
                security_validator=self.create_security_validator(),
            )
        return self._components["analysis_service"]

    def create_batch_analysis_service(self) -> BatchAnalysisService:
        """Create batch analysis service."""
        if "batch_analysis_service" not in self._components:
            self._components["batch_analysis_service"] = BatchAnalysisService(
                analysis_service=self.create_analysis_service(),
                idempotency_manager=self.create_idempotency_manager(),
                max_concurrent=self.config.max_concurrent_requests,
                request_timeout=self.config.request_timeout_seconds,
            )
        return self._components["batch_analysis_service"]

    def create_validation_service(self) -> ValidationService:
        """Create validation service."""
        if "validation_service" not in self._components:
            self._components["validation_service"] = ValidationService(
                validator=self.create_validator(),
                security_validator=self.create_security_validator(),
                opa_generator=self.create_opa_generator(),
            )
        return self._components["validation_service"]

    def create_analyze_metrics_use_case(self) -> AnalyzeMetricsUseCase:
        """Create analyze metrics use case."""
        if "analyze_metrics_use_case" not in self._components:
            self._components["analyze_metrics_use_case"] = AnalyzeMetricsUseCase(
                analysis_service=self.create_analysis_service(),
                validation_service=self.create_validation_service(),
            )
        return self._components["analyze_metrics_use_case"]

    def create_batch_analyze_metrics_use_case(self) -> BatchAnalyzeMetricsUseCase:
        """Create batch analyze metrics use case."""
        if "batch_analyze_metrics_use_case" not in self._components:
            self._components["batch_analyze_metrics_use_case"] = (
                BatchAnalyzeMetricsUseCase(
                    batch_analysis_service=self.create_batch_analysis_service(),
                    validation_service=self.create_validation_service(),
                )
            )
        return self._components["batch_analyze_metrics_use_case"]

    def create_quality_evaluation_use_case(self) -> QualityEvaluationUseCase:
        """Create quality evaluation use case."""
        if "quality_evaluation_use_case" not in self._components:
            self._components["quality_evaluation_use_case"] = QualityEvaluationUseCase(
                quality_evaluator=self.create_quality_evaluator()
            )
        return self._components["quality_evaluation_use_case"]

    def create_cache_management_use_case(self) -> CacheManagementUseCase:
        """Create cache management use case."""
        if "cache_management_use_case" not in self._components:
            self._components["cache_management_use_case"] = CacheManagementUseCase(
                idempotency_manager=self.create_idempotency_manager()
            )
        return self._components["cache_management_use_case"]

    def create_health_check_use_case(self) -> HealthCheckUseCase:
        """Create health check use case."""
        if "health_check_use_case" not in self._components:
            self._components["health_check_use_case"] = HealthCheckUseCase(
                model_server=self.create_model_server(),
                validator=self.create_validator(),
                opa_generator=self.create_opa_generator(),
            )
        return self._components["health_check_use_case"]

    def create_analysis_application_service(self) -> AnalysisApplicationService:
        """Create analysis application service."""
        if "analysis_application_service" not in self._components:
            self._components["analysis_application_service"] = (
                AnalysisApplicationService(
                    analyze_metrics_use_case=self.create_analyze_metrics_use_case(),
                    batch_analyze_metrics_use_case=self.create_batch_analyze_metrics_use_case(),
                    quality_evaluation_use_case=self.create_quality_evaluation_use_case(),
                    cache_management_use_case=self.create_cache_management_use_case(),
                    health_check_use_case=self.create_health_check_use_case(),
                )
            )
        return self._components["analysis_application_service"]

    def create_batch_analysis_application_service(
        self,
    ) -> BatchAnalysisApplicationService:
        """Create batch analysis application service."""
        if "batch_analysis_application_service" not in self._components:
            self._components["batch_analysis_application_service"] = (
                BatchAnalysisApplicationService(
                    batch_analyze_metrics_use_case=self.create_batch_analyze_metrics_use_case(),
                    cache_management_use_case=self.create_cache_management_use_case(),
                )
            )
        return self._components["batch_analysis_application_service"]

    def get_component(self, component_name: str) -> Any:
        """
        Get a component by name.

        Args:
            component_name: Name of the component

        Returns:
            Component instance

        Raises:
            KeyError: If component is not found
        """
        if component_name not in self._components:
            raise KeyError(f"Component '{component_name}' not found")
        return self._components[component_name]

    def list_components(self) -> list[str]:
        """List all created components."""
        return list(self._components.keys())

    def clear_components(self) -> None:
        """Clear all components (useful for testing)."""
        self._components.clear()

    async def initialize(self) -> None:
        """
        Initialize the factory and its components.

        This method initializes any components that require async setup,
        such as model servers or external connections.
        """
        logger.info("Initializing AnalysisModuleFactory")

        # Initialize model server if it requires async setup
        model_server = self.create_model_server()
        if hasattr(model_server, "initialize"):
            await model_server.initialize()

        logger.info("AnalysisModuleFactory initialization complete")

    async def cleanup(self) -> None:
        """
        Cleanup factory resources.

        This method performs cleanup of any resources that require
        explicit cleanup, such as closing connections or releasing resources.
        """
        logger.info("Cleaning up AnalysisModuleFactory")

        # Cleanup components that require explicit cleanup
        for component_name, component in self._components.items():
            if hasattr(component, "cleanup"):
                try:
                    if hasattr(component, "__aenter__"):  # async context manager
                        await component.cleanup()
                    else:
                        component.cleanup()
                    logger.debug("Cleaned up component: %s", component_name)
                except Exception as e:
                    logger.warning(
                        "Error cleaning up component %s: %s", component_name, e
                    )

        logger.info("AnalysisModuleFactory cleanup complete")

    def get_analysis_service(self) -> AnalysisService:
        """Get analysis service (alias for create_analysis_service)."""
        return self.create_analysis_service()

    def get_batch_analysis_service(self) -> BatchAnalysisService:
        """Get batch analysis service (alias for create_batch_analysis_service)."""
        return self.create_batch_analysis_service()

    @classmethod
    def create_from_config(cls, config: AnalysisConfig) -> "AnalysisModuleFactory":
        """
        Create factory from configuration.

        Args:
            config: Analysis module configuration

        Returns:
            Configured factory instance
        """
        return cls(config)

    @classmethod
    def create_from_env(cls) -> "AnalysisModuleFactory":
        """
        Create factory from environment variables.

        Returns:
            Factory instance configured from environment
        """
        from .settings import AnalysisConfig

        config = AnalysisConfig.from_env()
        return cls(config)
