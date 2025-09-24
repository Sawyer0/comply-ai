"""
Analysis Service Factory for component creation and dependency injection.

This factory manages the creation and wiring of analysis engines with
proper dependency injection and configuration management.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from ..domain import (
    AnalysisConfiguration,
    IAnalysisEngine,
    IComplianceIntelligenceEngine,
    IPatternRecognitionEngine,
    IRiskScoringEngine,
    ITemplateOrchestrator,
)
from ..engines import (
    ComplianceIntelligence,
    PatternRecognitionEngine,
    RiskScoringEngine,
    TemplateOrchestrator,
)
from ..lifecycle import ServiceLifecycleManager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class IDependencyProvider(ABC):
    """Interface for dependency providers."""
    
    @abstractmethod
    def get_dependency(self, dependency_type: Type[T], name: Optional[str] = None) -> T:
        """Get a dependency by type and optional name."""
        pass
    
    @abstractmethod
    def register_dependency(self, dependency_type: Type[T], instance: T, name: Optional[str] = None) -> None:
        """Register a dependency instance."""
        pass


class IServiceFactory(ABC):
    """Interface for service factories."""
    
    @abstractmethod
    def create_service(self, service_type: Type[T], config: Optional[Dict[str, Any]] = None) -> T:
        """Create a service instance with dependency injection."""
        pass
    
    @abstractmethod
    def register_service_type(self, service_type: Type[T], factory_func: Callable[..., T]) -> None:
        """Register a service type with its factory function."""
        pass


class DependencyProvider(IDependencyProvider):
    """
    Dependency provider for managing service dependencies.
    
    Supports singleton and transient dependency lifetimes with
    proper cleanup and lifecycle management.
    """
    
    def __init__(self):
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
    
    def get_dependency(self, dependency_type: Type[T], name: Optional[str] = None) -> T:
        """Get a dependency by type and optional name."""
        key = self._get_dependency_key(dependency_type, name)
        
        # Return singleton if exists
        if key in self._singletons:
            return self._singletons[key]
        
        # Create using factory if available
        if key in self._factories:
            factory = self._factories[key]
            instance = factory()
            
            # Store as singleton
            self._singletons[key] = instance
            return instance
        
        raise ValueError(f"No dependency registered for {dependency_type.__name__} (name: {name})")
    
    def register_dependency(self, dependency_type: Type[T], instance: T, name: Optional[str] = None) -> None:
        """Register a dependency instance."""
        key = self._get_dependency_key(dependency_type, name)
        self._singletons[key] = instance
        logger.debug(f"Registered dependency: {key}")
    
    def register_factory(self, dependency_type: Type[T], factory: Callable[[], T], name: Optional[str] = None) -> None:
        """Register a factory function for a dependency type."""
        key = self._get_dependency_key(dependency_type, name)
        self._factories[key] = factory
        logger.debug(f"Registered factory: {key}")
    
    def register_dependency_graph(self, service_type: Type, dependencies: List[str]) -> None:
        """Register dependency graph for a service type."""
        key = service_type.__name__
        self._dependency_graph[key] = dependencies
    
    def get_dependency_order(self, service_type: Type) -> List[str]:
        """Get dependency initialization order for a service type."""
        return self._dependency_graph.get(service_type.__name__, [])
    
    def clear_dependencies(self) -> None:
        """Clear all registered dependencies."""
        # Cleanup singletons if they have cleanup methods
        for instance in self._singletons.values():
            if hasattr(instance, 'cleanup'):
                try:
                    instance.cleanup()
                except Exception as e:
                    logger.warning(f"Error during dependency cleanup: {e}")
        
        self._singletons.clear()
        self._factories.clear()
        self._dependency_graph.clear()
    
    def _get_dependency_key(self, dependency_type: Type, name: Optional[str]) -> str:
        """Generate a unique key for a dependency."""
        base_key = dependency_type.__name__
        return f"{base_key}:{name}" if name else base_key


class AnalysisServiceFactory(IServiceFactory):
    """
    Enhanced factory for creating and wiring analysis service components.
    
    Manages dependency injection, configuration, and lifecycle of
    analysis engines with proper error handling and validation.
    Supports hot-reloading, health monitoring, and graceful shutdown.
    """
    
    def __init__(self, 
                 base_config: Optional[Dict[str, Any]] = None,
                 dependency_provider: Optional[IDependencyProvider] = None,
                 lifecycle_manager: Optional[ServiceLifecycleManager] = None):
        """
        Initialize the analysis service factory.
        
        Args:
            base_config: Base configuration for all services
            dependency_provider: Dependency injection provider
            lifecycle_manager: Service lifecycle manager
        """
        self.base_config = base_config or {}
        self.dependency_provider = dependency_provider or DependencyProvider()
        self.lifecycle_manager = lifecycle_manager or ServiceLifecycleManager()
        
        # Service registry and instances
        self.service_registry: Dict[str, Type[IAnalysisEngine]] = {}
        self.service_factories: Dict[str, Callable] = {}
        self.created_services: Dict[str, IAnalysisEngine] = {}
        
        # Configuration and wiring
        self.config_watchers: Dict[str, Callable] = {}
        self.initialization_hooks: Dict[str, List[Callable]] = {}
        self.shutdown_hooks: Dict[str, List[Callable]] = {}
        
        # State management
        self._initialized = False
        self._shutting_down = False
        
        # Register default engines and dependencies
        self._register_default_services()
        self._register_default_dependencies()
    
    def _register_default_services(self) -> None:
        """Register default analysis services."""
        self.register_service_type("pattern_recognition", PatternRecognitionEngine, self._create_pattern_engine)
        self.register_service_type("risk_scoring", RiskScoringEngine, self._create_risk_engine)
        self.register_service_type("compliance_intelligence", ComplianceIntelligence, self._create_compliance_engine)
        self.register_service_type("template_orchestrator", TemplateOrchestrator, self._create_orchestrator)
    
    def _register_default_dependencies(self) -> None:
        """Register default dependencies and their relationships."""
        # Register dependency graphs for proper initialization order
        self.dependency_provider.register_dependency_graph(
            PatternRecognitionEngine, 
            ["data_repository", "statistical_analyzer", "temporal_analyzer"]
        )
        self.dependency_provider.register_dependency_graph(
            RiskScoringEngine,
            ["business_context", "regulatory_weights", "scoring_algorithms"]
        )
        self.dependency_provider.register_dependency_graph(
            ComplianceIntelligence,
            ["compliance_rules", "framework_mappings", "gap_analyzer"]
        )
        self.dependency_provider.register_dependency_graph(
            TemplateOrchestrator,
            ["pattern_recognition", "risk_scoring", "compliance_intelligence"]
        )
    
    def register_service_type(self, name: str, service_class: Type[IAnalysisEngine], 
                            factory_func: Optional[Callable] = None) -> None:
        """
        Register a service type with optional custom factory function.
        
        Args:
            name: Service name
            service_class: Service class to register
            factory_func: Optional custom factory function
        """
        self.service_registry[name] = service_class
        if factory_func:
            self.service_factories[name] = factory_func
        logger.info(f"Registered service type: {name}")
    
    def create_service(self, service_type: Type[T], config: Optional[Dict[str, Any]] = None) -> T:
        """
        Create a service instance with dependency injection.
        
        Args:
            service_type: Type of service to create
            config: Optional configuration overrides
            
        Returns:
            Configured service instance with dependencies injected
        """
        service_name = service_type.__name__.lower().replace('engine', '')
        
        if service_name in self.created_services:
            return self.created_services[service_name]
        
        try:
            # Get or create configuration
            service_config = self._create_service_config(service_name, config)
            
            # Use custom factory if available
            if service_name in self.service_factories:
                factory_func = self.service_factories[service_name]
                service = factory_func(service_config)
            else:
                # Default creation with dependency injection
                service = self._create_service_with_injection(service_type, service_config)
            
            # Register with lifecycle manager
            self.lifecycle_manager.register_service(service_name, service)
            
            # Store created service
            self.created_services[service_name] = service
            
            logger.info(f"Created service: {service_name}")
            return service
            
        except Exception as e:
            logger.error(f"Failed to create service {service_name}: {e}")
            raise RuntimeError(f"Service creation failed: {e}") from e
    
    def register_dependency(self, dependency_type: Type[T], instance: T, name: Optional[str] = None) -> None:
        """
        Register a dependency instance.
        
        Args:
            dependency_type: Type of the dependency
            instance: Dependency instance
            name: Optional name for named dependencies
        """
        self.dependency_provider.register_dependency(dependency_type, instance, name)
    
    def register_dependency_factory(self, dependency_type: Type[T], factory: Callable[[], T], 
                                  name: Optional[str] = None) -> None:
        """
        Register a factory function for a dependency type.
        
        Args:
            dependency_type: Type of the dependency
            factory: Factory function
            name: Optional name for named dependencies
        """
        self.dependency_provider.register_factory(dependency_type, factory, name)
    
    def add_initialization_hook(self, service_name: str, hook: Callable) -> None:
        """
        Add an initialization hook for a service.
        
        Args:
            service_name: Name of the service
            hook: Hook function to call after initialization
        """
        if service_name not in self.initialization_hooks:
            self.initialization_hooks[service_name] = []
        self.initialization_hooks[service_name].append(hook)
    
    def add_shutdown_hook(self, service_name: str, hook: Callable) -> None:
        """
        Add a shutdown hook for a service.
        
        Args:
            service_name: Name of the service
            hook: Hook function to call before shutdown
        """
        if service_name not in self.shutdown_hooks:
            self.shutdown_hooks[service_name] = []
        self.shutdown_hooks[service_name].append(hook)
    
    def add_config_watcher(self, service_name: str, watcher: Callable) -> None:
        """
        Add a configuration watcher for hot-reloading.
        
        Args:
            service_name: Name of the service
            watcher: Function to call when configuration changes
        """
        self.config_watchers[service_name] = watcher
    
    async def initialize_all_services(self) -> Dict[str, bool]:
        """
        Initialize all registered services with proper dependency order.
        
        Returns:
            Dictionary mapping service names to initialization success status
        """
        if self._initialized:
            logger.warning("Services already initialized")
            return {}
        
        logger.info("Initializing all services...")
        
        # Start lifecycle manager
        results = await self.lifecycle_manager.start_all_services()
        
        # Run initialization hooks
        for service_name, hooks in self.initialization_hooks.items():
            if service_name in results and results[service_name]:
                for hook in hooks:
                    try:
                        if asyncio.iscoroutinefunction(hook):
                            await hook()
                        else:
                            hook()
                    except Exception as e:
                        logger.error(f"Initialization hook failed for {service_name}: {e}")
        
        self._initialized = True
        logger.info(f"Initialized {sum(results.values())}/{len(results)} services")
        
        return results
    
    async def shutdown_all_services(self) -> Dict[str, bool]:
        """
        Shutdown all services gracefully.
        
        Returns:
            Dictionary mapping service names to shutdown success status
        """
        if self._shutting_down:
            logger.warning("Shutdown already in progress")
            return {}
        
        self._shutting_down = True
        logger.info("Shutting down all services...")
        
        # Run shutdown hooks
        for service_name, hooks in self.shutdown_hooks.items():
            for hook in hooks:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook()
                    else:
                        hook()
                except Exception as e:
                    logger.error(f"Shutdown hook failed for {service_name}: {e}")
        
        # Stop lifecycle manager
        results = await self.lifecycle_manager.stop_all_services()
        
        # Clear dependencies
        self.dependency_provider.clear_dependencies()
        self.created_services.clear()
        
        self._initialized = False
        self._shutting_down = False
        
        logger.info(f"Shutdown {sum(results.values())}/{len(results)} services")
        return results
    
    async def reload_service_config(self, service_name: str, new_config: Dict[str, Any]) -> bool:
        """
        Reload configuration for a specific service (hot-reload).
        
        Args:
            service_name: Name of the service
            new_config: New configuration
            
        Returns:
            True if reload successful, False otherwise
        """
        if service_name not in self.created_services:
            logger.error(f"Service {service_name} not found for config reload")
            return False
        
        try:
            # Call config watcher if registered
            if service_name in self.config_watchers:
                watcher = self.config_watchers[service_name]
                if asyncio.iscoroutinefunction(watcher):
                    await watcher(new_config)
                else:
                    watcher(new_config)
            
            # Update service configuration if it supports it
            service = self.created_services[service_name]
            if hasattr(service, 'update_config'):
                if asyncio.iscoroutinefunction(service.update_config):
                    await service.update_config(new_config)
                else:
                    service.update_config(new_config)
            
            logger.info(f"Reloaded configuration for service: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload config for {service_name}: {e}")
            return False
    
    def get_service_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status of all services.
        
        Returns:
            Dictionary mapping service names to health information
        """
        return self.lifecycle_manager.get_all_service_status()
    
    def create_engine(self, engine_name: str, 
                     config_override: Optional[Dict[str, Any]] = None) -> IAnalysisEngine:
        """
        Create an analysis engine with dependency injection.
        
        Args:
            engine_name: Name of the engine to create
            config_override: Optional configuration overrides
            
        Returns:
            Configured analysis engine instance
            
        Raises:
            ValueError: If engine is not registered
            RuntimeError: If engine creation fails
        """
        if engine_name not in self.engine_registry:
            raise ValueError(f"Engine '{engine_name}' is not registered")
        
        try:
            # Create configuration
            config = self._create_engine_config(engine_name, config_override)
            
            # Get engine class
            engine_class = self.engine_registry[engine_name]
            
            # Create engine instance
            if engine_name == "template_orchestrator":
                # Special handling for orchestrator - needs other engines
                other_engines = self._create_dependent_engines(config_override)
                engine = engine_class(config, other_engines)
            else:
                engine = engine_class(config)
            
            # Inject dependencies
            self._inject_dependencies(engine, engine_name)
            
            # Store created engine
            self.created_engines[engine_name] = engine
            
            logger.info(f"Created analysis engine: {engine_name}")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create engine {engine_name}: {e}")
            raise RuntimeError(f"Engine creation failed: {e}") from e
    
    def create_orchestrator(self, config_override: Optional[Dict[str, Any]] = None) -> ITemplateOrchestrator:
        """
        Create a template orchestrator with all engines.
        
        Args:
            config_override: Optional configuration overrides
            
        Returns:
            Configured template orchestrator
        """
        orchestrator = self.create_engine("template_orchestrator", config_override)
        
        if not isinstance(orchestrator, ITemplateOrchestrator):
            raise RuntimeError("Created orchestrator does not implement ITemplateOrchestrator")
        
        return orchestrator
    
    def create_pattern_engine(self, config_override: Optional[Dict[str, Any]] = None) -> IPatternRecognitionEngine:
        """
        Create a pattern recognition engine.
        
        Args:
            config_override: Optional configuration overrides
            
        Returns:
            Configured pattern recognition engine
        """
        engine = self.create_engine("pattern_recognition", config_override)
        
        if not isinstance(engine, IPatternRecognitionEngine):
            raise RuntimeError("Created engine does not implement IPatternRecognitionEngine")
        
        return engine
    
    def create_risk_engine(self, config_override: Optional[Dict[str, Any]] = None) -> IRiskScoringEngine:
        """
        Create a risk scoring engine.
        
        Args:
            config_override: Optional configuration overrides
            
        Returns:
            Configured risk scoring engine
        """
        engine = self.create_engine("risk_scoring", config_override)
        
        if not isinstance(engine, IRiskScoringEngine):
            raise RuntimeError("Created engine does not implement IRiskScoringEngine")
        
        return engine
    
    def create_compliance_engine(self, config_override: Optional[Dict[str, Any]] = None) -> IComplianceIntelligenceEngine:
        """
        Create a compliance intelligence engine.
        
        Args:
            config_override: Optional configuration overrides
            
        Returns:
            Configured compliance intelligence engine
        """
        engine = self.create_engine("compliance_intelligence", config_override)
        
        if not isinstance(engine, IComplianceIntelligenceEngine):
            raise RuntimeError("Created engine does not implement IComplianceIntelligenceEngine")
        
        return engine
    
    def get_engine(self, engine_name: str) -> Optional[IAnalysisEngine]:
        """
        Get a previously created engine.
        
        Args:
            engine_name: Name of the engine
            
        Returns:
            Engine instance if exists, None otherwise
        """
        return self.created_engines.get(engine_name)
    
    def shutdown_all_engines(self) -> None:
        """Shutdown all created engines."""
        for engine_name, engine in self.created_engines.items():
            try:
                if hasattr(engine, 'shutdown'):
                    engine.shutdown()
                logger.info(f"Shutdown engine: {engine_name}")
            except Exception as e:
                logger.error(f"Failed to shutdown engine {engine_name}: {e}")
        
        self.created_engines.clear()
    
    def _create_engine_config(self, engine_name: str, 
                            config_override: Optional[Dict[str, Any]]) -> AnalysisConfiguration:
        """Create configuration for an engine."""
        # Start with base config
        config_dict = self.base_config.copy()
        
        # Add engine-specific config
        engine_config = config_dict.get('engines', {}).get(engine_name, {})
        config_dict.update(engine_config)
        
        # Apply overrides
        if config_override:
            config_dict.update(config_override)
        
        # Create AnalysisConfiguration
        return AnalysisConfiguration(
            engine_name=engine_name,
            enabled=config_dict.get('enabled', True),
            confidence_threshold=config_dict.get('confidence_threshold', 0.7),
            parameters=config_dict.get('parameters', {}),
            weights=config_dict.get('weights', {}),
            fallback_enabled=config_dict.get('fallback_enabled', True)
        )
    
    def _create_dependent_engines(self, config_override: Optional[Dict[str, Any]]) -> Dict[str, IAnalysisEngine]:
        """Create engines that the orchestrator depends on."""
        engines = {}
        
        # Create core engines for orchestrator
        engine_names = ["pattern_recognition", "risk_scoring", "compliance_intelligence"]
        
        for engine_name in engine_names:
            try:
                engine = self.create_engine(engine_name, config_override)
                engines[engine_name] = engine
            except Exception as e:
                logger.warning(f"Failed to create dependent engine {engine_name}: {e}")
                engines[engine_name] = None
        
        return engines
    
    def _create_service_with_injection(self, service_type: Type[T], config: AnalysisConfiguration) -> T:
        """Create a service with dependency injection."""
        # Get dependency order for this service type
        dependency_names = self.dependency_provider.get_dependency_order(service_type)
        
        # Resolve dependencies
        dependencies = {}
        for dep_name in dependency_names:
            try:
                # Try to get dependency by name first, then by type
                dependency = None
                try:
                    dependency = self.dependency_provider.get_dependency(object, dep_name)
                except ValueError:
                    # If named dependency not found, try to infer type and get it
                    pass
                
                if dependency:
                    dependencies[dep_name] = dependency
            except Exception as e:
                logger.warning(f"Failed to resolve dependency {dep_name}: {e}")
        
        # Create service instance
        if dependencies:
            # Try to pass dependencies as constructor arguments
            try:
                service = service_type(config, **dependencies)
            except TypeError:
                # Fallback to basic constructor
                service = service_type(config)
                # Inject dependencies as attributes
                for dep_name, dependency in dependencies.items():
                    setattr(service, dep_name, dependency)
        else:
            service = service_type(config)
        
        return service
    
    def _create_pattern_engine(self, config: AnalysisConfiguration) -> PatternRecognitionEngine:
        """Factory method for pattern recognition engine."""
        return PatternRecognitionEngine(config)
    
    def _create_risk_engine(self, config: AnalysisConfiguration) -> RiskScoringEngine:
        """Factory method for risk scoring engine."""
        return RiskScoringEngine(config)
    
    def _create_compliance_engine(self, config: AnalysisConfiguration) -> ComplianceIntelligence:
        """Factory method for compliance intelligence engine."""
        return ComplianceIntelligence(config)
    
    def _create_orchestrator(self, config: AnalysisConfiguration) -> TemplateOrchestrator:
        """Factory method for template orchestrator."""
        # Create dependent engines
        pattern_engine = self.create_service(PatternRecognitionEngine, config.parameters)
        risk_engine = self.create_service(RiskScoringEngine, config.parameters)
        compliance_engine = self.create_service(ComplianceIntelligence, config.parameters)
        
        engines = {
            "pattern_recognition": pattern_engine,
            "risk_scoring": risk_engine,
            "compliance_intelligence": compliance_engine
        }
        
        return TemplateOrchestrator(config, engines)


class EngineBuilder:
    """
    Builder pattern for creating analysis engines with fluent interface.
    """
    
    def __init__(self, factory: AnalysisServiceFactory):
        """
        Initialize the engine builder.
        
        Args:
            factory: Analysis service factory to use
        """
        self.factory = factory
        self.engine_name: Optional[str] = None
        self.config_overrides: Dict[str, Any] = {}
        self.dependencies: Dict[str, Any] = {}
    
    def engine(self, name: str) -> 'EngineBuilder':
        """
        Set the engine name to build.
        
        Args:
            name: Name of the engine
            
        Returns:
            Self for method chaining
        """
        self.engine_name = name
        return self
    
    def with_config(self, **config) -> 'EngineBuilder':
        """
        Add configuration overrides.
        
        Args:
            **config: Configuration key-value pairs
            
        Returns:
            Self for method chaining
        """
        self.config_overrides.update(config)
        return self
    
    def with_dependency(self, name: str, dependency: Any) -> 'EngineBuilder':
        """
        Add a dependency.
        
        Args:
            name: Dependency name
            dependency: Dependency object
            
        Returns:
            Self for method chaining
        """
        self.dependencies[name] = dependency
        return self
    
    def build(self) -> IAnalysisEngine:
        """
        Build the engine with configured settings.
        
        Returns:
            Configured analysis engine
            
        Raises:
            ValueError: If engine name is not set
        """
        if not self.engine_name:
            raise ValueError("Engine name must be set")
        
        # Register dependencies
        for dep_name, dependency in self.dependencies.items():
            self.factory.register_dependency(self.engine_name, dep_name, dependency)
        
        # Create engine
        return self.factory.create_engine(self.engine_name, self.config_overrides)


def create_default_factory(config: Optional[Dict[str, Any]] = None) -> AnalysisServiceFactory:
    """
    Create a factory with default configuration.
    
    Args:
        config: Optional base configuration
        
    Returns:
        Configured analysis service factory
    """
    default_config = {
        'engines': {
            'pattern_recognition': {
                'enabled': True,
                'confidence_threshold': 0.7,
                'parameters': {
                    'pattern_recognition': {
                        'temporal_window_hours': 24,
                        'correlation_threshold': 0.7,
                        'anomaly_sensitivity': 2.0
                    },
                    'statistical': {
                        'significance_threshold': 0.05,
                        'min_sample_size': 30
                    }
                }
            },
            'risk_scoring': {
                'enabled': True,
                'confidence_threshold': 0.7,
                'parameters': {
                    'risk_scoring': {
                        'temporal_decay_days': 30
                    },
                    'business': {
                        'impact_weights': {},
                        'business_hours': {
                            'start_hour': 9,
                            'end_hour': 17,
                            'days': [0, 1, 2, 3, 4]
                        }
                    }
                }
            },
            'compliance_intelligence': {
                'enabled': True,
                'confidence_threshold': 0.7,
                'parameters': {
                    'compliance_intelligence': {
                        'enabled_frameworks': ['soc2', 'iso27001', 'hipaa', 'gdpr', 'pci_dss'],
                        'gap_thresholds': {
                            'critical': 0.9,
                            'high': 0.7,
                            'medium': 0.5,
                            'low': 0.3
                        }
                    }
                }
            },
            'template_orchestrator': {
                'enabled': True,
                'confidence_threshold': 0.6,
                'parameters': {
                    'orchestration': {
                        'ai_threshold': 0.85
                    }
                }
            }
        }
    }
    
    # Merge with provided config
    if config:
        # Simple merge - in production, use deep merge
        default_config.update(config)
    
    return AnalysisServiceFactory(default_config)