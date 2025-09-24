"""
Example demonstrating enhanced dependency injection and factory patterns.

This example shows how to use the AnalysisServiceFactory with
dynamic configuration management and health monitoring.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from ..config import (
    DynamicConfigurationManager,
    PydanticConfigValidator,
    create_default_config_manager
)
from ..factories import (
    AnalysisServiceFactory,
    DependencyProvider,
    create_default_factory
)
from ..lifecycle import ServiceLifecycleManager
from ..monitoring import (
    ServiceHealthMonitor,
    HealthAlert,
    create_default_health_monitor
)

logger = logging.getLogger(__name__)


class ExampleDependencyInjectionDemo:
    """
    Demonstration of enhanced dependency injection and factory patterns.
    
    Shows integration of:
    - Dynamic configuration management with hot-reloading
    - Service factory with dependency injection
    - Service lifecycle management
    - Health monitoring and alerting
    """
    
    def __init__(self):
        self.config_manager: DynamicConfigurationManager = None
        self.service_factory: AnalysisServiceFactory = None
        self.health_monitor: ServiceHealthMonitor = None
        self.lifecycle_manager: ServiceLifecycleManager = None
    
    async def setup_demo(self) -> None:
        """Set up the demonstration environment."""
        logger.info("Setting up dependency injection demo...")
        
        # 1. Initialize configuration manager with hot-reloading
        self.config_manager = create_default_config_manager("config/analysis.yaml")
        
        # Register configuration sources
        await self._setup_configuration()
        
        # 2. Initialize dependency provider and service factory
        dependency_provider = DependencyProvider()
        self.lifecycle_manager = ServiceLifecycleManager(health_check_interval=10)
        
        self.service_factory = AnalysisServiceFactory(
            base_config=self.config_manager.get_config("base"),
            dependency_provider=dependency_provider,
            lifecycle_manager=self.lifecycle_manager
        )
        
        # 3. Register custom dependencies
        await self._register_dependencies()
        
        # 4. Set up health monitoring
        self.health_monitor = create_default_health_monitor(check_interval=15)
        self.health_monitor.add_alert_handler(self._handle_health_alert)
        
        # 5. Register configuration change handlers
        self._setup_config_watchers()
        
        logger.info("Demo setup completed")
    
    async def run_demo(self) -> None:
        """Run the demonstration."""
        logger.info("Starting dependency injection demo...")
        
        try:
            # Start configuration file watching
            self.config_manager.start_file_watching()
            
            # Initialize all services
            init_results = await self.service_factory.initialize_all_services()
            logger.info(f"Service initialization results: {init_results}")
            
            # Register services with health monitor
            for service_name in init_results.keys():
                if init_results[service_name]:
                    service = self.service_factory.created_services.get(service_name)
                    if service:
                        self.health_monitor.register_service(service_name, service)
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Demonstrate service creation and usage
            await self._demonstrate_service_creation()
            
            # Demonstrate configuration hot-reloading
            await self._demonstrate_config_hot_reload()
            
            # Demonstrate health monitoring
            await self._demonstrate_health_monitoring()
            
            # Run for a while to show monitoring in action
            logger.info("Demo running... (monitoring for 60 seconds)")
            await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            await self._cleanup_demo()
    
    async def _setup_configuration(self) -> None:
        """Set up configuration sources."""
        # Create example configuration files
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Base analysis configuration
        base_config = {
            "engines": {
                "pattern_recognition": {
                    "enabled": True,
                    "confidence_threshold": 0.7,
                    "parameters": {
                        "temporal_window_hours": 24,
                        "correlation_threshold": 0.7
                    }
                },
                "risk_scoring": {
                    "enabled": True,
                    "confidence_threshold": 0.8,
                    "parameters": {
                        "temporal_decay_days": 30
                    }
                }
            },
            "health_monitoring": {
                "check_interval": 15,
                "alert_cooldown": 300,
                "thresholds": {
                    "response_time_ms": 1000,
                    "error_rate_percent": 5.0
                }
            }
        }
        
        # Register configuration with validation
        self.config_manager.register_config_source(
            "analysis",
            config_dir / "analysis.yaml",
            watch=True
        )
        
        # Update with base config
        self.config_manager.update_config("analysis", base_config)
    
    async def _register_dependencies(self) -> None:
        """Register custom dependencies."""
        # Example: Register a custom data repository
        class MockDataRepository:
            def __init__(self):
                self.data = {}
            
            async def get_data(self, key: str):
                return self.data.get(key)
            
            async def store_data(self, key: str, value: Any):
                self.data[key] = value
        
        # Register as singleton dependency
        data_repo = MockDataRepository()
        self.service_factory.register_dependency(MockDataRepository, data_repo, "data_repository")
        
        # Example: Register a factory for statistical analyzer
        class MockStatisticalAnalyzer:
            def __init__(self):
                self.calculations = 0
            
            async def analyze(self, data):
                self.calculations += 1
                return {"result": "analyzed", "calculations": self.calculations}
        
        self.service_factory.register_dependency_factory(
            MockStatisticalAnalyzer,
            lambda: MockStatisticalAnalyzer(),
            "statistical_analyzer"
        )
        
        logger.info("Registered custom dependencies")
    
    def _setup_config_watchers(self) -> None:
        """Set up configuration change watchers."""
        async def on_analysis_config_change(event):
            logger.info(f"Analysis configuration changed: {event.config_path}")
            
            # Reload service configurations
            for service_name in self.service_factory.created_services.keys():
                success = await self.service_factory.reload_service_config(
                    service_name, 
                    event.new_config
                )
                logger.info(f"Config reload for {service_name}: {'success' if success else 'failed'}")
        
        self.config_manager.add_change_listener("analysis", on_analysis_config_change)
    
    async def _demonstrate_service_creation(self) -> None:
        """Demonstrate service creation with dependency injection."""
        logger.info("Demonstrating service creation...")
        
        # Create services using the factory
        try:
            from ..engines import PatternRecognitionEngine, RiskScoringEngine
            
            # Create pattern recognition engine
            pattern_engine = self.service_factory.create_service(
                PatternRecognitionEngine,
                {"confidence_threshold": 0.75}
            )
            logger.info(f"Created pattern engine: {pattern_engine.__class__.__name__}")
            
            # Create risk scoring engine
            risk_engine = self.service_factory.create_service(
                RiskScoringEngine,
                {"confidence_threshold": 0.8}
            )
            logger.info(f"Created risk engine: {risk_engine.__class__.__name__}")
            
            # Demonstrate service interaction
            if hasattr(pattern_engine, 'analyze') and hasattr(risk_engine, 'analyze'):
                # Mock analysis request
                mock_request = {"data": "sample analysis data"}
                
                # This would normally be actual analysis calls
                logger.info("Services created and ready for analysis")
            
        except Exception as e:
            logger.error(f"Service creation demonstration failed: {e}")
    
    async def _demonstrate_config_hot_reload(self) -> None:
        """Demonstrate configuration hot-reloading."""
        logger.info("Demonstrating configuration hot-reload...")
        
        # Update configuration programmatically
        new_config = self.config_manager.get_config("analysis")
        if new_config:
            # Modify some parameters
            new_config["engines"]["pattern_recognition"]["confidence_threshold"] = 0.9
            new_config["health_monitoring"]["check_interval"] = 20
            
            # Update configuration (this will trigger watchers)
            success = self.config_manager.update_config("analysis", new_config)
            logger.info(f"Configuration update: {'success' if success else 'failed'}")
            
            # Wait a moment for changes to propagate
            await asyncio.sleep(2)
    
    async def _demonstrate_health_monitoring(self) -> None:
        """Demonstrate health monitoring capabilities."""
        logger.info("Demonstrating health monitoring...")
        
        # Perform manual health checks
        health_results = await self.health_monitor.check_all_services()
        
        for service_name, result in health_results.items():
            logger.info(f"Health check for {service_name}: {result.overall_status.value}")
            for metric in result.metrics:
                logger.info(f"  {metric.name}: {metric.value}{metric.unit} (threshold: {metric.threshold}{metric.unit})")
        
        # Get overall health status
        all_status = self.health_monitor.get_all_service_status()
        logger.info(f"Overall service status: {all_status}")
        
        # Check for unhealthy services
        unhealthy = self.health_monitor.get_unhealthy_services()
        if unhealthy:
            logger.warning(f"Unhealthy services detected: {unhealthy}")
        else:
            logger.info("All services are healthy")
    
    def _handle_health_alert(self, alert: HealthAlert) -> None:
        """Handle health alerts."""
        logger.warning(
            f"HEALTH ALERT [{alert.severity.value.upper()}] "
            f"{alert.service_name}.{alert.metric_name}: {alert.message}"
        )
        
        # In a real implementation, you might:
        # - Send notifications (email, Slack, etc.)
        # - Trigger automated remediation
        # - Update monitoring dashboards
        # - Log to external systems
    
    async def _cleanup_demo(self) -> None:
        """Clean up demo resources."""
        logger.info("Cleaning up demo...")
        
        try:
            # Stop health monitoring
            if self.health_monitor:
                await self.health_monitor.stop_monitoring()
            
            # Shutdown all services
            if self.service_factory:
                await self.service_factory.shutdown_all_services()
            
            # Stop configuration file watching
            if self.config_manager:
                self.config_manager.stop_file_watching()
            
            logger.info("Demo cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def run_dependency_injection_demo():
    """Run the complete dependency injection demonstration."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demo = ExampleDependencyInjectionDemo()
    
    try:
        await demo.setup_demo()
        await demo.run_demo()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(run_dependency_injection_demo())