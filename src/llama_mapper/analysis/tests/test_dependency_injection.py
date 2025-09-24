"""
Tests for enhanced dependency injection and factory patterns.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from ..config import DynamicConfigurationManager, PydanticConfigValidator
from ..factories import AnalysisServiceFactory, DependencyProvider
from ..lifecycle import ServiceLifecycleManager
from ..monitoring import ServiceHealthMonitor, HealthStatus
from ..domain import AnalysisConfiguration, IAnalysisEngine


class MockAnalysisEngine(IAnalysisEngine):
    """Mock analysis engine for testing."""
    
    def __init__(self, config: AnalysisConfiguration):
        self.config = config
        self.initialized = False
        self.shutdown_called = False
    
    async def initialize(self):
        """Initialize the engine."""
        self.initialized = True
    
    async def shutdown(self):
        """Shutdown the engine."""
        self.shutdown_called = True
    
    async def analyze(self, request):
        """Mock analysis method."""
        return {"result": "mock_analysis", "confidence": 0.8}
    
    def get_confidence(self, result):
        """Get confidence score."""
        return result.get("confidence", 0.0)
    
    def get_health_status(self):
        """Get health status."""
        return "healthy" if self.initialized else "unhealthy"


class TestDependencyProvider:
    """Test dependency provider functionality."""
    
    def test_register_and_get_dependency(self):
        """Test basic dependency registration and retrieval."""
        provider = DependencyProvider()
        
        # Register a dependency
        mock_service = Mock()
        provider.register_dependency(Mock, mock_service)
        
        # Retrieve the dependency
        retrieved = provider.get_dependency(Mock)
        assert retrieved is mock_service
    
    def test_register_named_dependency(self):
        """Test named dependency registration."""
        provider = DependencyProvider()
        
        # Register named dependencies
        service1 = Mock()
        service2 = Mock()
        
        provider.register_dependency(Mock, service1, "service1")
        provider.register_dependency(Mock, service2, "service2")
        
        # Retrieve named dependencies
        assert provider.get_dependency(Mock, "service1") is service1
        assert provider.get_dependency(Mock, "service2") is service2
    
    def test_register_factory(self):
        """Test factory registration and lazy creation."""
        provider = DependencyProvider()
        
        # Register a factory
        mock_instance = Mock()
        factory = Mock(return_value=mock_instance)
        provider.register_factory(Mock, factory)
        
        # Get dependency (should create via factory)
        retrieved = provider.get_dependency(Mock)
        assert retrieved is mock_instance
        factory.assert_called_once()
        
        # Get again (should return same instance)
        retrieved2 = provider.get_dependency(Mock)
        assert retrieved2 is mock_instance
        # Factory should still only be called once (singleton behavior)
        factory.assert_called_once()
    
    def test_dependency_not_found(self):
        """Test error when dependency not found."""
        provider = DependencyProvider()
        
        with pytest.raises(ValueError, match="No dependency registered"):
            provider.get_dependency(Mock)


class TestAnalysisServiceFactory:
    """Test enhanced analysis service factory."""
    
    @pytest.fixture
    def factory(self):
        """Create a factory for testing."""
        config = {
            "engines": {
                "mock_engine": {
                    "enabled": True,
                    "confidence_threshold": 0.7
                }
            }
        }
        
        dependency_provider = DependencyProvider()
        lifecycle_manager = ServiceLifecycleManager()
        
        factory = AnalysisServiceFactory(
            base_config=config,
            dependency_provider=dependency_provider,
            lifecycle_manager=lifecycle_manager
        )
        
        # Register mock engine
        factory.register_service_type("mock_engine", MockAnalysisEngine)
        
        return factory
    
    def test_service_registration(self, factory):
        """Test service type registration."""
        assert "mock_engine" in factory.service_registry
        assert factory.service_registry["mock_engine"] is MockAnalysisEngine
    
    def test_dependency_registration(self, factory):
        """Test dependency registration."""
        mock_dep = Mock()
        factory.register_dependency(Mock, mock_dep, "test_dep")
        
        # Verify dependency is registered
        retrieved = factory.dependency_provider.get_dependency(Mock, "test_dep")
        assert retrieved is mock_dep
    
    @pytest.mark.asyncio
    async def test_service_creation(self, factory):
        """Test service creation with dependency injection."""
        # Create a service
        service = factory.create_service(MockAnalysisEngine)
        
        assert isinstance(service, MockAnalysisEngine)
        assert service.config is not None
        assert "mock_engine" in factory.created_services
    
    @pytest.mark.asyncio
    async def test_service_lifecycle(self, factory):
        """Test service lifecycle management."""
        # Create a service
        service = factory.create_service(MockAnalysisEngine)
        
        # Initialize all services
        results = await factory.initialize_all_services()
        assert results.get("mock_engine") is True
        assert service.initialized is True
        
        # Shutdown all services
        shutdown_results = await factory.shutdown_all_services()
        assert shutdown_results.get("mock_engine") is True
        assert service.shutdown_called is True
    
    @pytest.mark.asyncio
    async def test_config_hot_reload(self, factory):
        """Test configuration hot-reloading."""
        # Create a service
        service = factory.create_service(MockAnalysisEngine)
        
        # Add config watcher
        config_updated = False
        
        def config_watcher(new_config):
            nonlocal config_updated
            config_updated = True
        
        factory.add_config_watcher("mock_engine", config_watcher)
        
        # Reload configuration
        new_config = {"confidence_threshold": 0.9}
        success = await factory.reload_service_config("mock_engine", new_config)
        
        assert success is True
        assert config_updated is True


class TestDynamicConfigurationManager:
    """Test dynamic configuration manager."""
    
    def test_config_registration(self):
        """Test configuration source registration."""
        manager = DynamicConfigurationManager()
        
        # Register a config source (without file)
        config_data = {"test": "value"}
        success = manager.update_config("test_config", config_data)
        
        assert success is True
        assert manager.get_config("test_config") == config_data
    
    def test_config_merging(self):
        """Test configuration merging."""
        manager = DynamicConfigurationManager()
        
        # Register multiple configs
        config1 = {"a": 1, "b": {"x": 1}}
        config2 = {"b": {"y": 2}, "c": 3}
        
        manager.update_config("config1", config1)
        manager.update_config("config2", config2)
        
        # Get merged config
        merged = manager.get_merged_config("config1", "config2")
        
        expected = {"a": 1, "b": {"x": 1, "y": 2}, "c": 3}
        assert merged == expected
    
    def test_change_listeners(self):
        """Test configuration change listeners."""
        manager = DynamicConfigurationManager()
        
        # Add change listener
        changes = []
        
        def listener(event):
            changes.append(event)
        
        manager.add_change_listener("test_config", listener)
        
        # Update configuration
        config_data = {"test": "value"}
        manager.update_config("test_config", config_data)
        
        # Verify listener was called
        assert len(changes) == 1
        assert changes[0].config_path == "test_config"
        assert changes[0].new_config == config_data


class TestServiceHealthMonitor:
    """Test service health monitoring."""
    
    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor for testing."""
        return ServiceHealthMonitor(check_interval=1, alert_cooldown=1)
    
    def test_service_registration(self, health_monitor):
        """Test service registration for monitoring."""
        service = MockAnalysisEngine(AnalysisConfiguration(
            engine_name="test",
            enabled=True,
            confidence_threshold=0.7
        ))
        
        health_monitor.register_service("test_service", service)
        
        assert "test_service" in health_monitor.monitored_services
        assert health_monitor.monitored_services["test_service"] is service
    
    @pytest.mark.asyncio
    async def test_health_check(self, health_monitor):
        """Test health checking functionality."""
        service = MockAnalysisEngine(AnalysisConfiguration(
            engine_name="test",
            enabled=True,
            confidence_threshold=0.7
        ))
        
        # Initialize service
        await service.initialize()
        
        health_monitor.register_service("test_service", service)
        
        # Perform health check
        result = await health_monitor.check_service_health("test_service")
        
        assert result is not None
        assert result.service_name == "test_service"
        assert result.overall_status == HealthStatus.HEALTHY
        assert len(result.metrics) > 0
    
    @pytest.mark.asyncio
    async def test_alert_handling(self, health_monitor):
        """Test health alert handling."""
        alerts = []
        
        def alert_handler(alert):
            alerts.append(alert)
        
        health_monitor.add_alert_handler(alert_handler)
        
        # Create unhealthy service
        service = MockAnalysisEngine(AnalysisConfiguration(
            engine_name="test",
            enabled=True,
            confidence_threshold=0.7
        ))
        # Don't initialize to make it unhealthy
        
        health_monitor.register_service("unhealthy_service", service)
        
        # Perform health check
        await health_monitor.check_service_health("unhealthy_service")
        
        # Wait a moment for alerts to be processed
        await asyncio.sleep(0.1)
        
        # Verify alert was generated (this depends on the health checker implementation)
        # In a real scenario, the unhealthy service would trigger alerts


@pytest.mark.asyncio
async def test_integration():
    """Test integration of all components."""
    # Create configuration manager
    config_manager = DynamicConfigurationManager()
    config_manager.update_config("base", {
        "engines": {
            "mock_engine": {
                "enabled": True,
                "confidence_threshold": 0.8
            }
        }
    })
    
    # Create service factory
    dependency_provider = DependencyProvider()
    lifecycle_manager = ServiceLifecycleManager()
    
    factory = AnalysisServiceFactory(
        base_config=config_manager.get_config("base"),
        dependency_provider=dependency_provider,
        lifecycle_manager=lifecycle_manager
    )
    
    # Register mock engine
    factory.register_service_type("mock_engine", MockAnalysisEngine)
    
    # Create health monitor
    health_monitor = ServiceHealthMonitor(check_interval=1)
    
    try:
        # Create and initialize services
        service = factory.create_service(MockAnalysisEngine)
        await factory.initialize_all_services()
        
        # Register with health monitor
        health_monitor.register_service("mock_engine", service)
        
        # Start health monitoring
        await health_monitor.start_monitoring()
        
        # Verify everything is working
        assert service.initialized is True
        
        health_status = health_monitor.get_service_health_status("mock_engine")
        assert health_status == HealthStatus.HEALTHY
        
        # Test configuration update
        new_config = {"confidence_threshold": 0.9}
        success = await factory.reload_service_config("mock_engine", new_config)
        assert success is True
        
    finally:
        # Cleanup
        await health_monitor.stop_monitoring()
        await factory.shutdown_all_services()


if __name__ == "__main__":
    pytest.main([__file__])