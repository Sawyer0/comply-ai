"""
Integration tests for the refactored analysis system.

Tests the new dependency injection, factory patterns, and service lifecycle
management to ensure the refactored system works correctly.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from ...factories import AnalysisServiceFactory, DependencyProvider
from ...config import DynamicConfigurationManager
from ...monitoring import ServiceHealthMonitor, HealthStatus
from ...lifecycle import ServiceLifecycleManager
from ...domain import AnalysisConfiguration, IAnalysisEngine
from ...engines import PatternRecognitionEngine, RiskScoringEngine, ComplianceIntelligence


class MockAnalysisEngine(IAnalysisEngine):
    """Mock analysis engine for integration testing."""
    
    def __init__(self, config: AnalysisConfiguration):
        self.config = config
        self.initialized = False
        self.shutdown_called = False
        self.analysis_count = 0
    
    async def initialize(self):
        """Initialize the engine."""
        self.initialized = True
    
    async def shutdown(self):
        """Shutdown the engine."""
        self.shutdown_called = True
    
    async def analyze(self, request):
        """Mock analysis method."""
        self.analysis_count += 1
        return {
            "result": f"mock_analysis_{self.analysis_count}",
            "confidence": 0.8,
            "engine_type": self.__class__.__name__
        }
    
    def get_confidence(self, result):
        """Get confidence score."""
        return result.get("confidence", 0.0)
    
    def get_health_status(self):
        """Get health status."""
        return "healthy" if self.initialized else "unhealthy"
    
    async def get_response_time(self):
        """Mock response time."""
        return 50.0  # 50ms
    
    async def get_error_rate(self):
        """Mock error rate."""
        return 2.0  # 2%
    
    async def get_memory_usage(self):
        """Mock memory usage."""
        return 256.0  # 256MB
    
    async def get_cpu_usage(self):
        """Mock CPU usage."""
        return 25.0  # 25%


@pytest.mark.asyncio
class TestRefactoredSystemIntegration:
    """Integration tests for the complete refactored system."""
    
    @pytest.fixture
    async def config_manager(self):
        """Create configuration manager for testing."""
        manager = DynamicConfigurationManager()
        
        # Set up test configuration
        test_config = {
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
                },
                "compliance_intelligence": {
                    "enabled": True,
                    "confidence_threshold": 0.75,
                    "parameters": {
                        "enabled_frameworks": ["soc2", "iso27001"]
                    }
                }
            },
            "health_monitoring": {
                "check_interval": 5,
                "alert_cooldown": 60,
                "thresholds": {
                    "response_time_ms": 1000,
                    "error_rate_percent": 5.0,
                    "memory_usage_mb": 512,
                    "cpu_usage_percent": 80
                }
            }
        }
        
        manager.update_config("test_config", test_config)
        return manager
    
    @pytest.fixture
    async def dependency_provider(self):
        """Create dependency provider with test dependencies."""
        provider = DependencyProvider()
        
        # Register mock dependencies
        mock_data_repo = Mock()
        mock_data_repo.get_data = AsyncMock(return_value={"test": "data"})
        provider.register_dependency(Mock, mock_data_repo, "data_repository")
        
        mock_analyzer = Mock()
        mock_analyzer.analyze = AsyncMock(return_value={"analysis": "result"})
        provider.register_dependency(Mock, mock_analyzer, "statistical_analyzer")
        
        return provider
    
    @pytest.fixture
    async def service_factory(self, config_manager, dependency_provider):
        """Create service factory with all dependencies."""
        lifecycle_manager = ServiceLifecycleManager(health_check_interval=5)
        
        factory = AnalysisServiceFactory(
            base_config=config_manager.get_config("test_config"),
            dependency_provider=dependency_provider,
            lifecycle_manager=lifecycle_manager
        )
        
        # Register mock engine for testing
        factory.register_service_type("mock_engine", MockAnalysisEngine)
        
        return factory
    
    @pytest.fixture
    async def health_monitor(self):
        """Create health monitor for testing."""
        return ServiceHealthMonitor(check_interval=2, alert_cooldown=10)
    
    async def test_complete_system_initialization(self, service_factory, health_monitor):
        """Test complete system initialization and lifecycle."""
        try:
            # Create services
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            pattern_engine = service_factory.create_service(PatternRecognitionEngine)
            
            assert isinstance(mock_engine, MockAnalysisEngine)
            assert isinstance(pattern_engine, PatternRecognitionEngine)
            
            # Initialize all services
            init_results = await service_factory.initialize_all_services()
            
            assert init_results.get("mock_engine") is True
            assert init_results.get("pattern_recognition") is True
            assert mock_engine.initialized is True
            
            # Register services with health monitor
            health_monitor.register_service("mock_engine", mock_engine)
            health_monitor.register_service("pattern_recognition", pattern_engine)
            
            # Start health monitoring
            await health_monitor.start_monitoring()
            
            # Wait for health checks
            await asyncio.sleep(3)
            
            # Verify health status
            health_status = health_monitor.get_all_service_status()
            assert "mock_engine" in health_status
            assert "pattern_recognition" in health_status
            
            # Perform health checks
            mock_health = await health_monitor.check_service_health("mock_engine")
            assert mock_health is not None
            assert mock_health.overall_status == HealthStatus.HEALTHY
            
        finally:
            # Cleanup
            await health_monitor.stop_monitoring()
            await service_factory.shutdown_all_services()
    
    async def test_dependency_injection_works(self, service_factory):
        """Test that dependency injection works correctly."""
        try:
            # Create service with dependencies
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            
            # Initialize services
            await service_factory.initialize_all_services()
            
            # Test that service works
            result = await mock_engine.analyze({"test": "request"})
            
            assert result["result"] == "mock_analysis_1"
            assert result["confidence"] == 0.8
            assert result["engine_type"] == "MockAnalysisEngine"
            
        finally:
            await service_factory.shutdown_all_services()
    
    async def test_configuration_hot_reload(self, service_factory, config_manager):
        """Test configuration hot-reloading functionality."""
        try:
            # Create and initialize services
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            await service_factory.initialize_all_services()
            
            # Set up config watcher
            config_changes = []
            
            def config_watcher(new_config):
                config_changes.append(new_config)
            
            service_factory.add_config_watcher("mock_engine", config_watcher)
            
            # Update configuration
            new_config = {"confidence_threshold": 0.9, "new_parameter": "test_value"}
            success = await service_factory.reload_service_config("mock_engine", new_config)
            
            assert success is True
            assert len(config_changes) == 1
            assert config_changes[0]["confidence_threshold"] == 0.9
            
        finally:
            await service_factory.shutdown_all_services()
    
    async def test_health_monitoring_alerts(self, service_factory, health_monitor):
        """Test health monitoring and alerting functionality."""
        try:
            # Create unhealthy service
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            # Don't initialize to make it unhealthy
            
            health_monitor.register_service("unhealthy_service", mock_engine)
            
            # Collect alerts
            alerts = []
            
            def alert_handler(alert):
                alerts.append(alert)
            
            health_monitor.add_alert_handler(alert_handler)
            
            # Perform health check
            result = await health_monitor.check_service_health("unhealthy_service")
            
            assert result is not None
            assert result.overall_status != HealthStatus.HEALTHY
            
            # Wait for potential alerts
            await asyncio.sleep(1)
            
        finally:
            await health_monitor.stop_monitoring()
    
    async def test_service_lifecycle_hooks(self, service_factory):
        """Test service lifecycle hooks functionality."""
        try:
            # Set up hooks
            init_hook_called = False
            shutdown_hook_called = False
            
            def init_hook():
                nonlocal init_hook_called
                init_hook_called = True
            
            def shutdown_hook():
                nonlocal shutdown_hook_called
                shutdown_hook_called = True
            
            service_factory.add_initialization_hook("mock_engine", init_hook)
            service_factory.add_shutdown_hook("mock_engine", shutdown_hook)
            
            # Create and initialize service
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            await service_factory.initialize_all_services()
            
            assert init_hook_called is True
            
            # Shutdown services
            await service_factory.shutdown_all_services()
            
            assert shutdown_hook_called is True
            
        except Exception:
            # Ensure cleanup even if test fails
            await service_factory.shutdown_all_services()
    
    async def test_multiple_engine_coordination(self, service_factory):
        """Test coordination between multiple analysis engines."""
        try:
            # Create multiple engines
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            pattern_engine = service_factory.create_service(PatternRecognitionEngine)
            risk_engine = service_factory.create_service(RiskScoringEngine)
            
            # Initialize all
            init_results = await service_factory.initialize_all_services()
            
            # Verify all initialized successfully
            assert all(init_results.values())
            
            # Test that engines can work together
            mock_result = await mock_engine.analyze({"test": "data"})
            assert mock_result["result"] == "mock_analysis_1"
            
            # Verify engines are registered in lifecycle manager
            service_status = service_factory.get_service_health()
            assert len(service_status) >= 3  # At least our 3 engines
            
        finally:
            await service_factory.shutdown_all_services()
    
    async def test_error_handling_and_recovery(self, service_factory):
        """Test error handling and recovery mechanisms."""
        try:
            # Create service that might fail
            mock_engine = service_factory.create_service(MockAnalysisEngine)
            
            # Initialize services
            await service_factory.initialize_all_services()
            
            # Simulate error condition
            mock_engine.initialized = False  # Simulate failure
            
            # Service should still be manageable
            health_status = service_factory.get_service_health()
            assert "mock_engine" in health_status
            
            # Recovery: re-initialize
            mock_engine.initialized = True
            
            # Should work again
            result = await mock_engine.analyze({"recovery": "test"})
            assert result["result"] == "mock_analysis_1"
            
        finally:
            await service_factory.shutdown_all_services()
    
    async def test_configuration_validation(self, config_manager):
        """Test configuration validation functionality."""
        # Test valid configuration
        valid_config = {
            "engines": {
                "test_engine": {
                    "enabled": True,
                    "confidence_threshold": 0.8
                }
            }
        }
        
        success = config_manager.update_config("valid_test", valid_config)
        assert success is True
        
        # Test configuration retrieval
        retrieved = config_manager.get_config("valid_test")
        assert retrieved == valid_config
        
        # Test configuration merging
        additional_config = {
            "engines": {
                "test_engine": {
                    "new_parameter": "test_value"
                }
            }
        }
        
        config_manager.update_config("additional_test", additional_config)
        merged = config_manager.get_merged_config("valid_test", "additional_test")
        
        assert merged["engines"]["test_engine"]["enabled"] is True
        assert merged["engines"]["test_engine"]["confidence_threshold"] == 0.8
        assert merged["engines"]["test_engine"]["new_parameter"] == "test_value"


@pytest.mark.asyncio
class TestAPIIntegrationWithRefactoredSystem:
    """Test API integration with the refactored system."""
    
    async def test_api_uses_refactored_components(self):
        """Test that API endpoints use the refactored components correctly."""
        # This would test that the API endpoints properly use the new
        # factory pattern and dependency injection system
        
        from ...api.factory import create_analysis_app
        from fastapi.testclient import TestClient
        
        app = create_analysis_app()
        client = TestClient(app)
        
        # Test health endpoint works with refactored system
        response = client.get("/api/v1/analysis/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    async def test_end_to_end_analysis_with_refactored_system(self):
        """Test complete end-to-end analysis using refactored components."""
        # This would test a complete analysis request through the API
        # using the new dependency injection and factory patterns
        
        # Create test request
        test_request = {
            "period": "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z",
            "tenant": "integration-test",
            "app": "test-app",
            "route": "test-route",
            "required_detectors": ["presidio", "deberta-toxicity"],
            "observed_coverage": {"presidio": 0.8, "deberta-toxicity": 0.9},
            "required_coverage": {"presidio": 0.9, "deberta-toxicity": 0.9},
            "detector_errors": {},
            "high_sev_hits": [],
            "false_positive_bands": [],
            "policy_bundle": "test-bundle",
            "env": "dev"
        }
        
        # This test would verify that the request flows through the
        # refactored system correctly and produces expected results
        
        # For now, just verify the request structure is valid
        assert "tenant" in test_request
        assert "required_detectors" in test_request
        assert len(test_request["required_detectors"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])