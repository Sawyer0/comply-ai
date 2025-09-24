#!/usr/bin/env python3
"""
Refactoring validation test runner.

Runs integration tests to validate that the refactored system
works correctly and maintains expected behavior.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llama_mapper.analysis.factories import AnalysisServiceFactory, DependencyProvider
from llama_mapper.analysis.config import DynamicConfigurationManager
from llama_mapper.analysis.monitoring import ServiceHealthMonitor
from llama_mapper.analysis.lifecycle import ServiceLifecycleManager


async def test_basic_system_functionality():
    """Test basic system functionality."""
    print("🔧 Testing basic system functionality...")
    
    try:
        # Create configuration manager
        config_manager = DynamicConfigurationManager()
        config_manager.update_config("test", {
            "engines": {
                "test_engine": {
                    "enabled": True,
                    "confidence_threshold": 0.8
                }
            }
        })
        print("✓ Configuration manager created and configured")
        
        # Create dependency provider
        dependency_provider = DependencyProvider()
        print("✓ Dependency provider created")
        
        # Create service factory
        lifecycle_manager = ServiceLifecycleManager()
        factory = AnalysisServiceFactory(
            base_config=config_manager.get_config("test"),
            dependency_provider=dependency_provider,
            lifecycle_manager=lifecycle_manager
        )
        print("✓ Service factory created with dependency injection")
        
        # Create health monitor
        health_monitor = ServiceHealthMonitor(check_interval=5)
        print("✓ Health monitor created")
        
        print("✅ Basic system functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic system functionality test failed: {e}")
        return False


async def test_service_lifecycle():
    """Test service lifecycle management."""
    print("\n🔄 Testing service lifecycle management...")
    
    try:
        # Create minimal system
        config_manager = DynamicConfigurationManager()
        dependency_provider = DependencyProvider()
        lifecycle_manager = ServiceLifecycleManager()
        
        factory = AnalysisServiceFactory(
            base_config={},
            dependency_provider=dependency_provider,
            lifecycle_manager=lifecycle_manager
        )
        
        # Test initialization (without actual services for now)
        init_results = await factory.initialize_all_services()
        print(f"✓ Service initialization completed: {len(init_results)} services")
        
        # Test shutdown
        shutdown_results = await factory.shutdown_all_services()
        print(f"✓ Service shutdown completed: {len(shutdown_results)} services")
        
        print("✅ Service lifecycle test passed")
        return True
        
    except Exception as e:
        print(f"❌ Service lifecycle test failed: {e}")
        return False


async def test_configuration_management():
    """Test configuration management."""
    print("\n⚙️ Testing configuration management...")
    
    try:
        config_manager = DynamicConfigurationManager()
        
        # Test configuration registration
        test_config = {
            "test_key": "test_value",
            "nested": {
                "key": "value"
            }
        }
        
        success = config_manager.update_config("test_config", test_config)
        assert success, "Configuration update failed"
        print("✓ Configuration update successful")
        
        # Test configuration retrieval
        retrieved = config_manager.get_config("test_config")
        assert retrieved == test_config, "Configuration retrieval failed"
        print("✓ Configuration retrieval successful")
        
        # Test configuration merging
        additional_config = {"additional_key": "additional_value"}
        config_manager.update_config("additional", additional_config)
        
        merged = config_manager.get_merged_config("test_config", "additional")
        assert "test_key" in merged, "Configuration merging failed"
        assert "additional_key" in merged, "Configuration merging failed"
        print("✓ Configuration merging successful")
        
        print("✅ Configuration management test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        return False


async def test_health_monitoring():
    """Test health monitoring functionality."""
    print("\n🏥 Testing health monitoring...")
    
    try:
        health_monitor = ServiceHealthMonitor(check_interval=1)
        
        # Test basic health monitor functionality
        status = health_monitor.get_all_service_status()
        assert isinstance(status, dict), "Health status should be a dictionary"
        print("✓ Health monitor status retrieval successful")
        
        # Test alert handling
        alerts_received = []
        
        def test_alert_handler(alert):
            alerts_received.append(alert)
        
        health_monitor.add_alert_handler(test_alert_handler)
        print("✓ Alert handler registration successful")
        
        print("✅ Health monitoring test passed")
        return True
        
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False


async def run_all_validation_tests():
    """Run all validation tests."""
    print("🚀 Starting refactoring validation tests...\n")
    
    tests = [
        ("Basic System Functionality", test_basic_system_functionality),
        ("Service Lifecycle", test_service_lifecycle),
        ("Configuration Management", test_configuration_management),
        ("Health Monitoring", test_health_monitoring),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 REFACTORING VALIDATION SUMMARY")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"Total Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 ALL REFACTORING VALIDATION TESTS PASSED!")
        print("The refactored system is working correctly.")
        return True
    else:
        print(f"\n⚠️  {failed} TESTS FAILED")
        print("Please review the failed tests before proceeding.")
        return False


async def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        success = await run_all_validation_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error during validation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())