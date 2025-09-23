#!/usr/bin/env python3
"""
Basic test for runbook executor functionality.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


async def test_runbook_executor_basic():
    """Test basic runbook executor functionality."""
    print("🧪 Testing Runbook Executor Basic Functionality...")
    
    # Mock all the dependencies
    with patch('src.llama_mapper.analysis.monitoring.metrics_collector.AnalysisMetricsCollector') as mock_metrics, \
         patch('src.llama_mapper.analysis.infrastructure.model_server.Phi3AnalysisModelServer') as mock_model, \
         patch('src.llama_mapper.analysis.infrastructure.validator.AnalysisValidator') as mock_validator, \
         patch('src.llama_mapper.analysis.infrastructure.opa_generator.OPAPolicyGenerator') as mock_opa, \
         patch('src.llama_mapper.analysis.quality.quality_alerting_system.QualityAlertingSystem') as mock_quality, \
         patch('src.llama_mapper.cost_monitoring.CostMonitoringSystem') as mock_cost, \
         patch('src.llama_mapper.analysis.infrastructure.auth.APIKeyManager') as mock_auth, \
         patch('src.llama_mapper.analysis.security.waf.engine.rule_engine.WAFRuleEngine') as mock_waf, \
         patch('src.llama_mapper.analysis.domain.services.WeeklyEvaluationService') as mock_eval:
        
        # Mock the cost system start method
        mock_cost.return_value.start = Mock()
        
        # Import and create executor
        import importlib.util
        spec = importlib.util.spec_from_file_location("runbook_executor", project_root / "scripts" / "runbook-executor.py")
        runbook_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runbook_module)
        RunbookExecutor = runbook_module.RunbookExecutor
        executor = RunbookExecutor()
        
        print("✅ RunbookExecutor initialized successfully")
        
        # Test schema validation rate check
        mock_metrics.return_value.get_metric.return_value = 0.96
        result = await executor.check_schema_validation_rate()
        
        assert result["rate"] == 0.96
        assert result["status"] == "healthy"
        print("✅ Schema validation rate check works")
        
        # Test template fallback rate check
        mock_metrics.return_value.get_metric.return_value = 0.15
        result = await executor.check_template_fallback_rate()
        
        assert result["rate"] == 0.15
        assert result["status"] == "healthy"
        print("✅ Template fallback rate check works")
        
        # Test model server health check
        mock_health = Mock()
        mock_health.model_loaded = True
        mock_health.memory_usage_mb = 1024
        mock_health.last_inference_time_ms = 150
        mock_health.error_rate = 0.03
        
        mock_model.return_value.get_health_status.return_value = mock_health
        
        result = await executor.check_model_server_health()
        
        assert result["model_loaded"] is True
        assert result["status"] == "healthy"
        print("✅ Model server health check works")
        
        # Test health check execution
        executor.check_schema_validation_rate = AsyncMock(return_value={"rate": 0.96, "status": "healthy"})
        executor.check_template_fallback_rate = AsyncMock(return_value={"rate": 0.15, "status": "healthy"})
        executor.check_opa_compilation_rate = AsyncMock(return_value={"rate": 0.98, "status": "healthy"})
        executor.check_model_server_health = AsyncMock(return_value={"model_loaded": True, "status": "healthy"})
        executor.check_quality_score = AsyncMock(return_value={"score": 0.90, "status": "healthy"})
        executor.check_cost_metrics = AsyncMock(return_value={"daily_cost": 50.0, "status": "healthy"})
        executor.get_waf_statistics = AsyncMock(return_value={"total_rules": 25, "blocked_ips": 0})
        executor.get_circuit_breaker_status = AsyncMock(return_value={"state": "CLOSED", "failure_count": 0})
        executor.get_evaluation_status = AsyncMock(return_value={"monitoring_active": True})
        
        result = await executor.execute_health_check()
        
        assert "schema_validation" in result
        assert "template_fallback" in result
        assert "model_server" in result
        print("✅ Comprehensive health check works")
        
        # Test incident report generation
        results = {
            "validation_rate": {"rate": 0.85, "status": "critical"},
            "recent_failures": [
                {"request_id": "req-123", "error_message": "Schema validation failed"}
            ]
        }
        
        report = executor.generate_incident_report("schema_validation", results)
        
        assert "Schema Validation Investigation Report" in report
        assert "Schema validation rate: 85.00%" in report
        assert "Recent failures: 1" in report
        print("✅ Incident report generation works")
        
        print("\n🎉 All basic tests passed!")
        return True


async def test_runbook_executor_commands():
    """Test runbook executor command line interface."""
    print("\n🧪 Testing Runbook Executor Commands...")
    
    # Test help command
    import subprocess
    result = subprocess.run([
        sys.executable, "scripts/runbook-executor.py", "--help"
    ], capture_output=True, text=True, cwd=project_root)
    
    assert result.returncode == 0
    assert "Analysis Module Runbook Executor" in result.stdout
    print("✅ Help command works")
    
    # Test command validation
    result = subprocess.run([
        sys.executable, "scripts/runbook-executor.py", "invalid-command"
    ], capture_output=True, text=True, cwd=project_root)
    
    assert result.returncode == 2  # Argument error
    print("✅ Command validation works")
    
    print("🎉 All command tests passed!")
    return True


async def main():
    """Run all tests."""
    print("🚀 Starting Runbook Executor Tests...\n")
    
    try:
        # Test basic functionality
        await test_runbook_executor_basic()
        
        # Test command line interface
        await test_runbook_executor_commands()
        
        print("\n🎉 All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
