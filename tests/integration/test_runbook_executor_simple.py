"""
Simple integration tests for the runbook executor.
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))


class TestRunbookExecutorSimple:
    """Simple tests for the runbook executor functionality."""
    
    @pytest.mark.asyncio
    async def test_runbook_executor_import(self):
        """Test that the runbook executor can be imported."""
        try:
            from scripts.runbook_executor import RunbookExecutor
            assert RunbookExecutor is not None
        except ImportError as e:
            pytest.fail(f"Could not import RunbookExecutor: {e}")
    
    @pytest.mark.asyncio
    async def test_runbook_executor_initialization(self):
        """Test that the runbook executor can be initialized."""
        with patch('scripts.runbook_executor.AnalysisMetricsCollector') as mock_metrics, \
             patch('scripts.runbook_executor.Phi3AnalysisModelServer') as mock_model, \
             patch('scripts.runbook_executor.AnalysisValidator') as mock_validator, \
             patch('scripts.runbook_executor.OPAPolicyGenerator') as mock_opa, \
             patch('scripts.runbook_executor.QualityAlertingSystem') as mock_quality, \
             patch('scripts.runbook_executor.CostMonitoringSystem') as mock_cost, \
             patch('scripts.runbook_executor.APIKeyManager') as mock_auth, \
             patch('scripts.runbook_executor.WAFRuleEngine') as mock_waf, \
             patch('scripts.runbook_executor.WeeklyEvaluationService') as mock_eval:
            
            from scripts.runbook_executor import RunbookExecutor
            
            # Mock the cost system start method
            mock_cost.return_value.start = Mock()
            
            executor = RunbookExecutor()
            
            assert executor is not None
            assert executor.metrics_collector is not None
            assert executor.model_server is not None
            assert executor.validator is not None
            assert executor.opa_generator is not None
            assert executor.quality_system is not None
            assert executor.cost_system is not None
            assert executor.api_key_manager is not None
            assert executor.waf_engine is not None
            assert executor.evaluation_service is not None
    
    @pytest.mark.asyncio
    async def test_schema_validation_rate_check(self):
        """Test schema validation rate checking."""
        with patch('scripts.runbook_executor.AnalysisMetricsCollector') as mock_metrics, \
             patch('scripts.runbook_executor.Phi3AnalysisModelServer') as mock_model, \
             patch('scripts.runbook_executor.AnalysisValidator') as mock_validator, \
             patch('scripts.runbook_executor.OPAPolicyGenerator') as mock_opa, \
             patch('scripts.runbook_executor.QualityAlertingSystem') as mock_quality, \
             patch('scripts.runbook_executor.CostMonitoringSystem') as mock_cost, \
             patch('scripts.runbook_executor.APIKeyManager') as mock_auth, \
             patch('scripts.runbook_executor.WAFRuleEngine') as mock_waf, \
             patch('scripts.runbook_executor.WeeklyEvaluationService') as mock_eval:
            
            from scripts.runbook_executor import RunbookExecutor
            
            # Mock the cost system start method
            mock_cost.return_value.start = Mock()
            
            # Mock metrics collector to return a healthy rate
            mock_metrics.return_value.get_metric.return_value = 0.96
            
            executor = RunbookExecutor()
            result = await executor.check_schema_validation_rate()
            
            assert result["rate"] == 0.96
            assert result["status"] == "healthy"
            assert result["threshold_warning"] == 0.95
            assert result["threshold_critical"] == 0.90
    
    @pytest.mark.asyncio
    async def test_template_fallback_rate_check(self):
        """Test template fallback rate checking."""
        with patch('scripts.runbook_executor.AnalysisMetricsCollector') as mock_metrics, \
             patch('scripts.runbook_executor.Phi3AnalysisModelServer') as mock_model, \
             patch('scripts.runbook_executor.AnalysisValidator') as mock_validator, \
             patch('scripts.runbook_executor.OPAPolicyGenerator') as mock_opa, \
             patch('scripts.runbook_executor.QualityAlertingSystem') as mock_quality, \
             patch('scripts.runbook_executor.CostMonitoringSystem') as mock_cost, \
             patch('scripts.runbook_executor.APIKeyManager') as mock_auth, \
             patch('scripts.runbook_executor.WAFRuleEngine') as mock_waf, \
             patch('scripts.runbook_executor.WeeklyEvaluationService') as mock_eval:
            
            from scripts.runbook_executor import RunbookExecutor
            
            # Mock the cost system start method
            mock_cost.return_value.start = Mock()
            
            # Mock metrics collector to return a healthy fallback rate
            mock_metrics.return_value.get_metric.return_value = 0.15
            
            executor = RunbookExecutor()
            result = await executor.check_template_fallback_rate()
            
            assert result["rate"] == 0.15
            assert result["status"] == "healthy"
            assert result["threshold_warning"] == 0.20
            assert result["threshold_critical"] == 0.30
    
    @pytest.mark.asyncio
    async def test_model_server_health_check(self):
        """Test model server health checking."""
        with patch('scripts.runbook_executor.AnalysisMetricsCollector') as mock_metrics, \
             patch('scripts.runbook_executor.Phi3AnalysisModelServer') as mock_model, \
             patch('scripts.runbook_executor.AnalysisValidator') as mock_validator, \
             patch('scripts.runbook_executor.OPAPolicyGenerator') as mock_opa, \
             patch('scripts.runbook_executor.QualityAlertingSystem') as mock_quality, \
             patch('scripts.runbook_executor.CostMonitoringSystem') as mock_cost, \
             patch('scripts.runbook_executor.APIKeyManager') as mock_auth, \
             patch('scripts.runbook_executor.WAFRuleEngine') as mock_waf, \
             patch('scripts.runbook_executor.WeeklyEvaluationService') as mock_eval:
            
            from scripts.runbook_executor import RunbookExecutor
            
            # Mock the cost system start method
            mock_cost.return_value.start = Mock()
            
            # Mock model server health status
            mock_health = Mock()
            mock_health.model_loaded = True
            mock_health.memory_usage_mb = 1024
            mock_health.last_inference_time_ms = 150
            mock_health.error_rate = 0.03
            
            mock_model.return_value.get_health_status.return_value = mock_health
            
            executor = RunbookExecutor()
            result = await executor.check_model_server_health()
            
            assert result["model_loaded"] is True
            assert result["memory_usage_mb"] == 1024
            assert result["last_inference_time_ms"] == 150
            assert result["error_rate"] == 0.03
            assert result["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self):
        """Test comprehensive health check execution."""
        with patch('scripts.runbook_executor.AnalysisMetricsCollector') as mock_metrics, \
             patch('scripts.runbook_executor.Phi3AnalysisModelServer') as mock_model, \
             patch('scripts.runbook_executor.AnalysisValidator') as mock_validator, \
             patch('scripts.runbook_executor.OPAPolicyGenerator') as mock_opa, \
             patch('scripts.runbook_executor.QualityAlertingSystem') as mock_quality, \
             patch('scripts.runbook_executor.CostMonitoringSystem') as mock_cost, \
             patch('scripts.runbook_executor.APIKeyManager') as mock_auth, \
             patch('scripts.runbook_executor.WAFRuleEngine') as mock_waf, \
             patch('scripts.runbook_executor.WeeklyEvaluationService') as mock_eval:
            
            from scripts.runbook_executor import RunbookExecutor
            
            # Mock the cost system start method
            mock_cost.return_value.start = Mock()
            
            # Mock all health check methods to return healthy status
            executor = RunbookExecutor()
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
            assert "opa_compilation" in result
            assert "model_server" in result
            assert "quality_score" in result
            assert "cost_metrics" in result
            assert "waf_statistics" in result
            assert "circuit_breaker" in result
            assert "evaluation_status" in result
            
            # Check that all statuses are healthy
            for key, value in result.items():
                if isinstance(value, dict) and "status" in value:
                    assert value["status"] == "healthy", f"{key} status is not healthy: {value['status']}"
    
    @pytest.mark.asyncio
    async def test_incident_report_generation(self):
        """Test incident report generation."""
        with patch('scripts.runbook_executor.AnalysisMetricsCollector') as mock_metrics, \
             patch('scripts.runbook_executor.Phi3AnalysisModelServer') as mock_model, \
             patch('scripts.runbook_executor.AnalysisValidator') as mock_validator, \
             patch('scripts.runbook_executor.OPAPolicyGenerator') as mock_opa, \
             patch('scripts.runbook_executor.QualityAlertingSystem') as mock_quality, \
             patch('scripts.runbook_executor.CostMonitoringSystem') as mock_cost, \
             patch('scripts.runbook_executor.APIKeyManager') as mock_auth, \
             patch('scripts.runbook_executor.WAFRuleEngine') as mock_waf, \
             patch('scripts.runbook_executor.WeeklyEvaluationService') as mock_eval:
            
            from scripts.runbook_executor import RunbookExecutor
            
            # Mock the cost system start method
            mock_cost.return_value.start = Mock()
            
            executor = RunbookExecutor()
            
            # Test schema validation incident report
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
            assert "```json" in report
            assert "req-123" in report
            
            # Test template fallback incident report
            results = {
                "fallback_rate": {"rate": 0.35, "status": "critical"},
                "model_health": {"model_loaded": False}
            }
            
            report = executor.generate_incident_report("template_fallback", results)
            
            assert "Template Fallback Investigation Report" in report
            assert "Template fallback rate: 35.00%" in report
            assert "Model loaded: False" in report


if __name__ == "__main__":
    pytest.main([__file__])
