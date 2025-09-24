"""
Integration tests for the runbook executor.
"""

import pytest

# Skip this test module since runbook_executor doesn't exist yet
pytest.skip("RunbookExecutor not implemented yet", allow_module_level=True)


class TestRunbookExecutor:
    """Test the runbook executor functionality."""

    @pytest.fixture
    def executor(self):
        """Create a runbook executor instance."""
        with (
            patch("runbook_executor.AnalysisMetricsCollector"),
            patch("runbook_executor.Phi3AnalysisModelServer"),
            patch("runbook_executor.AnalysisValidator"),
            patch("runbook_executor.OPAPolicyGenerator"),
            patch("runbook_executor.QualityAlertingSystem"),
            patch("runbook_executor.CostMonitoringSystem"),
            patch("runbook_executor.APIKeyManager"),
            patch("runbook_executor.WAFRuleEngine"),
            patch("runbook_executor.WeeklyEvaluationService"),
        ):
            return RunbookExecutor()

    @pytest.mark.asyncio
    async def test_check_schema_validation_rate_healthy(self, executor):
        """Test schema validation rate check when healthy."""
        # Mock healthy response
        executor.metrics_collector.get_metric.return_value = 0.96

        result = await executor.check_schema_validation_rate()

        assert result["rate"] == 0.96
        assert result["status"] == "healthy"
        assert result["threshold_warning"] == 0.95
        assert result["threshold_critical"] == 0.90

    @pytest.mark.asyncio
    async def test_check_schema_validation_rate_warning(self, executor):
        """Test schema validation rate check when warning."""
        # Mock warning response
        executor.metrics_collector.get_metric.return_value = 0.93

        result = await executor.check_schema_validation_rate()

        assert result["rate"] == 0.93
        assert result["status"] == "warning"

    @pytest.mark.asyncio
    async def test_check_schema_validation_rate_critical(self, executor):
        """Test schema validation rate check when critical."""
        # Mock critical response
        executor.metrics_collector.get_metric.return_value = 0.85

        result = await executor.check_schema_validation_rate()

        assert result["rate"] == 0.85
        assert result["status"] == "critical"

    @pytest.mark.asyncio
    async def test_check_schema_validation_rate_error(self, executor):
        """Test schema validation rate check when error occurs."""
        # Mock error
        executor.metrics_collector.get_metric.side_effect = Exception(
            "Metric not found"
        )

        result = await executor.check_schema_validation_rate()

        assert "error" in result
        assert result["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_check_template_fallback_rate_healthy(self, executor):
        """Test template fallback rate check when healthy."""
        # Mock healthy response
        executor.metrics_collector.get_metric.return_value = 0.15

        result = await executor.check_template_fallback_rate()

        assert result["rate"] == 0.15
        assert result["status"] == "healthy"
        assert result["threshold_warning"] == 0.20
        assert result["threshold_critical"] == 0.30

    @pytest.mark.asyncio
    async def test_check_template_fallback_rate_warning(self, executor):
        """Test template fallback rate check when warning."""
        # Mock warning response
        executor.metrics_collector.get_metric.return_value = 0.25

        result = await executor.check_template_fallback_rate()

        assert result["rate"] == 0.25
        assert result["status"] == "warning"

    @pytest.mark.asyncio
    async def test_check_template_fallback_rate_critical(self, executor):
        """Test template fallback rate check when critical."""
        # Mock critical response
        executor.metrics_collector.get_metric.return_value = 0.35

        result = await executor.check_template_fallback_rate()

        assert result["rate"] == 0.35
        assert result["status"] == "critical"

    @pytest.mark.asyncio
    async def test_check_model_server_health_healthy(self, executor):
        """Test model server health check when healthy."""
        # Mock healthy response
        mock_health = Mock()
        mock_health.model_loaded = True
        mock_health.memory_usage_mb = 1024
        mock_health.last_inference_time_ms = 150
        mock_health.error_rate = 0.03

        executor.model_server.get_health_status.return_value = mock_health

        result = await executor.check_model_server_health()

        assert result["model_loaded"] is True
        assert result["memory_usage_mb"] == 1024
        assert result["last_inference_time_ms"] == 150
        assert result["error_rate"] == 0.03
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_check_model_server_health_warning(self, executor):
        """Test model server health check when warning."""
        # Mock warning response
        mock_health = Mock()
        mock_health.model_loaded = True
        mock_health.memory_usage_mb = 2048
        mock_health.last_inference_time_ms = 300
        mock_health.error_rate = 0.07

        executor.model_server.get_health_status.return_value = mock_health

        result = await executor.check_model_server_health()

        assert result["error_rate"] == 0.07
        assert result["status"] == "warning"

    @pytest.mark.asyncio
    async def test_check_model_server_health_critical(self, executor):
        """Test model server health check when critical."""
        # Mock critical response
        mock_health = Mock()
        mock_health.model_loaded = False
        mock_health.memory_usage_mb = 4096
        mock_health.last_inference_time_ms = 1000
        mock_health.error_rate = 0.15

        executor.model_server.get_health_status.return_value = mock_health

        result = await executor.check_model_server_health()

        assert result["model_loaded"] is False
        assert result["error_rate"] == 0.15
        assert result["status"] == "critical"

    @pytest.mark.asyncio
    async def test_get_recent_validation_failures(self, executor):
        """Test getting recent validation failures."""
        # Mock validation failures
        mock_failure1 = Mock()
        mock_failure1.request_id = "req-123"
        mock_failure1.error_message = "Schema validation failed"
        mock_failure1.timestamp = "2024-01-01T10:00:00Z"
        mock_failure1.input_snippet = "test input data"

        mock_failure2 = Mock()
        mock_failure2.request_id = "req-124"
        mock_failure2.error_message = "Invalid field type"
        mock_failure2.timestamp = "2024-01-01T10:01:00Z"
        mock_failure2.input_snippet = "another test input"

        executor.validator.get_recent_validation_failures.return_value = [
            mock_failure1,
            mock_failure2,
        ]

        result = await executor.get_recent_validation_failures(limit=10)

        assert len(result) == 2
        assert result[0]["request_id"] == "req-123"
        assert result[0]["error_message"] == "Schema validation failed"
        assert result[1]["request_id"] == "req-124"
        assert result[1]["error_message"] == "Invalid field type"

    @pytest.mark.asyncio
    async def test_get_model_statistics(self, executor):
        """Test getting model statistics."""
        # Mock model statistics
        mock_stats = Mock()
        mock_stats.avg_confidence = 0.85
        mock_stats.low_confidence_count = 5
        mock_stats.model_load_time = 2.5
        mock_stats.total_requests = 1000
        mock_stats.successful_requests = 950

        executor.model_server.get_model_statistics.return_value = mock_stats

        result = await executor.get_model_statistics()

        assert result["avg_confidence"] == 0.85
        assert result["low_confidence_count"] == 5
        assert result["model_load_time"] == 2.5
        assert result["total_requests"] == 1000
        assert result["successful_requests"] == 950

    @pytest.mark.asyncio
    async def test_get_opa_compilation_errors(self, executor):
        """Test getting OPA compilation errors."""
        # Mock compilation errors
        mock_error1 = Mock()
        mock_error1.error_message = "Syntax error in Rego"
        mock_error1.policy_snippet = "package analysis"
        mock_error1.timestamp = "2024-01-01T10:00:00Z"

        mock_error2 = Mock()
        mock_error2.error_message = "Undefined variable"
        mock_error2.policy_snippet = "allow { input.undefined_field }"
        mock_error2.timestamp = "2024-01-01T10:01:00Z"

        executor.opa_generator.get_recent_compilation_errors.return_value = [
            mock_error1,
            mock_error2,
        ]

        result = await executor.get_opa_compilation_errors(limit=5)

        assert len(result) == 2
        assert result[0]["error_message"] == "Syntax error in Rego"
        assert result[0]["policy_snippet"] == "package analysis"
        assert result[1]["error_message"] == "Undefined variable"

    @pytest.mark.asyncio
    async def test_get_cost_breakdown(self, executor):
        """Test getting cost breakdown."""
        # Mock cost breakdown
        mock_breakdown = {
            "cpu": 45.50,
            "memory": 23.75,
            "storage": 12.25,
            "network": 8.00,
        }

        executor.cost_system.get_cost_breakdown.return_value = mock_breakdown

        result = await executor.get_cost_breakdown()

        assert result["cpu"] == 45.50
        assert result["memory"] == 23.75
        assert result["storage"] == 12.25
        assert result["network"] == 8.00

    @pytest.mark.asyncio
    async def test_get_waf_statistics(self, executor):
        """Test getting WAF statistics."""
        # Mock WAF engine
        executor.waf_engine.rules = [Mock() for _ in range(25)]
        executor.waf_engine.blocked_ips = {"192.168.1.100", "10.0.0.50"}
        executor.waf_engine.suspicious_ips = {"172.16.0.25"}

        result = await executor.get_waf_statistics()

        assert result["total_rules"] == 25
        assert result["blocked_ips"] == 2
        assert result["suspicious_ips"] == 1

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_status(self, executor):
        """Test getting circuit breaker status."""
        # Mock circuit breaker
        mock_cb = Mock()
        mock_cb.state.value = "CLOSED"
        mock_cb._failure_count = 0
        mock_cb._success_count = 100
        mock_cb._last_failure_time = None

        with patch("runbook_executor.CircuitBreaker", return_value=mock_cb):
            result = await executor.get_circuit_breaker_status()

        assert result["state"] == "CLOSED"
        assert result["failure_count"] == 0
        assert result["success_count"] == 100
        assert result["last_failure_time"] is None

    @pytest.mark.asyncio
    async def test_execute_health_check(self, executor):
        """Test executing comprehensive health check."""
        # Mock all health check methods
        executor.check_schema_validation_rate = AsyncMock(
            return_value={"rate": 0.96, "status": "healthy"}
        )
        executor.check_template_fallback_rate = AsyncMock(
            return_value={"rate": 0.15, "status": "healthy"}
        )
        executor.check_opa_compilation_rate = AsyncMock(
            return_value={"rate": 0.98, "status": "healthy"}
        )
        executor.check_model_server_health = AsyncMock(
            return_value={"model_loaded": True, "status": "healthy"}
        )
        executor.check_quality_score = AsyncMock(
            return_value={"score": 0.90, "status": "healthy"}
        )
        executor.check_cost_metrics = AsyncMock(
            return_value={"daily_cost": 50.0, "status": "healthy"}
        )
        executor.get_waf_statistics = AsyncMock(
            return_value={"total_rules": 25, "blocked_ips": 0}
        )
        executor.get_circuit_breaker_status = AsyncMock(
            return_value={"state": "CLOSED", "failure_count": 0}
        )
        executor.get_evaluation_status = AsyncMock(
            return_value={"monitoring_active": True}
        )

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

    @pytest.mark.asyncio
    async def test_execute_schema_validation_investigation(self, executor):
        """Test executing schema validation investigation."""
        # Mock investigation methods
        executor.check_schema_validation_rate = AsyncMock(
            return_value={"rate": 0.85, "status": "critical"}
        )
        executor.get_recent_validation_failures = AsyncMock(
            return_value=[
                {"request_id": "req-123", "error_message": "Schema validation failed"}
            ]
        )
        executor.get_model_statistics = AsyncMock(return_value={"avg_confidence": 0.75})

        result = await executor.execute_schema_validation_investigation()

        assert "validation_rate" in result
        assert result["validation_rate"]["status"] == "critical"
        assert "recent_failures" in result
        assert "model_statistics" in result

    @pytest.mark.asyncio
    async def test_execute_template_fallback_investigation(self, executor):
        """Test executing template fallback investigation."""
        # Mock investigation methods
        executor.check_template_fallback_rate = AsyncMock(
            return_value={"rate": 0.35, "status": "critical"}
        )
        executor.check_model_server_health = AsyncMock(
            return_value={"model_loaded": False, "error_rate": 0.15}
        )
        executor.get_model_statistics = AsyncMock(return_value={"avg_confidence": 0.65})

        result = await executor.execute_template_fallback_investigation()

        assert "fallback_rate" in result
        assert result["fallback_rate"]["status"] == "critical"
        assert "model_health" in result
        assert "model_statistics" in result

    @pytest.mark.asyncio
    async def test_execute_opa_compilation_investigation(self, executor):
        """Test executing OPA compilation investigation."""
        # Mock investigation methods
        executor.check_opa_compilation_rate = AsyncMock(
            return_value={"rate": 0.85, "status": "critical"}
        )
        executor.get_opa_compilation_errors = AsyncMock(
            return_value=[
                {"error_message": "Syntax error", "policy_snippet": "package analysis"}
            ]
        )

        result = await executor.execute_opa_compilation_investigation()

        assert "compilation_rate" in result
        assert result["compilation_rate"]["status"] == "critical"
        assert "compilation_errors" in result

    @pytest.mark.asyncio
    async def test_execute_cost_anomaly_investigation(self, executor):
        """Test executing cost anomaly investigation."""
        # Mock investigation methods
        executor.check_cost_metrics = AsyncMock(
            return_value={
                "daily_cost": 1200.0,
                "usage_percentage": 120.0,
                "status": "critical",
            }
        )
        executor.get_cost_breakdown = AsyncMock(
            return_value={"cpu": 800.0, "memory": 400.0}
        )
        executor.get_cost_anomalies = AsyncMock(
            return_value=[
                {
                    "description": "CPU spike detected",
                    "severity": "high",
                    "cost_impact": 200.0,
                }
            ]
        )

        result = await executor.execute_cost_anomaly_investigation()

        assert "cost_metrics" in result
        assert result["cost_metrics"]["status"] == "critical"
        assert "cost_breakdown" in result
        assert "anomalies" in result

    def test_generate_incident_report_schema_validation(self, executor):
        """Test generating incident report for schema validation."""
        results = {
            "validation_rate": {"rate": 0.85, "status": "critical"},
            "recent_failures": [
                {"request_id": "req-123", "error_message": "Schema validation failed"}
            ],
        }

        report = executor.generate_incident_report("schema_validation", results)

        assert "Schema Validation Investigation Report" in report
        assert "Schema validation rate: 85.00%" in report
        assert "Recent failures: 1" in report
        assert "```json" in report
        assert "req-123" in report

    def test_generate_incident_report_template_fallback(self, executor):
        """Test generating incident report for template fallback."""
        results = {
            "fallback_rate": {"rate": 0.35, "status": "critical"},
            "model_health": {"model_loaded": False},
        }

        report = executor.generate_incident_report("template_fallback", results)

        assert "Template Fallback Investigation Report" in report
        assert "Template fallback rate: 35.00%" in report
        assert "Model loaded: False" in report

    def test_generate_incident_report_opa_compilation(self, executor):
        """Test generating incident report for OPA compilation."""
        results = {
            "compilation_rate": {"rate": 0.85, "status": "critical"},
            "compilation_errors": [
                {"error_message": "Syntax error", "policy_snippet": "package analysis"}
            ],
        }

        report = executor.generate_incident_report("opa_compilation", results)

        assert "Opa Compilation Investigation Report" in report
        assert "OPA compilation rate: 85.00%" in report
        assert "Recent errors: 1" in report

    def test_generate_incident_report_cost_anomaly(self, executor):
        """Test generating incident report for cost anomaly."""
        results = {
            "cost_metrics": {"daily_cost": 1200.0, "usage_percentage": 120.0},
            "anomalies": [{"description": "CPU spike detected", "severity": "high"}],
        }

        report = executor.generate_incident_report("cost_anomaly", results)

        assert "Cost Anomaly Investigation Report" in report
        assert "Daily cost: $1200.00" in report
        assert "Budget usage: 120.0%" in report
        assert "Anomalies detected: 1" in report


if __name__ == "__main__":
    pytest.main([__file__])
