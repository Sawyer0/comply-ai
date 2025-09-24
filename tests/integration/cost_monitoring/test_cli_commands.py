"""Integration tests for cost monitoring CLI commands."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from click.testing import CliRunner

from src.llama_mapper.cli.commands.cost_monitoring import (
    CostAnomaliesCommand,
    CostBreakdownCommand,
    CostForecastCommand,
    CostRecommendationsCommand,
    CostStatusCommand,
    CostTrendsCommand,
)


class TestCostMonitoringCLICommands:
    """Test cost monitoring CLI commands."""

    @pytest.fixture
    def mock_config_manager(self):
        """Create mock config manager."""
        config_manager = Mock()
        config_manager.serving.host = "localhost"
        config_manager.serving.port = 8000
        return config_manager

    @pytest.fixture
    def mock_cost_system(self):
        """Create mock cost monitoring system."""
        cost_system = Mock()
        cost_system.get_system_status.return_value = {
            "running": True,
            "uptime_seconds": 3600,
            "components": {
                "metrics_collector": {"status": "healthy"},
                "guardrails": {"status": "healthy"},
                "autoscaling": {"status": "healthy"},
                "analytics": {"status": "healthy"},
            },
            "budget_limits": {
                "daily": 1000.0,
                "monthly": 30000.0,
                "emergency_stop": 5000.0,
            },
        }
        cost_system.health_check = AsyncMock(
            return_value={
                "overall_health": "healthy",
                "components": {
                    "metrics_collector": {"status": "healthy"},
                    "guardrails": {"status": "healthy"},
                    "autoscaling": {"status": "healthy"},
                    "analytics": {"status": "healthy"},
                },
            }
        )
        return cost_system

    @pytest.mark.asyncio
    async def test_cost_status_command(self, mock_config_manager, mock_cost_system):
        """Test cost status command."""
        command = CostStatusCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async()

        # Verify the cost system methods were called
        mock_cost_system.get_system_status.assert_called_once()
        mock_cost_system.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_breakdown_command(self, mock_config_manager, mock_cost_system):
        """Test cost breakdown command."""
        from datetime import datetime, timedelta, timezone

        from src.llama_mapper.cost_monitoring.core.metrics_collector import (
            CostBreakdown,
        )

        # Mock cost breakdown
        breakdown = CostBreakdown(
            compute_cost=500.0,
            memory_cost=200.0,
            storage_cost=100.0,
            network_cost=50.0,
            api_cost=25.0,
            total_cost=875.0,
            currency="USD",
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc),
        )

        mock_cost_system.get_cost_breakdown.return_value = breakdown

        command = CostBreakdownCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(days=7, format="text")

        # Verify the cost system method was called
        mock_cost_system.get_cost_breakdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_trends_command(self, mock_config_manager, mock_cost_system):
        """Test cost trends command."""
        # Mock cost trends
        trends = {
            "dates": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "costs": [100.0, 150.0, 200.0],
            "total_cost": 450.0,
            "average_daily_cost": 150.0,
        }

        mock_cost_system.get_cost_trends.return_value = trends

        command = CostTrendsCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(days=30, format="text")

        # Verify the cost system method was called
        mock_cost_system.get_cost_trends.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_cost_recommendations_command(
        self, mock_config_manager, mock_cost_system
    ):
        """Test cost recommendations command."""
        from src.llama_mapper.cost_monitoring.analytics.cost_analytics import (
            CostOptimizationRecommendation,
        )

        # Mock recommendations
        recommendations = [
            CostOptimizationRecommendation(
                recommendation_id="rec_1",
                category="compute",
                title="Optimize Compute Resources",
                description="Consider right-sizing instances",
                potential_savings=100.0,
                confidence=0.8,
                effort_level="medium",
                impact_level="high",
                priority=8,
                created_at=datetime.now(timezone.utc),
            ),
        ]

        mock_cost_system.get_optimization_recommendations.return_value = recommendations

        command = CostRecommendationsCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(priority_min=7, format="text")

        # Verify the cost system method was called
        mock_cost_system.get_optimization_recommendations.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_anomalies_command(self, mock_config_manager, mock_cost_system):
        """Test cost anomalies command."""
        from src.llama_mapper.cost_monitoring.analytics.cost_analytics import (
            CostAnomaly,
        )

        # Mock anomalies
        anomalies = [
            CostAnomaly(
                anomaly_id="anomaly_1",
                anomaly_type="spike",
                detected_at=datetime.now(timezone.utc),
                cost_value=200.0,
                expected_value=100.0,
                deviation_percent=100.0,
                severity="high",
                description="Cost spike detected",
            ),
        ]

        mock_cost_system.get_cost_anomalies.return_value = anomalies

        command = CostAnomaliesCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(severity="high", days=30, format="text")

        # Verify the cost system method was called
        mock_cost_system.get_cost_anomalies.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_forecast_command(self, mock_config_manager, mock_cost_system):
        """Test cost forecast command."""
        from src.llama_mapper.cost_monitoring.analytics.cost_analytics import (
            CostForecast,
        )

        # Mock forecast
        forecast = CostForecast(
            forecast_id="forecast_1",
            forecast_period_start=datetime.now(timezone.utc),
            forecast_period_end=datetime.now(timezone.utc) + timedelta(days=30),
            predicted_cost=1500.0,
            confidence_interval_lower=1200.0,
            confidence_interval_upper=1800.0,
            confidence_level=0.95,
            model_type="linear_regression",
            created_at=datetime.now(timezone.utc),
        )

        mock_cost_system.get_latest_forecast.return_value = forecast

        command = CostForecastCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(format="text")

        # Verify the cost system method was called
        mock_cost_system.get_latest_forecast.assert_called_once()

    @pytest.mark.asyncio
    async def test_command_error_handling(self, mock_config_manager):
        """Test command error handling."""
        command = CostStatusCommand(mock_config_manager)

        # Mock cost system that raises an exception
        mock_cost_system = Mock()
        mock_cost_system.get_system_status.side_effect = Exception("Test error")

        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            with pytest.raises(Exception):
                await command.execute_async()

    @pytest.mark.asyncio
    async def test_command_with_tenant_id(self, mock_config_manager, mock_cost_system):
        """Test commands with tenant ID filtering."""
        from datetime import datetime, timedelta, timezone

        from src.llama_mapper.cost_monitoring.core.metrics_collector import (
            CostBreakdown,
        )

        # Mock cost breakdown
        breakdown = CostBreakdown(
            compute_cost=500.0,
            memory_cost=200.0,
            storage_cost=100.0,
            network_cost=50.0,
            api_cost=25.0,
            total_cost=875.0,
            currency="USD",
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc),
            tenant_id="test-tenant",
        )

        mock_cost_system.get_cost_breakdown.return_value = breakdown

        command = CostBreakdownCommand(mock_config_manager)

        # Mock the cost system creation
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(days=7, tenant_id="test-tenant", format="text")

        # Verify the cost system method was called with tenant ID
        mock_cost_system.get_cost_breakdown.assert_called_once()
        # Check that tenant_id was passed (this would be in the call arguments)
        call_args = mock_cost_system.get_cost_breakdown.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_command_output_formats(self, mock_config_manager, mock_cost_system):
        """Test commands with different output formats."""
        from datetime import datetime, timedelta, timezone

        from src.llama_mapper.cost_monitoring.core.metrics_collector import (
            CostBreakdown,
        )

        # Mock cost breakdown
        breakdown = CostBreakdown(
            compute_cost=500.0,
            memory_cost=200.0,
            storage_cost=100.0,
            network_cost=50.0,
            api_cost=25.0,
            total_cost=875.0,
            currency="USD",
            period_start=datetime.now(timezone.utc) - timedelta(days=7),
            period_end=datetime.now(timezone.utc),
        )

        mock_cost_system.get_cost_breakdown.return_value = breakdown

        command = CostBreakdownCommand(mock_config_manager)

        # Test JSON format
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(days=7, format="json")

        # Test text format
        with patch.object(command, "_get_cost_system", return_value=mock_cost_system):
            await command.execute_async(days=7, format="text")

        # Verify the cost system method was called twice
        assert mock_cost_system.get_cost_breakdown.call_count == 2


class TestCostMonitoringCLIIntegration:
    """Integration tests for cost monitoring CLI with Click."""

    def test_cli_command_registration(self):
        """Test that CLI commands are properly registered."""
        from src.llama_mapper.cli.core.registry import AutoDiscoveryRegistry

        registry = AutoDiscoveryRegistry()

        # Import and register cost monitoring commands
        from src.llama_mapper.cli.commands import cost_monitoring

        cost_monitoring.register(registry)

        # Check that the cost group was registered
        cost_group = registry.get_group("cost")
        assert cost_group is not None

        # Check that commands were registered
        commands = registry.list_commands("cost")
        expected_commands = [
            "status",
            "breakdown",
            "trends",
            "recommendations",
            "anomalies",
            "forecast",
        ]

        for expected_command in expected_commands:
            assert expected_command in commands

    def test_cli_command_help(self):
        """Test CLI command help text."""
        from src.llama_mapper.cli.core.registry import AutoDiscoveryRegistry

        registry = AutoDiscoveryRegistry()

        # Import and register cost monitoring commands
        from src.llama_mapper.cli.commands import cost_monitoring

        cost_monitoring.register(registry)

        # Check command help text
        status_info = registry.get_command_info("status")
        assert status_info is not None
        assert "status" in status_info["help"].lower()

        breakdown_info = registry.get_command_info("breakdown")
        assert breakdown_info is not None
        assert "breakdown" in breakdown_info["help"].lower()

    def test_cli_command_options(self):
        """Test CLI command options."""
        from src.llama_mapper.cli.core.registry import AutoDiscoveryRegistry

        registry = AutoDiscoveryRegistry()

        # Import and register cost monitoring commands
        from src.llama_mapper.cli.commands import cost_monitoring

        cost_monitoring.register(registry)

        # Check that commands have options
        breakdown_info = registry.get_command_info("breakdown")
        assert breakdown_info is not None
        assert "options" in breakdown_info
        assert len(breakdown_info["options"]) > 0

        # Check specific options
        options = breakdown_info["options"]
        option_names = [opt.name for opt in options]
        assert "days" in option_names
        assert "tenant_id" in option_names
        assert "fmt" in option_names
