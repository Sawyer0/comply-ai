"""Cost monitoring and autoscaling CLI commands."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import click

from ...cost_monitoring import CostMonitoringFactory
from ...cost_monitoring.cost_monitoring_system import CostMonitoringSystem
from ..core import AsyncCommand, BaseCommand, CLIError
from ..decorators.common import handle_errors, timing
from ..utils import display_success, format_output


class CostMonitoringCommand(AsyncCommand):
    """Base command for cost monitoring operations."""

    def __init__(self, config_manager):
        super().__init__(config_manager)
        self.cost_system = None

    async def _get_cost_system(self) -> CostMonitoringSystem:
        """Get or create the cost monitoring system."""
        if self.cost_system is None:
            # Create default configuration
            cost_config = CostMonitoringFactory.create_default_config()
            self.cost_system = CostMonitoringSystem(cost_config)
            await self.cost_system.start()
        return self.cost_system


class CostStatusCommand(CostMonitoringCommand):
    """Get cost monitoring system status."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the status command."""
        cost_system = await self._get_cost_system()

        # Get system status
        status = cost_system.get_system_status()

        # Get health check
        health = await cost_system.health_check()

        # Display status
        click.echo("Cost Monitoring System Status")
        click.echo("=" * 35)
        click.echo(f"Running: {status['running']}")
        click.echo(f"Uptime: {status['uptime_seconds']:.0f} seconds")
        click.echo(f"Overall Health: {health['overall_health']}")

        click.echo("\nComponents:")
        for component, info in health["components"].items():
            status_icon = "✓" if info["status"] == "healthy" else "✗"
            click.echo(f"  {status_icon} {component}: {info['status']}")

        click.echo("\nBudget Limits:")
        for limit_type, value in status["budget_limits"].items():
            click.echo(f"  {limit_type}: ${value:.2f}")

        display_success("Status retrieved successfully")


class CostBreakdownCommand(CostMonitoringCommand):
    """Get cost breakdown for a time period."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the breakdown command."""
        days = kwargs.get("days", 7)
        tenant_id = kwargs.get("tenant_id")
        format_type = kwargs.get("format", "json")

        cost_system = await self._get_cost_system()

        # Calculate time period
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Get cost breakdown
        breakdown = cost_system.get_cost_breakdown(start_time, end_time, tenant_id)

        # Display results
        if format_type == "json":
            format_output(breakdown.model_dump(), "json")
        else:
            click.echo(f"Cost Breakdown (Last {days} days)")
            click.echo("=" * 30)
            click.echo(f"Total Cost: ${breakdown.total_cost:.2f}")
            click.echo(f"Compute Cost: ${breakdown.compute_cost:.2f}")
            click.echo(f"Memory Cost: ${breakdown.memory_cost:.2f}")
            click.echo(f"Storage Cost: ${breakdown.storage_cost:.2f}")
            click.echo(f"Network Cost: ${breakdown.network_cost:.2f}")
            click.echo(f"API Cost: ${breakdown.api_cost:.2f}")
            click.echo(f"Currency: {breakdown.currency}")

        display_success("Cost breakdown retrieved successfully")


class CostTrendsCommand(CostMonitoringCommand):
    """Get cost trends over time."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the trends command."""
        days = kwargs.get("days", 30)
        format_type = kwargs.get("format", "json")

        cost_system = await self._get_cost_system()

        # Get cost trends
        trends = cost_system.get_cost_trends(days)

        # Display results
        if format_type == "json":
            format_output(trends, "json")
        else:
            click.echo(f"Cost Trends (Last {days} days)")
            click.echo("=" * 25)
            click.echo(f"Total Cost: ${trends['total_cost']:.2f}")
            click.echo(f"Average Daily Cost: ${trends['average_daily_cost']:.2f}")
            click.echo(f"Number of Data Points: {len(trends['costs'])}")

            if trends["costs"]:
                click.echo(f"Peak Daily Cost: ${max(trends['costs']):.2f}")
                click.echo(f"Lowest Daily Cost: ${min(trends['costs']):.2f}")

        display_success("Cost trends retrieved successfully")


class CostRecommendationsCommand(CostMonitoringCommand):
    """Get cost optimization recommendations."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the recommendations command."""
        category = kwargs.get("category")
        priority_min = kwargs.get("priority_min", 1)
        tenant_id = kwargs.get("tenant_id")
        format_type = kwargs.get("format", "json")

        cost_system = await self._get_cost_system()

        # Get recommendations
        recommendations = cost_system.get_optimization_recommendations(
            category, priority_min, tenant_id
        )

        # Display results
        if format_type == "json":
            format_output([rec.model_dump() for rec in recommendations], "json")
        else:
            click.echo("Cost Optimization Recommendations")
            click.echo("=" * 35)

            if not recommendations:
                click.echo("No recommendations available")
                return

            total_savings = sum(rec.potential_savings for rec in recommendations)
            click.echo(f"Total Potential Savings: ${total_savings:.2f}")
            click.echo(f"Number of Recommendations: {len(recommendations)}")

            for i, rec in enumerate(recommendations, 1):
                click.echo(f"\n{i}. {rec.title}")
                click.echo(f"   Category: {rec.category}")
                click.echo(f"   Priority: {rec.priority}/10")
                click.echo(f"   Potential Savings: ${rec.potential_savings:.2f}")
                click.echo(f"   Confidence: {rec.confidence:.1%}")
                click.echo(f"   Effort: {rec.effort_level}")
                click.echo(f"   Impact: {rec.impact_level}")
                click.echo(f"   Description: {rec.description}")

        display_success("Recommendations retrieved successfully")


class CostAnomaliesCommand(CostMonitoringCommand):
    """Get cost anomalies."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the anomalies command."""
        severity = kwargs.get("severity")
        days = kwargs.get("days", 30)
        tenant_id = kwargs.get("tenant_id")
        format_type = kwargs.get("format", "json")

        cost_system = await self._get_cost_system()

        # Get anomalies
        anomalies = cost_system.get_cost_anomalies(severity, days, tenant_id)

        # Display results
        if format_type == "json":
            format_output([anomaly.model_dump() for anomaly in anomalies], "json")
        else:
            click.echo(f"Cost Anomalies (Last {days} days)")
            click.echo("=" * 25)

            if not anomalies:
                click.echo("No anomalies detected")
                return

            click.echo(f"Total Anomalies: {len(anomalies)}")

            # Count by severity
            severity_counts = {}
            for anomaly in anomalies:
                severity_counts[anomaly.severity] = (
                    severity_counts.get(anomaly.severity, 0) + 1
                )

            click.echo("\nBy Severity:")
            for sev, count in severity_counts.items():
                click.echo(f"  {sev}: {count}")

            # Show recent anomalies
            click.echo("\nRecent Anomalies:")
            for anomaly in anomalies[:5]:  # Show last 5
                click.echo(
                    f"  • {anomaly.anomaly_type}: ${anomaly.cost_value:.2f} "
                    f"(expected ${anomaly.expected_value:.2f}) - {anomaly.severity}"
                )

        display_success("Anomalies retrieved successfully")


class CostForecastCommand(CostMonitoringCommand):
    """Get cost forecast."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the forecast command."""
        tenant_id = kwargs.get("tenant_id")
        format_type = kwargs.get("format", "json")

        cost_system = await self._get_cost_system()

        # Get forecast
        forecast = cost_system.get_latest_forecast(tenant_id)

        # Display results
        if format_type == "json":
            if forecast:
                format_output(forecast.model_dump(), "json")
            else:
                click.echo("{}")
        else:
            click.echo("Cost Forecast")
            click.echo("=" * 15)

            if not forecast:
                click.echo("No forecast available")
                return

            click.echo(
                f"Forecast Period: {forecast.forecast_period_start.date()} to {forecast.forecast_period_end.date()}"
            )
            click.echo(f"Predicted Cost: ${forecast.predicted_cost:.2f}")
            click.echo(f"Confidence Level: {forecast.confidence_level:.1%}")
            click.echo(
                f"Confidence Interval: ${forecast.confidence_interval_lower:.2f} - ${forecast.confidence_interval_upper:.2f}"
            )
            click.echo(f"Model Type: {forecast.model_type}")
            click.echo(f"Created: {forecast.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        display_success("Forecast retrieved successfully")


def register(registry) -> None:
    """Register cost monitoring commands with the new registry system."""
    # Register command group
    cost_group = registry.register_group(
        "cost", "Cost monitoring and autoscaling commands"
    )

    # Register status command
    registry.register_command(
        "status",
        CostStatusCommand,
        group="cost",
        help="Get cost monitoring system status",
    )

    # Register breakdown command
    registry.register_command(
        "breakdown",
        CostBreakdownCommand,
        group="cost",
        help="Get cost breakdown for a time period",
        options=[
            click.Option(
                ["--days"], type=int, default=7, help="Number of days to analyze"
            ),
            click.Option(["--tenant-id"], help="Tenant ID to filter by"),
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["json", "text"]),
                default="json",
                help="Output format",
            ),
        ],
    )

    # Register trends command
    registry.register_command(
        "trends",
        CostTrendsCommand,
        group="cost",
        help="Get cost trends over time",
        options=[
            click.Option(
                ["--days"], type=int, default=30, help="Number of days to analyze"
            ),
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["json", "text"]),
                default="json",
                help="Output format",
            ),
        ],
    )

    # Register recommendations command
    registry.register_command(
        "recommendations",
        CostRecommendationsCommand,
        group="cost",
        help="Get cost optimization recommendations",
        options=[
            click.Option(["--category"], help="Filter by recommendation category"),
            click.Option(
                ["--priority-min"], type=int, default=1, help="Minimum priority level"
            ),
            click.Option(["--tenant-id"], help="Tenant ID to filter by"),
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["json", "text"]),
                default="json",
                help="Output format",
            ),
        ],
    )

    # Register anomalies command
    registry.register_command(
        "anomalies",
        CostAnomaliesCommand,
        group="cost",
        help="Get cost anomalies",
        options=[
            click.Option(
                ["--severity"],
                type=click.Choice(["low", "medium", "high", "critical"]),
                help="Filter by severity",
            ),
            click.Option(
                ["--days"], type=int, default=30, help="Number of days to analyze"
            ),
            click.Option(["--tenant-id"], help="Tenant ID to filter by"),
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["json", "text"]),
                default="json",
                help="Output format",
            ),
        ],
    )

    # Register forecast command
    registry.register_command(
        "forecast",
        CostForecastCommand,
        group="cost",
        help="Get cost forecast",
        options=[
            click.Option(["--tenant-id"], help="Tenant ID to filter by"),
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["json", "text"]),
                default="json",
                help="Output format",
            ),
        ],
    )
