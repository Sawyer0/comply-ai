"""API client commands for testing and interacting with the Llama Mapper API."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import click
import httpx

from ..core import BaseCommand, CLIError
from ..decorators.common import handle_errors, timing


class ApiTestCommand(BaseCommand):
    """Test API connectivity and basic functionality."""

    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the API test command."""
        endpoint = kwargs.get("endpoint", "/health")
        method = kwargs.get("method", "GET")
        timeout = kwargs.get("timeout", 10)

        api_host = self.config_manager.serving.host
        api_port = self.config_manager.serving.port
        api_url = f"http://{api_host}:{api_port}{endpoint}"

        try:
            with httpx.Client(timeout=timeout) as client:
                if method.upper() == "GET":
                    response = client.get(api_url)
                elif method.upper() == "POST":
                    response = client.post(api_url)
                else:
                    raise CLIError(f"Unsupported HTTP method: {method}")

                click.echo(f"✓ API {method} {api_url}")
                click.echo(f"Status: {response.status_code}")
                click.echo(f"Response: {response.text}")

        except Exception as e:
            raise CLIError(f"API test failed: {e}")


class ApiHealthCommand(BaseCommand):
    """Check API health status."""

    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the API health check command."""
        api_host = self.config_manager.serving.host
        api_port = self.config_manager.serving.port
        api_url = f"http://{api_host}:{api_port}/health"

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(api_url)
                response.raise_for_status()

                health_data = response.json()

                click.echo("API Health Check")
                click.echo("=" * 20)
                click.echo(f"Status: {health_data.get('status', 'unknown')}")
                click.echo(f"Timestamp: {health_data.get('timestamp', 'unknown')}")
                click.echo(f"URL: {api_url}")
                click.echo("✓ API is healthy")

        except Exception as e:
            raise CLIError(f"Health check failed: {e}")


class ApiMetricsCommand(BaseCommand):
    """Get API metrics."""

    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the API metrics command."""
        format_type = kwargs.get("format", "json")
        api_host = self.config_manager.serving.host
        api_port = self.config_manager.serving.port
        api_url = f"http://{api_host}:{api_port}/metrics/summary"

        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(api_url)
                response.raise_for_status()

                metrics = response.json()

                if format_type == "json":
                    click.echo(json.dumps(metrics, indent=2))
                else:
                    click.echo("API Metrics")
                    click.echo("=" * 15)
                    for key, value in metrics.items():
                        click.echo(f"{key}: {value}")

        except Exception as e:
            raise CLIError(f"Failed to retrieve metrics: {e}")


def register(registry) -> None:
    """Register API client commands with the new registry system."""
    # Register command group
    api_client_group = registry.register_group(
        "api-client", "API client commands for testing and interaction"
    )

    # Register the test command
    registry.register_command(
        "test",
        ApiTestCommand,
        group="api-client",
        help="Test API connectivity and basic functionality",
        options=[
            click.Option(
                ["--endpoint"], default="/health", help="API endpoint to test"
            ),
            click.Option(
                ["--method"],
                type=click.Choice(["GET", "POST"]),
                default="GET",
                help="HTTP method to use",
            ),
            click.Option(
                ["--timeout"], type=int, default=10, help="Request timeout in seconds"
            ),
        ],
    )

    # Register health command
    registry.register_command(
        "health",
        ApiHealthCommand,
        group="api-client",
        help="Check API health status",
    )

    # Register metrics command
    registry.register_command(
        "metrics",
        ApiMetricsCommand,
        group="api-client",
        help="Get API metrics",
        options=[
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["json", "text"]),
                default="json",
                help="Output format",
            ),
        ],
    )
