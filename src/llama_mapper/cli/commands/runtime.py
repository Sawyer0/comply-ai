"""Runtime control commands."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import click
import httpx

from ...config.manager import ServingConfig
from ...logging import get_logger
from ..core import AsyncCommand, CLIError
from ..decorators.common import handle_errors, timing
from ..utils import get_config_manager


class RuntimeStatusCommand(AsyncCommand):
    """Show comprehensive system status including health, metrics, and configuration."""

    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the runtime status command."""
        fmt = kwargs.get("format", "text")
        include_metrics = kwargs.get("include_metrics", False)
        timeout = kwargs.get("timeout", 5)

        status = await self._get_system_status(include_metrics, timeout)

        if fmt == "json":
            click.echo(json.dumps(status, indent=2))
        elif fmt == "yaml":
            import yaml

            click.echo(yaml.dump(status, default_flow_style=False))
        else:  # text format
            self._display_status_text(status)

    async def _get_system_status(
        self, include_metrics: bool, timeout: int
    ) -> Dict[str, Any]:
        """Collect comprehensive system status."""
        # Basic system info
        status = {
            "timestamp": time.time(),
            "runtime_mode": getattr(self.config_manager.serving, "mode", "hybrid"),
            "configuration": {
                "config_path": str(self.config_manager.config_path),
                "environment": getattr(self.config_manager, "environment", "unknown"),
            },
            "services": {},
            "health": "unknown",
            "metrics": {} if include_metrics else None,
        }

        # Check main API service
        try:
            api_host = self.config_manager.serving.host
            api_port = self.config_manager.serving.port
            api_url = f"http://{api_host}:{api_port}"

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Health check
                try:
                    health_response = await client.get(f"{api_url}/health")
                    if health_response.status_code == 200:
                        status["services"]["main_api"] = {
                            "status": "healthy",
                            "url": api_url,
                            "response_time_ms": health_response.elapsed.total_seconds()
                            * 1000,
                        }
                        status["health"] = "healthy"
                    else:
                        status["services"]["main_api"] = {
                            "status": "unhealthy",
                            "url": api_url,
                            "error": f"HTTP {health_response.status_code}",
                        }
                        status["health"] = "unhealthy"
                except Exception as e:
                    status["services"]["main_api"] = {
                        "status": "unreachable",
                        "url": api_url,
                        "error": str(e),
                    }
                    status["health"] = "unhealthy"

                # Metrics if requested
                if include_metrics:
                    try:
                        metrics_response = await client.get(
                            f"{api_url}/metrics/summary"
                        )
                        if metrics_response.status_code == 200:
                            status["metrics"] = metrics_response.json()
                    except Exception as e:
                        status["metrics"] = {"error": str(e)}

        except Exception as e:
            self.logger.warning("Failed to check API service", error=str(e))
            status["services"]["main_api"] = {
                "status": "error",
                "error": str(e),
            }

        return status

    def _display_status_text(self, status: Dict[str, Any]) -> None:
        """Display status information in human-readable text format."""
        click.echo("Llama Mapper System Status")
        click.echo("=" * 30)

        # Overall health
        health_icon = "✓" if status["health"] == "healthy" else "✗"
        click.echo(f"Overall Health: {health_icon} {status['health'].upper()}")

        # Runtime mode
        click.echo(f"Runtime Mode: {status['runtime_mode']}")

        # Configuration
        config = status["configuration"]
        click.echo(f"Config Path: {config['config_path']}")
        click.echo(f"Environment: {config['environment']}")

        # Services
        click.echo("\nServices:")
        for service_name, service_info in status["services"].items():
            service_status = service_info["status"]
            status_icon = (
                "✓"
                if service_status == "healthy"
                else "✗" if service_status == "unhealthy" else "⚠"
            )
            click.echo(f"  {status_icon} {service_name}: {service_status}")

            if "url" in service_info:
                click.echo(f"    URL: {service_info['url']}")
            if "response_time_ms" in service_info:
                click.echo(
                    f"    Response Time: {service_info['response_time_ms']:.1f}ms"
                )
            if "error" in service_info:
                click.echo(f"    Error: {service_info['error']}")

        click.echo(
            f"\nStatus checked at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['timestamp']))}"
        )


def register(registry) -> None:
    """Register runtime commands with the new registry system."""
    # Register command group
    runtime_group = registry.register_group(
        "runtime", "Runtime controls (kill-switch, modes)"
    )

    # Register the status command using the new system
    registry.register_command(
        "status",
        RuntimeStatusCommand,
        group="runtime",
        help="Show comprehensive system status including health, metrics, and configuration",
        options=[
            click.Option(
                ["--format", "fmt"],
                type=click.Choice(["text", "json", "yaml"]),
                default="text",
                help="Output format for status information",
            ),
            click.Option(
                ["--include-metrics"],
                is_flag=True,
                help="Include detailed metrics in status output",
            ),
            click.Option(
                ["--timeout"],
                type=int,
                default=5,
                help="Timeout in seconds for health checks",
            ),
        ],
    )
