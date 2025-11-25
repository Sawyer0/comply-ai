"""
Service management CLI commands for Mapper Service.

Single Responsibility: Handle service-level operations via CLI.
Provides commands for service health, configuration, and monitoring.
"""

import asyncio
import json
import sys
from typing import Optional

import click

from ..core.mapper import CoreMapper
from ..config.settings import MapperSettings
from ..infrastructure.taxonomy_adapter import (
    SharedCanonicalTaxonomyAdapter,
    SharedFrameworkMappingAdapter,
)
from ..infrastructure.model_inference_adapter import SharedModelInferenceAdapter
from ..monitoring import HealthMonitor, MetricsCollector, PerformanceTracker


@click.group()
def service():
    """Service management - health checks, configuration, and monitoring."""
    pass


@service.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
@click.option(
    "--component",
    "-c",
    help="Check specific component (core_mapper, model_server, etc.)",
)
def health(output_format: str, component: Optional[str]):
    """Check service health status.

    Examples:
        mapper service health
        mapper service health -c model_server --format json
    """

    async def _health():
        try:
            settings = MapperSettings()
            mapper = CoreMapper(
                settings,
                canonical_taxonomy_port=SharedCanonicalTaxonomyAdapter(),
                framework_mapping_port=SharedFrameworkMappingAdapter(),
                model_inference_port=SharedModelInferenceAdapter(settings),
            )

            # Get health status
            health_status = await mapper.health_check()

            if component:
                # Show specific component
                if component not in health_status.get("components", {}):
                    click.echo(f"❌ Component '{component}' not found")
                    available = list(health_status.get("components", {}).keys())
                    click.echo(f"Available components: {', '.join(available)}")
                    sys.exit(1)

                comp_status = health_status["components"][component]

                if output_format == "json":
                    click.echo(json.dumps({component: comp_status}, indent=2))
                else:
                    status_icon = "✓" if comp_status else "❌"
                    click.echo(
                        f"{status_icon} {component}: {'Healthy' if comp_status else 'Unhealthy'}"
                    )

            else:
                # Show overall health
                if output_format == "json":
                    click.echo(json.dumps(health_status, indent=2))
                else:
                    overall_status = health_status.get("status", "unknown")
                    status_icon = "✓" if overall_status == "healthy" else "❌"

                    click.echo("Service Health Status")
                    click.echo("=" * 40)
                    click.echo(f"Overall: {status_icon} {overall_status.title()}")
                    click.echo("")

                    click.echo("Components:")
                    for comp_name, comp_status in health_status.get(
                        "components", {}
                    ).items():
                        comp_icon = "✓" if comp_status else "❌"
                        click.echo(
                            f"  {comp_icon} {comp_name}: {'Healthy' if comp_status else 'Unhealthy'}"
                        )

                    # Show model server details if available
                    if "model_server" in health_status:
                        click.echo(
                            f"\nModel Server: {'✓ Available' if health_status['model_server'] else '❌ Unavailable'}"
                        )

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        finally:
            if "mapper" in locals():
                await mapper.shutdown()

    asyncio.run(_health())


@service.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
def config(output_format: str):
    """Show current service configuration.

    Examples:
        mapper service config
        mapper service config --format json
    """

    try:
        settings = MapperSettings()

        if output_format == "json":
            # Convert settings to dict (simplified)
            config_dict = {
                "model_backend": settings.model_backend,
                "model_path": settings.model_path,
                "confidence_threshold": settings.confidence_threshold,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_new_tokens": settings.max_new_tokens,
                "detector_configs_path": str(settings.detector_configs_path),
                "frameworks_path": str(settings.frameworks_path),
                "model_version": settings.model_version,
                "taxonomy_version": settings.taxonomy_version,
            }
            click.echo(json.dumps(config_dict, indent=2))

        elif output_format == "yaml":
            import yaml

            config_dict = {
                "model_backend": settings.model_backend,
                "model_path": settings.model_path,
                "confidence_threshold": settings.confidence_threshold,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_new_tokens": settings.max_new_tokens,
                "detector_configs_path": str(settings.detector_configs_path),
                "frameworks_path": str(settings.frameworks_path),
                "model_version": settings.model_version,
                "taxonomy_version": settings.taxonomy_version,
            }
            click.echo(yaml.dump(config_dict, default_flow_style=False))

        else:  # table format
            click.echo("Service Configuration")
            click.echo("=" * 50)
            click.echo(f"Model Backend: {settings.model_backend}")
            click.echo(f"Model Path: {settings.model_path}")
            click.echo(f"Confidence Threshold: {settings.confidence_threshold}")
            click.echo(f"Temperature: {settings.temperature}")
            click.echo(f"Top P: {settings.top_p}")
            click.echo(f"Max New Tokens: {settings.max_new_tokens}")
            click.echo(f"Detector Configs: {settings.detector_configs_path}")
            click.echo(f"Frameworks Path: {settings.frameworks_path}")
            click.echo(f"Model Version: {settings.model_version}")
            click.echo(f"Taxonomy Version: {settings.taxonomy_version}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@service.command()
@click.option("--hours", "-h", type=int, default=1, help="Hours of metrics to show")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def metrics(hours: int, output_format: str):
    """Show service performance metrics.

    Examples:
        mapper service metrics
        mapper service metrics -h 24 --format json
    """

    # This is a simplified version - in a real implementation,
    # you'd connect to the actual metrics collector
    click.echo("Service Metrics (Simulated)")
    click.echo("=" * 40)
    click.echo(f"Time Range: Last {hours} hour(s)")
    click.echo("")

    if output_format == "json":
        metrics_data = {
            "time_range_hours": hours,
            "total_requests": 1250,
            "successful_requests": 1198,
            "failed_requests": 52,
            "success_rate": 95.84,
            "avg_response_time_ms": 145.2,
            "p95_response_time_ms": 320.1,
            "p99_response_time_ms": 450.7,
        }
        click.echo(json.dumps(metrics_data, indent=2))

    else:  # table format
        click.echo("Request Metrics:")
        click.echo(f"  Total Requests: 1,250")
        click.echo(f"  Successful: 1,198 (95.84%)")
        click.echo(f"  Failed: 52 (4.16%)")
        click.echo("")
        click.echo("Performance Metrics:")
        click.echo(f"  Avg Response Time: 145.2ms")
        click.echo(f"  P95 Response Time: 320.1ms")
        click.echo(f"  P99 Response Time: 450.7ms")


@service.command()
@click.option("--port", "-p", type=int, default=8000, help="Port to run the service on")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind the service to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def start(port: int, host: str, reload: bool):
    """Start the mapper service.

    Examples:
        mapper service start
        mapper service start -p 8080 --reload
    """

    try:
        import uvicorn

        click.echo(f"Starting Mapper Service on {host}:{port}")
        if reload:
            click.echo("Auto-reload enabled (development mode)")

        # Import the FastAPI app
        from ..main import app

        uvicorn.run(
            "mapper.main:app", host=host, port=port, reload=reload, log_level="info"
        )

    except ImportError:
        click.echo(
            "Error: uvicorn not installed. Install with: pip install uvicorn", err=True
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting service: {e}", err=True)
        sys.exit(1)


@service.command()
def version():
    """Show service version information."""

    click.echo("Mapper Service")
    click.echo("=" * 20)
    click.echo("Version: 1.0.0")
    click.echo("API Version: v1")
    click.echo("Build: production")

    try:
        settings = MapperSettings()
        click.echo(f"Model Version: {settings.model_version}")
        click.echo(f"Taxonomy Version: {settings.taxonomy_version}")
    except Exception:
        click.echo("Model Version: unknown")
        click.echo("Taxonomy Version: unknown")
