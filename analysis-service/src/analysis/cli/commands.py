"""
Main CLI entry point for Analysis Service.

This module provides the main CLI group and coordinates all command modules.
Single Responsibility: CLI command coordination and entry point.
"""

import click

from .tenant_commands import tenant
from .plugin_commands import plugin
from .analysis_commands import analyze
from .service_commands import service


@click.group()
@click.version_option(version="1.0.0", prog_name="analysis-service")
def cli():
    """Analysis Service CLI - Advanced analysis, risk scoring, and compliance intelligence."""
    pass


# Register command groups
cli.add_command(tenant)
cli.add_command(plugin)
cli.add_command(analyze)
cli.add_command(service)


if __name__ == "__main__":
    cli()
