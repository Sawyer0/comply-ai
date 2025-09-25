"""
Main CLI entry point for Mapper Service.

This module provides the main CLI group and coordinates all command modules.
Single Responsibility: CLI command coordination and entry point.
"""

import click

from .mapping_commands import mapping
from .model_commands import model
from .service_commands import service

# from .tenant_commands import tenant  # Temporarily disabled due to shared module issue


@click.group()
@click.version_option(version="1.0.0", prog_name="mapper-service")
def cli():
    """Mapper Service CLI - Privacy-first detector output mapping to canonical taxonomy."""
    pass


# Register command groups
cli.add_command(mapping)
cli.add_command(model)
cli.add_command(service)
# cli.add_command(tenant)  # Temporarily disabled due to shared module issue


if __name__ == "__main__":
    cli()
