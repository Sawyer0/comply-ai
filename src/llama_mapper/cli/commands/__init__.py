"""Command registration helpers for the CLI."""

from __future__ import annotations

import click

from . import analysis, api_client, auth, config, detectors, logs, metrics, performance, quality, runtime, serve, service_discovery, taxonomy, tenant, versions, weekly_evaluation
from ..core import BaseCommand, CLIError


def register_all(registry) -> None:
    """Register all command groups with the new registry system."""
    # Register commands that have been converted to the new system
    runtime.register(registry)
    api_client.register(registry)
    service_discovery.register(registry)
    performance.register(registry)
    weekly_evaluation.register(registry)
    
    # Temporarily disable legacy commands to debug the new system
    # _register_legacy_commands(registry)


def _register_legacy_commands(registry) -> None:
    """Register legacy commands that haven't been converted to the new system yet."""
    # For now, we'll create a simple bridge that allows legacy commands to work
    # This creates a temporary Click group and registers it with the main CLI
    
    # Create a temporary main group to capture legacy registrations
    temp_main = click.Group()
    
    # Register legacy commands with the temporary group
    analysis.register(temp_main)
    auth.register(temp_main)
    config.register(temp_main)
    detectors.register(temp_main)
    quality.register(temp_main)
    serve.register(temp_main)
    taxonomy.register(temp_main)
    tenant.register(temp_main)
    versions.register(temp_main)
    
    # Convert the temporary group's commands to registry commands
    for command_name, command_obj in temp_main.commands.items():
        if hasattr(command_obj, 'callback'):
            # Create a wrapper command class that can handle the legacy Click command
            class LegacyCommandWrapper(BaseCommand):
                def __init__(self, config_manager, click_command):
                    super().__init__(config_manager)
                    self.click_command = click_command
                
                def execute(self, **kwargs):
                    # Execute the legacy Click command
                    # This is a simplified approach - in production you'd want more sophisticated conversion
                    try:
                        # Create a mock context for the legacy command
                        from click.testing import CliRunner
                        runner = CliRunner()
                        
                        # Get the command's help to show what it would do
                        help_text = getattr(self.click_command, 'help', f"Legacy {self.click_command.name} command")
                        click.echo(f"Legacy command: {self.click_command.name}")
                        click.echo(f"Help: {help_text}")
                        click.echo("Note: This command needs to be converted to the new registry system")
                        
                    except Exception as e:
                        raise CLIError(f"Legacy command execution failed: {e}")
            
            registry.register_command(
                command_name,
                LegacyCommandWrapper,
                help=getattr(command_obj, 'help', f"Legacy {command_name} command")
            )
