"""Sample CLI plugin demonstrating the new architecture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from src.llama_mapper.cli.core import BaseCommand, AsyncCommand, CLIError
from src.llama_mapper.cli.decorators.common import handle_errors, timing, confirm_action
from src.llama_mapper.cli.validators.params import ParameterValidator
from src.llama_mapper.cli.utils import (
    display_success,
    display_error,
    display_warning,
    display_info,
    format_output,
    display_table,
)


class SampleInfoCommand(BaseCommand):
    """Sample command to display system information."""
    
    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the info command."""
        self.logger.info("Displaying system information")
        
        # Gather system information
        info = {
            "config_path": str(self.config_manager.config_path),
            "model_name": getattr(self.config_manager.model, "name", "unknown"),
            "serving_host": getattr(self.config_manager.serving, "host", "unknown"),
            "serving_port": getattr(self.config_manager.serving, "port", "unknown"),
        }
        
        # Display information
        click.echo("System Information")
        click.echo("=" * 20)
        
        for key, value in info.items():
            click.echo(f"{key}: {value}")
        
        display_success("System information displayed")


class SampleAsyncCommand(AsyncCommand):
    """Sample async command demonstrating async functionality."""
    
    @handle_errors
    @timing
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the async command."""
        self.logger.info("Executing async sample command")
        
        # Simulate async work
        import asyncio
        await asyncio.sleep(1)
        
        display_success("Async command completed successfully")


class SampleValidateCommand(BaseCommand):
    """Sample command demonstrating parameter validation."""
    
    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the validate command with parameter validation."""
        self.logger.info("Validating parameters")
        
        # Validate parameters
        validator = ParameterValidator()
        
        # Example validations
        if "tenant_id" in kwargs:
            tenant_id = validator.validate_tenant_id(kwargs["tenant_id"])
            display_success(f"Tenant ID '{tenant_id}' is valid")
        
        if "port" in kwargs:
            port = validator.validate_port(kwargs["port"])
            display_success(f"Port '{port}' is valid")
        
        if "environment" in kwargs:
            env = validator.validate_environment(kwargs["environment"])
            display_success(f"Environment '{env}' is valid")
        
        display_success("All parameters validated successfully")


class SampleTableCommand(BaseCommand):
    """Sample command demonstrating table output."""
    
    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the table command."""
        self.logger.info("Displaying sample table")
        
        # Sample data
        headers = ["Name", "Status", "Last Updated"]
        rows = [
            ["Service A", "Running", "2024-01-15 10:30:00"],
            ["Service B", "Stopped", "2024-01-15 09:15:00"],
            ["Service C", "Running", "2024-01-15 11:45:00"],
        ]
        
        display_table(headers, rows, title="Service Status")
        display_success("Table displayed successfully")


class SampleConfirmCommand(BaseCommand):
    """Sample command demonstrating user confirmation."""
    
    @handle_errors
    @timing
    @confirm_action("Are you sure you want to perform this action?", default=False)
    def execute(self, **kwargs: Any) -> None:
        """Execute the confirm command."""
        self.logger.info("Executing confirmed action")
        
        display_success("Action confirmed and executed")


def register(registry) -> None:
    """Register plugin commands with the CLI registry."""
    
    # Register commands
    registry.register_command("sample-info", SampleInfoCommand)
    registry.register_command("sample-async", SampleAsyncCommand)
    registry.register_command("sample-validate", SampleValidateCommand)
    registry.register_command("sample-table", SampleTableCommand)
    registry.register_command("sample-confirm", SampleConfirmCommand)
    
    # Create a plugin group
    plugin_group = registry.register_group(
        "sample",
        description="Sample plugin commands for demonstration",
    )
    
    # Add commands to the group
    registry.register_command("info", SampleInfoCommand, group="sample")
    registry.register_command("async", SampleAsyncCommand, group="sample")
    registry.register_command("validate", SampleValidateCommand, group="sample")
    registry.register_command("table", SampleTableCommand, group="sample")
    registry.register_command("confirm", SampleConfirmCommand, group="sample")


# Plugin metadata
__version__ = "1.0.0"
__description__ = "Sample CLI plugin demonstrating the new architecture"
__author__ = "Llama Mapper Team"
