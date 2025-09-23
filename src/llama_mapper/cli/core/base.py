"""Base classes and utilities for CLI commands."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

import click
from pydantic import BaseModel

from ...config.manager import ConfigManager
from ...logging import get_logger

T = TypeVar("T", bound=BaseModel)


class CLIError(Exception):
    """Base exception for CLI-related errors."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class BaseCommand(ABC):
    """Base class for CLI commands with common functionality."""

    def __init__(self, config_manager: ConfigManager) -> None:
        self.config_manager = config_manager
        self.logger = get_logger(self.__class__.__module__)

    @abstractmethod
    def execute(self, **kwargs: Any) -> None:
        """Execute the command with the given parameters."""
        pass

    def handle_error(self, error: Exception, ctx: click.Context) -> None:
        """Handle errors consistently across commands."""
        if isinstance(error, CLIError):
            self.logger.error("CLI error", error=str(error), exit_code=error.exit_code)
            click.echo(f"✗ {error}")
            ctx.exit(error.exit_code)
        else:
            self.logger.error("Unexpected error", error=str(error))
            click.echo(f"✗ Unexpected error: {error}")
            ctx.exit(1)


class AsyncCommand(BaseCommand):
    """Base class for async CLI commands."""

    @abstractmethod
    async def execute_async(self, **kwargs: Any) -> None:
        """Execute the async command with the given parameters."""
        pass

    def execute(self, **kwargs: Any) -> None:
        """Execute the async command by running it in an event loop."""
        try:
            asyncio.run(self.execute_async(**kwargs))
        except Exception as e:
            # Re-raise to be handled by the decorator
            raise e


class OutputFormatter:
    """Utility class for formatting command output."""

    @staticmethod
    def format_json(data: Union[Dict[str, Any], BaseModel], indent: int = 2) -> str:
        """Format data as JSON."""
        if isinstance(data, BaseModel):
            data = data.model_dump()
        
        # Custom JSON encoder to handle datetime objects
        class DateTimeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return super().default(obj)
        
        return json.dumps(data, indent=indent, cls=DateTimeEncoder)

    @staticmethod
    def format_yaml(data: Union[Dict[str, Any], BaseModel]) -> str:
        """Format data as YAML."""
        try:
            import yaml
        except ImportError:
            raise CLIError("PyYAML is required for YAML output format")

        if isinstance(data, BaseModel):
            data = data.model_dump()
        return yaml.dump(data, default_flow_style=False)

    @staticmethod
    def format_table(
        headers: list[str],
        rows: list[list[str]],
        max_width: int = 80,
    ) -> str:
        """Format data as a table."""
        if not headers or not rows:
            return ""

        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Limit total width
        total_width = sum(col_widths) + (len(headers) - 1) * 3  # 3 for separators
        if total_width > max_width:
            # Proportionally reduce column widths
            scale = max_width / total_width
            col_widths = [int(width * scale) for width in col_widths]

        # Build table
        lines = []
        separator = " | ".join("-" * width for width in col_widths)
        header_line = " | ".join(header.ljust(width) for header, width in zip(headers, col_widths))
        
        lines.append(header_line)
        lines.append(separator)
        
        for row in rows:
            row_line = " | ".join(str(cell)[:width].ljust(width) for cell, width in zip(row, col_widths))
            lines.append(row_line)

        return "\n".join(lines)

    @staticmethod
    def save_output(content: str, output_path: Optional[Path]) -> None:
        """Save content to file or print to stdout."""
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            click.echo(f"✓ Output saved to: {output_path}")
        else:
            click.echo(content)


def command_decorator(
    command_class: type[BaseCommand],
    **click_options: Any,
) -> Callable:
    """Decorator to create Click commands from BaseCommand classes."""
    
    def decorator(func: Callable) -> Callable:
        @click.pass_context
        def wrapper(ctx: click.Context, **kwargs: Any) -> None:
            try:
                config_manager = ctx.obj["config"]
                command = command_class(config_manager)
                command.execute(**kwargs)
            except Exception as e:
                command = command_class(ctx.obj["config"])
                command.handle_error(e, ctx)
        
        # Apply Click options
        for option_name, option_config in click_options.items():
            wrapper = click.option(option_name, **option_config)(wrapper)
        
        return wrapper
    
    return decorator


def async_command_decorator(
    command_class: type[AsyncCommand],
    **click_options: Any,
) -> Callable:
    """Decorator to create Click commands from AsyncCommand classes."""
    
    def decorator(func: Callable) -> Callable:
        @click.pass_context
        def wrapper(ctx: click.Context, **kwargs: Any) -> None:
            try:
                config_manager = ctx.obj["config"]
                command = command_class(config_manager)
                command.execute(**kwargs)
            except Exception as e:
                command = command_class(ctx.obj["config"])
                command.handle_error(e, ctx)
        
        # Apply Click options
        for option_name, option_config in click_options.items():
            wrapper = click.option(option_name, **option_config)(wrapper)
        
        return wrapper
    
    return decorator


def validate_file_path(path: str, must_exist: bool = True) -> Path:
    """Validate and return a Path object."""
    path_obj = Path(path)
    
    if must_exist and not path_obj.exists():
        raise CLIError(f"File not found: {path}")
    
    return path_obj


def validate_output_path(path: str) -> Path:
    """Validate and return an output Path object."""
    path_obj = Path(path)
    
    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    return path_obj


def get_config_manager_from_context(ctx: click.Context) -> ConfigManager:
    """Get ConfigManager from Click context."""
    return ctx.obj["config"]


def format_success_message(message: str) -> str:
    """Format a success message with checkmark."""
    return f"✓ {message}"


def format_error_message(message: str) -> str:
    """Format an error message with X mark."""
    return f"✗ {message}"


def format_warning_message(message: str) -> str:
    """Format a warning message with warning symbol."""
    return f"⚠ {message}"
