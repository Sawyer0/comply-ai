"""Shared helpers for the CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import click

from ..config import ConfigManager
from .core import CLIError, OutputFormatter


def get_config_manager(ctx: click.Context) -> ConfigManager:
    """Return the config manager stored on the click context."""
    return cast(ConfigManager, ctx.obj["config"])


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise CLIError(f"File not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise CLIError(f"Invalid JSON in {path}: {e}")
    except Exception as e:
        raise CLIError(f"Error reading {path}: {e}")


def save_json_file(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise CLIError(f"Error writing to {path}: {e}")


def format_output(
    data: Union[Dict[str, Any], List[Any]],
    format_type: str = "json",
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Format and output data in the specified format."""
    formatter = OutputFormatter()
    
    if format_type == "json":
        content = formatter.format_json(data)
    elif format_type == "yaml":
        content = formatter.format_yaml(data)
    else:
        raise CLIError(f"Unsupported output format: {format_type}")
    
    formatter.save_output(content, Path(output_path) if output_path else None)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    return click.confirm(message, default=default)


def select_from_list(
    items: List[str],
    prompt: str = "Select an item",
    default: Optional[str] = None,
) -> str:
    """Allow user to select from a list of items."""
    if not items:
        raise CLIError("No items to select from")
    
    if len(items) == 1:
        return items[0]
    
    for i, item in enumerate(items, 1):
        click.echo(f"{i}. {item}")
    
    while True:
        try:
            choice = click.prompt(prompt, default=default)
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(items):
                    return items[index]
            elif choice in items:
                return choice
            click.echo("Invalid selection. Please try again.")
        except (ValueError, KeyboardInterrupt):
            click.echo("Invalid input. Please try again.")


def display_table(
    headers: List[str],
    rows: List[List[str]],
    title: Optional[str] = None,
) -> None:
    """Display data in a formatted table."""
    if title:
        click.echo(f"\n{title}")
        click.echo("=" * len(title))
    
    formatter = OutputFormatter()
    table_content = formatter.format_table(headers, rows)
    click.echo(table_content)


def display_success(message: str) -> None:
    """Display a success message."""
    click.echo(click.style(f"✓ {message}", fg="green"))


def display_error(message: str) -> None:
    """Display an error message."""
    click.echo(click.style(f"✗ {message}", fg="red"))


def display_warning(message: str) -> None:
    """Display a warning message."""
    click.echo(click.style(f"⚠ {message}", fg="yellow"))


def display_info(message: str) -> None:
    """Display an info message."""
    click.echo(click.style(f"ℹ {message}", fg="blue"))


def validate_required_params(params: Dict[str, Any], required: List[str]) -> None:
    """Validate that required parameters are provided."""
    missing = [param for param in required if not params.get(param)]
    if missing:
        raise CLIError(f"Missing required parameters: {', '.join(missing)}")


def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """Validate that a file exists and return the Path object."""
    path = Path(file_path)
    if not path.exists():
        raise CLIError(f"File not found: {path}")
    return path


def validate_directory_exists(dir_path: Union[str, Path]) -> Path:
    """Validate that a directory exists and return the Path object."""
    path = Path(dir_path)
    if not path.exists():
        raise CLIError(f"Directory not found: {path}")
    if not path.is_dir():
        raise CLIError(f"Path is not a directory: {path}")
    return path


def ensure_directory_exists(dir_path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """Get the relative path from base to path."""
    return str(Path(path).relative_to(Path(base)))


def expand_path(path: Union[str, Path]) -> Path:
    """Expand user home directory and environment variables in path."""
    return Path(str(path).expanduser().expandvars()).resolve()
