"""Core CLI components and base classes."""

from .base import (
    AsyncCommand,
    BaseCommand,
    CLIError,
    OutputFormatter,
    async_command_decorator,
    command_decorator,
    format_error_message,
    format_success_message,
    format_warning_message,
    get_config_manager_from_context,
    validate_file_path,
    validate_output_path,
)
from .plugins import (
    PluginInterface,
    PluginManager,
    create_plugin_interface,
)
from .registry import (
    AutoDiscoveryRegistry,
    CommandRegistry,
    registry,
)

__all__ = [
    # Base classes
    "BaseCommand",
    "AsyncCommand",
    "CLIError",
    "OutputFormatter",
    # Decorators
    "command_decorator",
    "async_command_decorator",
    # Utilities
    "validate_file_path",
    "validate_output_path",
    "get_config_manager_from_context",
    "format_success_message",
    "format_error_message",
    "format_warning_message",
    # Registry
    "CommandRegistry",
    "AutoDiscoveryRegistry",
    "registry",
    # Plugins
    "PluginManager",
    "PluginInterface",
    "create_plugin_interface",
]
