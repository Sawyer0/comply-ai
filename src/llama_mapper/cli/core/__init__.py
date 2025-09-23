"""Core CLI components and base classes."""

from .base import (
    BaseCommand,
    AsyncCommand,
    CLIError,
    OutputFormatter,
    command_decorator,
    async_command_decorator,
    validate_file_path,
    validate_output_path,
    get_config_manager_from_context,
    format_success_message,
    format_error_message,
    format_warning_message,
)
from .registry import (
    CommandRegistry,
    AutoDiscoveryRegistry,
    registry,
)
from .plugins import (
    PluginManager,
    PluginInterface,
    create_plugin_interface,
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
