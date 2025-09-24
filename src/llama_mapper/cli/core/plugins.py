"""Plugin system for CLI commands."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import click

from .base import AsyncCommand, BaseCommand
from .registry import CommandRegistry


class PluginManager:
    """Manager for CLI command plugins."""

    def __init__(self, registry: CommandRegistry) -> None:
        self.registry = registry
        self._loaded_plugins: Dict[str, Any] = {}
        self._plugin_directories: List[Path] = []

    def add_plugin_directory(self, directory: Union[str, Path]) -> None:
        """Add a directory to search for plugins."""
        path = Path(directory)
        if path.exists() and path.is_dir():
            self._plugin_directories.append(path)
        else:
            raise ValueError(f"Plugin directory does not exist: {path}")

    def load_plugin(self, plugin_name: str, plugin_path: Optional[str] = None) -> None:
        """Load a plugin by name or path."""
        if plugin_name in self._loaded_plugins:
            return  # Already loaded

        try:
            if plugin_path:
                # Load from specific path
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                else:
                    raise ImportError(f"Could not load plugin from {plugin_path}")
            else:
                # Load from installed package
                module = importlib.import_module(plugin_name)

            # Register commands from the plugin
            self._register_plugin_commands(plugin_name, module)
            self._loaded_plugins[plugin_name] = module

        except Exception as e:
            raise ImportError(f"Failed to load plugin {plugin_name}: {e}")

    def load_plugins_from_directory(
        self, directory: Optional[Union[str, Path]] = None
    ) -> None:
        """Load all plugins from a directory."""
        if directory:
            search_dir = Path(directory)
        else:
            # Search all registered plugin directories
            for plugin_dir in self._plugin_directories:
                self._load_plugins_from_single_directory(plugin_dir)
            return

        self._load_plugins_from_single_directory(search_dir)

    def _load_plugins_from_single_directory(self, directory: Path) -> None:
        """Load plugins from a single directory."""
        if not directory.exists():
            return

        for item in directory.iterdir():
            if (
                item.is_file()
                and item.suffix == ".py"
                and not item.name.startswith("__")
            ):
                plugin_name = item.stem
                try:
                    self.load_plugin(plugin_name, str(item))
                except Exception as e:
                    print(f"Warning: Could not load plugin {plugin_name}: {e}")
            elif item.is_dir() and not item.name.startswith("__"):
                # Try to load as a package
                plugin_name = item.name
                try:
                    self.load_plugin(plugin_name)
                except Exception as e:
                    print(f"Warning: Could not load plugin package {plugin_name}: {e}")

    def _register_plugin_commands(self, plugin_name: str, module: Any) -> None:
        """Register commands from a plugin module."""
        # Look for command classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, (BaseCommand, AsyncCommand))
                and obj not in (BaseCommand, AsyncCommand)
                and not inspect.isabstract(obj)
            ):
                command_name = self._extract_command_name(name)
                self.registry.register_command(command_name, obj, group=plugin_name)

        # Look for register function
        if hasattr(module, "register"):
            register_func = getattr(module, "register")
            if callable(register_func):
                try:
                    register_func(self.registry)
                except Exception as e:
                    print(
                        f"Warning: Plugin {plugin_name} register function failed: {e}"
                    )

    @staticmethod
    def _extract_command_name(class_name: str) -> str:
        """Extract command name from class name."""
        if class_name.endswith("Command"):
            class_name = class_name[:-7]

        import re

        name = re.sub(r"(?<!^)(?=[A-Z])", "-", class_name)
        return name.lower()

    def list_loaded_plugins(self) -> List[str]:
        """List all loaded plugins."""
        return list(self._loaded_plugins.keys())

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded plugin."""
        if plugin_name not in self._loaded_plugins:
            return None

        module = self._loaded_plugins[plugin_name]
        return {
            "name": plugin_name,
            "module": module,
            "version": getattr(module, "__version__", "unknown"),
            "description": getattr(module, "__doc__", ""),
            "commands": self.registry.list_commands(group=plugin_name),
        }


class PluginInterface:
    """Interface for CLI plugins."""

    def __init__(self, registry: CommandRegistry) -> None:
        self.registry = registry

    def register_command(
        self,
        name: str,
        command_class: Type[Union[BaseCommand, AsyncCommand]],
        **click_options: Any,
    ) -> None:
        """Register a command with the CLI."""
        self.registry.register_command(name, command_class, **click_options)

    def register_group(
        self,
        name: str,
        description: Optional[str] = None,
        **group_options: Any,
    ) -> click.Group:
        """Register a command group."""
        return self.registry.register_group(name, description, **group_options)

    def create_command_decorator(
        self,
        command_class: Type[Union[BaseCommand, AsyncCommand]],
        **click_options: Any,
    ) -> Any:
        """Create a command decorator for the given command class."""
        from .base import async_command_decorator, command_decorator

        if issubclass(command_class, AsyncCommand):
            return async_command_decorator(command_class, **click_options)
        else:
            return command_decorator(command_class, **click_options)


def create_plugin_interface(registry: CommandRegistry) -> PluginInterface:
    """Create a plugin interface for the given registry."""
    return PluginInterface(registry)
