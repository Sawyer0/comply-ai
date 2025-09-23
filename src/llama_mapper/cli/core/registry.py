"""Command registry system for dynamic CLI command registration."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import click

from .base import BaseCommand, AsyncCommand


class CommandRegistry:
    """Registry for managing CLI commands."""
    
    def __init__(self) -> None:
        self._commands: Dict[str, Dict[str, Any]] = {}
        self._groups: Dict[str, click.Group] = {}
    
    def register_command(
        self,
        name: str,
        command_class: Type[Union[BaseCommand, AsyncCommand]],
        group: Optional[str] = None,
        help: Optional[str] = None,
        options: Optional[List[Any]] = None,
        **click_options: Any,
    ) -> None:
        """Register a command class with the registry.
        
        Args:
            name: Command name
            command_class: Command class to register
            group: Optional group name to register under
            help: Help text for the command
            options: List of Click options to apply to the command
            **click_options: Additional Click options (deprecated, use options parameter)
        """
        if group and group not in self._groups:
            self._groups[group] = click.Group(name=group)
        
        # Handle options parameter
        command_options = options or []
        
        # Handle legacy click_options parameter
        if click_options:
            # Convert click_options dict to list of options
            for key, value in click_options.items():
                if key == "help":
                    help = value
                else:
                    # This is a legacy approach, skip for now
                    pass
        
        command_info = {
            "class": command_class,
            "group": group,
            "options": command_options,
            "help": help,
        }
        
        self._commands[name] = command_info
    
    def register_group(
        self,
        name: str,
        description: Optional[str] = None,
        **group_options: Any,
    ) -> click.Group:
        """Register a command group.
        
        Args:
            name: Group name
            description: Group description
            **group_options: Additional group options
            
        Returns:
            The created Click group
        """
        group = click.Group(name=name, help=description, **group_options)
        self._groups[name] = group
        return group
    
    def get_group(self, name: str) -> Optional[click.Group]:
        """Get a registered group by name."""
        return self._groups.get(name)
    
    def get_command_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get command information by name."""
        return self._commands.get(name)
    
    def list_commands(self, group: Optional[str] = None) -> List[str]:
        """List all registered commands, optionally filtered by group."""
        if group:
            return [
                name for name, info in self._commands.items()
                if info.get("group") == group
            ]
        return list(self._commands.keys())
    
    def list_groups(self) -> List[str]:
        """List all registered groups."""
        return list(self._groups.keys())
    
    def create_click_command(
        self,
        name: str,
        description: Optional[str] = None,
    ) -> click.Command:
        """Create a Click command from a registered command class."""
        command_info = self.get_command_info(name)
        if not command_info:
            raise ValueError(f"Command '{name}' not found in registry")
        
        command_class = command_info["class"]
        click_options = command_info.get("options", [])
        
        # Create the command function that properly executes the BaseCommand
        @click.pass_context
        def base_command_func(ctx: click.Context, **kwargs: Any) -> None:
            # Get config manager from context
            from ..utils import get_config_manager
            config_manager = get_config_manager(ctx)
            
            # Instantiate and execute the command
            command_instance = command_class(config_manager)
            command_instance.execute(**kwargs)
        
        # Start with the base function
        command_func: Any = base_command_func
        
        # Apply options in reverse order (decorators are applied bottom-up)
        for option_config in reversed(click_options):
            if isinstance(option_config, click.Option):
                # For click.Option objects, we need to create a new option decorator
                # Extract the basic parameters that are safe to use
                option_params = {
                    'help': getattr(option_config, 'help', None),
                    'default': getattr(option_config, 'default', None),
                    'type': getattr(option_config, 'type', None),
                    'is_flag': getattr(option_config, 'is_flag', False),
                    'multiple': getattr(option_config, 'multiple', False),
                    'required': getattr(option_config, 'required', False),
                }
                
                # Remove None values
                option_params = {k: v for k, v in option_params.items() if v is not None}
                
                # Create the option decorator
                option_decorator = click.option(*option_config.opts, **option_params)
                command_func = option_decorator(command_func)
            elif isinstance(option_config, dict):
                # Handle dict-based option config
                command_func = click.option(**option_config)(command_func)
            elif callable(option_config):
                # Handle callable option configs (decorators)
                command_func = option_config(command_func)
            else:
                # Skip invalid option configs
                continue
        
        # Create Click command
        # Extract parameters from the decorated function
        params = getattr(command_func, '__click_params__', [])
        
        return click.Command(
            name=name,
            callback=command_func,
            params=params,
            help=description or command_info.get("help", f"Execute {name} command"),
        )
    
    def attach_to_main(self, main_group: click.Group) -> None:
        """Attach all registered commands and groups to the main CLI group."""
        # First, attach all groups
        for group_name, group in self._groups.items():
            main_group.add_command(group)
        
        # Then, attach commands to their respective groups or main
        for command_name, command_info in self._commands.items():
            group_name = command_info.get("group")
            
            if group_name and group_name in self._groups:
                # Add to specific group
                click_command = self.create_click_command(command_name)
                self._groups[group_name].add_command(click_command)
            else:
                # Add to main group
                click_command = self.create_click_command(command_name)
                main_group.add_command(click_command)


class AutoDiscoveryRegistry(CommandRegistry):
    """Registry that can automatically discover commands from modules."""
    
    def discover_commands_from_module(
        self,
        module_path: str,
        base_class: Type[Union[BaseCommand, AsyncCommand]] = BaseCommand,
    ) -> None:
        """Discover and register commands from a module.
        
        Args:
            module_path: Path to the module (e.g., 'src.llama_mapper.cli.commands')
            base_class: Base class to filter commands by
        """
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ValueError(f"Could not import module {module_path}: {e}")
        
        # Find all classes that inherit from base_class
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, base_class)
                and obj is not base_class
                and not inspect.isabstract(obj)
            ):
                # Extract command name from class name
                command_name = self._extract_command_name(name)
                
                # Register the command
                self.register_command(command_name, obj)
    
    def discover_commands_from_directory(
        self,
        directory_path: Path,
        base_class: Type[Union[BaseCommand, AsyncCommand]] = BaseCommand,
    ) -> None:
        """Discover and register commands from a directory of modules.
        
        Args:
            directory_path: Path to directory containing command modules
            base_class: Base class to filter commands by
        """
        if not directory_path.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        # Find all Python files in the directory
        for py_file in directory_path.glob("*.py"):
            if py_file.name.startswith("__"):
                continue
            
            # Convert file path to module path
            module_name = py_file.stem
            module_path = f"{directory_path.parent.name}.{directory_path.name}.{module_name}"
            
            try:
                self.discover_commands_from_module(module_path, base_class)
            except Exception as e:
                # Log warning but continue with other modules
                print(f"Warning: Could not discover commands from {module_path}: {e}")
    
    @staticmethod
    def _extract_command_name(class_name: str) -> str:
        """Extract command name from class name.
        
        Converts PascalCase to kebab-case.
        Example: 'AnalyzeCommand' -> 'analyze'
        """
        # Remove 'Command' suffix if present
        if class_name.endswith("Command"):
            class_name = class_name[:-7]
        
        # Convert PascalCase to kebab-case
        import re
        # Insert hyphens before uppercase letters (except the first one)
        name = re.sub(r'(?<!^)(?=[A-Z])', '-', class_name)
        return name.lower()


# Global registry instance
registry = AutoDiscoveryRegistry()
