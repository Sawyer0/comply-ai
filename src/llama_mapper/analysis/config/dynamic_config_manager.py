"""
Dynamic Configuration Manager for Analysis Services.

Provides hot-reloading, validation, and change notification
for analysis service configurations with proper error handling.
"""

import asyncio
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import yaml
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ConfigurationChangeEvent:
    """Event representing a configuration change."""
    
    def __init__(self, config_path: str, old_config: Dict[str, Any], 
                 new_config: Dict[str, Any], change_type: str = "update"):
        self.config_path = config_path
        self.old_config = old_config
        self.new_config = new_config
        self.change_type = change_type
        self.timestamp = time.time()


class IConfigurationValidator(ABC):
    """Interface for configuration validators."""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration and return True if valid."""
        pass
    
    @abstractmethod
    def get_validation_errors(self, config: Dict[str, Any]) -> List[str]:
        """Get list of validation errors."""
        pass


class PydanticConfigValidator(IConfigurationValidator):
    """Configuration validator using Pydantic models."""
    
    def __init__(self, model_class: type):
        self.model_class = model_class
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration using Pydantic model."""
        try:
            self.model_class(**config)
            return True
        except ValidationError:
            return False
    
    def get_validation_errors(self, config: Dict[str, Any]) -> List[str]:
        """Get validation errors from Pydantic model."""
        try:
            self.model_class(**config)
            return []
        except ValidationError as e:
            return [str(error) for error in e.errors()]


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration files."""
    
    def __init__(self, config_manager: 'DynamicConfigurationManager'):
        self.config_manager = config_manager
        self.debounce_time = 1.0  # seconds
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = event.src_path
        current_time = time.time()
        
        # Debounce rapid file changes
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < self.debounce_time:
                return
        
        self.last_modified[file_path] = current_time
        
        # Check if this is a watched config file
        if self.config_manager.is_watched_file(file_path):
            logger.info(f"Configuration file changed: {file_path}")
            asyncio.create_task(self.config_manager.reload_config_file(file_path))


class DynamicConfigurationManager:
    """
    Dynamic configuration manager with hot-reloading capabilities.
    
    Supports multiple configuration sources, validation, change notifications,
    and graceful error handling for production environments.
    """
    
    def __init__(self, base_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the dynamic configuration manager.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = Path(base_config_path) if base_config_path else Path("config.yaml")
        
        # Configuration storage
        self.configurations: Dict[str, Dict[str, Any]] = {}
        self.config_validators: Dict[str, IConfigurationValidator] = {}
        self.change_listeners: Dict[str, List[Callable]] = {}
        
        # File watching
        self.watched_files: Set[str] = set()
        self.file_observer: Optional[Observer] = None
        self.file_watcher: Optional[ConfigFileWatcher] = None
        
        # State management
        self.lock = threading.RLock()
        self.hot_reload_enabled = True
        self.validation_enabled = True
        
        # Load initial configuration
        self._load_initial_config()
    
    def _load_initial_config(self) -> None:
        """Load initial configuration from base config file."""
        try:
            if self.base_config_path.exists():
                config = self._load_config_file(self.base_config_path)
                self.configurations["base"] = config
                logger.info(f"Loaded base configuration from {self.base_config_path}")
            else:
                logger.warning(f"Base configuration file not found: {self.base_config_path}")
                self.configurations["base"] = {}
        except Exception as e:
            logger.error(f"Failed to load initial configuration: {e}")
            self.configurations["base"] = {}
    
    def register_config_source(self, name: str, config_path: Union[str, Path], 
                             validator: Optional[IConfigurationValidator] = None,
                             watch: bool = True) -> bool:
        """
        Register a configuration source.
        
        Args:
            name: Configuration source name
            config_path: Path to configuration file
            validator: Optional validator for the configuration
            watch: Whether to watch the file for changes
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            config_path = Path(config_path)
            
            # Load configuration
            if config_path.exists():
                config = self._load_config_file(config_path)
                
                # Validate if validator provided
                if validator and self.validation_enabled:
                    if not validator.validate(config):
                        errors = validator.get_validation_errors(config)
                        logger.error(f"Configuration validation failed for {name}: {errors}")
                        return False
                
                with self.lock:
                    self.configurations[name] = config
                    if validator:
                        self.config_validators[name] = validator
                    
                    # Add to watched files
                    if watch:
                        self.watched_files.add(str(config_path))
                
                logger.info(f"Registered configuration source: {name}")
                return True
            else:
                logger.error(f"Configuration file not found: {config_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to register config source {name}: {e}")
            return False
    
    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration by name.
        
        Args:
            name: Configuration name
            
        Returns:
            Configuration dictionary or None if not found
        """
        with self.lock:
            return self.configurations.get(name, {}).copy()
    
    def get_merged_config(self, *names: str) -> Dict[str, Any]:
        """
        Get merged configuration from multiple sources.
        
        Args:
            *names: Configuration source names in merge order
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        with self.lock:
            for name in names:
                if name in self.configurations:
                    self._deep_merge(merged, self.configurations[name])
        
        return merged
    
    def update_config(self, name: str, config: Dict[str, Any], 
                     validate: bool = True) -> bool:
        """
        Update configuration programmatically.
        
        Args:
            name: Configuration name
            config: New configuration
            validate: Whether to validate the configuration
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate if required
            if validate and self.validation_enabled and name in self.config_validators:
                validator = self.config_validators[name]
                if not validator.validate(config):
                    errors = validator.get_validation_errors(config)
                    logger.error(f"Configuration validation failed for {name}: {errors}")
                    return False
            
            with self.lock:
                old_config = self.configurations.get(name, {})
                self.configurations[name] = config.copy()
            
            # Notify listeners
            self._notify_config_change(name, old_config, config)
            
            logger.info(f"Updated configuration: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration {name}: {e}")
            return False
    
    def add_change_listener(self, config_name: str, listener: Callable[[ConfigurationChangeEvent], None]) -> None:
        """
        Add a listener for configuration changes.
        
        Args:
            config_name: Configuration name to listen for
            listener: Callback function for changes
        """
        with self.lock:
            if config_name not in self.change_listeners:
                self.change_listeners[config_name] = []
            self.change_listeners[config_name].append(listener)
        
        logger.debug(f"Added change listener for {config_name}")
    
    def remove_change_listener(self, config_name: str, listener: Callable) -> None:
        """
        Remove a configuration change listener.
        
        Args:
            config_name: Configuration name
            listener: Listener function to remove
        """
        with self.lock:
            if config_name in self.change_listeners:
                try:
                    self.change_listeners[config_name].remove(listener)
                except ValueError:
                    pass
    
    def start_file_watching(self) -> bool:
        """
        Start watching configuration files for changes.
        
        Returns:
            True if watching started successfully, False otherwise
        """
        if not self.hot_reload_enabled:
            return False
        
        try:
            if self.file_observer is None:
                self.file_watcher = ConfigFileWatcher(self)
                self.file_observer = Observer()
                
                # Watch directories containing config files
                watched_dirs = set()
                for file_path in self.watched_files:
                    dir_path = Path(file_path).parent
                    if dir_path not in watched_dirs:
                        self.file_observer.schedule(self.file_watcher, str(dir_path), recursive=False)
                        watched_dirs.add(dir_path)
                
                self.file_observer.start()
                logger.info("Started configuration file watching")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start file watching: {e}")
            return False
    
    def stop_file_watching(self) -> None:
        """Stop watching configuration files."""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
            self.file_watcher = None
            logger.info("Stopped configuration file watching")
    
    def is_watched_file(self, file_path: str) -> bool:
        """Check if a file is being watched for changes."""
        return file_path in self.watched_files
    
    async def reload_config_file(self, file_path: str) -> bool:
        """
        Reload configuration from a file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            # Find config name for this file path
            config_name = None
            for name, config in self.configurations.items():
                # This is a simplified lookup - in practice, you'd maintain a file->name mapping
                if str(file_path).endswith(f"{name}.yaml") or str(file_path).endswith(f"{name}.yml"):
                    config_name = name
                    break
            
            if not config_name:
                logger.warning(f"No configuration name found for file: {file_path}")
                return False
            
            # Load new configuration
            new_config = self._load_config_file(Path(file_path))
            
            # Validate if validator exists
            if config_name in self.config_validators and self.validation_enabled:
                validator = self.config_validators[config_name]
                if not validator.validate(new_config):
                    errors = validator.get_validation_errors(new_config)
                    logger.error(f"Configuration validation failed for {config_name}: {errors}")
                    return False
            
            with self.lock:
                old_config = self.configurations.get(config_name, {})
                self.configurations[config_name] = new_config
            
            # Notify listeners
            self._notify_config_change(config_name, old_config, new_config)
            
            logger.info(f"Reloaded configuration from file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload config file {file_path}: {e}")
            return False
    
    def enable_hot_reload(self, enabled: bool = True) -> None:
        """Enable or disable hot-reloading."""
        self.hot_reload_enabled = enabled
        if enabled:
            self.start_file_watching()
        else:
            self.stop_file_watching()
    
    def enable_validation(self, enabled: bool = True) -> None:
        """Enable or disable configuration validation."""
        self.validation_enabled = enabled
    
    def get_config_status(self) -> Dict[str, Any]:
        """
        Get status information about configuration management.
        
        Returns:
            Status information dictionary
        """
        with self.lock:
            return {
                "config_sources": list(self.configurations.keys()),
                "watched_files": list(self.watched_files),
                "hot_reload_enabled": self.hot_reload_enabled,
                "validation_enabled": self.validation_enabled,
                "file_watching_active": self.file_observer is not None and self.file_observer.is_alive(),
                "change_listeners": {name: len(listeners) for name, listeners in self.change_listeners.items()}
            }
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            raise
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge override into base dictionary."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _notify_config_change(self, config_name: str, old_config: Dict[str, Any], 
                            new_config: Dict[str, Any]) -> None:
        """Notify listeners of configuration changes."""
        if config_name in self.change_listeners:
            event = ConfigurationChangeEvent(config_name, old_config, new_config)
            
            for listener in self.change_listeners[config_name]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        asyncio.create_task(listener(event))
                    else:
                        listener(event)
                except Exception as e:
                    logger.error(f"Configuration change listener failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_file_watching()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_file_watching()


def create_default_config_manager(base_config_path: Optional[Union[str, Path]] = None) -> DynamicConfigurationManager:
    """
    Create a configuration manager with default settings.
    
    Args:
        base_config_path: Path to base configuration file
        
    Returns:
        Configured dynamic configuration manager
    """
    manager = DynamicConfigurationManager(base_config_path)
    
    # Register common configuration sources
    config_dir = Path("config")
    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            name = config_file.stem
            manager.register_config_source(name, config_file, watch=True)
    
    return manager