"""
Plugin manager for the mapper service.

This module provides the core plugin management functionality including
registration, lifecycle management, and plugin execution coordination.
"""

import asyncio
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import json
import yaml
from concurrent.futures import ThreadPoolExecutor

# Import shared components
from shared.utils.logging import get_logger
from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import BaseServiceException, ValidationError

from .interfaces import (
    IPlugin,
    IPluginRegistry,
    PluginType,
    PluginMetadata,
    PluginResult,
    IMappingEnginePlugin,
    IModelServingPlugin,
    IValidationPlugin,
    IPreprocessingPlugin,
    IPostprocessingPlugin,
)


logger = get_logger(__name__)


class PluginLoadError(Exception):
    """Raised when plugin loading fails."""

    pass


class PluginExecutionError(Exception):
    """Raised when plugin execution fails."""

    pass


class PluginRegistry(IPluginRegistry):
    """Registry for managing plugins."""

    def __init__(self):
        self._plugins: Dict[str, IPlugin] = {}
        self._plugin_metadata: Dict[str, PluginMetadata] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }

    def register_plugin(self, plugin: IPlugin) -> bool:
        """Register a plugin."""
        try:
            metadata = plugin.get_metadata()

            # Validate plugin metadata
            if not self._validate_plugin_metadata(metadata):
                logger.error(f"Invalid plugin metadata for {metadata.plugin_id}")
                return False

            # Check for conflicts
            if metadata.plugin_id in self._plugins:
                logger.warning(
                    f"Plugin {metadata.plugin_id} already registered, replacing"
                )

            # Register plugin
            self._plugins[metadata.plugin_id] = plugin
            self._plugin_metadata[metadata.plugin_id] = metadata

            # Add to type index
            if metadata.plugin_id not in self._plugins_by_type[metadata.plugin_type]:
                self._plugins_by_type[metadata.plugin_type].append(metadata.plugin_id)

            logger.info(f"Successfully registered plugin: {metadata.plugin_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False

    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin."""
        try:
            if plugin_id not in self._plugins:
                logger.warning(f"Plugin {plugin_id} not found for unregistration")
                return False

            plugin = self._plugins[plugin_id]
            metadata = self._plugin_metadata[plugin_id]

            # Cleanup plugin
            plugin.cleanup()

            # Remove from registry
            del self._plugins[plugin_id]
            del self._plugin_metadata[plugin_id]

            # Remove from type index
            if plugin_id in self._plugins_by_type[metadata.plugin_type]:
                self._plugins_by_type[metadata.plugin_type].remove(plugin_id)

            logger.info(f"Successfully unregistered plugin: {plugin_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_id}: {e}")
            return False

    def get_plugin(self, plugin_id: str) -> Optional[IPlugin]:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)

    def list_plugins(
        self, plugin_type: Optional[PluginType] = None
    ) -> List[PluginMetadata]:
        """List all registered plugins, optionally filtered by type."""
        if plugin_type is None:
            return list(self._plugin_metadata.values())

        plugin_ids = self._plugins_by_type.get(plugin_type, [])
        return [self._plugin_metadata[pid] for pid in plugin_ids]

    def get_plugins_by_capability(self, capability: str) -> List[IPlugin]:
        """Get plugins that support a specific capability."""
        matching_plugins = []

        for plugin_id, metadata in self._plugin_metadata.items():
            for cap in metadata.capabilities:
                if cap.name == capability:
                    plugin = self._plugins.get(plugin_id)
                    if plugin:
                        matching_plugins.append(plugin)
                    break

        return matching_plugins

    def _validate_plugin_metadata(self, metadata: PluginMetadata) -> bool:
        """Validate plugin metadata."""
        if not metadata.plugin_id or not metadata.name or not metadata.version:
            return False

        if not metadata.plugin_type or metadata.plugin_type not in PluginType:
            return False

        return True


class PluginManager:
    """Main plugin manager for the mapper service."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = PluginRegistry()
        self.executor = ThreadPoolExecutor(max_workers=config.get("max_workers", 4))
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._initialized_plugins: set = set()

    async def initialize(self):
        """Initialize the plugin manager."""
        try:
            # Load plugin configurations
            await self._load_plugin_configs()

            # Discover and load plugins
            await self._discover_plugins()

            # Initialize loaded plugins
            await self._initialize_plugins()

            logger.info("Plugin manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize plugin manager: {e}")
            raise

    async def shutdown(self):
        """Shutdown the plugin manager."""
        try:
            # Cleanup all plugins
            for plugin_id in list(self._initialized_plugins):
                await self._cleanup_plugin(plugin_id)

            # Shutdown executor
            self.executor.shutdown(wait=True)

            logger.info("Plugin manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during plugin manager shutdown: {e}")

    async def execute_mapping_plugin(
        self, plugin_id: str, detector_output: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginResult:
        """Execute a mapping engine plugin."""
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin or not isinstance(plugin, IMappingEnginePlugin):
            raise PluginExecutionError(
                f"Mapping plugin {plugin_id} not found or invalid type"
            )

        try:
            return await plugin.map_to_canonical(detector_output, context)
        except Exception as e:
            logger.error(f"Error executing mapping plugin {plugin_id}: {e}")
            raise PluginExecutionError(f"Mapping plugin execution failed: {e}")

    async def execute_validation_plugin(
        self, plugin_id: str, data: Any, schema: Dict[str, Any]
    ) -> PluginResult:
        """Execute a validation plugin."""
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin or not isinstance(plugin, IValidationPlugin):
            raise PluginExecutionError(
                f"Validation plugin {plugin_id} not found or invalid type"
            )

        try:
            return await plugin.validate_input(data, schema)
        except Exception as e:
            logger.error(f"Error executing validation plugin {plugin_id}: {e}")
            raise PluginExecutionError(f"Validation plugin execution failed: {e}")

    async def execute_model_serving_plugin(
        self, plugin_id: str, input_data: Any, model_id: str
    ) -> PluginResult:
        """Execute a model serving plugin."""
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin or not isinstance(plugin, IModelServingPlugin):
            raise PluginExecutionError(
                f"Model serving plugin {plugin_id} not found or invalid type"
            )

        try:
            return await plugin.predict(input_data, model_id)
        except Exception as e:
            logger.error(f"Error executing model serving plugin {plugin_id}: {e}")
            raise PluginExecutionError(f"Model serving plugin execution failed: {e}")

    def get_available_plugins(
        self, plugin_type: Optional[PluginType] = None
    ) -> List[PluginMetadata]:
        """Get list of available plugins."""
        return self.registry.list_plugins(plugin_type)

    def get_plugin_by_capability(self, capability: str) -> List[IPlugin]:
        """Get plugins that support a specific capability."""
        return self.registry.get_plugins_by_capability(capability)

    async def _load_plugin_configs(self):
        """Load plugin configurations from files."""
        config_dir = Path(self.config.get("plugin_config_dir", "config/plugins"))

        if not config_dir.exists():
            logger.warning(f"Plugin config directory {config_dir} does not exist")
            return

        for config_file in config_dir.glob("*.yaml"):
            try:
                with open(config_file, "r") as f:
                    plugin_config = yaml.safe_load(f)

                plugin_id = config_file.stem
                self._plugin_configs[plugin_id] = plugin_config

                logger.debug(f"Loaded config for plugin: {plugin_id}")

            except Exception as e:
                logger.error(f"Failed to load plugin config {config_file}: {e}")

    async def _discover_plugins(self):
        """Discover and load plugins from configured directories."""
        plugin_dirs = self.config.get("plugin_directories", ["plugins"])

        for plugin_dir in plugin_dirs:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                logger.warning(f"Plugin directory {plugin_path} does not exist")
                continue

            await self._load_plugins_from_directory(plugin_path)

    async def _load_plugins_from_directory(self, plugin_dir: Path):
        """Load plugins from a specific directory."""
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue

            try:
                await self._load_plugin_from_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")

    async def _load_plugin_from_file(self, plugin_file: Path):
        """Load a plugin from a Python file."""
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, IPlugin)
                    and obj != IPlugin
                    and not inspect.isabstract(obj)
                ):

                    # Instantiate and register plugin
                    plugin_instance = obj()
                    if self.registry.register_plugin(plugin_instance):
                        logger.info(f"Loaded plugin: {name} from {plugin_file}")

        except Exception as e:
            logger.error(f"Error loading plugin from {plugin_file}: {e}")
            raise PluginLoadError(f"Failed to load plugin: {e}")

    async def _initialize_plugins(self):
        """Initialize all loaded plugins."""
        for plugin_id, plugin in self.registry._plugins.items():
            try:
                config = self._plugin_configs.get(plugin_id, {})

                if plugin.initialize(config):
                    self._initialized_plugins.add(plugin_id)
                    logger.info(f"Initialized plugin: {plugin_id}")
                else:
                    logger.error(f"Failed to initialize plugin: {plugin_id}")

            except Exception as e:
                logger.error(f"Error initializing plugin {plugin_id}: {e}")

    async def _cleanup_plugin(self, plugin_id: str):
        """Cleanup a specific plugin."""
        try:
            plugin = self.registry.get_plugin(plugin_id)
            if plugin:
                plugin.cleanup()
                self._initialized_plugins.discard(plugin_id)
                logger.info(f"Cleaned up plugin: {plugin_id}")

        except Exception as e:
            logger.error(f"Error cleaning up plugin {plugin_id}: {e}")


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> Optional[PluginManager]:
    """Get the global plugin manager instance."""
    return _plugin_manager


def initialize_plugin_manager(config: Dict[str, Any]) -> PluginManager:
    """Initialize the global plugin manager."""
    global _plugin_manager
    _plugin_manager = PluginManager(config)
    return _plugin_manager
