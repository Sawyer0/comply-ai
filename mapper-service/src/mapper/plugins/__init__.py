"""
Plugin system for the mapper service.

This module provides a comprehensive plugin architecture that allows
extending the mapper service with custom functionality for mapping engines,
model serving, validation, and data processing.
"""

from .interfaces import (
    PluginType,
    PluginCapability,
    PluginMetadata,
    PluginResult,
    IPlugin,
    IMappingEnginePlugin,
    IModelServingPlugin,
    IValidationPlugin,
    IPreprocessingPlugin,
    IPostprocessingPlugin,
    IPluginRegistry,
)

from .manager import (
    PluginManager,
    PluginRegistry,
    PluginLoadError,
    PluginExecutionError,
    get_plugin_manager,
    initialize_plugin_manager,
)

from .builtin import (
    DefaultMappingEnginePlugin,
    JSONSchemaValidationPlugin,
    ContentScrubberPlugin,
    BUILTIN_PLUGINS,
    register_builtin_plugins,
)

__all__ = [
    # Interfaces
    "PluginType",
    "PluginCapability",
    "PluginMetadata",
    "PluginResult",
    "IPlugin",
    "IMappingEnginePlugin",
    "IModelServingPlugin",
    "IValidationPlugin",
    "IPreprocessingPlugin",
    "IPostprocessingPlugin",
    "IPluginRegistry",
    # Manager
    "PluginManager",
    "PluginRegistry",
    "PluginLoadError",
    "PluginExecutionError",
    "get_plugin_manager",
    "initialize_plugin_manager",
    # Built-in plugins
    "DefaultMappingEnginePlugin",
    "JSONSchemaValidationPlugin",
    "ContentScrubberPlugin",
    "BUILTIN_PLUGINS",
    "register_builtin_plugins",
]
