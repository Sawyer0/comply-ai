"""
Plugin manager for the Analysis Service.

This module provides plugin lifecycle management, loading, configuration,
and execution coordination for analysis plugins.
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Union
import traceback
from datetime import datetime, timezone

from .interfaces import (
    IPlugin,
    IAnalysisEnginePlugin,
    IMLModelPlugin,
    IQualityEvaluatorPlugin,
    IPatternDetectorPlugin,
    IRiskScorerPlugin,
    IComplianceMapperPlugin,
    PluginType,
    PluginStatus,
    PluginMetadata,
    AnalysisRequest,
    AnalysisResult,
)
from .database import PluginDatabaseManager

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for managing plugin instances and metadata."""

    def __init__(self):
        self.plugins: Dict[str, IPlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_errors: Dict[str, List[str]] = {}

    def register_plugin(
        self, plugin_name: str, plugin: IPlugin, config: Dict[str, Any] = None
    ) -> bool:
        """
        Register a plugin instance.

        Args:
            plugin_name: Unique plugin name
            plugin: Plugin instance
            config: Plugin configuration

        Returns:
            True if registration successful
        """
        try:
            metadata = plugin.get_metadata()

            self.plugins[plugin_name] = plugin
            self.plugin_metadata[plugin_name] = metadata
            self.plugin_status[plugin_name] = PluginStatus.INACTIVE
            self.plugin_configs[plugin_name] = config or {}
            self.plugin_errors[plugin_name] = []

            logger.info(
                "Registered plugin",
                plugin_name=plugin_name,
                plugin_type=metadata.plugin_type.value,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to register plugin", plugin_name=plugin_name, error=str(e)
            )
            return False

    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin.

        Args:
            plugin_name: Plugin name to unregister

        Returns:
            True if unregistration successful
        """
        try:
            if plugin_name in self.plugins:
                del self.plugins[plugin_name]
                del self.plugin_metadata[plugin_name]
                del self.plugin_status[plugin_name]
                del self.plugin_configs[plugin_name]
                del self.plugin_errors[plugin_name]

                logger.info("Unregistered plugin", plugin_name=plugin_name)
                return True
            return False

        except Exception as e:
            logger.error(
                "Failed to unregister plugin", plugin_name=plugin_name, error=str(e)
            )
            return False

    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get plugin instance by name."""
        return self.plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[IPlugin]:
        """Get all plugins of a specific type."""
        return [
            plugin
            for plugin_name, plugin in self.plugins.items()
            if self.plugin_metadata[plugin_name].plugin_type == plugin_type
        ]

    def get_active_plugins(self) -> List[IPlugin]:
        """Get all active plugins."""
        return [
            plugin
            for plugin_name, plugin in self.plugins.items()
            if self.plugin_status[plugin_name] == PluginStatus.ACTIVE
        ]

    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all registered plugins with their status."""
        return {
            plugin_name: {
                "metadata": metadata.dict(),
                "status": self.plugin_status[plugin_name].value,
                "errors": self.plugin_errors[plugin_name],
            }
            for plugin_name, metadata in self.plugin_metadata.items()
        }


class PluginManager:
    """
    Plugin manager for the Analysis Service.

    Handles plugin discovery, loading, lifecycle management, and execution.
    """

    def __init__(
        self,
        plugin_directories: List[str] = None,
        tenant_manager=None,
        db_service_name: str = "analysis",
    ):
        """
        Initialize the plugin manager.

        Args:
            plugin_directories: List of directories to search for plugins
            tenant_manager: Tenant manager for tenant-specific plugin configurations
            db_service_name: Database service name for plugin data
        """
        self.registry = PluginRegistry()
        self.plugin_directories = plugin_directories or []
        self.tenant_manager = tenant_manager
        self.plugin_locks: Dict[str, asyncio.Lock] = {}
        self.db_manager = PluginDatabaseManager(db_service_name)

        # Default plugin directory
        default_plugin_dir = Path(__file__).parent / "builtin"
        if default_plugin_dir.exists():
            self.plugin_directories.append(str(default_plugin_dir))

    async def initialize(self) -> bool:
        """
        Initialize the plugin manager and discover plugins.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing plugin manager")

            # Discover and load plugins
            await self.discover_plugins()

            # Initialize all plugins
            await self.initialize_all_plugins()

            logger.info("Plugin manager initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize plugin manager", error=str(e))
            return False

    async def discover_plugins(self) -> List[str]:
        """
        Discover plugins in configured directories.

        Returns:
            List of discovered plugin names
        """
        discovered_plugins = []

        for plugin_dir in self.plugin_directories:
            try:
                plugin_path = Path(plugin_dir)
                if not plugin_path.exists():
                    continue

                # Add plugin directory to Python path
                if str(plugin_path) not in sys.path:
                    sys.path.insert(0, str(plugin_path))

                # Discover Python files
                for py_file in plugin_path.glob("*.py"):
                    if py_file.name.startswith("__"):
                        continue

                    plugin_name = py_file.stem
                    try:
                        await self.load_plugin_from_file(str(py_file), plugin_name)
                        discovered_plugins.append(plugin_name)

                    except Exception as e:
                        logger.error(
                            "Failed to load plugin from file",
                            file=str(py_file),
                            error=str(e),
                        )

                # Discover plugin packages
                for pkg_dir in plugin_path.iterdir():
                    if pkg_dir.is_dir() and (pkg_dir / "__init__.py").exists():
                        plugin_name = pkg_dir.name
                        try:
                            await self.load_plugin_from_package(
                                str(pkg_dir), plugin_name
                            )
                            discovered_plugins.append(plugin_name)

                        except Exception as e:
                            logger.error(
                                "Failed to load plugin from package",
                                package=str(pkg_dir),
                                error=str(e),
                            )

            except Exception as e:
                logger.error(
                    "Failed to discover plugins in directory",
                    directory=plugin_dir,
                    error=str(e),
                )

        logger.info(
            "Discovered plugins",
            count=len(discovered_plugins),
            plugins=discovered_plugins,
        )
        return discovered_plugins

    async def load_plugin_from_file(self, file_path: str, plugin_name: str) -> bool:
        """
        Load plugin from a Python file.

        Args:
            file_path: Path to plugin file
            plugin_name: Plugin name

        Returns:
            True if loading successful
        """
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location(plugin_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, IPlugin)
                    and obj != IPlugin
                    and not inspect.isabstract(obj)
                ):
                    plugin_classes.append(obj)

            if not plugin_classes:
                logger.warning("No plugin classes found in file", file=file_path)
                return False

            # Instantiate and register plugins
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    class_plugin_name = f"{plugin_name}_{plugin_class.__name__}"

                    self.registry.register_plugin(class_plugin_name, plugin_instance)
                    self.plugin_locks[class_plugin_name] = asyncio.Lock()

                except Exception as e:
                    logger.error(
                        "Failed to instantiate plugin class",
                        class_name=plugin_class.__name__,
                        error=str(e),
                    )

            return True

        except Exception as e:
            logger.error(
                "Failed to load plugin from file", file=file_path, error=str(e)
            )
            return False

    async def load_plugin_from_package(
        self, package_path: str, plugin_name: str
    ) -> bool:
        """
        Load plugin from a Python package.

        Args:
            package_path: Path to plugin package
            plugin_name: Plugin name

        Returns:
            True if loading successful
        """
        try:
            # Import the package
            spec = importlib.util.spec_from_file_location(
                plugin_name, os.path.join(package_path, "__init__.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for plugin factory function or class
            plugin_instance = None

            if hasattr(module, "create_plugin"):
                plugin_instance = module.create_plugin()
            elif hasattr(module, "Plugin"):
                plugin_instance = module.Plugin()
            else:
                # Find first plugin class
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, IPlugin)
                        and obj != IPlugin
                        and not inspect.isabstract(obj)
                    ):
                        plugin_instance = obj()
                        break

            if plugin_instance:
                self.registry.register_plugin(plugin_name, plugin_instance)
                self.plugin_locks[plugin_name] = asyncio.Lock()
                return True
            else:
                logger.warning("No plugin found in package", package=package_path)
                return False

        except Exception as e:
            logger.error(
                "Failed to load plugin from package", package=package_path, error=str(e)
            )
            return False

    async def initialize_all_plugins(self) -> Dict[str, bool]:
        """
        Initialize all registered plugins.

        Returns:
            Dictionary of plugin names to initialization success status
        """
        results = {}

        for plugin_name in self.registry.plugins.keys():
            results[plugin_name] = await self.initialize_plugin(plugin_name)

        return results

    async def initialize_plugin(
        self, plugin_name: str, config: Dict[str, Any] = None
    ) -> bool:
        """
        Initialize a specific plugin.

        Args:
            plugin_name: Plugin name
            config: Plugin configuration

        Returns:
            True if initialization successful
        """
        try:
            plugin = self.registry.get_plugin(plugin_name)
            if not plugin:
                return False

            # Update status to loading
            self.registry.plugin_status[plugin_name] = PluginStatus.LOADING

            # Use provided config or stored config
            plugin_config = config or self.registry.plugin_configs.get(plugin_name, {})

            # Initialize plugin
            async with self.plugin_locks[plugin_name]:
                success = await plugin.initialize(plugin_config)

            if success:
                self.registry.plugin_status[plugin_name] = PluginStatus.ACTIVE
                self.registry.plugin_errors[plugin_name] = []
                logger.info("Initialized plugin", plugin_name=plugin_name)
            else:
                self.registry.plugin_status[plugin_name] = PluginStatus.ERROR
                self.registry.plugin_errors[plugin_name].append("Initialization failed")
                logger.error("Failed to initialize plugin", plugin_name=plugin_name)

            return success

        except Exception as e:
            self.registry.plugin_status[plugin_name] = PluginStatus.ERROR
            error_msg = f"Initialization error: {str(e)}"
            self.registry.plugin_errors[plugin_name].append(error_msg)
            logger.error(
                "Plugin initialization exception", plugin_name=plugin_name, error=str(e)
            )
            return False

    async def execute_analysis(
        self, plugin_name: str, request: AnalysisRequest
    ) -> Optional[AnalysisResult]:
        """
        Execute analysis using a specific plugin.

        Args:
            plugin_name: Plugin name
            request: Analysis request

        Returns:
            Analysis result if successful, None otherwise
        """
        try:
            plugin = self.registry.get_plugin(plugin_name)
            if not plugin or not isinstance(plugin, IAnalysisEnginePlugin):
                return None

            # Check plugin status
            if self.registry.plugin_status[plugin_name] != PluginStatus.ACTIVE:
                logger.warning("Plugin not active", plugin_name=plugin_name)
                return None

            # Check tenant permissions if tenant manager available
            if self.tenant_manager:
                tenant_config = await self.tenant_manager.get_tenant_config(
                    request.tenant_id
                )
                if tenant_config and plugin_name not in tenant_config.custom_engines:
                    # Check if plugin is allowed for this tenant
                    pass

            # Execute analysis
            async with self.plugin_locks[plugin_name]:
                result = await plugin.analyze(request)

            logger.info(
                "Plugin analysis completed",
                plugin_name=plugin_name,
                request_id=request.request_id,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            error_msg = f"Analysis execution error: {str(e)}"
            self.registry.plugin_errors[plugin_name].append(error_msg)
            logger.error(
                "Plugin analysis failed",
                plugin_name=plugin_name,
                request_id=request.request_id,
                error=str(e),
            )
            return None

    async def execute_batch_analysis(
        self, plugin_name: str, requests: List[AnalysisRequest]
    ) -> List[AnalysisResult]:
        """
        Execute batch analysis using a specific plugin.

        Args:
            plugin_name: Plugin name
            requests: List of analysis requests

        Returns:
            List of analysis results
        """
        try:
            plugin = self.registry.get_plugin(plugin_name)
            if not plugin or not isinstance(plugin, IAnalysisEnginePlugin):
                return []

            # Check plugin status
            if self.registry.plugin_status[plugin_name] != PluginStatus.ACTIVE:
                logger.warning("Plugin not active", plugin_name=plugin_name)
                return []

            # Execute batch analysis
            async with self.plugin_locks[plugin_name]:
                results = await plugin.batch_analyze(requests)

            logger.info(
                "Plugin batch analysis completed",
                plugin_name=plugin_name,
                request_count=len(requests),
                result_count=len(results),
            )

            return results

        except Exception as e:
            error_msg = f"Batch analysis execution error: {str(e)}"
            self.registry.plugin_errors[plugin_name].append(error_msg)
            logger.error(
                "Plugin batch analysis failed",
                plugin_name=plugin_name,
                request_count=len(requests),
                error=str(e),
            )
            return []

    async def get_plugin_health(self, plugin_name: str) -> Dict[str, Any]:
        """
        Get health status of a specific plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            Health status information
        """
        try:
            plugin = self.registry.get_plugin(plugin_name)
            if not plugin:
                return {"status": "not_found"}

            health_info = await plugin.health_check()
            health_info["plugin_status"] = self.registry.plugin_status[
                plugin_name
            ].value
            health_info["errors"] = self.registry.plugin_errors[plugin_name]

            return health_info

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "plugin_status": self.registry.plugin_status.get(
                    plugin_name, PluginStatus.ERROR
                ).value,
            }

    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a specific plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            True if reload successful
        """
        try:
            # Cleanup existing plugin
            plugin = self.registry.get_plugin(plugin_name)
            if plugin:
                await plugin.cleanup()

            # Unregister plugin
            self.registry.unregister_plugin(plugin_name)

            # Rediscover and load plugin
            await self.discover_plugins()

            # Initialize the reloaded plugin
            return await self.initialize_plugin(plugin_name)

        except Exception as e:
            logger.error(
                "Failed to reload plugin", plugin_name=plugin_name, error=str(e)
            )
            return False

    async def shutdown(self) -> bool:
        """
        Shutdown the plugin manager and cleanup all plugins.

        Returns:
            True if shutdown successful
        """
        try:
            logger.info("Shutting down plugin manager")

            # Cleanup all plugins
            for plugin_name, plugin in self.registry.plugins.items():
                try:
                    await plugin.cleanup()
                    logger.info("Cleaned up plugin", plugin_name=plugin_name)
                except Exception as e:
                    logger.error(
                        "Failed to cleanup plugin",
                        plugin_name=plugin_name,
                        error=str(e),
                    )

            # Clear registry
            self.registry.plugins.clear()
            self.registry.plugin_metadata.clear()
            self.registry.plugin_status.clear()
            self.registry.plugin_configs.clear()
            self.registry.plugin_errors.clear()

            logger.info("Plugin manager shutdown completed")
            return True

        except Exception as e:
            logger.error("Failed to shutdown plugin manager", error=str(e))
            return False

    def get_registry(self) -> PluginRegistry:
        """Get the plugin registry."""
        return self.registry
