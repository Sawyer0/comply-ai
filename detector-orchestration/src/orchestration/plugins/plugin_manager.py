"""Plugin manager following SRP.

This module provides ONLY plugin management functionality.
Single Responsibility: Manage plugin lifecycle and registration.
"""

import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Type
from pathlib import Path

from .detector_plugin import DetectorPluginInterface
from .policy_plugin import PolicyPluginInterface

logger = logging.getLogger(__name__)


class PluginManager:
    """Plugin manager for orchestration service.

    Single Responsibility: Manage plugin lifecycle and registration.
    Does NOT handle: plugin implementation, business logic.
    """

    def __init__(self):
        """Initialize plugin manager."""
        self._detector_plugins: Dict[str, DetectorPluginInterface] = {}
        self._policy_plugins: Dict[str, PolicyPluginInterface] = {}
        self._plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = False

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin manager.

        Args:
            config: Plugin manager configuration

        Returns:
            True if initialization successful
        """
        try:
            self._plugin_configs = config.get("plugins", {})

            # Load detector plugins
            detector_configs = self._plugin_configs.get("detectors", {})
            for plugin_name, plugin_config in detector_configs.items():
                await self._load_detector_plugin(plugin_name, plugin_config)

            # Load policy plugins
            policy_configs = self._plugin_configs.get("policies", {})
            for plugin_name, plugin_config in policy_configs.items():
                await self._load_policy_plugin(plugin_name, plugin_config)

            self._initialized = True
            logger.info(
                "Plugin manager initialized with %d detector and %d policy plugins",
                len(self._detector_plugins),
                len(self._policy_plugins),
            )
            return True

        except Exception as e:
            logger.error("Failed to initialize plugin manager: %s", str(e))
            return False

    async def register_detector_plugin(
        self, plugin: DetectorPluginInterface, config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a detector plugin.

        Args:
            plugin: Detector plugin instance
            config: Optional plugin configuration

        Returns:
            True if registration successful
        """
        try:
            plugin_name = plugin.plugin_name

            if plugin_name in self._detector_plugins:
                logger.warning(
                    "Detector plugin %s already registered, replacing", plugin_name
                )
                await self._detector_plugins[plugin_name].cleanup()

            # Initialize plugin
            plugin_config = config or {}
            if await plugin.initialize(plugin_config):
                self._detector_plugins[plugin_name] = plugin
                logger.info(
                    "Registered detector plugin: %s v%s",
                    plugin_name,
                    plugin.plugin_version,
                )
                return True

            logger.error("Failed to initialize detector plugin: %s", plugin_name)
            return False

        except Exception as e:
            logger.error("Failed to register detector plugin: %s", str(e))
            return False

    async def register_policy_plugin(
        self, plugin: PolicyPluginInterface, config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a policy plugin.

        Args:
            plugin: Policy plugin instance
            config: Optional plugin configuration

        Returns:
            True if registration successful
        """
        try:
            plugin_name = plugin.plugin_name

            if plugin_name in self._policy_plugins:
                logger.warning(
                    "Policy plugin %s already registered, replacing", plugin_name
                )
                await self._policy_plugins[plugin_name].cleanup()

            # Initialize plugin
            plugin_config = config or {}
            if await plugin.initialize(plugin_config):
                self._policy_plugins[plugin_name] = plugin
                logger.info(
                    "Registered policy plugin: %s v%s",
                    plugin_name,
                    plugin.plugin_version,
                )
                return True

            logger.error("Failed to initialize policy plugin: %s", plugin_name)
            return False

        except Exception as e:
            logger.error("Failed to register policy plugin: %s", str(e))
            return False

    async def unregister_detector_plugin(self, plugin_name: str) -> bool:
        """Unregister a detector plugin.

        Args:
            plugin_name: Name of plugin to unregister

        Returns:
            True if unregistration successful
        """
        try:
            if plugin_name not in self._detector_plugins:
                logger.warning("Detector plugin %s not found", plugin_name)
                return False

            plugin = self._detector_plugins[plugin_name]
            await plugin.cleanup()
            del self._detector_plugins[plugin_name]

            logger.info("Unregistered detector plugin: %s", plugin_name)
            return True

        except Exception as e:
            logger.error(
                "Failed to unregister detector plugin %s: %s", plugin_name, str(e)
            )
            return False

    async def unregister_policy_plugin(self, plugin_name: str) -> bool:
        """Unregister a policy plugin.

        Args:
            plugin_name: Name of plugin to unregister

        Returns:
            True if unregistration successful
        """
        try:
            if plugin_name not in self._policy_plugins:
                logger.warning("Policy plugin %s not found", plugin_name)
                return False

            plugin = self._policy_plugins[plugin_name]
            await plugin.cleanup()
            del self._policy_plugins[plugin_name]

            logger.info("Unregistered policy plugin: %s", plugin_name)
            return True

        except Exception as e:
            logger.error(
                "Failed to unregister policy plugin %s: %s", plugin_name, str(e)
            )
            return False

    def get_detector_plugin(
        self, plugin_name: str
    ) -> Optional[DetectorPluginInterface]:
        """Get a detector plugin by name.

        Args:
            plugin_name: Name of plugin to get

        Returns:
            Detector plugin instance or None
        """
        return self._detector_plugins.get(plugin_name)

    def get_policy_plugin(self, plugin_name: str) -> Optional[PolicyPluginInterface]:
        """Get a policy plugin by name.

        Args:
            plugin_name: Name of plugin to get

        Returns:
            Policy plugin instance or None
        """
        return self._policy_plugins.get(plugin_name)

    def list_detector_plugins(self) -> List[str]:
        """List all registered detector plugins.

        Returns:
            List of detector plugin names
        """
        return list(self._detector_plugins.keys())

    def list_policy_plugins(self) -> List[str]:
        """List all registered policy plugins.

        Returns:
            List of policy plugin names
        """
        return list(self._policy_plugins.keys())

    async def health_check_plugins(self) -> Dict[str, bool]:
        """Check health of all plugins.

        Returns:
            Dictionary mapping plugin names to health status
        """
        health_status = {}

        # Check detector plugins
        for plugin_name, plugin in self._detector_plugins.items():
            try:
                health_status[f"detector:{plugin_name}"] = await plugin.health_check()
            except Exception as e:
                logger.error(
                    "Health check failed for detector plugin %s: %s",
                    plugin_name,
                    str(e),
                )
                health_status[f"detector:{plugin_name}"] = False

        # Check policy plugins (if they have health check method)
        for plugin_name, plugin in self._policy_plugins.items():
            try:
                # Policy plugins don't have health_check in interface, so just check if initialized
                health_status[f"policy:{plugin_name}"] = (
                    hasattr(plugin, "_initialized") and plugin._initialized
                )
            except Exception as e:
                logger.error(
                    "Health check failed for policy plugin %s: %s", plugin_name, str(e)
                )
                health_status[f"policy:{plugin_name}"] = False

        return health_status

    async def cleanup(self) -> None:
        """Cleanup all plugins."""
        try:
            # Cleanup detector plugins
            for plugin_name, plugin in self._detector_plugins.items():
                try:
                    await plugin.cleanup()
                except Exception as e:
                    logger.error(
                        "Failed to cleanup detector plugin %s: %s", plugin_name, str(e)
                    )

            # Cleanup policy plugins
            for plugin_name, plugin in self._policy_plugins.items():
                try:
                    await plugin.cleanup()
                except Exception as e:
                    logger.error(
                        "Failed to cleanup policy plugin %s: %s", plugin_name, str(e)
                    )

            self._detector_plugins.clear()
            self._policy_plugins.clear()
            self._initialized = False

            logger.info("Plugin manager cleaned up")

        except Exception as e:
            logger.error("Failed to cleanup plugin manager: %s", str(e))

    async def _load_detector_plugin(
        self, plugin_name: str, plugin_config: Dict[str, Any]
    ) -> bool:
        """Load a detector plugin from configuration.

        Args:
            plugin_name: Name of plugin to load
            plugin_config: Plugin configuration

        Returns:
            True if loading successful
        """
        try:
            module_path = plugin_config.get("module")
            class_name = plugin_config.get("class")

            if not module_path or not class_name:
                logger.error(
                    "Missing module or class for detector plugin %s", plugin_name
                )
                return False

            # Import module and get class
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            # Validate plugin class
            if not issubclass(plugin_class, DetectorPluginInterface):
                logger.error(
                    "Plugin class %s does not implement DetectorPluginInterface",
                    class_name,
                )
                return False

            # Create and register plugin
            plugin_instance = plugin_class()
            return await self.register_detector_plugin(
                plugin_instance, plugin_config.get("config", {})
            )

        except Exception as e:
            logger.error("Failed to load detector plugin %s: %s", plugin_name, str(e))
            return False

    async def _load_policy_plugin(
        self, plugin_name: str, plugin_config: Dict[str, Any]
    ) -> bool:
        """Load a policy plugin from configuration.

        Args:
            plugin_name: Name of plugin to load
            plugin_config: Plugin configuration

        Returns:
            True if loading successful
        """
        try:
            module_path = plugin_config.get("module")
            class_name = plugin_config.get("class")

            if not module_path or not class_name:
                logger.error(
                    "Missing module or class for policy plugin %s", plugin_name
                )
                return False

            # Import module and get class
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)

            # Validate plugin class
            if not issubclass(plugin_class, PolicyPluginInterface):
                logger.error(
                    "Plugin class %s does not implement PolicyPluginInterface",
                    class_name,
                )
                return False

            # Create and register plugin
            plugin_instance = plugin_class()
            return await self.register_policy_plugin(
                plugin_instance, plugin_config.get("config", {})
            )

        except Exception as e:
            logger.error("Failed to load policy plugin %s: %s", plugin_name, str(e))
            return False

    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin statistics.

        Returns:
            Plugin statistics
        """
        return {
            "total_detector_plugins": len(self._detector_plugins),
            "total_policy_plugins": len(self._policy_plugins),
            "detector_plugins": list(self._detector_plugins.keys()),
            "policy_plugins": list(self._policy_plugins.keys()),
            "initialized": self._initialized,
        }


# Export only the plugin manager functionality
__all__ = [
    "PluginManager",
]
