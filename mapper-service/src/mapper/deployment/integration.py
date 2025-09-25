"""
Integration module for deployment features.

This module provides integration between plugins, deployment systems,
and pipelines for the mapper service.
"""

import asyncio
from typing import Any, Dict, List, Optional
import yaml
from pathlib import Path

# Import shared components
from shared.utils.logging import get_logger
from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import BaseServiceException

from ..plugins import (
    initialize_plugin_manager,
    get_plugin_manager,
    register_builtin_plugins,
)
from .canary import get_canary_controller
from .blue_green import get_blue_green_controller
from .feature_flags import initialize_feature_flag_manager, get_feature_flag_manager
from ..pipelines import get_pipeline_executor, get_optimization_pipeline

logger = get_logger(__name__)


class DeploymentManager:
    """Main deployment manager that coordinates all deployment features."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.plugin_manager = None
        self.canary_controller = None
        self.blue_green_controller = None
        self.feature_flag_manager = None
        self.pipeline_executor = None
        self.optimization_pipeline = None

        self._initialized = False

    async def initialize(self):
        """Initialize all deployment components."""
        try:
            logger.info("Initializing deployment manager...")

            # Initialize plugin manager
            await self._initialize_plugin_manager()

            # Initialize deployment controllers
            await self._initialize_deployment_controllers()

            # Initialize feature flag manager
            await self._initialize_feature_flag_manager()

            # Initialize pipelines
            await self._initialize_pipelines()

            self._initialized = True
            logger.info("Deployment manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize deployment manager: {e}")
            raise

    async def shutdown(self):
        """Shutdown all deployment components."""
        try:
            logger.info("Shutting down deployment manager...")

            # Shutdown in reverse order
            if self.feature_flag_manager:
                await self.feature_flag_manager.shutdown()

            if self.plugin_manager:
                await self.plugin_manager.shutdown()

            logger.info("Deployment manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during deployment manager shutdown: {e}")

    async def _initialize_plugin_manager(self):
        """Initialize the plugin manager."""
        plugin_config = self.config.get("plugins", {})

        # Initialize plugin manager
        self.plugin_manager = initialize_plugin_manager(plugin_config)
        await self.plugin_manager.initialize()

        # Register built-in plugins if enabled
        if plugin_config.get("auto_load_builtin", True):
            register_builtin_plugins(self.plugin_manager)

        logger.info("Plugin manager initialized")

    async def _initialize_deployment_controllers(self):
        """Initialize deployment controllers."""
        # Initialize canary controller
        self.canary_controller = get_canary_controller()

        # Initialize blue-green controller
        self.blue_green_controller = get_blue_green_controller()

        logger.info("Deployment controllers initialized")

    async def _initialize_feature_flag_manager(self):
        """Initialize feature flag manager."""
        feature_flag_config = self.config.get("feature_flags", {})

        self.feature_flag_manager = initialize_feature_flag_manager(feature_flag_config)
        await self.feature_flag_manager.initialize()

        logger.info("Feature flag manager initialized")

    async def _initialize_pipelines(self):
        """Initialize pipeline systems."""
        # Initialize pipeline executor
        self.pipeline_executor = get_pipeline_executor()

        # Initialize optimization pipeline
        self.optimization_pipeline = get_optimization_pipeline()

        logger.info("Pipeline systems initialized")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all deployment components."""
        if not self._initialized:
            return {"status": "not_initialized"}

        status = {"status": "initialized", "components": {}}

        # Plugin manager status
        if self.plugin_manager:
            plugins = self.plugin_manager.get_available_plugins()
            status["components"]["plugins"] = {
                "total_plugins": len(plugins),
                "plugins_by_type": {},
            }

            for plugin in plugins:
                plugin_type = plugin.plugin_type.value
                if (
                    plugin_type
                    not in status["components"]["plugins"]["plugins_by_type"]
                ):
                    status["components"]["plugins"]["plugins_by_type"][plugin_type] = 0
                status["components"]["plugins"]["plugins_by_type"][plugin_type] += 1

        # Canary deployments status
        if self.canary_controller:
            active_canaries = self.canary_controller.list_active_deployments()
            status["components"]["canary"] = {
                "active_deployments": len(active_canaries),
                "deployments": [
                    {
                        "deployment_id": d.config.deployment_id,
                        "status": d.status.value,
                        "traffic_percent": d.current_traffic_percent,
                    }
                    for d in active_canaries
                ],
            }

        # Blue-green deployments status
        if self.blue_green_controller:
            active_bg = self.blue_green_controller.list_active_deployments()
            status["components"]["blue_green"] = {
                "active_deployments": len(active_bg),
                "deployments": [
                    {
                        "deployment_id": d.config.deployment_id,
                        "status": d.status.value,
                        "active_environment": d.active_environment.value,
                        "target_environment": d.target_environment.value,
                    }
                    for d in active_bg
                ],
            }

        # Feature flags status
        if self.feature_flag_manager:
            flags = self.feature_flag_manager.list_flags()
            status["components"]["feature_flags"] = {
                "total_flags": len(flags),
                "active_flags": len([f for f in flags if f.get("enabled", False)]),
            }

        # Pipeline status
        if self.pipeline_executor:
            active_pipelines = self.pipeline_executor.list_active_executions()
            status["components"]["pipelines"] = {
                "active_executions": len(active_pipelines)
            }

        # Optimization status
        if self.optimization_pipeline:
            active_optimizations = self.optimization_pipeline.list_active_executions()
            status["components"]["optimization"] = {
                "active_executions": len(active_optimizations)
            }

        return status

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all deployment components."""
        health = {"healthy": True, "components": {}}

        # Check plugin manager
        try:
            if self.plugin_manager:
                plugins = self.plugin_manager.get_available_plugins()
                health["components"]["plugins"] = {
                    "healthy": True,
                    "plugin_count": len(plugins),
                }
            else:
                health["components"]["plugins"] = {
                    "healthy": False,
                    "error": "Plugin manager not initialized",
                }
                health["healthy"] = False
        except Exception as e:
            health["components"]["plugins"] = {"healthy": False, "error": str(e)}
            health["healthy"] = False

        # Check feature flag manager
        try:
            if self.feature_flag_manager:
                # Try to evaluate a test flag
                from ..deployment.feature_flags import EvaluationContext

                test_context = EvaluationContext(user_id="health_check")

                # This will return a result even if flag doesn't exist
                await self.feature_flag_manager.evaluate_flag(
                    "health_check_flag", test_context
                )

                health["components"]["feature_flags"] = {"healthy": True}
            else:
                health["components"]["feature_flags"] = {
                    "healthy": False,
                    "error": "Feature flag manager not initialized",
                }
                health["healthy"] = False
        except Exception as e:
            health["components"]["feature_flags"] = {"healthy": False, "error": str(e)}
            health["healthy"] = False

        # Check deployment controllers
        health["components"]["canary"] = {"healthy": self.canary_controller is not None}

        health["components"]["blue_green"] = {
            "healthy": self.blue_green_controller is not None
        }

        health["components"]["pipelines"] = {
            "healthy": self.pipeline_executor is not None
        }

        health["components"]["optimization"] = {
            "healthy": self.optimization_pipeline is not None
        }

        return health


def load_deployment_config(
    config_path: str = "config/deployment.yaml",
) -> Dict[str, Any]:
    """Load deployment configuration from file."""
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Deployment config file not found: {config_path}")
            return {}

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded deployment configuration from {config_path}")
        return config

    except Exception as e:
        logger.error(f"Failed to load deployment config from {config_path}: {e}")
        return {}


# Global deployment manager instance
_deployment_manager: Optional[DeploymentManager] = None


async def initialize_deployment_manager(
    config: Optional[Dict[str, Any]] = None,
) -> DeploymentManager:
    """Initialize the global deployment manager."""
    global _deployment_manager

    if config is None:
        config = load_deployment_config()

    _deployment_manager = DeploymentManager(config)
    await _deployment_manager.initialize()

    return _deployment_manager


def get_deployment_manager() -> Optional[DeploymentManager]:
    """Get the global deployment manager instance."""
    return _deployment_manager


async def shutdown_deployment_manager():
    """Shutdown the global deployment manager."""
    global _deployment_manager

    if _deployment_manager:
        await _deployment_manager.shutdown()
        _deployment_manager = None
