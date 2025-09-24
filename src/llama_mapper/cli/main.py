"""Root Click group for the Llama Mapper CLI."""

from __future__ import annotations

from typing import Optional

import click

from ..config.manager import ConfigManager
from ..logging import get_logger, setup_logging
from .commands import register_all
from .core import AutoDiscoveryRegistry, PluginManager


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=False), help="Configuration file path"
)
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
)
@click.option(
    "--plugin-dir",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing CLI plugins",
)
@click.pass_context
def main(
    ctx: click.Context, config: Optional[str], log_level: str, plugin_dir: Optional[str]
) -> None:
    """Llama Mapper CLI - Fine-tuned model for detector output mapping."""
    ctx.ensure_object(dict)

    config_manager = ConfigManager(config_path=config)
    ctx.obj["config"] = config_manager

    setup_logging(
        log_level=log_level or config_manager.monitoring.log_level,
        log_format="console",
        enable_privacy_filter=True,
    )

    logger = get_logger(__name__)
    logger.info(
        "Llama Mapper CLI initialized",
        config_path=str(config_manager.config_path),
        log_level=log_level,
    )

    # Initialize command registry (for plugins only)
    registry = AutoDiscoveryRegistry()
    ctx.obj["registry"] = registry

    # Load plugins if plugin directory is specified
    if plugin_dir:
        plugin_manager = PluginManager(registry)
        plugin_manager.add_plugin_directory(plugin_dir)
        plugin_manager.load_plugins_from_directory()
        logger.info("Loaded plugins from: %s", plugin_dir)

        # Attach plugin commands to main group
        registry.attach_to_main(main)


# Register built-in commands at module import time
_builtin_registry = AutoDiscoveryRegistry()
register_all(_builtin_registry)
_builtin_registry.attach_to_main(main)


if __name__ == "__main__":  # pragma: no cover
    main()  # pylint: disable=no-value-for-parameter
