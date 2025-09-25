"""
Plugin management CLI commands for Analysis Service.

This module provides essential CLI commands for plugin operations.
Single Responsibility: Handle plugin management CLI operations.
"""

import asyncio
import json
import logging

import click

from ..dependencies import get_plugin_manager, get_plugin_db_manager
from shared.database.connection_manager import initialize_databases, close_all_databases

logger = logging.getLogger(__name__)


@click.group()
def plugin():
    """Plugin management commands."""
    pass


@plugin.command("list")
def list_plugins():
    """List all plugins."""

    async def list_all_plugins():
        try:
            await initialize_databases()
            plugin_db_manager = await get_plugin_db_manager()

            plugins = await plugin_db_manager.list_plugins()

            if not plugins:
                click.echo("No plugins found.")
                return

            click.echo(f"{'Plugin Name':<30} {'Status':<10} {'Type':<20}")
            click.echo("-" * 60)
            for plugin in plugins:
                click.echo(
                    f"{plugin['plugin_name']:<30} {plugin['status']:<10} {plugin['plugin_type']:<20}"
                )

        except Exception as e:
            logger.error("Failed to list plugins", error=str(e))
            click.echo(f"✗ Failed to list plugins: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(list_all_plugins())


@plugin.command()
@click.argument("plugin_name")
def status(plugin_name: str):
    """Get plugin status and health."""

    async def get_plugin_status():
        try:
            await initialize_databases()
            plugin_manager = await get_plugin_manager()

            health_info = await plugin_manager.get_plugin_health(plugin_name)

            click.echo(f"Plugin: {plugin_name}")
            click.echo("-" * 30)

            status = health_info.get("status", "unknown")
            status_icon = "✓" if status == "healthy" else "✗"
            click.echo(f"Status: {status_icon} {status}")

            if "last_check" in health_info:
                click.echo(f"Last Check: {health_info['last_check']}")

            if "error" in health_info:
                click.echo(f"Error: {health_info['error']}")

        except Exception as e:
            logger.error(
                "Failed to get plugin status", error=str(e), plugin_name=plugin_name
            )
            click.echo(f"✗ Failed to get plugin status: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(get_plugin_status())


@plugin.command()
def init():
    """Initialize all plugins."""

    async def initialize_all_plugins():
        try:
            await initialize_databases()
            plugin_manager = await get_plugin_manager()

            results = await plugin_manager.initialize_all_plugins()

            successful = sum(1 for success in results.values() if success)
            failed = len(results) - successful

            click.echo("Plugin initialization completed:")
            click.echo(f"  Successful: {successful}")
            click.echo(f"  Failed: {failed}")

            if failed > 0:
                click.echo("\nFailed plugins:")
                for plugin_name, success in results.items():
                    if not success:
                        click.echo(f"  ✗ {plugin_name}")

        except Exception as e:
            logger.error("Failed to initialize plugins", error=str(e))
            click.echo(f"✗ Failed to initialize plugins: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(initialize_all_plugins())


if __name__ == "__main__":
    plugin()
