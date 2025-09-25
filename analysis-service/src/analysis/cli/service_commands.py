"""
Service management CLI commands for Analysis Service.

This module provides essential CLI commands for service operations.
Single Responsibility: Handle service management CLI operations.
"""

import asyncio
import logging

import click

from shared.database.connection_manager import (
    initialize_databases,
    close_all_databases,
    db_manager,
)

logger = logging.getLogger(__name__)


@click.group()
def service():
    """Service management commands."""
    pass


@service.command()
def health():
    """Check service health status."""

    async def check_service_health():
        try:
            await initialize_databases()

            # Check database health
            db_health = await db_manager.health_check()

            # Display health status
            click.echo("Analysis Service Health Check")
            click.echo("-" * 30)

            # Database status
            all_healthy = True
            for service_name, health in db_health.items():
                status_icon = "✓" if health else "✗"
                status_text = "Healthy" if health else "Unhealthy"
                click.echo(f"{status_icon} {service_name}: {status_text}")
                if not health:
                    all_healthy = False

            # Overall status
            if all_healthy:
                click.echo("\n✓ Service is healthy")
            else:
                click.echo("\n✗ Service has issues")
                raise click.ClickException("Service health check failed")

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            click.echo(f"✗ Health check failed: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(check_service_health())


@service.command()
@click.option("--days", type=int, default=30, help="Days of data to retain")
def cleanup(days: int):
    """Cleanup old data and logs."""

    async def cleanup_service():
        try:
            await initialize_databases()

            from ..tenancy import AnalyticsManager
            from ..plugins import PluginDatabaseManager

            analytics_manager = AnalyticsManager()
            plugin_db_manager = PluginDatabaseManager()

            # Cleanup old analytics
            analytics_cleaned = await analytics_manager.cleanup_old_analytics(days)

            # Cleanup old plugin executions
            executions_cleaned = await plugin_db_manager.cleanup_old_executions(days)

            click.echo("✓ Cleanup completed:")
            click.echo(f"  Analytics records cleaned: {analytics_cleaned}")
            click.echo(f"  Plugin execution records cleaned: {executions_cleaned}")

        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            click.echo(f"✗ Cleanup failed: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(cleanup_service())


if __name__ == "__main__":
    service()
