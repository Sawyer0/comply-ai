"""
Tenant management CLI commands for Analysis Service.

This module provides essential CLI commands for tenant operations.
Single Responsibility: Handle tenant management CLI operations.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta

import click

from ..dependencies import get_tenant_manager, get_analytics_manager
from ..tenancy import TenantRequest, QuotaRequest
from shared.database.connection_manager import initialize_databases, close_all_databases

logger = logging.getLogger(__name__)


@click.group()
def tenant():
    """Tenant management commands."""
    pass


@tenant.command()
@click.argument("tenant_id")
@click.option("--name", required=True, help="Tenant name")
def create(tenant_id: str, name: str):
    """Create a new tenant with default settings."""

    async def create_tenant():
        try:
            await initialize_databases()
            tenant_manager = await get_tenant_manager()

            request = TenantRequest(
                name=name,
                status="active",
                default_confidence_threshold=0.8,
                enable_ml_analysis=True,
                enable_statistical_analysis=True,
                enable_pattern_recognition=True,
                quality_alert_threshold=0.7,
                log_level="INFO",
            )

            tenant_config = await tenant_manager.create_tenant(tenant_id, request)

            click.echo(f"✓ Created tenant: {tenant_config.tenant_id}")
            click.echo(f"  Name: {tenant_config.name}")
            click.echo(f"  Status: {tenant_config.status.value}")

        except Exception as e:
            logger.error("Failed to create tenant", error=str(e), tenant_id=tenant_id)
            click.echo(f"✗ Failed to create tenant: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(create_tenant())


@tenant.command()
@click.argument("tenant_id")
@click.option(
    "--format", "output_format", type=click.Choice(["json", "table"]), default="table"
)
def get(tenant_id: str, output_format: str):
    """Get tenant configuration."""

    async def get_tenant():
        try:
            await initialize_databases()
            tenant_manager = await get_tenant_manager()

            tenant_config = await tenant_manager.get_tenant_config(tenant_id)
            if not tenant_config:
                raise click.ClickException(f"Tenant not found: {tenant_id}")

            if output_format == "json":
                click.echo(json.dumps(tenant_config.to_dict(), indent=2, default=str))
            else:
                click.echo(f"Tenant ID: {tenant_config.tenant_id}")
                click.echo(f"Name: {tenant_config.name}")
                click.echo(f"Status: {tenant_config.status.value}")
                click.echo(
                    f"Confidence Threshold: {tenant_config.default_confidence_threshold}"
                )
                click.echo(
                    f"ML Analysis: {'Enabled' if tenant_config.enable_ml_analysis else 'Disabled'}"
                )
                click.echo(
                    f"Statistical Analysis: {'Enabled' if tenant_config.enable_statistical_analysis else 'Disabled'}"
                )
                click.echo(
                    f"Pattern Recognition: {'Enabled' if tenant_config.enable_pattern_recognition else 'Disabled'}"
                )
                click.echo(f"Created: {tenant_config.created_at}")
                click.echo(f"Updated: {tenant_config.updated_at}")

                click.echo("\nResource Quotas:")
                for resource_type, quota in tenant_config.quotas.items():
                    click.echo(
                        f"  {resource_type.value}: {quota.current_usage}/{quota.limit} ({quota.remaining()} remaining)"
                    )

        except Exception as e:
            click.echo(f"✗ Failed to get tenant: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(get_tenant())


@tenant.command("list")
def list_tenants():
    """List all tenants."""

    async def list_all_tenants():
        try:
            await initialize_databases()
            tenant_manager = await get_tenant_manager()

            tenants = await tenant_manager.list_tenants()

            if not tenants:
                click.echo("No tenants found.")
                return

            click.echo(f"{'Tenant ID':<20} {'Name':<30} {'Status':<10}")
            click.echo("-" * 60)
            for tenant in tenants:
                click.echo(
                    f"{tenant.tenant_id:<20} {tenant.name:<30} {tenant.status.value:<10}"
                )

        except Exception as e:
            logger.error("Failed to list tenants", error=str(e))
            click.echo(f"✗ Failed to list tenants: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(list_all_tenants())


@tenant.command()
@click.argument("tenant_id")
@click.argument(
    "resource_type", type=click.Choice(["analysis_requests", "batch_requests"])
)
@click.argument("limit", type=int)
def quota(tenant_id: str, resource_type: str, limit: int):
    """Set resource quota for a tenant."""

    async def set_tenant_quota():
        try:
            await initialize_databases()
            tenant_manager = await get_tenant_manager()

            quota_request = QuotaRequest(
                resource_type=resource_type, limit=limit, period_hours=24
            )

            success = await tenant_manager.set_resource_quota(tenant_id, quota_request)
            if not success:
                raise click.ClickException(
                    f"Failed to set quota for tenant: {tenant_id}"
                )

            click.echo(f"✓ Set {resource_type} quota to {limit} for tenant {tenant_id}")

        except Exception as e:
            logger.error("Failed to set quota", error=str(e), tenant_id=tenant_id)
            click.echo(f"✗ Failed to set quota: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(set_tenant_quota())


@tenant.command()
@click.argument("tenant_id")
def stats(tenant_id: str):
    """Get tenant usage statistics."""

    async def get_tenant_stats():
        try:
            await initialize_databases()
            analytics_manager = await get_analytics_manager()

            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=7)

            analytics = await analytics_manager.get_analytics(
                tenant_id, start_date, end_date
            )

            if not analytics:
                click.echo(f"No statistics found for tenant {tenant_id}")
                return

            click.echo(f"Statistics for tenant {tenant_id} (last 7 days):")
            click.echo("-" * 40)
            click.echo(f"Total Requests: {analytics.total_requests}")
            click.echo(f"Successful: {analytics.successful_requests}")
            click.echo(f"Failed: {analytics.failed_requests}")

            if analytics.total_requests > 0:
                success_rate = (
                    analytics.successful_requests / analytics.total_requests
                ) * 100
                click.echo(f"Success Rate: {success_rate:.1f}%")

            click.echo(f"Avg Response Time: {analytics.avg_response_time_ms:.0f}ms")
            click.echo(f"Avg Confidence: {analytics.avg_confidence_score:.2f}")

        except Exception as e:
            logger.error("Failed to get analytics", error=str(e), tenant_id=tenant_id)
            click.echo(f"✗ Failed to get statistics: {e}")
            raise click.ClickException(str(e))
        finally:
            await close_all_databases()

    asyncio.run(get_tenant_stats())


if __name__ == "__main__":
    tenant()
