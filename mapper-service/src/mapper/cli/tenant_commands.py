"""
Tenant management CLI commands for Mapper Service.

Single Responsibility: Handle tenant operations via CLI.
Provides commands for tenant management, configuration, and monitoring.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from typing import Optional

import click

from ..tenancy import get_tenant_service


@click.group()
def tenant():
    """Tenant management - create, configure, and monitor tenants."""
    pass


@tenant.command()
@click.option("--tenant-id", "-t", required=True, help="Unique tenant identifier")
@click.option("--name", "-n", required=True, help="Tenant display name")
@click.option("--description", "-d", help="Tenant description")
@click.option(
    "--tier",
    type=click.Choice(["free", "basic", "premium", "enterprise"]),
    default="basic",
    help="Tenant tier",
)
@click.option("--enabled/--disabled", default=True, help="Enable or disable the tenant")
def create(
    tenant_id: str, name: str, description: Optional[str], tier: str, enabled: bool
):
    """Create a new tenant.

    Examples:
        mapper tenant create -t acme-corp -n "ACME Corporation" --tier enterprise
        mapper tenant create -t test-tenant -n "Test Tenant" -d "Development tenant"
    """

    async def _create():
        try:
            tenant_service = get_tenant_service()
            if not tenant_service:
                click.echo("❌ Tenant service not available", err=True)
                sys.exit(1)

            click.echo(f"Creating tenant '{tenant_id}'...", err=True)

            # Create tenant
            tenant = await tenant_service.create_tenant(
                tenant_id=tenant_id,
                name=name,
                description=description,
                settings={"tier": tier},
                enabled=enabled,
            )

            click.echo(f"✓ Tenant '{tenant_id}' created successfully")
            click.echo(f"Name: {tenant.name}")
            click.echo(f"Tier: {tier}")
            click.echo(f"Status: {'Enabled' if tenant.enabled else 'Disabled'}")
            click.echo(f"Created: {tenant.created_at}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_create())


@tenant.command()
@click.option("--enabled", type=bool, help="Filter by enabled status (true/false)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def list(enabled: Optional[bool], output_format: str):
    """List all tenants.

    Examples:
        mapper tenant list
        mapper tenant list --enabled true --format json
    """

    async def _list():
        try:
            tenant_service = get_tenant_service()
            if not tenant_service:
                click.echo("❌ Tenant service not available", err=True)
                sys.exit(1)

            tenants = await tenant_service.list_tenants(enabled=enabled)

            if not tenants:
                click.echo("No tenants found")
                return

            if output_format == "json":
                tenants_data = [
                    {
                        "tenant_id": t.tenant_id,
                        "name": t.name,
                        "description": t.description,
                        "enabled": t.enabled,
                        "created_at": t.created_at.isoformat(),
                        "updated_at": t.updated_at.isoformat(),
                    }
                    for t in tenants
                ]
                click.echo(json.dumps(tenants_data, indent=2))

            else:  # table format
                click.echo("Tenants")
                click.echo("=" * 80)
                click.echo(
                    f"{'Tenant ID':<20} {'Name':<25} {'Status':<10} {'Created':<20}"
                )
                click.echo("-" * 80)

                for tenant in tenants:
                    status = "Enabled" if tenant.enabled else "Disabled"
                    created = tenant.created_at.strftime("%Y-%m-%d %H:%M")
                    click.echo(
                        f"{tenant.tenant_id:<20} {tenant.name:<25} {status:<10} {created:<20}"
                    )

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_list())


@tenant.command()
@click.option("--tenant-id", "-t", required=True, help="Tenant identifier")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def show(tenant_id: str, output_format: str):
    """Show detailed tenant information.

    Examples:
        mapper tenant show -t acme-corp
        mapper tenant show -t test-tenant --format json
    """

    async def _show():
        try:
            tenant_service = get_tenant_service()
            if not tenant_service:
                click.echo("❌ Tenant service not available", err=True)
                sys.exit(1)

            tenant = await tenant_service.get_tenant(tenant_id)
            if not tenant:
                click.echo(f"❌ Tenant '{tenant_id}' not found")
                sys.exit(1)

            if output_format == "json":
                tenant_data = {
                    "tenant_id": tenant.tenant_id,
                    "name": tenant.name,
                    "description": tenant.description,
                    "settings": tenant.settings,
                    "enabled": tenant.enabled,
                    "created_at": tenant.created_at.isoformat(),
                    "updated_at": tenant.updated_at.isoformat(),
                }
                click.echo(json.dumps(tenant_data, indent=2))

            else:  # table format
                click.echo(f"Tenant Details: {tenant_id}")
                click.echo("=" * 50)
                click.echo(f"Name: {tenant.name}")
                click.echo(f"Description: {tenant.description or 'N/A'}")
                click.echo(f"Status: {'Enabled' if tenant.enabled else 'Disabled'}")
                click.echo(f"Created: {tenant.created_at}")
                click.echo(f"Updated: {tenant.updated_at}")

                if tenant.settings:
                    click.echo("\nSettings:")
                    for key, value in tenant.settings.items():
                        click.echo(f"  {key}: {value}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_show())


@tenant.command()
@click.option("--tenant-id", "-t", required=True, help="Tenant identifier")
@click.option(
    "--days", "-d", type=int, default=30, help="Number of days of usage data to show"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format",
)
def usage(tenant_id: str, days: int, output_format: str):
    """Show tenant usage statistics.

    Examples:
        mapper tenant usage -t acme-corp
        mapper tenant usage -t test-tenant -d 7 --format json
    """

    async def _usage():
        try:
            tenant_service = get_tenant_service()
            if not tenant_service:
                click.echo("❌ Tenant service not available", err=True)
                sys.exit(1)

            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            usage_data = await tenant_service.get_tenant_usage(
                tenant_id=tenant_id, start_date=start_date, end_date=end_date
            )

            if not usage_data:
                click.echo(f"❌ No usage data found for tenant '{tenant_id}'")
                sys.exit(1)

            if output_format == "json":
                usage_dict = {
                    "tenant_id": usage_data.tenant_id,
                    "period_start": usage_data.period_start.isoformat(),
                    "period_end": usage_data.period_end.isoformat(),
                    "total_requests": usage_data.total_requests,
                    "successful_requests": usage_data.successful_requests,
                    "failed_requests": usage_data.failed_requests,
                    "avg_response_time_ms": usage_data.avg_response_time_ms,
                    "total_cost": usage_data.total_cost,
                    "usage_by_detector": usage_data.usage_by_detector,
                    "usage_by_framework": usage_data.usage_by_framework,
                }
                click.echo(json.dumps(usage_dict, indent=2))

            else:  # table format
                success_rate = (
                    (usage_data.successful_requests / usage_data.total_requests * 100)
                    if usage_data.total_requests > 0
                    else 0
                )

                click.echo(f"Usage Statistics: {tenant_id}")
                click.echo("=" * 50)
                click.echo(
                    f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                )
                click.echo(f"Total Requests: {usage_data.total_requests:,}")
                click.echo(
                    f"Successful: {usage_data.successful_requests:,} ({success_rate:.1f}%)"
                )
                click.echo(f"Failed: {usage_data.failed_requests:,}")
                click.echo(
                    f"Avg Response Time: {usage_data.avg_response_time_ms:.1f}ms"
                )
                click.echo(f"Total Cost: ${usage_data.total_cost:.2f}")

                if usage_data.usage_by_detector:
                    click.echo("\nUsage by Detector:")
                    for detector, count in usage_data.usage_by_detector.items():
                        click.echo(f"  {detector}: {count:,}")

                if usage_data.usage_by_framework:
                    click.echo("\nUsage by Framework:")
                    for framework, count in usage_data.usage_by_framework.items():
                        click.echo(f"  {framework}: {count:,}")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_usage())


@tenant.command()
@click.option("--tenant-id", "-t", required=True, help="Tenant identifier")
def enable(tenant_id: str):
    """Enable a tenant.

    Examples:
        mapper tenant enable -t acme-corp
    """

    async def _enable():
        try:
            tenant_service = get_tenant_service()
            if not tenant_service:
                click.echo("❌ Tenant service not available", err=True)
                sys.exit(1)

            success = await tenant_service.enable_tenant(tenant_id)

            if success:
                click.echo(f"✓ Tenant '{tenant_id}' enabled successfully")
            else:
                click.echo(f"❌ Failed to enable tenant '{tenant_id}' (not found)")
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_enable())


@tenant.command()
@click.option("--tenant-id", "-t", required=True, help="Tenant identifier")
def disable(tenant_id: str):
    """Disable a tenant.

    Examples:
        mapper tenant disable -t test-tenant
    """

    async def _disable():
        try:
            tenant_service = get_tenant_service()
            if not tenant_service:
                click.echo("❌ Tenant service not available", err=True)
                sys.exit(1)

            success = await tenant_service.disable_tenant(tenant_id)

            if success:
                click.echo(f"✓ Tenant '{tenant_id}' disabled successfully")
            else:
                click.echo(f"❌ Failed to disable tenant '{tenant_id}' (not found)")
                sys.exit(1)

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

    asyncio.run(_disable())
