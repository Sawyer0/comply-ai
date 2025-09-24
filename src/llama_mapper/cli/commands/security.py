"""
CLI commands for security operations including secrets rotation.
"""

import asyncio
from typing import Optional

import click
import structlog

from ...config.manager import ConfigManager
from ...security.secrets_manager import SecretsManager
from ...security.rotation import SecretsRotationManager, RotationStatus
from ...utils.correlation import generate_correlation_id

logger = structlog.get_logger(__name__).bind(component="security_cli")


@click.group()
def security():
    """Security management commands."""
    pass


@security.command()
@click.option('--database', '-d', help='Database name to rotate credentials for')
@click.option('--all-databases', is_flag=True, help='Rotate credentials for all databases')
@click.option('--dry-run', is_flag=True, help='Show what would be rotated without executing')
def rotate_db_credentials(database: Optional[str], all_databases: bool, dry_run: bool):
    """Rotate database credentials."""
    generate_correlation_id()
    
    if not database and not all_databases:
        click.echo("Error: Must specify either --database or --all-databases")
        return
    
    if database and all_databases:
        click.echo("Error: Cannot specify both --database and --all-databases")
        return
    
    async def _rotate():
        config_manager = ConfigManager()
        secrets_manager = SecretsManager(config_manager.settings)
        rotation_manager = SecretsRotationManager(secrets_manager)
        
        if dry_run:
            click.echo("DRY RUN: Would rotate credentials for:")
            if all_databases:
                databases = await rotation_manager._get_database_list()
                for db_name in databases:
                    click.echo(f"  - {db_name}")
            else:
                click.echo(f"  - {database}")
            return
        
        if all_databases:
            databases = await rotation_manager._get_database_list()
            click.echo(f"Rotating credentials for {len(databases)} databases...")
            
            for db_name in databases:
                click.echo(f"Rotating credentials for {db_name}...")
                result = await rotation_manager.rotate_database_credentials(db_name)
                
                if result.status == RotationStatus.COMPLETED:
                    click.echo(f"✓ Successfully rotated credentials for {db_name}")
                elif result.status == RotationStatus.ROLLED_BACK:
                    click.echo(f"⚠ Rolled back credentials for {db_name}: {result.error_message}")
                else:
                    click.echo(f"✗ Failed to rotate credentials for {db_name}: {result.error_message}")
        else:
            click.echo(f"Rotating credentials for {database}...")
            result = await rotation_manager.rotate_database_credentials(database)
            
            if result.status == RotationStatus.COMPLETED:
                click.echo(f"✓ Successfully rotated credentials for {database}")
                click.echo(f"  New version: {result.new_version}")
            elif result.status == RotationStatus.ROLLED_BACK:
                click.echo(f"⚠ Rolled back credentials for {database}")
                click.echo(f"  Reason: {result.error_message}")
            else:
                click.echo(f"✗ Failed to rotate credentials for {database}")
                click.echo(f"  Error: {result.error_message}")
    
    try:
        asyncio.run(_rotate())
    except Exception as e:
        logger.error("Database credential rotation failed", error=str(e))
        click.echo(f"✗ Rotation failed: {e}")


@security.command()
@click.option('--tenant-id', '-t', help='Tenant ID to rotate API key for')
@click.option('--all-tenants', is_flag=True, help='Rotate API keys for all active tenants')
@click.option('--dry-run', is_flag=True, help='Show what would be rotated without executing')
def rotate_api_keys(tenant_id: Optional[str], all_tenants: bool, dry_run: bool):
    """Rotate API keys."""
    generate_correlation_id()
    
    if not tenant_id and not all_tenants:
        click.echo("Error: Must specify either --tenant-id or --all-tenants")
        return
    
    if tenant_id and all_tenants:
        click.echo("Error: Cannot specify both --tenant-id and --all-tenants")
        return
    
    async def _rotate():
        config_manager = ConfigManager()
        secrets_manager = SecretsManager(config_manager.settings)
        rotation_manager = SecretsRotationManager(secrets_manager)
        
        if dry_run:
            click.echo("DRY RUN: Would rotate API keys for:")
            if all_tenants:
                tenants = await rotation_manager._get_active_tenants()
                for t_id in tenants:
                    click.echo(f"  - {t_id}")
            else:
                click.echo(f"  - {tenant_id}")
            return
        
        if all_tenants:
            tenants = await rotation_manager._get_active_tenants()
            click.echo(f"Rotating API keys for {len(tenants)} tenants...")
            
            for t_id in tenants:
                click.echo(f"Rotating API key for {t_id}...")
                result = await rotation_manager.rotate_api_keys(t_id)
                
                if result.status == RotationStatus.COMPLETED:
                    click.echo(f"✓ Successfully rotated API key for {t_id}")
                else:
                    click.echo(f"✗ Failed to rotate API key for {t_id}: {result.error_message}")
        else:
            click.echo(f"Rotating API key for {tenant_id}...")
            result = await rotation_manager.rotate_api_keys(tenant_id)
            
            if result.status == RotationStatus.COMPLETED:
                click.echo(f"✓ Successfully rotated API key for {tenant_id}")
                click.echo(f"  New version: {result.new_version}")
            else:
                click.echo(f"✗ Failed to rotate API key for {tenant_id}")
                click.echo(f"  Error: {result.error_message}")
    
    try:
        asyncio.run(_rotate())
    except Exception as e:
        logger.error("API key rotation failed", error=str(e))
        click.echo(f"✗ Rotation failed: {e}")


@security.command()
def start_rotation_scheduler():
    """Start the automated rotation scheduler."""
    generate_correlation_id()
    
    async def _start_scheduler():
        config_manager = ConfigManager()
        secrets_manager = SecretsManager(config_manager.settings)
        rotation_manager = SecretsRotationManager(secrets_manager)
        
        click.echo("Starting automated rotation scheduler...")
        rotation_manager.schedule_rotation_jobs()
        
        click.echo("Rotation scheduler started. Press Ctrl+C to stop.")
        try:
            # Keep the scheduler running
            while True:
                await asyncio.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            click.echo("\nStopping rotation scheduler...")
            await rotation_manager.stop_scheduled_rotations()
            click.echo("Rotation scheduler stopped.")
    
    try:
        asyncio.run(_start_scheduler())
    except Exception as e:
        logger.error("Rotation scheduler failed", error=str(e))
        click.echo(f"✗ Scheduler failed: {e}")


@security.command()
@click.option('--secret-name', '-s', help='Show history for specific secret')
def rotation_history(secret_name: Optional[str]):
    """Show rotation history."""
    generate_correlation_id()
    
    async def _show_history():
        config_manager = ConfigManager()
        secrets_manager = SecretsManager(config_manager.settings)
        rotation_manager = SecretsRotationManager(secrets_manager)
        
        history = rotation_manager.get_rotation_history(secret_name)
        
        if not history:
            click.echo("No rotation history found.")
            return
        
        for secret, results in history.items():
            click.echo(f"\n{secret}:")
            click.echo("-" * (len(secret) + 1))
            
            for result in results[-5:]:  # Show last 5 results
                status_icon = {
                    RotationStatus.COMPLETED: "✓",
                    RotationStatus.FAILED: "✗",
                    RotationStatus.ROLLED_BACK: "⚠",
                    RotationStatus.IN_PROGRESS: "⏳",
                    RotationStatus.PENDING: "⏸"
                }.get(result.status, "?")
                
                click.echo(f"  {status_icon} {result.status.value}")
                if result.new_version:
                    click.echo(f"    Version: {result.new_version}")
                if result.error_message:
                    click.echo(f"    Error: {result.error_message}")
    
    try:
        asyncio.run(_show_history())
    except Exception as e:
        logger.error("Failed to show rotation history", error=str(e))
        click.echo(f"✗ Failed to show history: {e}")


@security.command()
@click.argument('input_text')
@click.option('--level', type=click.Choice(['basic', 'strict', 'paranoid']), default='strict')
def test_sanitization(input_text: str, level: str):
    """Test input sanitization on provided text."""
    from ...security.input_sanitization import SecuritySanitizer, SanitizationLevel
    
    generate_correlation_id()
    
    level_map = {
        'basic': SanitizationLevel.BASIC,
        'strict': SanitizationLevel.STRICT,
        'paranoid': SanitizationLevel.PARANOID
    }
    
    sanitizer = SecuritySanitizer(level_map[level])
    
    click.echo(f"Original text: {input_text}")
    click.echo(f"Sanitization level: {level}")
    click.echo("-" * 50)
    
    try:
        # Detect attacks first
        attacks = sanitizer.detect_malicious_patterns(input_text)
        if attacks:
            click.echo(f"⚠ Detected attacks: {', '.join([a.value for a in attacks])}")
        else:
            click.echo("✓ No malicious patterns detected")
        
        # Sanitize the input
        sanitized = sanitizer.sanitize_string(input_text)
        click.echo(f"Sanitized text: {sanitized}")
        
        if input_text != sanitized:
            click.echo("⚠ Input was modified during sanitization")
        else:
            click.echo("✓ Input passed sanitization unchanged")
            
    except ValueError as e:
        click.echo(f"✗ Sanitization failed: {e}")


def register(main_group):
    """Register security commands with the main CLI group."""
    main_group.add_command(security)


if __name__ == '__main__':
    security()