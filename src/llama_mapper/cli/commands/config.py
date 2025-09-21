"""Configuration-related CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import click

from ...config import ConfigManager
from ...config.validator import validate_configuration
from ...logging import get_logger
from ..utils import get_config_manager


def register(main: click.Group) -> None:
    """Attach config-centric commands to the root CLI."""

    @main.command()
    @click.option(
        "--data-dir",
        type=click.Path(exists=False, file_okay=False),
        help="Directory containing taxonomy.yaml, frameworks.yaml, and detector YAMLs",
    )
    @click.pass_context
    def validate_config(ctx: click.Context, data_dir: Optional[str]) -> None:
        """Validate taxonomy, frameworks, and detector mappings."""
        logger = get_logger(__name__)
        base = Path(data_dir) if data_dir else None

        click.echo("Validating configuration (taxonomy/frameworks/detectors)...\n")
        result = validate_configuration(base)

        click.echo(f"Data directory: {result.data_dir}")

        click.echo("\n[Taxonomy]")
        if result.taxonomy.ok:
            click.echo("  ✓ taxonomy.yaml loaded successfully")
        else:
            click.echo("  ✗ taxonomy.yaml validation failed")
            for err in result.taxonomy.errors:
                click.echo(f"    - {err}")

        click.echo("\n[Frameworks]")
        if result.frameworks.ok:
            frameworks = ", ".join(
                cast(list[str], result.frameworks.details.get("frameworks", []))
            )
            click.echo(f"  ✓ frameworks.yaml valid (frameworks: {frameworks})")
        else:
            click.echo("  ✗ frameworks.yaml validation failed")
            for err in result.frameworks.errors:
                click.echo(f"    - {err}")

        click.echo("\n[Detectors]")
        if result.detectors.ok:
            detectors_found = cast(
                list[str], result.detectors.details.get("detectors_found", [])
            )
            dets = ", ".join(detectors_found)
            click.echo(f"  ✓ detector YAMLs valid ({len(detectors_found)} found)")
            if dets:
                click.echo(f"    - {dets}")
        else:
            click.echo("  ✗ detector YAML validation failed")
            for err in result.detectors.errors:
                click.echo(f"    - {err}")

        if not result.ok:
            logger.error("Configuration validation failed")
            ctx.exit(1)
        else:
            logger.info("Configuration validation successful")
            click.echo("\nAll configuration checks passed ✓")

    @main.command()
    @click.option("--tenant", type=str, help="Tenant ID to apply tenant overrides")
    @click.option(
        "--environment",
        type=str,
        help="Environment name to apply environment overrides (development|staging|production)",
    )
    @click.option("--format", "fmt", type=click.Choice(["json"]), default=None)
    @click.pass_context
    def show_config(
        ctx: click.Context,
        tenant: Optional[str],
        environment: Optional[str],
        fmt: Optional[str],
    ) -> None:
        """Display current configuration with optional tenant/environment overlays."""
        base_cm = get_config_manager(ctx)
        if tenant or environment:
            config_manager = ConfigManager(
                config_path=base_cm.config_path,
                tenant_id=tenant,
                environment=environment,
            )
        else:
            config_manager = base_cm

        config_dict = config_manager.get_config_dict()

        active_env = config_dict.get("environment") or getattr(
            getattr(config_manager, "_config_data", {}), "get", lambda *_: None
        )("environment")

        if fmt == "json":
            masked: dict[str, dict[str, object]] = {}
            for section, values in config_dict.items():
                if section == "environment":
                    masked[section] = values
                    continue
                sec_out: dict[str, object] = {}
                for key, value in values.items():
                    if key in {"api_key", "secret_key"}:
                        sec_out[key] = "***MASKED***" if value else None
                    else:
                        sec_out[key] = value
                masked[section] = sec_out
            click.echo(click.style("Configuration (masked):", bold=True))
            import json as _json

            click.echo(_json.dumps(masked, indent=2))
            return

        click.echo("Configuration overview:\n")
        click.echo(f"Active environment: {active_env}")

        for section, values in config_dict.items():
            click.echo(f"\n[{section}]")
            for key, value in values.items():
                if key in ["api_key", "secret_key"]:
                    value = "***MASKED***" if value else None
                click.echo(f"  {key}: {value}")

    @main.command()
    @click.option(
        "--output", "-o", type=click.Path(), help="Output path for configuration file"
    )
    @click.pass_context
    def init_config(ctx: click.Context, output: Optional[str]) -> None:
        """Initialize a new configuration file with defaults."""
        config_manager = get_config_manager(ctx)
        logger = get_logger(__name__)

        output_path = Path(output) if output else config_manager.config_path

        try:
            config_manager.save_config(output_path)
            logger.info("Configuration file created", path=str(output_path))
            click.echo(f"✓ Configuration file created at: {output_path}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create configuration file", error=str(exc))
            click.echo(f"✗ Failed to create configuration file: {exc}")
            ctx.exit(1)
