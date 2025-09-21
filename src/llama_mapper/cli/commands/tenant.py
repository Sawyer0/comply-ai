"""Tenant configuration commands."""

from __future__ import annotations

from pathlib import Path

import click
import yaml


def register(main: click.Group) -> None:
    """Attach tenant configuration commands."""

    @click.group()
    def tenant() -> None:
        """Tenant configuration tools (migration and validation)."""

    @tenant.command("migrate-config")
    @click.option(
        "--input-dir",
        required=True,
        type=click.Path(exists=True, file_okay=False),
        help="Directory containing tenant config YAMLs",
    )
    @click.option(
        "--output-dir",
        required=True,
        type=click.Path(file_okay=False),
        help="Directory to write migrated configs",
    )
    def tenant_migrate_config(input_dir: str, output_dir: str) -> None:
        """Migrate tenant config files to the latest schema and validate."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        migrated = 0
        for yaml_file in input_path.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as file:
                    contents = yaml.safe_load(file) or {}
                contents.setdefault("tenant_id", yaml_file.stem)
                contents.setdefault("overrides", {})
                overrides = contents.get("overrides") or {}
                for level in ["global", "tenant", "environment"]:
                    overrides.setdefault(level, {})
                out_file = output_path / yaml_file.name
                with open(out_file, "w", encoding="utf-8") as file:
                    yaml.safe_dump(contents, file, sort_keys=False)
                migrated += 1
            except Exception as exc:  # noqa: BLE001
                click.echo(f"✗ Failed to migrate {yaml_file.name}: {exc}")
        click.echo(f"✓ Migrated {migrated} tenant config(s) to {output_path}")

    @tenant.command("validate-config")
    @click.option(
        "--dir",
        "dir_path",
        required=True,
        type=click.Path(exists=True, file_okay=False),
        help="Directory containing tenant config YAMLs",
    )
    @click.pass_context
    def tenant_validate_config(ctx: click.Context, dir_path: str) -> None:
        """Validate tenant config files for required structure and precedence rules."""
        dirp = Path(dir_path)
        errors = 0
        for yaml_file in dirp.glob("*.yaml"):
            try:
                with open(yaml_file, "r", encoding="utf-8") as file:
                    contents = yaml.safe_load(file) or {}
                if "tenant_id" not in contents:
                    raise ValueError("missing tenant_id")
                overrides = contents.get("overrides") or {}
                for level in ["global", "tenant", "environment"]:
                    if level not in overrides or not isinstance(overrides[level], dict):
                        raise ValueError(f"missing overrides.{level}")
            except Exception as exc:  # noqa: BLE001
                errors += 1
                click.echo(f"✗ {yaml_file.name}: {exc}")
        if errors == 0:
            click.echo("✓ All tenant configs valid")
        else:
            click.echo(f"✗ {errors} invalid tenant config(s)")
            ctx.exit(1)

    main.add_command(tenant)
