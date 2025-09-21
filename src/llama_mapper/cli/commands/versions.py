"""Version management commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from ...versioning import VersionManager


def register(main: click.Group) -> None:
    """Attach version related commands."""

    @click.group()
    def versions() -> None:
        """Version management commands."""

    @versions.command("show")
    @click.option(
        "--data-dir",
        type=click.Path(exists=False),
        help="Directory with taxonomy/frameworks/detectors",
    )
    @click.option(
        "--registry",
        type=click.Path(exists=False),
        help="Path to model versions registry (versions.json)",
    )
    def versions_show(data_dir: Optional[str], registry: Optional[str]) -> None:
        """Show current version snapshot as JSON."""
        vm = VersionManager(
            Path(data_dir) if data_dir else None,
            Path(registry) if registry else None,
        )
        snap = vm.snapshot().to_dict()
        click.echo(json.dumps(snap, indent=2))

    main.add_command(versions)
