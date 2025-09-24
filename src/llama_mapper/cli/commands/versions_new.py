"""Refactored version management commands using the new CLI architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import click

from ...versioning import VersionManager
from ..core import BaseCommand, CLIError
from ..decorators.common import handle_errors, timing
from ..utils import display_success, format_output


class VersionsShowCommand(BaseCommand):
    """Command to show current version snapshot."""

    def __init__(
        self,
        config_manager,
        data_dir: Optional[str] = None,
        registry: Optional[str] = None,
    ):
        super().__init__(config_manager)
        self.data_dir = Path(data_dir) if data_dir else None
        self.registry = Path(registry) if registry else None

    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the versions show command."""
        self.logger.info(
            "Showing version snapshot", data_dir=self.data_dir, registry=self.registry
        )

        try:
            vm = VersionManager(self.data_dir, self.registry)
            snapshot = vm.snapshot().to_dict()

            # Output the snapshot
            format_output(snapshot, format_type="json")
            display_success("Version snapshot retrieved successfully")

        except Exception as e:
            raise CLIError(f"Failed to get version snapshot: {e}")


class VersionsListCommand(BaseCommand):
    """Command to list available versions."""

    def __init__(self, config_manager, data_dir: Optional[str] = None):
        super().__init__(config_manager)
        self.data_dir = Path(data_dir) if data_dir else None

    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the versions list command."""
        self.logger.info("Listing available versions", data_dir=self.data_dir)

        try:
            vm = VersionManager(self.data_dir)
            versions = vm.list_versions()

            if not versions:
                click.echo("No versions found")
                return

            # Display versions in a table format
            from ..utils import display_table

            headers = ["Version", "Created", "Description"]
            rows = []

            for version in versions:
                rows.append(
                    [
                        version.get("version", "unknown"),
                        version.get("created", "unknown"),
                        version.get("description", ""),
                    ]
                )

            display_table(headers, rows, title="Available Versions")
            display_success(f"Found {len(versions)} versions")

        except Exception as e:
            raise CLIError(f"Failed to list versions: {e}")


class VersionsCompareCommand(BaseCommand):
    """Command to compare two versions."""

    def __init__(
        self,
        config_manager,
        version1: str,
        version2: str,
        data_dir: Optional[str] = None,
    ):
        super().__init__(config_manager)
        self.version1 = version1
        self.version2 = version2
        self.data_dir = Path(data_dir) if data_dir else None

    @handle_errors
    @timing
    def execute(self, **kwargs: Any) -> None:
        """Execute the versions compare command."""
        self.logger.info(
            "Comparing versions",
            version1=self.version1,
            version2=self.version2,
            data_dir=self.data_dir,
        )

        try:
            vm = VersionManager(self.data_dir)
            comparison = vm.compare_versions(self.version1, self.version2)

            # Display comparison results
            click.echo(f"Comparison between {self.version1} and {self.version2}")
            click.echo("=" * 50)

            if comparison.get("identical", False):
                click.echo("✓ Versions are identical")
            else:
                changes = comparison.get("changes", [])
                if changes:
                    click.echo(f"Found {len(changes)} differences:")
                    for change in changes:
                        click.echo(f"  • {change}")
                else:
                    click.echo("No differences found")

            display_success("Version comparison completed")

        except Exception as e:
            raise CLIError(f"Failed to compare versions: {e}")


def register(main: click.Group) -> None:
    """Register version commands using the new architecture."""

    @click.group()
    def versions() -> None:
        """Version management commands."""
        pass

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
    @click.pass_context
    def versions_show(
        ctx: click.Context, data_dir: Optional[str], registry: Optional[str]
    ) -> None:
        """Show current version snapshot as JSON."""
        command = VersionsShowCommand(ctx.obj["config"], data_dir, registry)
        command.execute()

    @versions.command("list")
    @click.option(
        "--data-dir",
        type=click.Path(exists=False),
        help="Directory with taxonomy/frameworks/detectors",
    )
    @click.pass_context
    def versions_list(ctx: click.Context, data_dir: Optional[str]) -> None:
        """List available versions."""
        command = VersionsListCommand(ctx.obj["config"], data_dir)
        command.execute()

    @versions.command("compare")
    @click.argument("version1")
    @click.argument("version2")
    @click.option(
        "--data-dir",
        type=click.Path(exists=False),
        help="Directory with taxonomy/frameworks/detectors",
    )
    @click.pass_context
    def versions_compare(
        ctx: click.Context,
        version1: str,
        version2: str,
        data_dir: Optional[str],
    ) -> None:
        """Compare two versions."""
        command = VersionsCompareCommand(
            ctx.obj["config"], version1, version2, data_dir
        )
        command.execute()

    main.add_command(versions)
