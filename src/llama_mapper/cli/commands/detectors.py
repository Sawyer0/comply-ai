"""Detector configuration CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, cast

import click

from ...config.validator import (
    apply_detector_fix_plan,
    build_detector_fix_plan,
    scaffold_detector_yaml,
    validate_configuration,
)


def register(main: click.Group) -> None:
    """Attach detector-related subcommands."""

    @click.group()
    def detectors() -> None:
        """Detector configuration commands."""

    @detectors.command("add")
    @click.option(
        "--name", required=True, help="Detector name (e.g., openai-moderation)"
    )
    @click.option(
        "--version", default="v1", show_default=True, help="Detector config version"
    )
    @click.option(
        "--output-dir",
        type=click.Path(file_okay=False),
        help="Directory to place YAML (defaults to pillars-detectors or .kiro/pillars-detectors)",
    )
    @click.pass_context
    def detectors_add(
        ctx: click.Context, name: str, version: str, output_dir: Optional[str]
    ) -> None:
        """Scaffold a new detector YAML with required fields and guidance."""
        try:
            path, guidance = scaffold_detector_yaml(
                name=name,
                version=version,
                output_dir=Path(output_dir) if output_dir else None,
            )
            click.echo(f"✓ Created: {path}")
            click.echo(guidance)
        except FileExistsError as exc:
            click.echo(f"✗ {exc}")
            ctx.exit(1)
        except Exception as exc:  # noqa: BLE001
            click.echo(f"✗ Failed to scaffold detector: {exc}")
            ctx.exit(1)

    @detectors.command("lint")
    @click.option(
        "--data-dir",
        type=click.Path(exists=False, file_okay=False),
        help="Directory with taxonomy.yaml and detector YAMLs",
    )
    @click.option("--format", "fmt", type=click.Choice(["json"]), default=None)
    @click.option(
        "--strict", is_flag=True, default=False, help="Treat warnings as errors"
    )
    def detectors_lint(
        data_dir: Optional[str], fmt: Optional[str], strict: bool
    ) -> None:
        """Lint detector YAMLs against the taxonomy and report issues."""
        result = validate_configuration(Path(data_dir) if data_dir else None)

        issues = []
        if not result.taxonomy.ok:
            issues.append(
                "Taxonomy invalid; fix taxonomy.yaml before linting detectors."
            )
        if not result.detectors.ok:
            issues.extend(result.detectors.errors)

        if fmt == "json":
            payload = {
                "data_dir": str(result.data_dir),
                "taxonomy_ok": result.taxonomy.ok,
                "detectors_ok": result.detectors.ok,
                "detectors_found": result.detectors.details.get("detectors_found", []),
                "errors": issues,
                "warnings": result.detectors.warnings,
            }
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo(f"Data directory: {result.data_dir}")
            click.echo("\n[Detectors Lint]")
            if result.taxonomy.ok:
                click.echo("  ✓ taxonomy.yaml valid")
            else:
                click.echo("  ✗ taxonomy.yaml invalid")
                for err in result.taxonomy.errors:
                    click.echo(f"    - {err}")
            if result.detectors.ok:
                detectors_found = cast(
                    list[str], result.detectors.details.get("detectors_found", [])
                )
                names = ", ".join(detectors_found)
                click.echo(f"  ✓ detector YAMLs valid ({len(detectors_found)} found)")
                if names:
                    click.echo(f"    - {names}")
            else:
                click.echo("  ✗ detector YAML validation failed")
                for err in result.detectors.errors:
                    click.echo(f"    - {err}")

        if (
            (not result.taxonomy.ok)
            or (not result.detectors.ok)
            or (strict and result.detectors.warnings)
        ):
            raise SystemExit(1)

    @detectors.command("fix")
    @click.option(
        "--data-dir",
        type=click.Path(exists=False, file_okay=False),
        help="Directory with taxonomy.yaml and detector YAMLs",
    )
    @click.option(
        "--apply/--dry-run",
        default=False,
        help="Apply suggested fixes where confident enough",
    )
    @click.option(
        "--threshold",
        type=float,
        default=0.86,
        show_default=True,
        help="Confidence threshold (0-1) to auto-apply a suggestion",
    )
    @click.option("--format", "fmt", type=click.Choice(["json"]), default=None)
    def detectors_fix(
        data_dir: Optional[str], apply: bool, threshold: float, fmt: Optional[str]
    ) -> None:
        """Suggest (and optionally apply) fixes for invalid detector canonical labels."""
        plan = build_detector_fix_plan(Path(data_dir) if data_dir else None)

        if fmt == "json":
            click.echo(json.dumps(plan, indent=2))
        else:
            click.echo(f"Data directory: {plan['data_dir']}")
            click.echo(f"Total invalid labels: {plan['total_invalid']}")
            items = cast(list[dict], plan.get("items", []))
            for item in items:
                det = item.get("detector")
                file = item.get("file")
                click.echo(f"\nDetector: {det}\nFile: {file}")
                for bad in item.get("invalid", []):
                    click.echo(f"  - {bad}")
                    for suggestion in item.get("suggestions", {}).get(bad, []):
                        click.echo(
                            f"      suggestion: {suggestion['label']} (score={suggestion['score']})"
                        )

        if apply:
            summary = apply_detector_fix_plan(plan, apply_threshold=threshold)
            if fmt == "json":
                click.echo(json.dumps({"apply_summary": summary}, indent=2))
            else:
                click.echo("\nApply summary:")
                click.echo(f"  applied: {summary['applied']}")
                click.echo(f"  skipped: {summary['skipped']}")
                if summary.get("updated_files"):
                    click.echo("  updated_files:")
                    updated_files = cast(
                        dict[str, int], summary.get("updated_files", {})
                    )
                    for file_path, count in updated_files.items():
                        click.echo(f"    - {file_path}: {count} change(s)")
            new_plan = build_detector_fix_plan(Path(data_dir) if data_dir else None)
            total_invalid = cast(int, (new_plan.get("total_invalid", 0) or 0))
            if total_invalid > 0:
                raise SystemExit(1)

    main.add_command(detectors)
