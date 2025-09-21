"""Taxonomy migration commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import yaml

from ...versioning import MigrationPlan, TaxonomyMigrator


def register(main: click.Group) -> None:
    """Attach taxonomy commands to the root CLI."""

    @click.group()
    def taxonomy() -> None:
        """Taxonomy migration commands."""

    @taxonomy.command("migrate-plan")
    @click.option(
        "--from-taxonomy",
        "from_taxonomy",
        required=True,
        type=click.Path(exists=True),
        help="Path to old taxonomy.yaml",
    )
    @click.option(
        "--to-taxonomy",
        "to_taxonomy",
        required=True,
        type=click.Path(exists=True),
        help="Path to new taxonomy.yaml",
    )
    @click.option("--output", "-o", type=click.Path(), help="Path to write migration plan JSON")
    def taxonomy_migrate_plan(
        from_taxonomy: str, to_taxonomy: str, output: Optional[str]
    ) -> None:
        """Compute and optionally persist a taxonomy migration plan."""
        migrator = TaxonomyMigrator(Path(from_taxonomy), Path(to_taxonomy))
        plan = migrator.compute_plan()
        summary = migrator.validate_plan_completeness(plan)
        result = {
            "plan": {
                "from_version": plan.from_version,
                "to_version": plan.to_version,
                "created_at": plan.created_at,
                "label_map": plan.label_map,
                "label_map_count": len(plan.label_map),
                "unmapped_old_labels": plan.unmapped_old_labels,
                "new_labels_without_source": plan.new_labels_without_source,
            },
            "summary": summary,
        }
        text = json.dumps(result, indent=2)
        if output:
            with open(output, "w", encoding="utf-8") as file:
                file.write(text)
            click.echo(f"✓ Migration plan written to {output}")
        else:
            click.echo(text)

    @taxonomy.command("migrate-apply")
    @click.option(
        "--plan",
        "plan_path",
        required=True,
        type=click.Path(exists=True),
        help="Migration plan JSON produced by migrate-plan",
    )
    @click.option(
        "--detectors-dir",
        required=True,
        type=click.Path(exists=True),
        help="Directory containing detector YAMLs",
    )
    @click.option(
        "--write-dir",
        type=click.Path(),
        help="Optional directory to write migrated detector YAMLs (dry-run if omitted)",
    )
    def taxonomy_migrate_apply(
        plan_path: str, detectors_dir: str, write_dir: Optional[str]
    ) -> None:
        """Apply a migration plan to detector YAMLs."""
        with open(plan_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        plan_obj = data.get("plan") or {}
        plan = MigrationPlan(
            from_version=plan_obj.get("from_version", ""),
            to_version=plan_obj.get("to_version", ""),
            created_at=plan_obj.get("created_at", ""),
            label_map=plan_obj.get("label_map", {}) if "label_map" in plan_obj else {},
            unmapped_old_labels=plan_obj.get("unmapped_old_labels", []),
            new_labels_without_source=plan_obj.get("new_labels_without_source", []),
        )

        det_dir = Path(detectors_dir)
        yaml_files = [
            path
            for path in det_dir.glob("*.yaml")
            if path.name not in {"taxonomy.yaml", "frameworks.yaml"}
        ]
        detector_maps: dict[str, dict] = {}
        for yaml_file in yaml_files:
            with open(yaml_file, "r", encoding="utf-8") as file:
                contents = yaml.safe_load(file)
            if not isinstance(contents, dict) or "detector" not in contents or "maps" not in contents:
                continue
            detector_maps[contents["detector"]] = contents["maps"]

        migrator = TaxonomyMigrator(Path("noop_old.yaml"), Path("noop_new.yaml"))
        report = migrator.apply_to_detector_mappings(detector_maps, plan)

        click.echo(
            json.dumps(
                {
                    "summary": {
                        "total_mappings": report.total_mappings,
                        "remapped": report.remapped,
                        "unchanged": report.unchanged,
                        "unknown_after_migration": report.unknown_after_migration,
                    },
                    "details": report.details,
                },
                indent=2,
            )
        )

        if write_dir:
            out_dir = Path(write_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for yaml_file in yaml_files:
                with open(yaml_file, "r", encoding="utf-8") as file:
                    contents = yaml.safe_load(file)
                detector_name = contents.get("detector")
                if detector_name and detector_name in detector_maps:
                    new_maps = {
                        key: plan.label_map.get(value, value)
                        for key, value in detector_maps[detector_name].items()
                    }
                    contents["maps"] = new_maps
                    version = str(contents.get("version", "v1"))
                    if version.startswith("v"):
                        try:
                            parts = version[1:].split(".")
                            if len(parts) == 1:
                                contents["version"] = f"v{int(parts[0]) + 1}"
                            else:
                                major = int(parts[0])
                                minor = int(parts[1]) if len(parts) > 1 else 0
                                contents["version"] = f"v{major}.{minor + 1}"
                        except Exception:  # pragma: no cover - defensive bump
                            pass
                    out_file = out_dir / yaml_file.name
                    with open(out_file, "w", encoding="utf-8") as file:
                        yaml.safe_dump(contents, file, sort_keys=False)
            click.echo(f"✓ Migrated detector YAMLs written to {out_dir}")

    main.add_command(taxonomy)
