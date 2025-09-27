"""
Migration tools for taxonomy and schema evolution.

Single responsibility: Provide migration utilities for taxonomy and schema changes.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base_models import CompatibilityCheck, CompatibilityLevel
from .canonical_taxonomy import CanonicalTaxonomy
from .schema_evolution import SchemaEvolutionManager
from .framework_mappings import FrameworkMappingRegistry

logger = logging.getLogger(__name__)


class MigrationPlan:
    """Migration plan for taxonomy or schema changes."""

    def __init__(
        self,
        resource_type: str,
        resource_name: str,
        from_version: str,
        to_version: str,
        compatibility: CompatibilityCheck,
    ):
        self.resource_type = resource_type
        self.resource_name = resource_name
        self.from_version = from_version
        self.to_version = to_version
        self.compatibility = compatibility
        self.migration_steps: List[Dict[str, Any]] = []
        self.rollback_steps: List[Dict[str, Any]] = []

    def add_migration_step(
        self, step_type: str, description: str, action: Dict[str, Any]
    ) -> None:
        """Add a migration step."""
        self.migration_steps.append(
            {"type": step_type, "description": description, "action": action}
        )

    def add_rollback_step(
        self, step_type: str, description: str, action: Dict[str, Any]
    ) -> None:
        """Add a rollback step."""
        self.rollback_steps.append(
            {"type": step_type, "description": description, "action": action}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert migration plan to dictionary."""
        return {
            "resource_type": self.resource_type,
            "resource_name": self.resource_name,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "compatibility": {
                "compatible": self.compatibility.compatible,
                "compatibility_level": self.compatibility.compatibility_level.value,
                "breaking_changes": self.compatibility.breaking_changes,
                "warnings": self.compatibility.warnings,
                "migration_required": self.compatibility.migration_required,
                "migration_complexity": self.compatibility.migration_complexity,
            },
            "migration_steps": self.migration_steps,
            "rollback_steps": self.rollback_steps,
        }


class TaxonomyMigrationManager:
    """
    Taxonomy migration manager.

    Single responsibility: Handle taxonomy migrations and data transformations.
    """

    def __init__(self, taxonomy: CanonicalTaxonomy):
        """Initialize taxonomy migration manager."""
        self.taxonomy = taxonomy

    def create_migration_plan(
        self, from_version: str, to_version: str
    ) -> MigrationPlan:
        """Create migration plan for taxonomy version change."""
        # For taxonomy, we need to analyze label changes
        from_version_info = self.taxonomy.version_manager.get_version_info(from_version)
        to_version_info = self.taxonomy.version_manager.get_version_info(to_version)

        if not from_version_info or not to_version_info:
            raise ValueError("Version information not found")

        # Create compatibility check based on version info
        compatibility = CompatibilityCheck(
            from_version=from_version,
            to_version=to_version,
            compatible=to_version_info.backward_compatible,
            compatibility_level=(
                CompatibilityLevel.FULL
                if to_version_info.backward_compatible
                else CompatibilityLevel.BREAKING
            ),
            breaking_changes=[],
            warnings=[],
            migration_required=not to_version_info.backward_compatible,
            migration_complexity="simple",
        )

        plan = MigrationPlan(
            "taxonomy", "canonical_taxonomy", from_version, to_version, compatibility
        )

        # Add migration steps based on changes
        for change in to_version_info.changes:
            if "deprecated" in change.lower():
                plan.add_migration_step(
                    "deprecation",
                    f"Handle deprecated labels: {change}",
                    {"type": "deprecation", "change": change},
                )
            elif "added" in change.lower():
                plan.add_migration_step(
                    "addition",
                    f"Handle new labels: {change}",
                    {"type": "addition", "change": change},
                )

        return plan

    def migrate_labels(
        self, labels: List[str], from_version: str, to_version: str
    ) -> Tuple[List[str], List[str]]:
        """
        Migrate taxonomy labels from one version to another.

        Returns:
            Tuple of (migrated_labels, warnings)
        """
        migrated_labels = []
        warnings = []

        # Get deprecated labels for the target version
        deprecated_labels = self.taxonomy.get_deprecated_labels()
        deprecated_map = {
            label: replacement for label, _, replacement in deprecated_labels
        }

        for label in labels:
            if label in deprecated_map and deprecated_map[label]:
                # Use replacement
                migrated_labels.append(deprecated_map[label])
                warnings.append(
                    f"Migrated deprecated label '{label}' to '{deprecated_map[label]}'"
                )
            elif label in deprecated_map:
                # Deprecated with no replacement
                warnings.append(f"Label '{label}' is deprecated with no replacement")
                migrated_labels.append(label)  # Keep original
            else:
                # No migration needed
                migrated_labels.append(label)

        return migrated_labels, warnings

    def validate_migration(
        self, original_data: Dict[str, Any], migrated_data: Dict[str, Any]
    ) -> List[str]:
        """Validate migration results."""
        issues = []

        # Check that no data was lost
        original_labels = set(original_data.get("labels", []))
        migrated_labels = set(migrated_data.get("labels", []))

        if len(migrated_labels) < len(original_labels):
            issues.append("Some labels were lost during migration")

        # Check for invalid labels
        for label in migrated_labels:
            if not self.taxonomy.is_valid_label(label):
                issues.append(f"Invalid label after migration: {label}")

        return issues


class SchemaMigrationManager:
    """
    Schema migration manager.

    Single responsibility: Handle schema migrations and data transformations.
    """

    def __init__(self, schema_manager: SchemaEvolutionManager):
        """Initialize schema migration manager."""
        self.schema_manager = schema_manager

    def create_migration_plan(
        self, schema_name: str, from_version: str, to_version: str
    ) -> MigrationPlan:
        """Create migration plan for schema version change."""
        compatibility = self.schema_manager.check_compatibility(
            schema_name, from_version, to_version
        )

        plan = MigrationPlan(
            "schema", schema_name, from_version, to_version, compatibility
        )

        # Add migration steps based on breaking changes
        for breaking_change in compatibility.breaking_changes:
            if "removed required field" in breaking_change.lower():
                field_name = breaking_change.split(": ")[-1]
                plan.add_migration_step(
                    "field_removal",
                    f"Handle removed required field: {field_name}",
                    {"type": "field_removal", "field": field_name},
                )
                plan.add_rollback_step(
                    "field_addition",
                    f"Restore required field: {field_name}",
                    {"type": "field_addition", "field": field_name},
                )
            elif "added required field" in breaking_change.lower():
                field_name = breaking_change.split(": ")[-1]
                plan.add_migration_step(
                    "field_addition",
                    f"Handle new required field: {field_name}",
                    {"type": "field_addition", "field": field_name},
                )
                plan.add_rollback_step(
                    "field_removal",
                    f"Remove required field: {field_name}",
                    {"type": "field_removal", "field": field_name},
                )
            elif "changed type" in breaking_change.lower():
                plan.add_migration_step(
                    "type_conversion",
                    f"Handle type change: {breaking_change}",
                    {"type": "type_conversion", "change": breaking_change},
                )

        return plan

    def migrate_data(
        self, data: Dict[str, Any], schema_name: str, from_version: str, to_version: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Migrate data from one schema version to another.

        Returns:
            Tuple of (migrated_data, warnings)
        """
        migrated_data = data.copy()
        warnings = []

        compatibility = self.schema_manager.check_compatibility(
            schema_name, from_version, to_version
        )

        # Handle breaking changes
        for breaking_change in compatibility.breaking_changes:
            if "removed required field" in breaking_change.lower():
                field_name = breaking_change.split(": ")[-1]
                if field_name in migrated_data:
                    warnings.append(
                        f"Field '{field_name}' was removed from schema but exists in data"
                    )
            elif "added required field" in breaking_change.lower():
                field_name = breaking_change.split(": ")[-1]
                if field_name not in migrated_data:
                    # Add default value
                    migrated_data[field_name] = self._get_default_value_for_field(
                        schema_name, to_version, field_name
                    )
                    warnings.append(
                        f"Added default value for new required field '{field_name}'"
                    )

        return migrated_data, warnings

    def _get_default_value_for_field(
        self, schema_name: str, version: str, field_name: str
    ) -> Any:
        """Get default value for a field based on its type."""
        schema_def = self.schema_manager.get_schema_definition(schema_name, version)

        if not schema_def or "properties" not in schema_def:
            return None

        field_def = schema_def["properties"].get(field_name, {})
        field_type = field_def.get("type", "string")

        # Return appropriate default based on type
        defaults = {
            "string": "",
            "integer": 0,
            "number": 0.0,
            "boolean": False,
            "array": [],
            "object": {},
        }

        return defaults.get(field_type, None)

    def validate_migration(
        self,
        original_data: Dict[str, Any],
        migrated_data: Dict[str, Any],
        schema_name: str,
        to_version: str,
    ) -> List[str]:
        """Validate migration results."""
        issues = []

        # Validate against target schema
        is_valid, validation_errors = self.schema_manager.validate_data(
            migrated_data, schema_name, to_version
        )

        if not is_valid:
            issues.extend([f"Validation error: {error}" for error in validation_errors])

        return issues


class MigrationExecutor:
    """
    Migration executor for running migration plans.

    Single responsibility: Execute migration plans and handle rollbacks.
    """

    def __init__(self, backup_path: Optional[Path] = None):
        """Initialize migration executor."""
        self.backup_path = backup_path or Path("backups/migrations")
        self.execution_log: List[Dict[str, Any]] = []

    def execute_migration_plan(
        self, plan: MigrationPlan, dry_run: bool = False
    ) -> Dict[str, Any]:
        """Execute a migration plan."""
        execution_id = f"{plan.resource_name}_{plan.from_version}_to_{plan.to_version}"

        if not dry_run:
            # Create backup
            self._create_backup(execution_id, plan)

        results = {
            "execution_id": execution_id,
            "success": True,
            "steps_executed": 0,
            "errors": [],
            "warnings": [],
            "dry_run": dry_run,
        }

        try:
            for i, step in enumerate(plan.migration_steps):
                if dry_run:
                    logger.info(
                        "DRY RUN: Would execute step %d: %s", i + 1, step["description"]
                    )
                else:
                    logger.info("Executing step %d: %s", i + 1, step["description"])
                    self._execute_migration_step(step)

                results["steps_executed"] += 1

                # Log execution
                self.execution_log.append(
                    {
                        "execution_id": execution_id,
                        "step_number": i + 1,
                        "step": step,
                        "success": True,
                        "dry_run": dry_run,
                    }
                )

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            logger.error(
                "Migration failed at step %d: %s", results["steps_executed"] + 1, str(e)
            )

        return results

    def _execute_migration_step(self, step: Dict[str, Any]) -> None:
        """Execute individual migration step."""
        step_type = step["type"]
        action = step["action"]

        if step_type == "deprecation":
            # Handle deprecation - usually just logging
            logger.info("Handling deprecation: %s", action.get("change"))
        elif step_type == "addition":
            # Handle addition - usually validation
            logger.info("Handling addition: %s", action.get("change"))
        elif step_type == "field_removal":
            # Handle field removal
            logger.info("Handling field removal: %s", action.get("field"))
        elif step_type == "field_addition":
            # Handle field addition
            logger.info("Handling field addition: %s", action.get("field"))
        elif step_type == "type_conversion":
            # Handle type conversion
            logger.info("Handling type conversion: %s", action.get("change"))
        else:
            logger.warning("Unknown migration step type: %s", step_type)

    def _create_backup(self, execution_id: str, plan: MigrationPlan) -> None:
        """Create backup before migration."""
        try:
            self.backup_path.mkdir(parents=True, exist_ok=True)

            backup_data = {
                "execution_id": execution_id,
                "plan": plan.to_dict(),
                "timestamp": str(datetime.now()),
            }

            backup_file = self.backup_path / f"{execution_id}_backup.json"
            with open(backup_file, "w") as f:
                json.dump(backup_data, f, indent=2)

            logger.info("Created backup: %s", backup_file)

        except Exception as e:
            logger.error("Failed to create backup: %s", str(e))
            raise

    def rollback_migration(self, execution_id: str) -> Dict[str, Any]:
        """Rollback a migration."""
        # Find backup
        backup_file = self.backup_path / f"{execution_id}_backup.json"

        if not backup_file.exists():
            return {"success": False, "error": "Backup not found"}

        try:
            with open(backup_file, "r") as f:
                backup_data = json.load(f)

            plan_data = backup_data["plan"]

            # Execute rollback steps
            results = {"success": True, "steps_executed": 0, "errors": []}

            rollback_steps = plan_data.get("rollback_steps", [])
            for i, step in enumerate(rollback_steps):
                logger.info(
                    "Executing rollback step %d: %s", i + 1, step["description"]
                )
                self._execute_migration_step(step)
                results["steps_executed"] += 1

            return results

        except Exception as e:
            logger.error("Rollback failed: %s", str(e))
            return {"success": False, "error": str(e)}
