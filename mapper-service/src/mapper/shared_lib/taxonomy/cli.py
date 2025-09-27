"""
CLI tool for managing centralized taxonomy and schema system.

Single responsibility: Provide command-line interface for taxonomy management.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .canonical_taxonomy import canonical_taxonomy
from .schema_evolution import schema_evolution_manager
from .framework_mappings import framework_mapping_registry
from .migration_tools import (
    TaxonomyMigrationManager,
    SchemaMigrationManager,
    MigrationExecutor,
)
from .base_models import ChangeType


def taxonomy_commands(args):
    """Handle taxonomy-related commands."""
    if args.taxonomy_action == "stats":
        stats = canonical_taxonomy.get_taxonomy_stats()
        print(json.dumps(stats, indent=2))

    elif args.taxonomy_action == "list":
        if args.category:
            labels = canonical_taxonomy.get_labels_by_category(args.category)
            for label in labels:
                print(label)
        else:
            for category in canonical_taxonomy.categories:
                print(
                    f"{category}: {canonical_taxonomy.categories[category].description}"
                )

    elif args.taxonomy_action == "validate":
        if args.label:
            is_valid = canonical_taxonomy.is_valid_label(args.label)
            print(f"Label '{args.label}' is {'valid' if is_valid else 'invalid'}")
        else:
            print("Please provide a label to validate with --label")

    elif args.taxonomy_action == "add-category":
        if args.name and args.description:
            try:
                canonical_taxonomy.add_category(args.name, args.description)
                version = canonical_taxonomy.create_new_version(
                    ChangeType.MINOR, [f"Added category: {args.name}"], "cli-user"
                )
                print(f"Added category '{args.name}' in version {version}")
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("Please provide --name and --description")

    elif args.taxonomy_action == "add-subcategory":
        if args.category and args.name and args.description:
            try:
                types = args.types.split(",") if args.types else []
                canonical_taxonomy.add_subcategory(
                    args.category, args.name, args.description, types
                )
                version = canonical_taxonomy.create_new_version(
                    ChangeType.MINOR,
                    [f"Added subcategory: {args.category}.{args.name}"],
                    "cli-user",
                )
                print(
                    f"Added subcategory '{args.category}.{args.name}' in version {version}"
                )
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("Please provide --category, --name, and --description")


def schema_commands(args):
    """Handle schema-related commands."""
    if args.schema_action == "list":
        schemas = schema_evolution_manager.list_schemas()
        for schema in schemas:
            versions = schema_evolution_manager.get_schema_versions(schema)
            print(f"{schema}: {len(versions)} versions")

    elif args.schema_action == "versions":
        if args.schema_name:
            versions = schema_evolution_manager.get_schema_versions(args.schema_name)
            for version in versions:
                print(version)
        else:
            print("Please provide --schema-name")

    elif args.schema_action == "validate":
        if args.schema_name and args.data_file:
            try:
                with open(args.data_file, "r") as f:
                    data = json.load(f)

                is_valid, errors = schema_evolution_manager.validate_data(
                    data, args.schema_name, args.version
                )

                if is_valid:
                    print("Data is valid")
                else:
                    print("Data is invalid:")
                    for error in errors:
                        print(f"  - {error}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please provide --schema-name and --data-file")

    elif args.schema_action == "compatibility":
        if args.schema_name and args.from_version and args.to_version:
            try:
                compatibility = schema_evolution_manager.check_compatibility(
                    args.schema_name, args.from_version, args.to_version
                )

                print(f"Compatibility: {compatibility.compatibility_level.value}")
                print(f"Compatible: {compatibility.compatible}")

                if compatibility.breaking_changes:
                    print("Breaking changes:")
                    for change in compatibility.breaking_changes:
                        print(f"  - {change}")

                if compatibility.warnings:
                    print("Warnings:")
                    for warning in compatibility.warnings:
                        print(f"  - {warning}")

            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Please provide --schema-name, --from-version, and --to-version")


def framework_commands(args):
    """Handle framework-related commands."""
    if args.framework_action == "list":
        frameworks = framework_mapping_registry.get_supported_frameworks()
        for framework in frameworks:
            mappings = framework_mapping_registry.get_framework_mappings(framework)
            count = len(mappings) if mappings else 0
            print(f"{framework}: {count} mappings")

    elif args.framework_action == "mappings":
        if args.framework_name:
            mappings = framework_mapping_registry.get_framework_mappings(
                args.framework_name
            )
            if mappings:
                for canonical, framework_label in mappings.items():
                    print(f"{canonical} -> {framework_label}")
            else:
                print(f"No mappings found for framework: {args.framework_name}")
        else:
            print("Please provide --framework-name")

    elif args.framework_action == "coverage":
        if args.framework_name:
            all_labels = canonical_taxonomy.valid_labels
            stats = framework_mapping_registry.get_coverage_stats(
                args.framework_name, all_labels
            )
            print(f"Coverage: {stats['coverage']:.1f}%")
            print(f"Mapped: {stats['mapped']}/{stats['total']} labels")

            if args.show_unmapped and stats.get("unmapped_labels"):
                print("Unmapped labels:")
                for label in stats["unmapped_labels"][:10]:  # Show first 10
                    print(f"  - {label}")
        else:
            print("Please provide --framework-name")


def migration_commands(args):
    """Handle migration-related commands."""
    if args.migration_action == "plan":
        if args.resource_type == "taxonomy":
            migration_manager = TaxonomyMigrationManager(canonical_taxonomy)
            plan = migration_manager.create_migration_plan(
                args.from_version, args.to_version
            )
            print(json.dumps(plan.to_dict(), indent=2))

        elif args.resource_type == "schema":
            if args.schema_name:
                migration_manager = SchemaMigrationManager(schema_evolution_manager)
                plan = migration_manager.create_migration_plan(
                    args.schema_name, args.from_version, args.to_version
                )
                print(json.dumps(plan.to_dict(), indent=2))
            else:
                print("Please provide --schema-name for schema migrations")

    elif args.migration_action == "execute":
        print("Migration execution not implemented in CLI - use programmatic interface")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Centralized Taxonomy and Schema Management CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Taxonomy commands
    taxonomy_parser = subparsers.add_parser("taxonomy", help="Taxonomy management")
    taxonomy_parser.add_argument(
        "taxonomy_action",
        choices=["stats", "list", "validate", "add-category", "add-subcategory"],
    )
    taxonomy_parser.add_argument("--category", help="Category name")
    taxonomy_parser.add_argument("--label", help="Label to validate")
    taxonomy_parser.add_argument("--name", help="Name for new category/subcategory")
    taxonomy_parser.add_argument("--description", help="Description")
    taxonomy_parser.add_argument(
        "--types", help="Comma-separated types for subcategory"
    )

    # Schema commands
    schema_parser = subparsers.add_parser("schema", help="Schema management")
    schema_parser.add_argument(
        "schema_action", choices=["list", "versions", "validate", "compatibility"]
    )
    schema_parser.add_argument("--schema-name", help="Schema name")
    schema_parser.add_argument("--version", help="Schema version")
    schema_parser.add_argument(
        "--from-version", help="Source version for compatibility check"
    )
    schema_parser.add_argument(
        "--to-version", help="Target version for compatibility check"
    )
    schema_parser.add_argument("--data-file", help="JSON file to validate")

    # Framework commands
    framework_parser = subparsers.add_parser(
        "framework", help="Framework mapping management"
    )
    framework_parser.add_argument(
        "framework_action", choices=["list", "mappings", "coverage"]
    )
    framework_parser.add_argument("--framework-name", help="Framework name")
    framework_parser.add_argument(
        "--show-unmapped", action="store_true", help="Show unmapped labels"
    )

    # Migration commands
    migration_parser = subparsers.add_parser("migration", help="Migration management")
    migration_parser.add_argument("migration_action", choices=["plan", "execute"])
    migration_parser.add_argument(
        "--resource-type", choices=["taxonomy", "schema"], required=True
    )
    migration_parser.add_argument(
        "--schema-name", help="Schema name (for schema migrations)"
    )
    migration_parser.add_argument(
        "--from-version", required=True, help="Source version"
    )
    migration_parser.add_argument("--to-version", required=True, help="Target version")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "taxonomy":
            taxonomy_commands(args)
        elif args.command == "schema":
            schema_commands(args)
        elif args.command == "framework":
            framework_commands(args)
        elif args.command == "migration":
            migration_commands(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
