"""Versioning and migration utilities for Llama Mapper.

Exports:
- VersionManager: gathers component versions, creates snapshots, and helps embed
  version tags in outputs and reports.
- TaxonomyMigrator: computes migration plans between taxonomy versions and can
  remap detector mappings, with rollback support and validation.
"""

from .version_manager import VersionManager, VersionSnapshot
from .taxonomy_migrator import (
    TaxonomyMigrator,
    MigrationPlan,
    MigrationReport,
)

__all__ = [
    "VersionManager",
    "VersionSnapshot",
    "TaxonomyMigrator",
    "MigrationPlan",
    "MigrationReport",
]