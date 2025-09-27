"""
Centralized taxonomy and schema management for all microservices.

This module provides a unified taxonomy and schema management system
with versioning, evolution, and migration capabilities.
"""

from .base_models import (
    ChangeType,
    CompatibilityLevel,
    VersionInfo,
    TaxonomyCategory,
    TaxonomySubcategory,
    CompatibilityCheck,
)
from .canonical_taxonomy import CanonicalTaxonomy, canonical_taxonomy
from .schema_evolution import SchemaEvolutionManager, schema_evolution_manager
from .framework_mappings import FrameworkMappingRegistry, framework_mapping_registry
from .migration_tools import (
    TaxonomyMigrationManager,
    SchemaMigrationManager,
    MigrationExecutor,
    MigrationPlan,
)
from .version_manager import VersionManager, TaxonomyVersionManager

__all__ = [
    # Base models
    "ChangeType",
    "CompatibilityLevel",
    "VersionInfo",
    "TaxonomyCategory",
    "TaxonomySubcategory",
    "CompatibilityCheck",
    # Core managers
    "CanonicalTaxonomy",
    "canonical_taxonomy",
    "SchemaEvolutionManager",
    "schema_evolution_manager",
    "FrameworkMappingRegistry",
    "framework_mapping_registry",
    # Migration tools
    "TaxonomyMigrationManager",
    "SchemaMigrationManager",
    "MigrationExecutor",
    "MigrationPlan",
    # Version management
    "VersionManager",
    "TaxonomyVersionManager",
]
