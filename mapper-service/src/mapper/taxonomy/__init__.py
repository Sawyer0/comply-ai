"""
Taxonomy management for the Mapper Service.

Single responsibility: Taxonomy operations and framework mappings.
"""

# Import centralized taxonomy system
from shared.taxonomy import (
    canonical_taxonomy,
    schema_evolution_manager,
    framework_mapping_registry,
)

from .taxonomy_manager import TaxonomyManager
from .framework_adapter import FrameworkAdapter

__all__ = [
    "TaxonomyManager",
    "FrameworkAdapter",
    "canonical_taxonomy",
    "schema_evolution_manager",
    "framework_mapping_registry",
]
