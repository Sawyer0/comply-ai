"""
Taxonomy management for the Mapper Service.

Single responsibility: Taxonomy operations and framework mappings.
"""

from .taxonomy_manager import TaxonomyManager
from .framework_adapter import FrameworkAdapter

__all__ = [
    "TaxonomyManager",
    "FrameworkAdapter",
]
