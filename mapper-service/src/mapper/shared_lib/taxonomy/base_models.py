"""
Base data models for taxonomy and schema management.

Single responsibility: Define core data structures used across taxonomy system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class ChangeType(Enum):
    """Types of changes for versioning."""

    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features/categories
    PATCH = "patch"  # Bug fixes, clarifications


class CompatibilityLevel(Enum):
    """Compatibility levels between versions."""

    FULL = "full"  # Fully backward compatible
    PARTIAL = "partial"  # Partially compatible with warnings
    BREAKING = "breaking"  # Breaking changes


@dataclass
class VersionInfo:
    """Version information for any versioned resource."""

    version: str
    created_at: datetime
    created_by: str
    change_type: ChangeType
    changes: List[str] = field(default_factory=list)
    backward_compatible: bool = True
    migration_notes: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class TaxonomySubcategory:
    """Individual taxonomy subcategory definition."""

    name: str
    description: str
    types: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecated_since: Optional[str] = None
    replacement: Optional[str] = None


@dataclass
class TaxonomyCategory:
    """Individual taxonomy category definition."""

    name: str
    description: str
    subcategories: Dict[str, TaxonomySubcategory] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecated_since: Optional[str] = None
    replacement: Optional[str] = None


@dataclass
class CompatibilityCheck:
    """Result of compatibility check between versions."""

    from_version: str
    to_version: str
    compatible: bool
    compatibility_level: CompatibilityLevel
    breaking_changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    migration_required: bool = False
    migration_complexity: str = "simple"  # simple, moderate, complex
