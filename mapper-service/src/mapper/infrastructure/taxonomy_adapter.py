"""Infrastructure adapters for taxonomy-related ports.

These adapters implement the core taxonomy ports by delegating to
shared.taxonomy primitives. This keeps the core layer decoupled
from the shared implementation while still using the real
production components.
"""

from __future__ import annotations

from typing import Dict, List

from shared.taxonomy import framework_mapping_registry, canonical_taxonomy

from ..core.ports import CanonicalTaxonomyPort, FrameworkMappingPort


class SharedCanonicalTaxonomyAdapter(CanonicalTaxonomyPort):
    """Adapter that uses shared.taxonomy.canonical_taxonomy.

    This is a thin, production-ready wrapper – no stubs – so the
    core can validate canonical labels without knowing about the
    concrete taxonomy implementation.
    """

    def is_valid_label(self, label: str) -> bool:  # type: ignore[override]
        return canonical_taxonomy.is_valid_label(label)


class SharedFrameworkMappingAdapter(FrameworkMappingPort):
    """Adapter that uses shared.taxonomy.framework_mapping_registry.

    It directly delegates to the central framework mapping
    registry used by other parts of the system.
    """

    def map_to_framework(  # type: ignore[override]
        self, labels: List[str], framework: str
    ) -> Dict[str, str]:
        return framework_mapping_registry.map_to_framework(labels, framework)
