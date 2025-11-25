"""Core ports (interfaces) for mapper domain dependencies.

These ports define the mapper service's dependency surface for
canonical taxonomy and framework mapping concerns. Concrete
implementations live in infrastructure modules and delegate to
shared libraries (e.g. shared.taxonomy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CanonicalTaxonomyPort(ABC):
    """Port for canonical taxonomy operations.

    This abstraction lets the core mapping logic validate
    canonical labels without depending directly on the
    shared.taxonomy implementation.
    """

    @abstractmethod
    def is_valid_label(self, label: str) -> bool:
        """Return True if the label exists in the canonical taxonomy."""
        raise NotImplementedError


class FrameworkMappingPort(ABC):
    """Port for mapping canonical labels to framework-specific controls.

    This abstracts the central framework mapping registry behind a
    simple interface the core can depend on.
    """

    @abstractmethod
    def map_to_framework(self, labels: List[str], framework: str) -> Dict[str, str]:
        """Map canonical labels to a specific framework's labels.

        Args:
            labels: Canonical taxonomy labels.
            framework: Target compliance framework name.

        Returns:
            A mapping of canonical label -> framework-specific label.
        """
        raise NotImplementedError


class ModelInferencePort(ABC):
    """Port for model-based canonical mapping operations.

    This abstraction lets the core mapping logic request canonical
    taxonomy mappings from a model-backed inference service without
    depending on concrete model servers or resilience implementations.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize underlying model infrastructure if needed."""
        raise NotImplementedError

    @abstractmethod
    async def is_available(self) -> bool:
        """Return True if the model backend is available for inference."""
        raise NotImplementedError

    @abstractmethod
    async def generate_mapping(
        self,
        detector: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a canonical mapping JSON string for detector output."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        """Check health of the underlying model backend."""
        raise NotImplementedError

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanly shut down any model resources (connections, processes)."""
        raise NotImplementedError
