"""Domain types for the mapper core.

Single responsibility: domain-level value objects and aggregates used by CoreMapper.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..config.settings import MapperSettings
from ..schemas.models import MappingRequest


@dataclass(frozen=True)
class DetectorMappingCommand:
    """Domain command representing a detector-output mapping request."""

    detector: str
    output: str
    metadata: Optional[Dict[str, Any]]
    tenant_id: Optional[str]
    framework: Optional[str]
    confidence_threshold: float

    @classmethod
    def from_api_request(
        cls, request: MappingRequest, settings: MapperSettings
    ) -> "DetectorMappingCommand":
        """Create a domain command from an API MappingRequest and settings."""

        return cls(
            detector=request.detector,
            output=request.output,
            metadata=request.metadata,
            tenant_id=request.tenant_id,
            framework=request.framework,
            confidence_threshold=(
                request.confidence_threshold
                if request.confidence_threshold is not None
                else settings.confidence_threshold
            ),
        )


@dataclass
class MappingContext:
    """Context for mapping operations within the domain layer."""

    detector: str
    output: str
    metadata: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    framework: Optional[str] = None
    confidence_threshold: float = 0.7

    @classmethod
    def from_command(cls, command: DetectorMappingCommand) -> "MappingContext":
        """Build a context from a domain command."""

        return cls(
            detector=command.detector,
            output=command.output,
            metadata=command.metadata,
            tenant_id=command.tenant_id,
            framework=command.framework,
            confidence_threshold=command.confidence_threshold,
        )
