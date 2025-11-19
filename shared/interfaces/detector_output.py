"""Canonical detector output interfaces shared across services."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .common import RiskLevel, Severity
from .orchestration import DetectorResult, OrchestrationResponse
from .analysis import CanonicalTaxonomyResult


class CanonicalDetectorEntity(BaseModel):
    """Canonical view of a single detected entity."""

    text: str = Field(
        description="Extracted text span or value",
        min_length=1,
    )
    start_offset: Optional[int] = Field(
        None, description="Start character offset in original content", ge=0
    )
    end_offset: Optional[int] = Field(
        None, description="End character offset in original content", ge=0
    )

    label: str = Field(
        description="Canonical taxonomy label for the entity",
        min_length=1,
    )
    category: str = Field(
        description="Canonical taxonomy category",
        min_length=1,
    )
    subcategory: Optional[str] = Field(
        None, description="Canonical taxonomy subcategory"
    )
    type: Optional[str] = Field(None, description="Canonical taxonomy type")

    confidence: float = Field(
        description="Entity-level confidence score", ge=0.0, le=1.0
    )
    severity: Severity = Field(description="Entity severity level")
    risk_level: RiskLevel = Field(description="Entity risk level")

    detector_id: Optional[str] = Field(
        None, description="Identifier of the detector that produced this entity"
    )
    detector_type: Optional[str] = Field(
        None, description="Type of the detector that produced this entity"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata for the entity"
    )


class CanonicalDetectorOutput(BaseModel):
    """Canonical detector output for a single detector execution."""

    detector_id: str = Field(
        description="Detector identifier",
        min_length=1,
    )
    detector_type: str = Field(
        description="Detector type",
        min_length=1,
    )

    canonical_result: CanonicalTaxonomyResult = Field(
        description="Canonical taxonomy summary for this detector"
    )
    entities: List[CanonicalDetectorEntity] = Field(
        default_factory=list,
        description="Canonicalized entities detected by this detector",
    )

    max_severity: Severity = Field(
        description="Maximum severity across canonical result and entities"
    )
    max_risk_level: RiskLevel = Field(
        description="Maximum risk level across canonical result and entities"
    )

    raw_result: Optional[DetectorResult] = Field(
        None, description="Raw detector result snapshot for audit and debugging"
    )


class CanonicalDetectorOutputs(BaseModel):
    """Canonical detector outputs for a single orchestration request."""

    tenant_id: str = Field(
        description="Tenant identifier",
        min_length=1,
    )
    request_correlation_id: str = Field(
        description="Correlation identifier for the request",
        min_length=1,
    )
    outputs: List[CanonicalDetectorOutput] = Field(
        description="Canonical outputs per detector"
    )

    def to_canonical_results(self) -> List[CanonicalTaxonomyResult]:
        """Flatten canonical detector outputs into canonical taxonomy results.

        This is a convenience helper for downstream services that only need the
        per-detector canonical taxonomy view rather than full per-entity detail.
        """

        return [output.canonical_result for output in self.outputs]


def canonical_outputs_from_orchestration_response(
    response: OrchestrationResponse,
) -> Optional[CanonicalDetectorOutputs]:
    """Reconstruct CanonicalDetectorOutputs from an OrchestrationResponse.

    The orchestration service serializes CanonicalDetectorOutputs into the
    OrchestrationResponse.canonical_outputs field. This helper restores the
    structured model so downstream services can work with typed data.
    """

    data = getattr(response, "canonical_outputs", None)
    if data is None:
        return None

    if isinstance(data, CanonicalDetectorOutputs):
        return data

    if isinstance(data, dict):
        # Support both Pydantic v2 (model_validate) and v1 (parse_obj/constructor)
        if hasattr(CanonicalDetectorOutputs, "model_validate"):
            return CanonicalDetectorOutputs.model_validate(data)  # type: ignore[attr-defined]
        return CanonicalDetectorOutputs(**data)

    raise TypeError("canonical_outputs must be a dict or CanonicalDetectorOutputs instance")


def canonical_taxonomy_results_from_outputs(
    outputs: CanonicalDetectorOutputs,
) -> List[CanonicalTaxonomyResult]:
    """Extract CanonicalTaxonomyResult instances from CanonicalDetectorOutputs."""

    return [output.canonical_result for output in outputs.outputs]


def canonical_taxonomy_results_from_orchestration_response(
    response: OrchestrationResponse,
) -> List[CanonicalTaxonomyResult]:
    """Convenience helper to get canonical taxonomy results from a response.

    Returns an empty list if no canonical outputs are present.
    """

    outputs = canonical_outputs_from_orchestration_response(response)
    if outputs is None:
        return []
    return canonical_taxonomy_results_from_outputs(outputs)
