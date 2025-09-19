"""
Pydantic models for the FastAPI service layer.
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Provenance(BaseModel):
    """Provenance information for tracking detector outputs."""

    vendor: Optional[str] = None
    detector: Optional[str] = None
    detector_version: Optional[str] = None
    raw_ref: Optional[str] = Field(
        None, description="Pointer/ID to the raw event in Splunk/Datadog/S3"
    )
    route: Optional[str] = None
    model: Optional[str] = None
    tenant_id: Optional[str] = None
    ts: Optional[datetime] = None


class PolicyContext(BaseModel):
    """Policy context for detector expectations."""

    expected_detectors: Optional[List[str]] = None
    environment: Optional[str] = Field(None, pattern="^(dev|stage|prod)$")


class DetectorRequest(BaseModel):
    """Request model for the /map endpoint."""

    detector: str = Field(
        ..., description="Name of the detector that produced the output"
    )
    output: str = Field(..., description="Raw output from the detector")
    metadata: Optional[Dict] = Field(
        None, description="Optional metadata about the detection"
    )
    tenant_id: Optional[str] = Field(
        None, description="Tenant identifier for multi-tenancy"
    )


class BatchDetectorRequest(BaseModel):
    """Request model for batch processing multiple detector outputs."""

    requests: List[DetectorRequest] = Field(..., min_length=1, max_length=100)


class MappingResponse(BaseModel):
    """Response model following pillars-detectors/schema.json."""

    taxonomy: List[str] = Field(
        ..., min_length=1, description="Canonical taxonomy labels"
    )
    scores: Dict[str, float] = Field(
        ..., description="Map of taxonomy labels to normalized scores [0,1]"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model-calibrated confidence in the mapping"
    )
    notes: Optional[str] = Field(
        None, max_length=500, description="Optional debugging notes"
    )
    provenance: Optional[Provenance] = None
    policy_context: Optional[PolicyContext] = None

    @field_validator("taxonomy")
    @classmethod
    def validate_taxonomy_format(cls, v: List[str]) -> List[str]:
        """Validate taxonomy label format."""
        import re

        pattern = r"^[A-Z][A-Z0-9_]*(\.[A-Za-z0-9_]+)*$"
        for item in v:
            if not re.match(pattern, item):
                raise ValueError(
                    f'Taxonomy label "{item}" does not match required pattern'
                )
        return v

    @field_validator("scores")
    @classmethod
    def validate_scores_range(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate that all scores are in [0,1] range."""
        for label, score in v.items():
            if not (0.0 <= score <= 1.0):
                raise ValueError(f'Score for "{label}" must be between 0.0 and 1.0')
        return v


class BatchMappingResponse(BaseModel):
    """Response model for batch processing."""

    results: List[MappingResponse]
    errors: Optional[List[Dict]] = Field(
        None, description="Any errors encountered during batch processing"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
