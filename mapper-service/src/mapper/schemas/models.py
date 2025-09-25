"""
Data models for the Mapper Service.

Single responsibility: Pydantic model definitions for API contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Provenance(BaseModel):
    """Provenance information for mapping results."""

    detector: str = Field(..., description="Name of the detector")
    raw_ref: Optional[str] = Field(None, description="Reference to raw input")


class VersionInfo(BaseModel):
    """Version information for mapping results."""

    model_version: str = Field(..., description="Model version used")
    taxonomy_version: str = Field(..., description="Taxonomy version used")
    timestamp: datetime = Field(..., description="Mapping timestamp")


class MappingRequest(BaseModel):
    """Request for mapping detector output to canonical taxonomy."""

    detector: str = Field(..., description="Name of the detector")
    output: str = Field(..., description="Raw detector output")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    framework: Optional[str] = Field(None, description="Target compliance framework")
    confidence_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    @validator("detector")
    def validate_detector(cls, v):
        if not v or not v.strip():
            raise ValueError("Detector name cannot be empty")
        return v.strip()

    @validator("output")
    def validate_output(cls, v):
        if not v or not v.strip():
            raise ValueError("Output cannot be empty")
        return v.strip()


class MappingResponse(BaseModel):
    """Response from mapping detector output."""

    taxonomy: List[str] = Field(..., description="Canonical taxonomy labels")
    scores: Dict[str, float] = Field(
        ..., description="Confidence scores for each label"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    notes: Optional[str] = Field(None, description="Additional notes")
    provenance: Provenance = Field(..., description="Provenance information")
    version_info: Optional[VersionInfo] = Field(None, description="Version information")

    @validator("taxonomy")
    def validate_taxonomy(cls, v):
        if not v:
            raise ValueError("Taxonomy cannot be empty")
        return v

    @validator("scores")
    def validate_scores(cls, v, values):
        if not v:
            raise ValueError("Scores cannot be empty")

        # Check all scores are valid
        for label, score in v.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score for {label} must be between 0.0 and 1.0")

        return v


class BatchMappingRequest(BaseModel):
    """Request for batch mapping multiple detector outputs."""

    requests: List[MappingRequest] = Field(..., description="List of mapping requests")

    @validator("requests")
    def validate_requests(cls, v):
        if not v:
            raise ValueError("Requests list cannot be empty")
        if len(v) > 100:  # Reasonable batch size limit
            raise ValueError("Batch size cannot exceed 100 requests")
        return v


class BatchMappingResponse(BaseModel):
    """Response from batch mapping."""

    responses: List[MappingResponse] = Field(
        ..., description="List of mapping responses"
    )
    total_processed: int = Field(..., description="Total number of requests processed")
    success_count: int = Field(..., description="Number of successful mappings")
    error_count: int = Field(..., description="Number of failed mappings")
