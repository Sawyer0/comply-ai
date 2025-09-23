"""
Analysis models - compatibility layer for backward compatibility.

This module re-exports entities from the domain layer to maintain
backward compatibility with existing tests and code.
"""

from .domain.entities import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    AnalysisType,
    VersionInfo,
    EvidenceReference,
)
from .validation.evidence_refs import ALLOWED_EVIDENCE_REFS

__all__ = [
    "AnalysisRequest",
    "AnalysisResponse", 
    "BatchAnalysisRequest",
    "BatchAnalysisResponse",
    "AnalysisType",
    "VersionInfo",
    "EvidenceReference",
    "ALLOWED_EVIDENCE_REFS",
]
