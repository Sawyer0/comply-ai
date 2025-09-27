"""
Taxonomy management for the analysis service.

This module provides taxonomy systems for pattern recognition,
risk assessment, and compliance analysis.
"""

# Import centralized taxonomy system
from shared.taxonomy import (
    canonical_taxonomy,
    schema_evolution_manager,
    framework_mapping_registry,
)

from .pattern_taxonomy import (
    PatternTaxonomy,
    PatternType,
    PatternSeverity,
    PatternDefinition,
)
from .risk_taxonomy import RiskTaxonomy, RiskCategory, RiskLevel, RiskFactor

__all__ = [
    "PatternTaxonomy",
    "PatternType",
    "PatternSeverity",
    "PatternDefinition",
    "RiskTaxonomy",
    "RiskCategory",
    "RiskLevel",
    "RiskFactor",
    "canonical_taxonomy",
    "schema_evolution_manager",
    "framework_mapping_registry",
]
