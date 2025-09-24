"""
Type definitions and protocols for the Risk Scoring Framework.
"""

from typing import Protocol, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

from ...domain.analysis_models import SecurityFinding


class RiskDimension(Enum):
    """Enumeration of risk dimensions."""
    TECHNICAL = "technical"
    BUSINESS = "business"
    REGULATORY = "regulatory"
    TEMPORAL = "temporal"


@dataclass
class RiskCalculationContext:
    """Context for risk calculations."""
    findings: List[SecurityFinding]
    business_context: Dict[str, Any]
    regulatory_context: Dict[str, Any]
    temporal_context: Dict[str, Any]
    calculation_weights: Dict[str, float]


class IRiskScorer(Protocol):
    """Protocol for risk scoring components."""

    async def calculate_risk(self, context: RiskCalculationContext) -> float:
        """Calculate risk for this dimension."""
        ...

    def get_dimension_name(self) -> str:
        """Get the name of this risk dimension."""
        ...
