"""
Risk Scoring Types and Enums

Defines core types and enums for the advanced risk scoring system.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime


class RiskDimension(Enum):
    """Risk dimensions for multi-dimensional risk assessment."""
    
    TECHNICAL = "technical"
    BUSINESS = "business"
    REGULATORY = "regulatory"
    TEMPORAL = "temporal"


class RiskLevel(Enum):
    """Risk level classifications."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class RiskCalculationContext:
    """Context for risk calculation operations."""
    
    findings: List[Any]
    business_context: Optional[Dict[str, Any]] = None
    regulatory_context: Optional[Dict[str, Any]] = None
    temporal_context: Optional[Dict[str, Any]] = None
    tenant_id: Optional[str] = None
    calculation_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.calculation_timestamp is None:
            self.calculation_timestamp = datetime.utcnow()


@dataclass
class RiskScore:
    """Individual risk score with metadata."""
    
    dimension: RiskDimension
    score: float
    confidence: float
    factors: Dict[str, Any]
    calculation_method: str
    timestamp: datetime
    
    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Risk score must be between 0.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class RiskBreakdown:
    """Detailed breakdown of risk calculation."""
    
    total_score: float
    dimension_scores: Dict[RiskDimension, float]
    confidence_scores: Dict[RiskDimension, float]
    contributing_factors: Dict[RiskDimension, List[str]]
    calculation_metadata: Dict[str, Any]
    timestamp: datetime
    
    def get_primary_risk_dimension(self) -> RiskDimension:
        """Get the dimension with the highest score."""
        return max(self.dimension_scores.items(), key=lambda x: x[1])[0]
    
    def get_risk_level(self) -> RiskLevel:
        """Determine overall risk level based on total score."""
        if self.total_score >= 0.9:
            return RiskLevel.CRITICAL
        elif self.total_score >= 0.7:
            return RiskLevel.HIGH
        elif self.total_score >= 0.4:
            return RiskLevel.MEDIUM
        elif self.total_score >= 0.1:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFORMATIONAL
