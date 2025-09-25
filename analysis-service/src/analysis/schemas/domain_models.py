"""
Domain models for analysis service.

This module defines the core data models used across all analysis engines
to ensure consistent data structures and type safety.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# Enums for classification
class PatternType(Enum):
    """Types of patterns that can be detected."""

    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"


class PatternStrength(Enum):
    """Strength classification for detected patterns."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class RiskLevel(Enum):
    """Risk level classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    INFORMATIONAL = "informational"


class BusinessRelevance(Enum):
    """Business relevance classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"


class AnalysisStrategy(Enum):
    """Analysis strategy types."""

    RULE_BASED_ONLY = "rule_based_only"
    AI_ENHANCED = "ai_enhanced"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


# Core Data Models
class TimeRange(BaseModel):
    """Time range specification."""

    start: datetime = Field(description="Start time")
    end: datetime = Field(description="End time")

    @field_validator("end")
    @classmethod
    def validate_end_after_start(cls, v, info):
        if info.data and "start" in info.data and v <= info.data["start"]:
            raise ValueError("End time must be after start time")
        return v


class SecurityData(BaseModel):
    """Security data for pattern analysis."""

    time_series: List[Dict[str, Any]] = Field(
        description="Time-series security metrics"
    )
    events: List[Dict[str, Any]] = Field(description="Security events")
    multi_dimensional: List[Dict[str, Any]] = Field(
        description="Multi-dimensional data points"
    )
    metrics: List[Dict[str, Any]] = Field(description="Security metrics")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class Pattern(BaseModel):
    """Detected pattern in security data."""

    pattern_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique pattern ID"
    )
    pattern_type: PatternType = Field(description="Type of pattern detected")
    strength: PatternStrength = Field(description="Strength of the pattern")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in pattern detection"
    )
    description: str = Field(description="Human-readable pattern description")
    affected_detectors: List[str] = Field(
        description="Detectors affected by this pattern"
    )
    time_range: TimeRange = Field(description="Time range where pattern was observed")
    statistical_significance: float = Field(
        ge=0.0, le=1.0, description="Statistical significance"
    )
    business_relevance: BusinessRelevance = Field(
        description="Business relevance of pattern"
    )
    supporting_evidence: List[Dict[str, Any]] = Field(
        description="Evidence supporting pattern"
    )


class SecurityFinding(BaseModel):
    """Individual security finding."""

    finding_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique finding ID"
    )
    detector_id: str = Field(description="ID of detector that generated finding")
    severity: RiskLevel = Field(description="Severity level of finding")
    category: str = Field(description="Category of security finding")
    description: str = Field(description="Description of the finding")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional finding metadata"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in finding")


class RiskFactor(BaseModel):
    """Individual risk factor contributing to overall risk."""

    factor_name: str = Field(description="Name of the risk factor")
    weight: float = Field(ge=0.0, le=1.0, description="Weight of this factor")
    value: float = Field(ge=0.0, le=1.0, description="Value of this factor")
    contribution: float = Field(
        ge=0.0, le=1.0, description="Contribution to overall risk"
    )
    justification: str = Field(description="Justification for this factor's value")


class RiskBreakdown(BaseModel):
    """Detailed breakdown of risk score components."""

    technical_risk: float = Field(
        ge=0.0, le=1.0, description="Technical risk component"
    )
    business_risk: float = Field(ge=0.0, le=1.0, description="Business risk component")
    regulatory_risk: float = Field(
        ge=0.0, le=1.0, description="Regulatory risk component"
    )
    temporal_risk: float = Field(ge=0.0, le=1.0, description="Temporal risk component")
    contributing_factors: List[RiskFactor] = Field(
        description="Individual risk factors"
    )
    methodology: str = Field(description="Risk calculation methodology used")


class RiskScore(BaseModel):
    """Comprehensive risk score assessment."""

    composite_score: float = Field(
        ge=0.0, le=1.0, description="Overall composite risk score"
    )
    risk_level: RiskLevel = Field(description="Categorical risk level")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in risk assessment"
    )
    breakdown: RiskBreakdown = Field(description="Detailed risk breakdown")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validity_period: Optional[TimeRange] = Field(
        None, description="Period for which score is valid"
    )


class BusinessImpact(BaseModel):
    """Business impact assessment."""

    financial_impact: Dict[str, Any] = Field(description="Financial impact estimates")
    operational_impact: Dict[str, Any] = Field(
        description="Operational impact assessment"
    )
    reputational_impact: Dict[str, Any] = Field(
        description="Reputational impact assessment"
    )
    compliance_impact: Dict[str, Any] = Field(
        description="Compliance impact assessment"
    )
    total_risk_value: float = Field(ge=0.0, description="Total quantified risk value")
    confidence_interval: Dict[str, Any] = Field(
        description="Confidence intervals for estimates"
    )
    impact_timeline: Dict[str, Any] = Field(
        description="Timeline for impact realization"
    )


class ComplianceMapping(BaseModel):
    """Mapping of findings to compliance framework requirements."""

    mapping_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique mapping ID"
    )
    framework: ComplianceFramework = Field(description="Target compliance framework")
    control_id: str = Field(description="Specific control or requirement ID")
    control_description: str = Field(description="Description of the control")
    finding_ids: List[str] = Field(description="IDs of findings mapped to this control")
    compliance_status: str = Field(description="Current compliance status")
    gap_severity: RiskLevel = Field(description="Severity of any compliance gap")
    remediation_priority: int = Field(
        ge=1, le=5, description="Priority for remediation (1=highest)"
    )


class AnalysisResult(BaseModel):
    """Standardized analysis result from any analysis engine."""

    result_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique result ID"
    )
    engine_name: str = Field(description="Name of the analysis engine")
    analysis_type: str = Field(description="Type of analysis performed")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in results"
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_ms: int = Field(ge=0, description="Processing time in milliseconds")

    # Core analysis outputs
    patterns: List[Pattern] = Field(
        default_factory=list, description="Detected patterns"
    )
    risk_scores: List[RiskScore] = Field(
        default_factory=list, description="Risk assessments"
    )
    compliance_mappings: List[ComplianceMapping] = Field(
        default_factory=list, description="Compliance mappings"
    )

    # Supporting data
    evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="Supporting evidence"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    quality_indicators: Dict[str, float] = Field(
        default_factory=dict, description="Quality indicators"
    )


class AnalysisConfiguration(BaseModel):
    """Configuration for analysis engines."""

    engine_name: str = Field(description="Name of the analysis engine")
    enabled: bool = Field(default=True, description="Whether engine is enabled")
    confidence_threshold: float = Field(
        ge=0.0, le=1.0, default=0.7, description="Minimum confidence threshold"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Engine-specific parameters"
    )
    weights: Dict[str, float] = Field(
        default_factory=dict, description="Weighting factors"
    )
    fallback_enabled: bool = Field(
        default=True, description="Whether fallback is enabled"
    )
