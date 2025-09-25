"""
Risk taxonomy for the analysis service.

This module defines the risk classification system and taxonomy
used for risk assessment and scoring operations.
"""

from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass


class RiskCategory(Enum):
    """Risk categories for classification."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"


class RiskLevel(Enum):
    """Risk severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class RiskFactor:
    """Individual risk factor definition."""

    name: str
    category: RiskCategory
    weight: float
    description: str
    calculation_method: str


class RiskTaxonomy:
    """
    Risk taxonomy management system.

    Provides classification and categorization of risks
    for consistent risk assessment across the analysis service.
    """

    def __init__(self):
        self.risk_factors = self._initialize_risk_factors()
        self.category_weights = self._initialize_category_weights()
        self.level_thresholds = self._initialize_level_thresholds()

    def _initialize_risk_factors(self) -> Dict[str, RiskFactor]:
        """Initialize standard risk factors."""
        return {
            "data_exposure": RiskFactor(
                name="data_exposure",
                category=RiskCategory.TECHNICAL,
                weight=0.8,
                description="Risk of sensitive data exposure",
                calculation_method="severity_weighted",
            ),
            "compliance_violation": RiskFactor(
                name="compliance_violation",
                category=RiskCategory.REGULATORY,
                weight=0.9,
                description="Risk of regulatory compliance violations",
                calculation_method="framework_weighted",
            ),
            "business_disruption": RiskFactor(
                name="business_disruption",
                category=RiskCategory.BUSINESS,
                weight=0.7,
                description="Risk of business process disruption",
                calculation_method="impact_weighted",
            ),
            "security_breach": RiskFactor(
                name="security_breach",
                category=RiskCategory.TECHNICAL,
                weight=0.95,
                description="Risk of security breach or attack",
                calculation_method="threat_weighted",
            ),
        }

    def _initialize_category_weights(self) -> Dict[RiskCategory, float]:
        """Initialize category weights for composite scoring."""
        return {
            RiskCategory.TECHNICAL: 0.3,
            RiskCategory.BUSINESS: 0.25,
            RiskCategory.REGULATORY: 0.25,
            RiskCategory.OPERATIONAL: 0.15,
            RiskCategory.STRATEGIC: 0.05,
        }

    def _initialize_level_thresholds(self) -> Dict[RiskLevel, float]:
        """Initialize risk level thresholds."""
        return {
            RiskLevel.CRITICAL: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.LOW: 0.1,
            RiskLevel.NEGLIGIBLE: 0.0,
        }

    def get_risk_factor(self, name: str) -> RiskFactor:
        """Get risk factor by name."""
        return self.risk_factors.get(name)

    def get_category_weight(self, category: RiskCategory) -> float:
        """Get weight for risk category."""
        return self.category_weights.get(category, 0.0)

    def classify_risk_level(self, score: float) -> RiskLevel:
        """Classify numeric risk score into risk level."""
        for level in [
            RiskLevel.CRITICAL,
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
            RiskLevel.LOW,
        ]:
            if score >= self.level_thresholds[level]:
                return level
        return RiskLevel.NEGLIGIBLE

    def get_factors_by_category(self, category: RiskCategory) -> List[RiskFactor]:
        """Get all risk factors for a specific category."""
        return [
            factor
            for factor in self.risk_factors.values()
            if factor.category == category
        ]
