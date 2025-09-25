"""
Advanced Risk Scoring Engine - Comprehensive risk assessment orchestration.

This engine orchestrates specialized risk scoring components following SRP
by focusing solely on risk calculation coordination and result aggregation.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .types import RiskCalculationContext, RiskDimension, RiskBreakdown, RiskLevel
from .scorers import (
    TechnicalRiskScorer,
    BusinessRiskScorer,
    RegulatoryRiskScorer,
    TemporalRiskScorer
)
from .calculators import (
    CompositeRiskCalculator,
    RiskBreakdownGenerator,
    ConfidenceCalculator
)

logger = logging.getLogger(__name__)


class AdvancedRiskScoringEngine:
    """
    Advanced risk scoring engine that orchestrates specialized components.
    
    Provides comprehensive risk assessment by coordinating:
    - Technical risk scoring (CVSS-like methodology)
    - Business impact assessment
    - Regulatory compliance risk
    - Temporal risk factors
    - Composite risk calculation
    - Detailed risk breakdown
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get("risk_scoring", {})
        
        # Initialize specialized scorers
        self.technical_scorer = TechnicalRiskScorer(self.risk_config)
        self.business_scorer = BusinessRiskScorer(self.risk_config)
        self.regulatory_scorer = RegulatoryRiskScorer(self.risk_config)
        self.temporal_scorer = TemporalRiskScorer(self.risk_config)
        
        # Initialize calculators
        self.composite_calculator = CompositeRiskCalculator(
            self.risk_config.get("calculation_weights", {
                "technical": 0.3,
                "business": 0.25,
                "regulatory": 0.25,
                "temporal": 0.2
            })
        )
        self.breakdown_generator = RiskBreakdownGenerator()
        self.confidence_calculator = ConfidenceCalculator()
        
        # Risk thresholds
        self.risk_thresholds = self.risk_config.get("risk_thresholds", {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        })

    async def calculate_comprehensive_risk(self, context: RiskCalculationContext) -> Dict[str, Any]:
        """
        Calculate comprehensive risk assessment using all dimensions.
        
        Args:
            context: Risk calculation context with findings and metadata
            
        Returns:
            Comprehensive risk assessment with scores, breakdown, and metadata
        """
        try:
            logger.info("Starting comprehensive risk calculation", 
                       finding_count=len(context.findings))
            
            # Calculate risk for each dimension
            dimension_scores = await self._calculate_dimension_scores(context)
            
            # Calculate composite risk score
            composite_score = await self.composite_calculator.calculate_weighted_composite_score(
                dimension_scores
            )
            
            # Generate detailed breakdown
            breakdown = await self.breakdown_generator.create_comprehensive_risk_breakdown(
                dimension_scores, context.findings, context
            )
            
            # Calculate confidence score
            confidence = await self.confidence_calculator.calculate_enhanced_confidence(
                context.findings, breakdown, context
            )
            
            # Determine risk level
            risk_level = self._determine_risk_level(composite_score)
            
            # Create comprehensive result
            result = {
                "composite_score": composite_score,
                "risk_level": risk_level.value,
                "dimension_scores": dimension_scores,
                "confidence": confidence,
                "breakdown": {
                    "total_score": breakdown.total_score,
                    "dimension_scores": breakdown.dimension_scores,
                    "confidence_scores": breakdown.confidence_scores,
                    "contributing_factors": breakdown.contributing_factors,
                    "calculation_metadata": breakdown.calculation_metadata
                },
                "metadata": {
                    "calculation_timestamp": datetime.utcnow().isoformat(),
                    "engine_version": "2.0.0",
                    "methodology": "Advanced multi-dimensional risk scoring",
                    "finding_count": len(context.findings),
                    "tenant_id": context.tenant_id
                }
            }
            
            logger.info("Risk calculation completed", 
                       composite_score=composite_score,
                       risk_level=risk_level.value,
                       confidence=confidence)
            
            return result
            
        except Exception as e:
            logger.error("Error in comprehensive risk calculation: %s", e)
            return self._create_error_result(str(e))

    async def _calculate_dimension_scores(self, context: RiskCalculationContext) -> Dict[str, float]:
        """Calculate risk scores for each dimension."""
        dimension_scores = {}
        
        # Technical risk
        try:
            technical_score = await self.technical_scorer.calculate_risk(context)
            dimension_scores["technical"] = technical_score
        except Exception as e:
            logger.warning("Error calculating technical risk: %s", e)
            dimension_scores["technical"] = 0.0
        
        # Business risk
        try:
            business_score = await self.business_scorer.calculate_risk(context)
            dimension_scores["business"] = business_score
        except Exception as e:
            logger.warning("Error calculating business risk: %s", e)
            dimension_scores["business"] = 0.0
        
        # Regulatory risk
        try:
            regulatory_score = await self.regulatory_scorer.calculate_risk(context)
            dimension_scores["regulatory"] = regulatory_score
        except Exception as e:
            logger.warning("Error calculating regulatory risk: %s", e)
            dimension_scores["regulatory"] = 0.0
        
        # Temporal risk
        try:
            temporal_score = await self.temporal_scorer.calculate_risk(context)
            dimension_scores["temporal"] = temporal_score
        except Exception as e:
            logger.warning("Error calculating temporal risk: %s", e)
            dimension_scores["temporal"] = 0.0
        
        return dimension_scores

    def _determine_risk_level(self, composite_score: float) -> RiskLevel:
        """Determine risk level based on composite score."""
        if composite_score >= self.risk_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif composite_score >= self.risk_thresholds["high"]:
            return RiskLevel.HIGH
        elif composite_score >= self.risk_thresholds["medium"]:
            return RiskLevel.MEDIUM
        elif composite_score >= self.risk_thresholds["low"]:
            return RiskLevel.LOW
        else:
            return RiskLevel.INFORMATIONAL

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result when calculation fails."""
        return {
            "composite_score": 0.0,
            "risk_level": "unknown",
            "dimension_scores": {},
            "confidence": 0.0,
            "breakdown": {
                "total_score": 0.0,
                "dimension_scores": {},
                "confidence_scores": {},
                "contributing_factors": [],
                "calculation_metadata": {"error": error_message}
            },
            "metadata": {
                "calculation_timestamp": datetime.utcnow().isoformat(),
                "engine_version": "2.0.0",
                "methodology": "Error state",
                "finding_count": 0,
                "error": error_message
            }
        }

    async def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the risk scoring engine."""
        return {
            "engine_type": "advanced_risk_scoring",
            "version": "2.0.0",
            "dimensions": [dim.value for dim in RiskDimension],
            "scorers": {
                "technical": self.technical_scorer.get_dimension_name(),
                "business": self.business_scorer.get_dimension_name(),
                "regulatory": self.regulatory_scorer.get_dimension_name(),
                "temporal": self.temporal_scorer.get_dimension_name()
            },
            "calculation_weights": self.composite_calculator.get_calculation_weights(),
            "risk_thresholds": self.risk_thresholds
        }

    async def update_configuration(self, new_config: Dict[str, Any]):
        """Update engine configuration."""
        self.risk_config.update(new_config)
        
        # Update calculation weights if provided
        if "calculation_weights" in new_config:
            self.composite_calculator.update_weights(new_config["calculation_weights"])
        
        logger.info("Risk scoring engine configuration updated", config=new_config)
