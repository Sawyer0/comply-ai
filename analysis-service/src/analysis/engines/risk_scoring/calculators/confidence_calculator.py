"""
Confidence Calculator - Enhanced confidence assessment for risk scores.

Follows SRP by focusing solely on confidence calculation using
multiple factors and data quality assessment.
"""

import logging
from typing import List, Dict, Any

from ..types import RiskCalculationContext, RiskBreakdown

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """Calculator for enhanced confidence scoring with multiple factors."""

    def __init__(self):
        pass

    async def calculate_enhanced_confidence(self, 
                                          findings: List[Any],
                                          breakdown: RiskBreakdown,
                                          context: RiskCalculationContext) -> float:
        """Calculate enhanced confidence score with multiple factors."""
        if not findings:
            return 1.0  # High confidence for no risk

        try:
            # Calculate base confidence from findings
            base_confidence = await self._calculate_base_confidence(findings)

            # Calculate data quality confidence
            data_quality_confidence = await self._calculate_data_quality_confidence(findings)

            # Calculate context quality confidence
            context_confidence = await self._calculate_context_confidence(context)

            # Calculate breakdown consistency confidence
            breakdown_confidence = await self._calculate_breakdown_confidence(breakdown, findings)

            # Weighted combination of confidence factors
            confidence_weights = {
                'base': 0.3,
                'data_quality': 0.3,
                'context': 0.2,
                'breakdown': 0.2
            }

            weighted_confidence = (
                base_confidence * confidence_weights['base'] +
                data_quality_confidence * confidence_weights['data_quality'] +
                context_confidence * confidence_weights['context'] +
                breakdown_confidence * confidence_weights['breakdown']
            )

            return max(0.0, min(1.0, weighted_confidence))

        except Exception as e:
            logger.error("Error calculating enhanced confidence: %s", e)
            return 0.5  # Default moderate confidence

    async def _calculate_base_confidence(self, findings: List[Any]) -> float:
        """Calculate base confidence from finding confidence scores."""
        if not findings:
            return 1.0

        total_confidence = 0.0
        valid_findings = 0

        for finding in findings:
            confidence = getattr(finding, 'confidence', 0.5)
            if confidence > 0:
                total_confidence += confidence
                valid_findings += 1

        if valid_findings == 0:
            return 0.5  # Default moderate confidence

        return total_confidence / valid_findings

    async def _calculate_data_quality_confidence(self, findings: List[Any]) -> float:
        """Calculate confidence based on data quality indicators."""
        if not findings:
            return 1.0

        quality_indicators = []
        
        for finding in findings:
            # Check metadata completeness
            metadata = getattr(finding, 'metadata', {}) or {}
            metadata_completeness = len(metadata) / 10.0  # Normalize to 0-1
            quality_indicators.append(min(1.0, metadata_completeness))
            
            # Check for required fields
            required_fields = ['severity', 'category', 'description']
            field_completeness = sum(1 for field in required_fields if hasattr(finding, field)) / len(required_fields)
            quality_indicators.append(field_completeness)

        if not quality_indicators:
            return 0.5

        return sum(quality_indicators) / len(quality_indicators)

    async def _calculate_context_confidence(self, context: RiskCalculationContext) -> float:
        """Calculate confidence based on context quality."""
        confidence_factors = []
        
        # Check business context completeness
        if context.business_context:
            business_completeness = len(context.business_context) / 5.0  # Normalize
            confidence_factors.append(min(1.0, business_completeness))
        else:
            confidence_factors.append(0.3)  # Lower confidence without business context
        
        # Check regulatory context
        if context.regulatory_context:
            regulatory_completeness = len(context.regulatory_context) / 3.0  # Normalize
            confidence_factors.append(min(1.0, regulatory_completeness))
        else:
            confidence_factors.append(0.5)  # Moderate confidence without regulatory context
        
        # Check temporal context
        if context.temporal_context:
            temporal_completeness = len(context.temporal_context) / 2.0  # Normalize
            confidence_factors.append(min(1.0, temporal_completeness))
        else:
            confidence_factors.append(0.7)  # Higher confidence without temporal context
        
        return sum(confidence_factors) / len(confidence_factors)

    async def _calculate_breakdown_confidence(self, breakdown: RiskBreakdown, findings: List[Any]) -> float:
        """Calculate confidence based on breakdown consistency."""
        if not breakdown.dimension_scores:
            return 0.5
        
        # Check for score consistency
        scores = list(breakdown.dimension_scores.values())
        if not scores:
            return 0.5
        
        # Calculate variance (lower variance = higher confidence)
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to confidence (lower variance = higher confidence)
        consistency_confidence = max(0.1, 1.0 - (variance * 2))
        
        # Check for reasonable score distribution
        high_scores = sum(1 for score in scores if score > 0.7)
        if high_scores > len(scores) * 0.8:  # Too many high scores might indicate over-scoring
            consistency_confidence *= 0.8
        
        return consistency_confidence

    def get_confidence_factors(self, findings: List[Any], breakdown: RiskBreakdown, 
                             context: RiskCalculationContext) -> Dict[str, float]:
        """Get detailed confidence factors for analysis."""
        return {
            'base_confidence': self._calculate_base_confidence(findings),
            'data_quality_confidence': self._calculate_data_quality_confidence(findings),
            'context_confidence': self._calculate_context_confidence(context),
            'breakdown_confidence': self._calculate_breakdown_confidence(breakdown, findings)
        }
