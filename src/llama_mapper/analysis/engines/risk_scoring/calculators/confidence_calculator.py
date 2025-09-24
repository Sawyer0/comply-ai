"""
Confidence Calculator - Enhanced confidence assessment for risk scores.
"""

import logging
from typing import List, Dict, Any

from ....domain.analysis_models import SecurityFinding, RiskBreakdown
from ..types import RiskCalculationContext

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """Calculator for enhanced confidence scoring with multiple factors."""

    def __init__(self):
        pass

    async def calculate_enhanced_confidence(self, 
                                          findings: List[SecurityFinding],
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

            return max(0.1, min(1.0, weighted_confidence))

        except Exception as e:
            logger.error("Error calculating enhanced confidence: %s", e)
            return 0.5  # Default moderate confidence

    async def _calculate_base_confidence(self, findings: List[SecurityFinding]) -> float:
        """Calculate base confidence from finding characteristics."""
        if not findings:
            return 1.0

        # Average finding confidence
        avg_finding_confidence = sum(f.confidence for f in findings) / len(findings)

        # High confidence findings bonus
        high_confidence_findings = [f for f in findings if f.confidence > 0.8]
        high_confidence_ratio = len(high_confidence_findings) / len(findings)

        # Combine factors
        base_confidence = (avg_finding_confidence * 0.7) + (high_confidence_ratio * 0.3)

        return base_confidence

    async def _calculate_data_quality_confidence(self, findings: List[SecurityFinding]) -> float:
        """Calculate confidence based on data quality and completeness."""
        quality_factors = []

        # Metadata completeness
        metadata_completeness = self._assess_metadata_completeness(findings)
        quality_factors.append(metadata_completeness)

        # Finding diversity (multiple detector types)
        detector_diversity = self._assess_detector_diversity(findings)
        quality_factors.append(detector_diversity)

        # Data consistency
        data_consistency = self._assess_data_consistency(findings)
        quality_factors.append(data_consistency)

        return sum(quality_factors) / len(quality_factors)

    def _assess_metadata_completeness(self, findings: List[SecurityFinding]) -> float:
        """Assess how complete the metadata is across findings."""
        if not findings:
            return 1.0

        important_fields = [
            'attack_vector', 'attack_complexity', 'confidentiality_impact',
            'integrity_impact', 'availability_impact', 'affected_systems'
        ]

        total_completeness = 0.0
        for finding in findings:
            metadata = finding.metadata
            if metadata:
                present_fields = sum(1 for field in important_fields if field in metadata)
                completeness = present_fields / len(important_fields)
            else:
                completeness = 0.0
            total_completeness += completeness

        return total_completeness / len(findings)

    def _assess_detector_diversity(self, findings: List[SecurityFinding]) -> float:
        """Assess diversity of detector types (more diversity = higher confidence)."""
        if not findings:
            return 1.0

        unique_detectors = len(set(f.detector_id for f in findings))
        total_findings = len(findings)

        # Normalize diversity score
        diversity_ratio = unique_detectors / total_findings
        
        # Higher diversity up to a point increases confidence
        if diversity_ratio > 0.7:
            return 1.0
        elif diversity_ratio > 0.5:
            return 0.9
        elif diversity_ratio > 0.3:
            return 0.8
        else:
            return 0.7

    def _assess_data_consistency(self, findings: List[SecurityFinding]) -> float:
        """Assess consistency of data across findings."""
        if len(findings) <= 1:
            return 1.0

        # Check for consistent timestamp patterns
        timestamps = [f.timestamp for f in findings]
        timestamp_variance = self._calculate_timestamp_variance(timestamps)

        # Check for consistent confidence levels
        confidences = [f.confidence for f in findings]
        confidence_variance = self._calculate_confidence_variance(confidences)

        # Lower variance = higher consistency = higher confidence
        timestamp_consistency = max(0.0, 1.0 - (timestamp_variance / 24))  # 24 hour variance = 0 consistency
        confidence_consistency = max(0.0, 1.0 - (confidence_variance * 2))  # Scale variance

        return (timestamp_consistency + confidence_consistency) / 2

    def _calculate_timestamp_variance(self, timestamps) -> float:
        """Calculate variance in timestamps (in hours)."""
        if len(timestamps) <= 1:
            return 0.0

        # Convert to hours from earliest timestamp
        earliest = min(timestamps)
        hours = [(ts - earliest).total_seconds() / 3600 for ts in timestamps]
        
        mean_hours = sum(hours) / len(hours)
        variance = sum((h - mean_hours) ** 2 for h in hours) / len(hours)
        
        return variance ** 0.5  # Standard deviation

    def _calculate_confidence_variance(self, confidences) -> float:
        """Calculate variance in confidence scores."""
        if len(confidences) <= 1:
            return 0.0

        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
        
        return variance ** 0.5  # Standard deviation

    async def _calculate_context_confidence(self, context: RiskCalculationContext) -> float:
        """Calculate confidence based on context quality."""
        context_factors = []

        # Business context completeness
        business_completeness = len(context.business_context) / 10.0  # Normalize by expected fields
        context_factors.append(min(1.0, business_completeness))

        # Regulatory context completeness
        regulatory_completeness = len(context.regulatory_context) / 5.0  # Normalize by expected fields
        context_factors.append(min(1.0, regulatory_completeness))

        # Temporal context completeness
        temporal_completeness = len(context.temporal_context) / 3.0  # Normalize by expected fields
        context_factors.append(min(1.0, temporal_completeness))

        return sum(context_factors) / len(context_factors) if context_factors else 0.5

    async def _calculate_breakdown_confidence(self, breakdown: RiskBreakdown, findings: List[SecurityFinding]) -> float:
        """Calculate confidence based on breakdown consistency."""
        try:
            # Check if breakdown values are reasonable given input
            breakdown_factors = []

            # Check if technical risk aligns with finding severities
            technical_alignment = self._assess_technical_alignment(breakdown.technical_risk, findings)
            breakdown_factors.append(technical_alignment)

            # Check if contributing factors are well-justified
            factor_quality = self._assess_contributing_factor_quality(breakdown.contributing_factors)
            breakdown_factors.append(factor_quality)

            # Check for balanced risk distribution
            risk_balance = self._assess_risk_balance(breakdown)
            breakdown_factors.append(risk_balance)

            return sum(breakdown_factors) / len(breakdown_factors)

        except Exception as e:
            logger.warning("Error calculating breakdown confidence: %s", e)
            return 0.5

    def _assess_technical_alignment(self, technical_risk: float, findings: List[SecurityFinding]) -> float:
        """Assess if technical risk aligns with finding severities."""
        if not findings:
            return 1.0 if technical_risk == 0.0 else 0.5

        # Calculate expected technical risk from severities
        severity_scores = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }

        expected_risk = sum(
            severity_scores.get(f.severity.value.lower(), 0.5) * f.confidence
            for f in findings
        ) / len(findings)

        # Calculate alignment (closer values = higher confidence)
        diff = abs(technical_risk - expected_risk)
        alignment = max(0.0, 1.0 - (diff * 2))  # Scale difference

        return alignment

    def _assess_contributing_factor_quality(self, contributing_factors) -> float:
        """Assess the quality of contributing factors."""
        if not contributing_factors:
            return 0.5

        # Check if factors have reasonable weights
        total_weight = sum(f.weight for f in contributing_factors)
        weight_quality = 1.0 if abs(total_weight - 1.0) < 0.1 else 0.7

        # Check if justifications are present
        justified_factors = [f for f in contributing_factors if f.justification and len(f.justification) > 10]
        justification_quality = len(justified_factors) / len(contributing_factors)

        return (weight_quality + justification_quality) / 2

    def _assess_risk_balance(self, breakdown: RiskBreakdown) -> float:
        """Assess if risk is reasonably distributed across dimensions."""
        risk_values = [
            breakdown.technical_risk,
            breakdown.business_risk,
            breakdown.regulatory_risk,
            breakdown.temporal_risk
        ]

        # Check for extreme concentration in one dimension
        max_risk = max(risk_values)
        min_risk = min(risk_values)
        
        # Prefer some distribution over extreme concentration
        if max_risk > 0.9 and min_risk < 0.1:
            return 0.7  # Moderate confidence for extreme concentration
        elif max_risk - min_risk > 0.8:
            return 0.8  # Reasonable distribution
        else:
            return 1.0  # Good balance
