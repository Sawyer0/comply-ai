"""
Temporal Scorer - Time-based risk assessment.

Follows SRP by focusing solely on temporal risk calculation using
time-based patterns and urgency factors.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from ..types import RiskCalculationContext, RiskDimension

logger = logging.getLogger(__name__)


class TemporalRiskScorer:
    """
    Specialized scorer for temporal risk assessment.

    Calculates temporal risk based on finding age, urgency, recency,
    and temporal patterns.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.temporal_decay_days = config.get('temporal_decay_days', 30)
        self.urgency_weights = config.get('urgency_weights', {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2,
            'critical': 1.5
        })

    async def calculate_risk(self, context: RiskCalculationContext) -> float:
        """Calculate temporal risk component."""
        if not context.findings:
            return 0.0

        total_temporal_risk = 0.0
        valid_findings = 0

        for finding in context.findings:
            try:
                temporal_score = await self._calculate_finding_temporal_risk(finding, context)
                if temporal_score is not None:
                    total_temporal_risk += temporal_score
                    valid_findings += 1
            except Exception as e:
                logger.warning("Error calculating temporal risk for finding %s: %s", 
                             getattr(finding, 'finding_id', 'unknown'), e)
                continue

        if valid_findings == 0:
            logger.warning("No valid findings for temporal risk calculation")
            return 0.0

        return min(1.0, total_temporal_risk / valid_findings)

    async def _calculate_finding_temporal_risk(self, finding: Any, context: RiskCalculationContext) -> Optional[float]:
        """Calculate temporal risk for a single finding."""
        try:
            current_time = datetime.now(timezone.utc)

            # Calculate age-based risk
            finding_time = getattr(finding, 'timestamp', current_time)
            if isinstance(finding_time, str):
                finding_time = datetime.fromisoformat(finding_time.replace('Z', '+00:00'))
            elif finding_time.tzinfo is None:
                finding_time = finding_time.replace(tzinfo=timezone.utc)
                
            age_risk = await self._calculate_age_based_risk(finding_time, current_time)

            # Calculate urgency-based risk
            urgency_risk = await self._calculate_urgency_risk(finding, context)

            # Calculate recency bonus (newer findings are higher risk)
            recency_bonus = await self._calculate_recency_bonus(finding_time, current_time)

            # Calculate temporal patterns
            pattern_multiplier = await self._calculate_temporal_pattern_multiplier(finding, context)

            temporal_risk = (age_risk + urgency_risk + recency_bonus) * pattern_multiplier

            # Apply confidence weighting
            confidence = getattr(finding, 'confidence', 0.5)
            temporal_risk *= confidence

            return max(0.0, min(1.0, temporal_risk))

        except Exception as e:
            logger.error("Error in temporal risk calculation for finding %s: %s", 
                        getattr(finding, 'finding_id', 'unknown'), e)
            return None

    async def _calculate_age_based_risk(self, finding_time: datetime, current_time: datetime) -> float:
        """Calculate risk based on finding age with decay function."""
        age_hours = (current_time - finding_time).total_seconds() / 3600

        if age_hours <= 1:
            # Fresh findings are highest risk
            return 1.0
        elif age_hours <= 24:
            # Recent findings (within 24 hours)
            return 0.9
        elif age_hours <= 168:  # 7 days
            # Recent findings (within 1 week)
            return 0.8
        elif age_hours <= 720:  # 30 days
            # Older findings with decay
            return 0.6
        else:
            # Very old findings with significant decay
            decay_factor = max(0.1, 1.0 - (age_hours / (self.temporal_decay_days * 24)))
            return decay_factor

    async def _calculate_urgency_risk(self, finding: Any, context: RiskCalculationContext) -> float:
        """Calculate risk based on urgency indicators."""
        metadata = getattr(finding, 'metadata', {}) or {}
        
        # Check explicit urgency
        urgency = metadata.get('urgency', 'medium')
        urgency_multiplier = self.urgency_weights.get(urgency, 1.0)
        
        # Check for time-sensitive indicators
        time_sensitive = metadata.get('time_sensitive', False)
        if time_sensitive:
            urgency_multiplier *= 1.3
        
        # Check for SLA deadlines
        sla_deadline = metadata.get('sla_deadline')
        if sla_deadline:
            try:
                deadline = datetime.fromisoformat(sla_deadline.replace('Z', '+00:00'))
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=timezone.utc)
                
                time_to_deadline = (deadline - datetime.now(timezone.utc)).total_seconds()
                if time_to_deadline < 0:
                    # Past deadline - critical urgency
                    urgency_multiplier *= 2.0
                elif time_to_deadline < 3600:  # Less than 1 hour
                    urgency_multiplier *= 1.8
                elif time_to_deadline < 86400:  # Less than 1 day
                    urgency_multiplier *= 1.5
            except Exception as e:
                logger.warning("Error parsing SLA deadline: %s", e)
        
        return min(1.0, urgency_multiplier * 0.5)  # Scale to 0-1 range

    async def _calculate_recency_bonus(self, finding_time: datetime, current_time: datetime) -> float:
        """Calculate recency bonus for newer findings."""
        age_hours = (current_time - finding_time).total_seconds() / 3600
        
        if age_hours <= 0.25:  # 15 minutes
            return 0.3
        elif age_hours <= 1:  # 1 hour
            return 0.2
        elif age_hours <= 6:  # 6 hours
            return 0.1
        else:
            return 0.0

    async def _calculate_temporal_pattern_multiplier(self, finding: Any, context: RiskCalculationContext) -> float:
        """Calculate multiplier based on temporal patterns."""
        metadata = getattr(finding, 'metadata', {}) or {}
        
        # Check for recurring patterns
        is_recurring = metadata.get('is_recurring', False)
        if is_recurring:
            return 1.2
        
        # Check for escalation patterns
        escalation_level = metadata.get('escalation_level', 0)
        if escalation_level > 0:
            return 1.0 + (escalation_level * 0.1)
        
        # Check for business hours impact
        business_hours_impact = metadata.get('business_hours_impact', 'medium')
        impact_multipliers = {
            'none': 0.8,
            'low': 0.9,
            'medium': 1.0,
            'high': 1.1,
            'critical': 1.3
        }
        
        return impact_multipliers.get(business_hours_impact, 1.0)

    def get_dimension_name(self) -> str:
        """Get the name of this risk dimension."""
        return RiskDimension.TEMPORAL.value
