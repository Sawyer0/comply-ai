"""
Business Impact Scorer - Business impact and financial risk assessment.

Follows SRP by focusing solely on business impact calculation using
sophisticated business relevance assessment.
"""

import logging
from typing import Any, Dict, List, Optional

from ..types import RiskCalculationContext, RiskDimension

logger = logging.getLogger(__name__)


class BusinessRiskScorer:
    """
    Specialized scorer for business impact assessment.

    Calculates business risk based on financial impact, operational disruption,
    reputational damage, and compliance implications.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.impact_weights = config.get('impact_weights', {
            'revenue': 0.4,
            'operational': 0.3,
            'reputational': 0.2,
            'compliance': 0.1
        })

    async def calculate_risk(self, context: RiskCalculationContext) -> float:
        """Calculate business impact risk component."""
        if not context.findings:
            return 0.0

        total_business_risk = 0.0
        valid_findings = 0

        for finding in context.findings:
            try:
                business_score = await self._calculate_finding_business_impact(finding, context)
                if business_score is not None:
                    total_business_risk += business_score
                    valid_findings += 1
            except Exception as e:
                logger.warning("Error calculating business risk for finding %s: %s", 
                             getattr(finding, 'finding_id', 'unknown'), e)
                continue

        if valid_findings == 0:
            logger.warning("No valid findings for business risk calculation")
            return 0.0

        return min(1.0, total_business_risk / valid_findings)

    async def _calculate_finding_business_impact(self, finding: Any, context: RiskCalculationContext) -> Optional[float]:
        """Calculate business impact for a single finding."""
        try:
            # Base impact from severity
            severity_impact = {
                'low': 0.1,
                'medium': 0.3,
                'high': 0.7,
                'critical': 1.0
            }
            
            severity = getattr(finding, 'severity', None)
            if hasattr(severity, 'value'):
                severity = severity.value.lower()
            elif isinstance(severity, str):
                severity = severity.lower()
            else:
                severity = 'medium'
                
            base_impact = severity_impact.get(severity, 0.3)

            # Business criticality assessment
            criticality = await self._assess_business_criticality(finding, context)
            business_impact = base_impact * criticality

            # Affected business processes multiplier
            metadata = getattr(finding, 'metadata', {}) or {}
            affected_processes = metadata.get('affected_processes', [])
            process_multiplier = await self._calculate_process_impact_multiplier(
                affected_processes, context
            )
            business_impact *= process_multiplier

            # Data sensitivity multiplier
            data_sensitivity = metadata.get('data_sensitivity', 'medium')
            sensitivity_multiplier = await self._calculate_sensitivity_multiplier(
                data_sensitivity, context
            )
            business_impact *= sensitivity_multiplier

            # Financial impact assessment
            financial_impact = await self._calculate_financial_impact(finding, context)
            business_impact = self._combine_impact_scores(business_impact, financial_impact)

            # Apply confidence weighting
            confidence = getattr(finding, 'confidence', 0.5)
            business_impact *= confidence

            return max(0.0, min(1.0, business_impact))

        except Exception as e:
            logger.error("Error in business impact calculation for finding %s: %s", 
                        getattr(finding, 'finding_id', 'unknown'), e)
            return None

    async def _assess_business_criticality(self, finding: Any, context: RiskCalculationContext) -> float:
        """Assess business criticality of the finding."""
        metadata = getattr(finding, 'metadata', {}) or {}
        
        # Check if finding affects critical business functions
        critical_functions = context.business_context.get('critical_functions', []) if context.business_context else []
        affected_functions = metadata.get('affected_functions', [])
        
        if any(func in critical_functions for func in affected_functions):
            return 1.0
        
        # Check business hours impact
        business_hours_impact = metadata.get('business_hours_impact', 'medium')
        impact_multipliers = {
            'none': 0.1,
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9,
            'critical': 1.0
        }
        
        return impact_multipliers.get(business_hours_impact, 0.6)

    async def _calculate_process_impact_multiplier(self, affected_processes: List[str], 
                                                 context: RiskCalculationContext) -> float:
        """Calculate multiplier based on affected business processes."""
        if not affected_processes:
            return 1.0
            
        business_context = context.business_context or {}
        critical_processes = business_context.get('critical_processes', [])
        
        # Count critical processes affected
        critical_affected = sum(1 for proc in affected_processes if proc in critical_processes)
        
        if critical_affected > 0:
            return 1.0 + (critical_affected * 0.2)  # 20% increase per critical process
        
        return 1.0

    async def _calculate_sensitivity_multiplier(self, data_sensitivity: str, 
                                              context: RiskCalculationContext) -> float:
        """Calculate multiplier based on data sensitivity."""
        sensitivity_multipliers = {
            'public': 0.5,
            'internal': 1.0,
            'confidential': 1.5,
            'restricted': 2.0,
            'top_secret': 2.5
        }
        
        return sensitivity_multipliers.get(data_sensitivity, 1.0)

    async def _calculate_financial_impact(self, finding: Any, context: RiskCalculationContext) -> float:
        """Calculate financial impact of the finding."""
        metadata = getattr(finding, 'metadata', {}) or {}
        
        # Direct financial impact
        direct_cost = metadata.get('estimated_cost', 0)
        if isinstance(direct_cost, (int, float)) and direct_cost > 0:
            # Normalize to 0-1 scale (assuming max cost of $1M)
            return min(1.0, direct_cost / 1000000)
        
        # Indirect financial impact based on severity
        severity = getattr(finding, 'severity', None)
        if hasattr(severity, 'value'):
            severity = severity.value.lower()
        elif isinstance(severity, str):
            severity = severity.lower()
        else:
            severity = 'medium'
            
        severity_costs = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'critical': 1.0
        }
        
        return severity_costs.get(severity, 0.3)

    def _combine_impact_scores(self, business_impact: float, financial_impact: float) -> float:
        """Combine business and financial impact scores."""
        # Weighted combination
        return (business_impact * 0.7) + (financial_impact * 0.3)

    def get_dimension_name(self) -> str:
        """Get the name of this risk dimension."""
        return RiskDimension.BUSINESS.value
