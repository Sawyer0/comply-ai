"""
Risk Breakdown Generator - Detailed risk factor analysis and justification.
"""

import logging
from typing import List, Dict, Any

from ....domain.analysis_models import SecurityFinding, RiskBreakdown, RiskFactor
from ..types import RiskCalculationContext

logger = logging.getLogger(__name__)


class RiskBreakdownGenerator:
    """Generator for comprehensive risk breakdowns with detailed justifications."""

    def __init__(self):
        pass

    async def create_comprehensive_risk_breakdown(self, 
                                                risk_components: Dict[str, float],
                                                findings: List[SecurityFinding],
                                                context: RiskCalculationContext) -> RiskBreakdown:
        """Create comprehensive risk breakdown with detailed factors."""
        try:
            # Create contributing factors for each risk dimension
            contributing_factors = []
            
            for dimension_name, risk_value in risk_components.items():
                if risk_value > 0:
                    factor = await self._create_risk_factor(
                        dimension_name, risk_value, findings, context
                    )
                    contributing_factors.append(factor)

            # Sort factors by contribution (highest first)
            contributing_factors.sort(key=lambda f: f.contribution, reverse=True)

            return RiskBreakdown(
                technical_risk=risk_components.get('technical', 0.0),
                business_risk=risk_components.get('business', 0.0),
                regulatory_risk=risk_components.get('regulatory', 0.0),
                temporal_risk=risk_components.get('temporal', 0.0),
                contributing_factors=contributing_factors,
                methodology="Modular risk scoring with specialized components"
            )

        except Exception as e:
            logger.error("Error creating risk breakdown: %s", e)
            # Return basic breakdown
            return RiskBreakdown(
                technical_risk=risk_components.get('technical', 0.0),
                business_risk=risk_components.get('business', 0.0),
                regulatory_risk=risk_components.get('regulatory', 0.0),
                temporal_risk=risk_components.get('temporal', 0.0),
                contributing_factors=[],
                methodology="Basic risk scoring (error in detailed breakdown)"
            )

    async def _create_risk_factor(self, dimension_name: str, risk_value: float,
                                findings: List[SecurityFinding],
                                context: RiskCalculationContext) -> RiskFactor:
        """Create a risk factor for a specific dimension."""
        weight = context.calculation_weights.get(dimension_name, 0.0)
        contribution = risk_value * weight
        
        justification = await self._generate_justification(
            dimension_name, risk_value, findings, context
        )

        return RiskFactor(
            factor_name=f"{dimension_name}_risk",
            weight=weight,
            value=risk_value,
            contribution=contribution,
            justification=justification
        )

    async def _generate_justification(self, dimension_name: str, risk_value: float,
                                    findings: List[SecurityFinding],
                                    context: RiskCalculationContext) -> str:
        """Generate human-readable justification for a risk factor."""
        try:
            if dimension_name == 'technical':
                return self._generate_technical_justification(risk_value, findings)
            elif dimension_name == 'business':
                return self._generate_business_justification(risk_value, findings, context)
            elif dimension_name == 'regulatory':
                return self._generate_regulatory_justification(risk_value, findings, context)
            elif dimension_name == 'temporal':
                return self._generate_temporal_justification(risk_value, findings, context)
            else:
                return f"{dimension_name.title()} risk assessment: {risk_value:.2f}"

        except Exception as e:
            logger.warning("Error generating justification for %s: %s", dimension_name, e)
            return f"{dimension_name.title()} risk: {risk_value:.2f}"

    def _generate_technical_justification(self, risk_value: float, findings: List[SecurityFinding]) -> str:
        """Generate justification for technical risk."""
        severity_counts = {}
        for finding in findings:
            severity = finding.severity.value.lower()
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        justification_parts = []
        for severity, count in severity_counts.items():
            justification_parts.append(f"{count} {severity} severity finding{'s' if count > 1 else ''}")

        justification = f"Technical risk based on " + ", ".join(justification_parts)
        
        if risk_value > 0.8:
            justification += " with critical CVSS factors"
        elif risk_value > 0.6:
            justification += " with significant exploitability factors"
        
        return justification

    def _generate_business_justification(self, risk_value: float, findings: List[SecurityFinding], context: RiskCalculationContext) -> str:
        """Generate justification for business risk."""
        affected_systems = set()
        affected_processes = set()
        
        for finding in findings:
            affected_systems.update(finding.metadata.get('affected_systems', []))
            affected_processes.update(finding.metadata.get('affected_processes', []))

        justification = f"Business impact risk of {risk_value:.2f}"
        
        if affected_systems:
            justification += f" affecting {len(affected_systems)} system{'s' if len(affected_systems) > 1 else ''}"
        
        if affected_processes:
            justification += f" and {len(affected_processes)} business process{'es' if len(affected_processes) > 1 else ''}"

        return justification

    def _generate_regulatory_justification(self, risk_value: float, findings: List[SecurityFinding], context: RiskCalculationContext) -> str:
        """Generate justification for regulatory risk."""
        regulations = set()
        for finding in findings:
            regulations.update(finding.metadata.get('applicable_regulations', []))

        if regulations:
            reg_list = ", ".join(sorted(regulations))
            return f"Regulatory compliance risk for {reg_list} frameworks"
        else:
            return f"General regulatory compliance risk: {risk_value:.2f}"

    def _generate_temporal_justification(self, risk_value: float, findings: List[SecurityFinding], context: RiskCalculationContext) -> str:
        """Generate justification for temporal risk."""
        if not findings:
            return f"Temporal risk: {risk_value:.2f}"

        # Calculate average age in hours
        now = context.temporal_context.get('current_time')
        if now:
            ages_hours = [(now - f.timestamp).total_seconds() / 3600 for f in findings]
            avg_age_hours = sum(ages_hours) / len(ages_hours)
            return f"Temporal risk based on finding age (avg {avg_age_hours:.1f} hours) and urgency indicators"
        else:
            return f"Temporal risk assessment: {risk_value:.2f}"
