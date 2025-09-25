"""
Risk Breakdown Generator - Detailed risk factor analysis and justification.

Follows SRP by focusing solely on risk breakdown generation with
detailed factor analysis and justification.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from ..types import RiskCalculationContext, RiskBreakdown, RiskFactor

logger = logging.getLogger(__name__)


class RiskBreakdownGenerator:
    """Generator for comprehensive risk breakdowns with detailed justifications."""

    def __init__(self):
        pass

    async def create_comprehensive_risk_breakdown(self, 
                                                risk_components: Dict[str, float],
                                                findings: List[Any],
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
                total_score=sum(risk_components.values()) / len(risk_components) if risk_components else 0.0,
                dimension_scores=risk_components,
                confidence_scores={dim: 0.8 for dim in risk_components.keys()},  # Default confidence
                contributing_factors=[f.description for f in contributing_factors],
                calculation_metadata={
                    "methodology": "Modular risk scoring with specialized components",
                    "timestamp": datetime.utcnow().isoformat(),
                    "finding_count": len(findings)
                },
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error("Error creating risk breakdown: %s", e)
            # Return basic breakdown
            return RiskBreakdown(
                total_score=0.0,
                dimension_scores=risk_components,
                confidence_scores={},
                contributing_factors=[],
                calculation_metadata={"error": str(e)},
                timestamp=datetime.utcnow()
            )

    async def _create_risk_factor(self, dimension_name: str, risk_value: float, 
                                findings: List[Any], context: RiskCalculationContext) -> RiskFactor:
        """Create detailed risk factor for a specific dimension."""
        try:
            # Analyze findings for this dimension
            relevant_findings = await self._filter_findings_by_dimension(findings, dimension_name)
            
            # Generate factor description
            description = await self._generate_factor_description(
                dimension_name, risk_value, relevant_findings, context
            )
            
            # Calculate contribution percentage
            total_risk = sum(risk_value for risk_value in [risk_value] if risk_value > 0)
            contribution = (risk_value / total_risk) * 100 if total_risk > 0 else 0
            
            return RiskFactor(
                dimension=dimension_name,
                contribution=contribution,
                description=description,
                evidence_count=len(relevant_findings),
                severity_level=self._determine_severity_level(risk_value)
            )

        except Exception as e:
            logger.error("Error creating risk factor for %s: %s", dimension_name, e)
            return RiskFactor(
                dimension=dimension_name,
                contribution=0.0,
                description=f"Error analyzing {dimension_name} risk: {str(e)}",
                evidence_count=0,
                severity_level="unknown"
            )

    async def _filter_findings_by_dimension(self, findings: List[Any], dimension_name: str) -> List[Any]:
        """Filter findings relevant to a specific risk dimension."""
        relevant_findings = []
        
        for finding in findings:
            metadata = getattr(finding, 'metadata', {}) or {}
            
            # Check if finding is relevant to this dimension
            if dimension_name == 'technical':
                # Technical findings typically have severity and technical metadata
                if hasattr(finding, 'severity') or 'cvss' in metadata:
                    relevant_findings.append(finding)
            elif dimension_name == 'business':
                # Business findings have business impact metadata
                if any(key in metadata for key in ['business_impact', 'affected_processes', 'financial_impact']):
                    relevant_findings.append(finding)
            elif dimension_name == 'regulatory':
                # Regulatory findings have compliance metadata
                if any(key in metadata for key in ['applicable_regulations', 'compliance_framework', 'regulatory_impact']):
                    relevant_findings.append(finding)
            elif dimension_name == 'temporal':
                # Temporal findings have time-based metadata
                if any(key in metadata for key in ['urgency', 'sla_deadline', 'time_sensitive']):
                    relevant_findings.append(finding)
        
        return relevant_findings

    async def _generate_factor_description(self, dimension_name: str, risk_value: float,
                                         relevant_findings: List[Any], context: RiskCalculationContext) -> str:
        """Generate detailed description for a risk factor."""
        finding_count = len(relevant_findings)
        
        if finding_count == 0:
            return f"No specific {dimension_name} findings identified"
        
        # Base description
        description = f"{dimension_name.title()} risk level: {self._format_risk_level(risk_value)}"
        
        # Add finding-specific details
        if dimension_name == 'technical':
            severity_counts = self._count_by_severity(relevant_findings)
            description += f" based on {finding_count} technical findings"
            if severity_counts:
                description += f" ({severity_counts})"
        elif dimension_name == 'business':
            description += f" affecting {finding_count} business processes"
        elif dimension_name == 'regulatory':
            regulations = self._extract_regulations(relevant_findings)
            description += f" involving {finding_count} compliance issues"
            if regulations:
                description += f" ({', '.join(regulations[:3])})"
        elif dimension_name == 'temporal':
            urgency_levels = self._count_by_urgency(relevant_findings)
            description += f" with {finding_count} time-sensitive issues"
            if urgency_levels:
                description += f" ({urgency_levels})"
        
        return description

    def _determine_severity_level(self, risk_value: float) -> str:
        """Determine severity level based on risk value."""
        if risk_value >= 0.9:
            return "critical"
        elif risk_value >= 0.7:
            return "high"
        elif risk_value >= 0.4:
            return "medium"
        elif risk_value >= 0.1:
            return "low"
        else:
            return "informational"

    def _format_risk_level(self, risk_value: float) -> str:
        """Format risk value as percentage."""
        return f"{risk_value * 100:.1f}%"

    def _count_by_severity(self, findings: List[Any]) -> str:
        """Count findings by severity level."""
        severity_counts = {}
        for finding in findings:
            severity = getattr(finding, 'severity', None)
            if hasattr(severity, 'value'):
                severity = severity.value.lower()
            elif isinstance(severity, str):
                severity = severity.lower()
            else:
                severity = 'unknown'
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if not severity_counts:
            return ""
        
        return ", ".join([f"{count} {severity}" for severity, count in severity_counts.items()])

    def _extract_regulations(self, findings: List[Any]) -> List[str]:
        """Extract unique regulations from findings."""
        regulations = set()
        for finding in findings:
            metadata = getattr(finding, 'metadata', {}) or {}
            applicable_regs = metadata.get('applicable_regulations', [])
            regulations.update(applicable_regs)
        
        return list(regulations)

    def _count_by_urgency(self, findings: List[Any]) -> str:
        """Count findings by urgency level."""
        urgency_counts = {}
        for finding in findings:
            metadata = getattr(finding, 'metadata', {}) or {}
            urgency = metadata.get('urgency', 'medium')
            urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
        
        if not urgency_counts:
            return ""
        
        return ", ".join([f"{count} {urgency}" for urgency, count in urgency_counts.items()])
