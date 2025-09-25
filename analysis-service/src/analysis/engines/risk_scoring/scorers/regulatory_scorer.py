"""
Regulatory Scorer - Compliance framework risk assessment.

Follows SRP by focusing solely on regulatory compliance risk calculation
using compliance framework mapping.
"""

import logging
from typing import Any, Dict, List, Optional

from ..types import RiskCalculationContext, RiskDimension

logger = logging.getLogger(__name__)


class RegulatoryRiskScorer:
    """
    Specialized scorer for regulatory compliance risk assessment.

    Calculates regulatory risk based on applicable compliance frameworks,
    violation severity, and regulatory impact.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.regulatory_weights = config.get('regulatory_weights', {
            'soc2': 1.2,
            'iso27001': 1.1,
            'hipaa': 1.5,
            'gdpr': 1.4,
            'pci_dss': 1.3,
            'sox': 1.4,
            'fedramp': 1.6
        })

    async def calculate_risk(self, context: RiskCalculationContext) -> float:
        """Calculate regulatory risk component."""
        if not context.findings:
            return 0.0

        total_regulatory_risk = 0.0
        valid_findings = 0

        for finding in context.findings:
            try:
                regulatory_score = await self._calculate_finding_regulatory_risk(finding, context)
                if regulatory_score is not None:
                    total_regulatory_risk += regulatory_score
                    valid_findings += 1
            except Exception as e:
                logger.warning("Error calculating regulatory risk for finding %s: %s", 
                             getattr(finding, 'finding_id', 'unknown'), e)
                continue

        if valid_findings == 0:
            logger.warning("No valid findings for regulatory risk calculation")
            return 0.0

        return min(1.0, total_regulatory_risk / valid_findings)

    async def _calculate_finding_regulatory_risk(self, finding: Any, context: RiskCalculationContext) -> Optional[float]:
        """Calculate regulatory risk for a single finding."""
        try:
            # Identify applicable regulations
            applicable_regulations = await self._identify_applicable_regulations(finding, context)

            if not applicable_regulations:
                # Use default regulatory risk
                return await self._calculate_default_regulatory_risk(finding)

            # Calculate regulatory risk for each applicable regulation
            regulatory_risks = []
            for regulation in applicable_regulations:
                risk = await self._calculate_regulation_specific_risk(finding, regulation, context)
                regulatory_risks.append(risk)

            # Take the maximum regulatory risk (most severe)
            regulatory_risk = max(regulatory_risks)

            # Apply confidence weighting
            confidence = getattr(finding, 'confidence', 0.5)
            regulatory_risk *= confidence

            return max(0.0, min(1.0, regulatory_risk))

        except Exception as e:
            logger.error("Error in regulatory risk calculation for finding %s: %s", 
                        getattr(finding, 'finding_id', 'unknown'), e)
            return None

    async def _identify_applicable_regulations(self, finding: Any, context: RiskCalculationContext) -> List[str]:
        """Identify which regulations apply to this finding."""
        regulations = []

        # Check explicit regulation metadata
        metadata = getattr(finding, 'metadata', {}) or {}
        explicit_regulations = metadata.get('applicable_regulations', [])
        regulations.extend(explicit_regulations)

        # Infer regulations based on detector type and category
        inferred_regulations = await self._infer_regulations_from_finding(finding, context)
        regulations.extend(inferred_regulations)

        # Remove duplicates and return
        return list(set(regulations))

    async def _infer_regulations_from_finding(self, finding: Any, context: RiskCalculationContext) -> List[str]:
        """Infer applicable regulations from finding characteristics."""
        regulations = []
        metadata = getattr(finding, 'metadata', {}) or {}
        
        # Check detector type for regulation mapping
        detector_type = metadata.get('detector_type', '').lower()
        detector_category = metadata.get('category', '').lower()
        
        # PII-related findings
        if any(keyword in detector_type for keyword in ['pii', 'privacy', 'personal']):
            regulations.extend(['gdpr', 'hipaa', 'ccpa'])
        
        # Access control findings
        if any(keyword in detector_type for keyword in ['access', 'auth', 'permission']):
            regulations.extend(['soc2', 'iso27001', 'fedramp'])
        
        # Data protection findings
        if any(keyword in detector_type for keyword in ['encryption', 'data', 'storage']):
            regulations.extend(['soc2', 'iso27001', 'hipaa', 'gdpr'])
        
        # Financial data findings
        if any(keyword in detector_type for keyword in ['payment', 'financial', 'card']):
            regulations.extend(['pci_dss', 'sox'])
        
        # Security control findings
        if any(keyword in detector_type for keyword in ['security', 'vulnerability', 'threat']):
            regulations.extend(['soc2', 'iso27001', 'fedramp'])
        
        return regulations

    async def _calculate_regulation_specific_risk(self, finding: Any, regulation: str, 
                                                 context: RiskCalculationContext) -> float:
        """Calculate risk for a specific regulation."""
        # Base risk from severity
        severity = getattr(finding, 'severity', None)
        if hasattr(severity, 'value'):
            severity = severity.value.lower()
        elif isinstance(severity, str):
            severity = severity.lower()
        else:
            severity = 'medium'
            
        severity_risk = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }.get(severity, 0.5)

        # Apply regulation-specific weight
        regulation_weight = self.regulatory_weights.get(regulation, 1.0)
        weighted_risk = severity_risk * regulation_weight

        # Check for regulation-specific metadata
        metadata = getattr(finding, 'metadata', {}) or {}
        regulation_metadata = metadata.get('regulatory_impact', {})
        
        if regulation in regulation_metadata:
            impact_level = regulation_metadata[regulation]
            impact_multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5,
                'critical': 2.0
            }
            weighted_risk *= impact_multipliers.get(impact_level, 1.0)

        return min(1.0, weighted_risk)

    async def _calculate_default_regulatory_risk(self, finding: Any) -> float:
        """Calculate default regulatory risk when no specific regulations apply."""
        # Use severity as base risk
        severity = getattr(finding, 'severity', None)
        if hasattr(severity, 'value'):
            severity = severity.value.lower()
        elif isinstance(severity, str):
            severity = severity.lower()
        else:
            severity = 'medium'
            
        return {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'critical': 0.9
        }.get(severity, 0.3)

    def get_dimension_name(self) -> str:
        """Get the name of this risk dimension."""
        return RiskDimension.REGULATORY.value
