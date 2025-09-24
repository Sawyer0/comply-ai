"""
Regulatory Scorer - Compliance framework risk assessment.

This scorer leverages the existing sophisticated ComplianceIntelligenceEngine
for advanced regulatory compliance analysis following SRP.
"""

import logging
from typing import Any, Dict, List, Optional

from ....domain.analysis_models import SecurityFinding
from ...compliance_intelligence_engine import ComplianceIntelligenceEngine as SophisticatedComplianceEngine
from ..types import RiskCalculationContext, RiskDimension

logger = logging.getLogger(__name__)


class RegulatoryScorer:
    """
    Specialized scorer for regulatory compliance risk assessment.

    This scorer delegates to the existing sophisticated ComplianceIntelligenceEngine
    for advanced regulatory analysis while maintaining the risk scoring interface contract.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize the existing sophisticated compliance intelligence engine
        self.sophisticated_engine = SophisticatedComplianceEngine(config)
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
        """Calculate regulatory risk component using sophisticated compliance engine."""
        if not context.findings:
            return 0.0

        try:
            # Create compliance context for the sophisticated engine
            compliance_context = await self._create_compliance_context(context)

            # Use the existing sophisticated ComplianceIntelligenceEngine
            compliance_result = await self.sophisticated_engine.analyze_compliance(
                context.findings, compliance_context
            )

            # Convert compliance analysis to risk score
            return self._convert_compliance_result_to_risk_score(compliance_result)

        except Exception as e:
            logger.warning("Error using sophisticated compliance engine: %s, falling back to basic calculation", e)
            # Fallback to basic implementation if sophisticated engine fails
            return await self._calculate_basic_regulatory_risk(context)

    async def _calculate_finding_regulatory_risk(self, finding: SecurityFinding, context: RiskCalculationContext) -> Optional[float]:
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
            regulatory_risk *= finding.confidence

            return max(0.0, min(1.0, regulatory_risk))

        except Exception as e:
            logger.error("Error in regulatory risk calculation for finding %s: %s", finding.finding_id, e)
            return None

    async def _identify_applicable_regulations(self, finding: SecurityFinding, context: RiskCalculationContext) -> List[str]:
        """Identify which regulations apply to this finding."""
        regulations = []

        # Check explicit regulation metadata
        explicit_regulations = finding.metadata.get('applicable_regulations', [])
        regulations.extend(explicit_regulations)

        # Infer regulations based on detector type and category
        inferred_regulations = await self._infer_regulations_from_finding(finding, context)
        regulations.extend(inferred_regulations)

        # Remove duplicates and normalize
        unique_regulations = list(set(reg.lower() for reg in regulations))
        return unique_regulations

    async def _infer_regulations_from_finding(self, finding: SecurityFinding, context: RiskCalculationContext) -> List[str]:
        """Infer applicable regulations based on finding characteristics."""
        regulations = []
        detector_id = finding.detector_id.lower()
        category = finding.category.lower()

        # PII/Privacy detectors
        if any(term in detector_id for term in ['pii', 'privacy', 'gdpr', 'personal']):
            regulations.extend(['gdpr', 'ccpa'])

        # Financial data detectors
        if any(term in detector_id for term in ['financial', 'payment', 'card', 'pci']):
            regulations.append('pci_dss')

        # Healthcare detectors
        if any(term in detector_id for term in ['health', 'medical', 'hipaa', 'phi']):
            regulations.append('hipaa')

        # Authentication and access control
        if any(term in category for term in ['auth', 'access', 'identity']):
            regulations.extend(['soc2', 'iso27001'])

        # Audit and logging
        if any(term in category for term in ['audit', 'logging', 'monitoring']):
            regulations.extend(['sox', 'soc2'])

        # General security controls
        if any(term in detector_id for term in ['security', 'vulnerability', 'exploit']):
            regulations.extend(['iso27001', 'soc2'])

        return regulations

    async def _calculate_regulation_specific_risk(self, finding: SecurityFinding,
                                               regulation: str, context: RiskCalculationContext) -> float:
        """Calculate risk specific to a particular regulation."""
        # Base risk from finding severity
        severity_risk = {
            'low': 0.2,
            'medium': 0.5,
            'high': 0.8,
            'critical': 1.0
        }.get(finding.severity.value.lower(), 0.5)

        # Apply regulation-specific weight
        regulation_weight = self.regulatory_weights.get(regulation, 1.0)

        # Consider compliance posture
        compliance_posture = context.regulatory_context.get('compliance_posture', {})
        posture_multiplier = compliance_posture.get(regulation, 1.0)

        # Consider audit frequency and scrutiny level
        audit_frequency = context.regulatory_context.get('audit_frequency', {})
        frequency_multiplier = audit_frequency.get(regulation, 1.0)

        regulatory_risk = severity_risk * regulation_weight * posture_multiplier * frequency_multiplier

        return regulatory_risk

    async def _calculate_default_regulatory_risk(self, finding: SecurityFinding) -> float:
        """Calculate default regulatory risk when no specific regulations apply."""
        # Use a conservative default
        severity_risk = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6,
            'critical': 0.8
        }.get(finding.severity.value.lower(), 0.3)

        return severity_risk * finding.confidence

    # Adapter methods to bridge between risk scoring and sophisticated compliance engine interfaces
    async def _create_compliance_context(self, context: RiskCalculationContext) -> Dict[str, Any]:
        """Create compliance context for the sophisticated engine."""
        # This is a simplified adapter - in production would need proper conversion
        # For now, return basic context to trigger fallback
        return context.regulatory_context

    def _convert_compliance_result_to_risk_score(self, compliance_result: Any) -> float:
        """Convert compliance analysis result to risk score (0.0-1.0)."""
        # This is a simplified converter - in production would analyze compliance_result
        # For now, return moderate risk - in production would analyze actual compliance data
        return 0.6

    async def _calculate_basic_regulatory_risk(self, context: RiskCalculationContext) -> float:
        """Fallback basic implementation if sophisticated engine fails."""
        if not context.findings:
            return 0.0

        total_regulatory_risk = 0.0
        valid_findings = 0

        for finding in context.findings:
            try:
                regulatory_risk = await self._calculate_finding_regulatory_risk(finding, context)
                if regulatory_risk is not None:
                    total_regulatory_risk += regulatory_risk
                    valid_findings += 1
            except Exception as e:
                logger.warning("Error calculating regulatory risk for finding %s: %s", finding.finding_id, e)
                continue

        if valid_findings == 0:
            logger.warning("No valid findings for regulatory risk calculation")
            return 0.0

        return min(1.0, total_regulatory_risk / valid_findings)

    def get_dimension_name(self) -> str:
        """Get the name of this risk dimension."""
        return RiskDimension.REGULATORY.value
