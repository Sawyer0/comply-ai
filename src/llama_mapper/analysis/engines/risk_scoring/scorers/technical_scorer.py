"""
Technical Risk Scorer - CVSS-like methodology for technical risk assessment.
"""

import logging
from typing import Any, Dict, List, Optional

from ....domain.analysis_models import SecurityFinding
from ..types import RiskCalculationContext, RiskDimension

logger = logging.getLogger(__name__)


class TechnicalRiskScorer:
    """
    Specialized scorer for technical risk assessment.

    Uses CVSS-like methodology with enhanced factors for comprehensive
    technical risk evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cvss_weights = config.get('cvss_weights', {
            'confidentiality': 0.3,
            'integrity': 0.3,
            'availability': 0.3,
            'scope': 0.1
        })

    async def calculate_risk(self, context: RiskCalculationContext) -> float:
        """Calculate technical risk component."""
        if not context.findings:
            return 0.0

        total_technical_risk = 0.0
        valid_findings = 0

        for finding in context.findings:
            try:
                technical_score = await self._calculate_finding_technical_risk(finding, context)
                if technical_score is not None:
                    total_technical_risk += technical_score
                    valid_findings += 1
            except Exception as e:
                logger.warning("Error calculating technical risk for finding %s: %s", finding.finding_id, e)
                continue

        if valid_findings == 0:
            logger.warning("No valid findings for technical risk calculation")
            return 0.0

        return min(1.0, total_technical_risk / valid_findings)

    async def _calculate_finding_technical_risk(self, finding: SecurityFinding, context: RiskCalculationContext) -> Optional[float]:
        """Calculate technical risk for a single finding."""
        try:
            # Base score from severity
            severity_scores = {
                'low': 0.2,
                'medium': 0.5,
                'high': 0.8,
                'critical': 1.0
            }
            base_score = severity_scores.get(finding.severity.value.lower(), 0.5)

            if base_score == 0.0:
                return 0.0

            # Enhanced CVSS-like calculation with metadata factors
            metadata = finding.metadata

            # Confidentiality impact with enhanced granularity
            confidentiality = metadata.get('confidentiality_impact', 'medium')
            conf_multiplier = {
                'none': 0.0,
                'low': 0.22,
                'medium': 0.56,
                'high': 0.56,
                'complete': 0.66
            }.get(confidentiality, 0.56)

            # Integrity impact
            integrity = metadata.get('integrity_impact', 'medium')
            int_multiplier = {
                'none': 0.0,
                'low': 0.22,
                'medium': 0.56,
                'high': 0.56,
                'complete': 0.66
            }.get(integrity, 0.56)

            # Availability impact
            availability = metadata.get('availability_impact', 'medium')
            avail_multiplier = {
                'none': 0.0,
                'low': 0.22,
                'medium': 0.56,
                'high': 0.56,
                'complete': 0.66
            }.get(availability, 0.56)

            # Scope factor (changed scope vs unchanged)
            scope = metadata.get('scope', 'unchanged')
            scope_multiplier = {'unchanged': 1.0, 'changed': 1.08}.get(scope, 1.0)

            # Exploitability factors
            exploitability = await self._calculate_exploitability_factors(finding, metadata)

            # Calculate technical score using CVSS v3.1-like formula
            technical_score = base_score * (
                self.cvss_weights['confidentiality'] * conf_multiplier +
                self.cvss_weights['integrity'] * int_multiplier +
                self.cvss_weights['availability'] * avail_multiplier +
                self.cvss_weights['scope'] * scope_multiplier
            ) * exploitability

            # Apply confidence weighting with safeguards
            confidence = max(0.1, finding.confidence)  # Minimum confidence floor
            technical_score *= confidence

            # Apply business context adjustments if available
            technical_score = await self._apply_business_context_adjustments(
                technical_score, finding, context
            )

            return max(0.0, min(1.0, technical_score))

        except Exception as e:
            logger.error("Error in technical risk calculation for finding %s: %s", finding.finding_id, e)
            return None

    async def _calculate_exploitability_factors(self, finding: SecurityFinding, metadata: Dict[str, Any]) -> float:
        """Calculate exploitability factors that affect technical risk."""
        exploitability = 1.0

        # Attack vector complexity
        attack_vector = metadata.get('attack_vector', 'network')
        av_multipliers = {
            'physical': 0.2,
            'local': 0.55,
            'adjacent': 0.62,
            'network': 0.85
        }
        exploitability *= av_multipliers.get(attack_vector, 0.85)

        # Attack complexity
        attack_complexity = metadata.get('attack_complexity', 'low')
        ac_multipliers = {
            'low': 0.77,
            'high': 0.44
        }
        exploitability *= ac_multipliers.get(attack_complexity, 0.77)

        # Privileges required
        privileges = metadata.get('privileges_required', 'none')
        pr_multipliers = {
            'none': 0.85,
            'low': 0.68,
            'high': 0.50
        }
        exploitability *= pr_multipliers.get(privileges, 0.85)

        # User interaction
        user_interaction = metadata.get('user_interaction', 'none')
        ui_multipliers = {
            'none': 0.85,
            'required': 0.62
        }
        exploitability *= ui_multipliers.get(user_interaction, 0.85)

        return exploitability

    async def _apply_business_context_adjustments(self, technical_score: float,
                                                finding: SecurityFinding,
                                                context: RiskCalculationContext) -> float:
        """Apply business context adjustments to technical risk."""
        # Consider affected business processes
        affected_processes = finding.metadata.get('affected_processes', [])
        if affected_processes:
            # Higher risk if critical business processes are affected
            critical_processes = context.business_context.get('critical_processes', [])
            if any(proc in critical_processes for proc in affected_processes):
                technical_score *= 1.2

        # Consider data classification
        data_classification = finding.metadata.get('data_classification', 'internal')
        classification_multipliers = {
            'public': 0.8,
            'internal': 1.0,
            'confidential': 1.3,
            'restricted': 1.5
        }
        technical_score *= classification_multipliers.get(data_classification, 1.0)

        return technical_score

    def get_dimension_name(self) -> str:
        """Get the name of this risk dimension."""
        return RiskDimension.TECHNICAL.value
