"""
Risk Scoring Engine for intelligent risk assessment with business context.

This module now serves as a compatibility layer that imports the modular
Risk Scoring Framework following the Single Responsibility Principle.

DEPRECATED: This monolithic file has been refactored into a modular structure.
Please use: from .risk_scoring import RiskScoringEngine
"""

import logging

from .risk_scoring import RiskScoringEngine as ModularRiskScoringEngine
from .risk_scoring import RiskCalculationError

logger = logging.getLogger(__name__)

# Export the modular implementation for backward compatibility
RiskScoringEngine = ModularRiskScoringEngine

# Re-export other types for compatibility
__all__ = [
    'RiskScoringEngine',
    'RiskCalculationError'
]