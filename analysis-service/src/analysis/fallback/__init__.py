"""
Fallback mechanisms for the Analysis Service.

This module provides:
- Rule-based fallbacks for ML models
- Template-based responses
- Fallback coordination
- Graceful degradation
"""

from .rule_based import (
    RuleBasedRiskAssessment,
    RuleBasedPatternAnalysis,
    RuleBasedComplianceMapping,
    RuleResult,
)

__all__ = [
    "RuleBasedRiskAssessment",
    "RuleBasedPatternAnalysis",
    "RuleBasedComplianceMapping",
    "RuleResult",
]
