"""
Analysis Engines Module

This module contains specialized analysis engines that were refactored from
the monolithic template provider to provide modular, maintainable components.

Each engine has a single responsibility:
- PatternRecognitionEngine: Detects patterns in security data
- RiskScoringEngine: Calculates intelligent risk scores
- ComplianceIntelligence: Maps findings to regulatory frameworks
- TemplateOrchestrator: Coordinates analysis engines
"""

from .interfaces import (
    IAnalysisEngine,
    IComplianceIntelligence,
    IPatternRecognitionEngine,
    IRiskScoringEngine,
    ITemplateOrchestrator,
)
from .pattern_recognition import PatternRecognitionEngine
from .risk_scoring import RiskScoringEngine
from .compliance_intelligence import ComplianceIntelligence
from .template_orchestrator import TemplateOrchestrator
from .threshold_optimization import ThresholdOptimizationEngine

__all__ = [
    # Interfaces
    "IAnalysisEngine",
    "IPatternRecognitionEngine",
    "IRiskScoringEngine",
    "IComplianceIntelligence",
    "ITemplateOrchestrator",
    # Implementations
    "PatternRecognitionEngine",
    "RiskScoringEngine",
    "ComplianceIntelligence",
    "TemplateOrchestrator",
    "ThresholdOptimizationEngine",
]
