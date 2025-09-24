"""
Risk Scoring Framework - Modular Implementation

This package provides a production-ready risk scoring framework with specialized
components following the Single Responsibility Principle (SRP).

Architecture:
- scorers/: Individual risk dimension scorers (Technical, Business, Regulatory, Temporal)
- calculators/: Risk calculation and composition algorithms
- engine.py: Main orchestration engine
- exceptions.py: Custom exception classes
- types.py: Type definitions and protocols
"""

from .engine import RiskScoringEngine
from .exceptions import RiskCalculationError
from .types import RiskDimension, IRiskScorer

__all__ = [
    'RiskScoringEngine',
    'RiskCalculationError', 
    'RiskDimension',
    'IRiskScorer'
]
