"""
Llama Mapper: Fine-tuned model for mapping detector outputs to canonical taxonomy.

This package provides a privacy-first, audit-ready service that normalizes
raw detector outputs into a canonical taxonomy for compliance evidence generation.
"""

__version__ = "0.1.0"
__author__ = "AI Safety Team"

from .config import ConfigManager
from .logging import setup_logging

# Make compliance infrastructure available at package level
try:
    from .compliance import (
        ComplianceOutputValidator,
        GroundingEnforcer,
        ComplianceTemplateFallbacks,
        ComplianceConstitution,
        ConstitutionalEnforcer,
        PreferenceDataGenerator,
        RegulatoryTimelineTracker,
        TemporalAwarenessEvaluator,
        ComplianceMetricsCollector,
        MetricsDashboard,
    )
    
    __all__ = [
        "ConfigManager", 
        "setup_logging",
        # Compliance infrastructure
        "ComplianceOutputValidator",
        "GroundingEnforcer", 
        "ComplianceTemplateFallbacks",
        "ComplianceConstitution",
        "ConstitutionalEnforcer",
        "PreferenceDataGenerator",
        "RegulatoryTimelineTracker",
        "TemporalAwarenessEvaluator",
        "ComplianceMetricsCollector",
        "MetricsDashboard",
    ]
except ImportError:
    # Compliance modules not available (development/optional)
    __all__ = ["ConfigManager", "setup_logging"]
