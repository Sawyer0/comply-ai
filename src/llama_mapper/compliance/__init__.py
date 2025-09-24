"""
Compliance infrastructure for enterprise-grade AI models.

This module provides constitutional AI, grounding validation, template fallbacks,
behavioral preference tuning, temporal awareness, and metrics collection
for senior-level compliance analysis.
"""

from .grounding_validator import ComplianceOutputValidator, GroundingEnforcer
from .template_fallbacks import ComplianceTemplateFallbacks
from .constitution_rails import ComplianceConstitution, ConstitutionalEnforcer
from .preference_tuning import PreferenceDataGenerator
from .temporal_awareness import RegulatoryTimelineTracker, TemporalAwarenessEvaluator
from .metrics_dashboard import ComplianceMetricsCollector, MetricsDashboard
from .tool_hooks import (
    simulate_retrieval_with_filters,
    simulate_citation_checking,
    simulate_policy_generation,
)

__all__ = [
    # Core validation and grounding
    "ComplianceOutputValidator",
    "GroundingEnforcer",
    
    # Template fallbacks
    "ComplianceTemplateFallbacks",
    
    # Constitutional AI
    "ComplianceConstitution",
    "ConstitutionalEnforcer",
    
    # Behavioral preference tuning
    "PreferenceDataGenerator",
    
    # Temporal awareness
    "RegulatoryTimelineTracker",
    "TemporalAwarenessEvaluator",
    
    # Metrics and monitoring
    "ComplianceMetricsCollector",
    "MetricsDashboard",
    
    # Tool simulation hooks
    "simulate_retrieval_with_filters",
    "simulate_citation_checking", 
    "simulate_policy_generation",
]
