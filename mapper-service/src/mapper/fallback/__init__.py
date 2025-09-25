"""
Fallback systems for the Mapper Service.

Single responsibility: Fallback mechanisms when primary mapping fails.
"""

from .rule_based_fallback import RuleBasedFallback
from .template_fallback import TemplateFallback
from .fallback_coordinator import FallbackCoordinator

__all__ = [
    "RuleBasedFallback",
    "TemplateFallback",
    "FallbackCoordinator",
]
