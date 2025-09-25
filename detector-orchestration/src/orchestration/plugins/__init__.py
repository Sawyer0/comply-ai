"""Plugin system for orchestration service following SRP.

This module provides ONLY plugin system interfaces and base classes.
Single Responsibility: Define plugin system contracts and base implementations.
"""

from .plugin_manager import PluginManager
from .detector_plugin import DetectorPlugin, DetectorPluginInterface
from .policy_plugin import (
    PolicyPlugin,
    PolicyPluginInterface,
    PolicyEvaluationDecision,
    PolicyResult,
)

__all__ = [
    "PluginManager",
    "DetectorPlugin",
    "DetectorPluginInterface",
    "PolicyPlugin",
    "PolicyPluginInterface",
    "PolicyEvaluationDecision",
    "PolicyResult",
]
