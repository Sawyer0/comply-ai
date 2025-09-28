"""Policy management functionality for orchestration service.

This module provides policy management capabilities following SRP:
- PolicyManager: Handle policy enforcement and OPA integration
- ConflictResolver: Resolve conflicts between detector results
"""

from .policy_manager import PolicyManager
from .models import PolicyDecision, PolicyConfig, EnforcementResult
from .policy_loader import PolicyLoader, PolicyLoadError
from .violations import evaluate_policy_violations
from .evaluator import evaluate_policy_decision, evaluate_policy_fallback
from .repository import PolicyRepository
from .initialization import PolicyInitializer
from .file_loader import PolicyLoadError as FileLoaderError
from .parser import validate_policy_syntax, validate_tenant_policy_data
from .metadata import parse_policy_structure, extract_policy_metadata

__all__ = [
    "PolicyManager",
    "PolicyDecision", 
    "PolicyConfig",
    "EnforcementResult",
    "PolicyLoader",
    "PolicyLoadError",
    "FileLoaderError",
    "PolicyRepository",
    "PolicyInitializer",
    "evaluate_policy_violations",
    "evaluate_policy_decision",
    "evaluate_policy_fallback",
    "validate_policy_syntax",
    "validate_tenant_policy_data",
    "parse_policy_structure",
    "extract_policy_metadata",
]
