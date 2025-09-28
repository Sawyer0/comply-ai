"""Domain models for policy management.

These dataclasses hold immutable-like configuration / decision data that
other helpers use. Keeping them in a dedicated module avoids circular
imports between loader, manager, and violations helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "PolicyDecision",
    "PolicyConfig",
    "EnforcementResult",
]


@dataclass(slots=True)
class PolicyDecision:
    """Outcome of policy evaluation used by the router.

    Mirrors the previous in-file definition but lives in its own
    module so that helpers and tests can import without touching the
    heavy `policy_manager` implementation.
    """

    selected_detectors: List[str]
    coverage_method: str
    coverage_requirements: Dict[str, Any]
    routing_reason: str
    policy_violations: List[str] = field(default_factory=list)

    # Convenience helpers kept for backwards-compatibility with callers
    def has_violations(self) -> bool:
        """Check if there are any policy violations."""
        return bool(self.policy_violations)

    def add_violation(self, violation: str) -> None:
        """Add a policy violation to the decision."""
        self.policy_violations.append(violation)

    def get_violation_count(self) -> int:
        """Get the total number of policy violations."""
        return len(self.policy_violations)


@dataclass(slots=True)
class PolicyConfig:
    """Validated policy configuration used during decision making.

    This is a *placeholder* until the larger refactor moves validation
    inside a dedicated `validator` module.  For now we only store the
    raw mapping so that downstream code can migrate seamlessly.
    """

    raw: Dict[str, Any]


@dataclass(slots=True)
class EnforcementResult:
    """Result returned by post-run violation evaluation.

    Another placeholder to be fleshed out when `violations.py` is
    introduced.  It allows us to type-hint returns without repeatedly
    using Dict / List aliases.
    """

    violations: List[str]
    compliant: bool
