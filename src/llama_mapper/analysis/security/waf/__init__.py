"""
WAF (Web Application Firewall) package.

This package provides comprehensive security filtering for web applications
with configurable rules, middleware, and monitoring capabilities.
"""

from .engine.rule_engine import WAFRule, WAFRuleEngine
from .factory import WAFFactory
from .interfaces import (
    AttackType,
    IWAFMetricsCollector,
    IWAFMiddleware,
    IWAFRule,
    IWAFRuleEngine,
    ViolationSeverity,
    WAFViolation,
)
from .middleware.waf_middleware import WAFMiddleware

__all__ = [
    # Interfaces
    "AttackType",
    "ViolationSeverity",
    "WAFViolation",
    "IWAFRule",
    "IWAFRuleEngine",
    "IWAFMiddleware",
    "IWAFMetricsCollector",
    # Implementations
    "WAFRuleEngine",
    "WAFRule",
    "WAFMiddleware",
    # Factory
    "WAFFactory",
]
