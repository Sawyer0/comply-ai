"""
WAF (Web Application Firewall) package.

This package provides comprehensive security filtering for web applications
with configurable rules, middleware, and monitoring capabilities.
"""

from .interfaces import (
    AttackType, ViolationSeverity, WAFViolation,
    IWAFRule, IWAFRuleEngine, IWAFMiddleware, IWAFMetricsCollector
)
from .engine.rule_engine import WAFRuleEngine, WAFRule
from .middleware.waf_middleware import WAFMiddleware
from .factory import WAFFactory

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
