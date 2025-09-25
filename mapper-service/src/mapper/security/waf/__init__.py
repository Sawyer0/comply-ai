"""
WAF (Web Application Firewall) module for mapper service.

This package provides comprehensive WAF functionality including
attack detection, rule engines, and security middleware.
"""

from .interfaces import (
    AttackType,
    ViolationSeverity,
    WAFViolation,
    IWAFRule,
    IWAFRuleEngine,
    IWAFMiddleware,
    IWAFMetricsCollector,
)
from .engine.rule_engine import WAFRule, WAFRuleEngine
from .middleware.waf_middleware import WAFMiddleware
from .patterns import (
    SQLInjectionPatterns,
    XSSPatterns,
    PathTraversalPatterns,
    CommandInjectionPatterns,
    LDAPInjectionPatterns,
    NoSQLInjectionPatterns,
    XMLInjectionPatterns,
    SSIInjectionPatterns,
    MaliciousPayloadPatterns,
)
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
    "WAFRule",
    "WAFRuleEngine",
    "WAFMiddleware",
    # Attack Patterns
    "SQLInjectionPatterns",
    "XSSPatterns",
    "PathTraversalPatterns",
    "CommandInjectionPatterns",
    "LDAPInjectionPatterns",
    "NoSQLInjectionPatterns",
    "XMLInjectionPatterns",
    "SSIInjectionPatterns",
    "MaliciousPayloadPatterns",
    # Factory
    "WAFFactory",
]
