"""
WAF rule engine package.

This package provides the core WAF rule engine implementation
for pattern matching and violation detection.
"""

from .rule_engine import WAFRule, WAFRuleEngine

__all__ = [
    "WAFRule",
    "WAFRuleEngine",
]
