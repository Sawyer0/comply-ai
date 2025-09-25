"""
WAF Factory for creating WAF components.

This module provides factory methods for creating WAF components
with proper configuration and dependencies.
"""

from typing import List, Optional
from .rule import WAFRule
from .engine.rule_engine import WAFRuleEngine
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


class WAFFactory:
    """Factory for creating WAF components."""
    
    @staticmethod
    def create_rule_engine() -> WAFRuleEngine:
        """Create a WAF rule engine."""
        return WAFRuleEngine()
    
    @staticmethod
    def create_middleware(rules: Optional[List[WAFRule]] = None) -> WAFMiddleware:
        """Create WAF middleware with optional rules."""
        return WAFMiddleware(rules=rules)
    
    @staticmethod
    def create_default_rules() -> List[WAFRule]:
        """Create default WAF rules."""
        rules = []
        
        # Add all pattern types
        rules.extend(SQLInjectionPatterns.get_patterns())
        rules.extend(XSSPatterns.get_patterns())
        rules.extend(PathTraversalPatterns.get_patterns())
        rules.extend(CommandInjectionPatterns.get_patterns())
        rules.extend(LDAPInjectionPatterns.get_patterns())
        rules.extend(NoSQLInjectionPatterns.get_patterns())
        rules.extend(XMLInjectionPatterns.get_patterns())
        rules.extend(SSIInjectionPatterns.get_patterns())
        rules.extend(MaliciousPayloadPatterns.get_patterns())
        
        return rules
    
    @staticmethod
    def create_configured_middleware() -> WAFMiddleware:
        """Create WAF middleware with default configuration."""
        rules = WAFFactory.create_default_rules()
        return WAFFactory.create_middleware(rules)
