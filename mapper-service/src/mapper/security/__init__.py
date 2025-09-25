"""
Security module for mapper service.

This package provides comprehensive security features including
authentication, authorization, WAF, and security management.
"""

from .security_manager import SecurityManager
from .waf import (
    WAFRule,
    WAFRuleEngine,
    WAFMiddleware,
    AttackDetector,
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
from .authentication import AuthenticationManager
from .authorization import AuthorizationManager
from .api_key_manager import APIKeyManager

__all__ = [
    # Core Security
    "SecurityManager",
    "AuthenticationManager", 
    "AuthorizationManager",
    "APIKeyManager",
    # WAF Components
    "WAFRule",
    "WAFRuleEngine", 
    "WAFMiddleware",
    "AttackDetector",
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
]