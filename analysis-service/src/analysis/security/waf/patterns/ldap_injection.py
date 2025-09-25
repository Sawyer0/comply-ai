"""
LDAP injection security patterns.

This module contains patterns for detecting LDAP injection attacks
and directory service manipulation attempts.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class LDAPInjectionPatterns(PatternCollection):
    """Collection of LDAP injection security patterns."""

    def __init__(self):
        """Initialize LDAP injection patterns."""
        super().__init__(AttackType.LDAP_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all LDAP injection patterns."""

        # LDAP filter injection
        self.add_pattern(
            name="ldap_filter_injection",
            pattern=r"[()&|!]",
            severity=ViolationSeverity.HIGH,
            description="LDAP Injection - Filter operators",
        )

        self.add_pattern(
            name="ldap_wildcard_injection",
            pattern=r"\*",
            severity=ViolationSeverity.MEDIUM,
            description="LDAP Injection - Wildcard injection",
        )

        # LDAP attribute injection
        self.add_pattern(
            name="ldap_attribute_injection",
            pattern=r"\([^)]*\)",
            severity=ViolationSeverity.HIGH,
            description="LDAP Injection - Attribute filter injection",
        )

        # LDAP search injection
        self.add_pattern(
            name="ldap_search_injection",
            pattern=r"\(&[^)]*\)",
            severity=ViolationSeverity.HIGH,
            description="LDAP Injection - AND filter injection",
        )

        self.add_pattern(
            name="ldap_or_injection",
            pattern=r"\(|[^)]*\)",
            severity=ViolationSeverity.HIGH,
            description="LDAP Injection - OR filter injection",
        )
