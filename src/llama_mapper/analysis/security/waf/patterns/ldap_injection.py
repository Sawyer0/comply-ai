"""LDAP injection security patterns."""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class LDAPInjectionPatterns(PatternCollection):
    """Collection of LDAP injection security patterns."""

    def __init__(self):
        super().__init__(AttackType.LDAP_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        # LDAP metacharacters
        self.add_pattern(
            name="ldap_metacharacters",
            pattern=r"[()=*!&|]",
            severity=ViolationSeverity.MEDIUM,
            description="LDAP Injection - Metacharacters",
        )

        # LDAP attributes
        self.add_pattern(
            name="ldap_attributes",
            pattern=r"\b(uid|cn|sn|givenName|mail|objectClass)\b",
            severity=ViolationSeverity.MEDIUM,
            description="LDAP Injection - Attributes",
        )

        # Boolean operators
        self.add_pattern(
            name="ldap_boolean_operators",
            pattern=r"\b(AND|OR|NOT)\b",
            severity=ViolationSeverity.MEDIUM,
            description="LDAP Injection - Boolean operators",
        )
