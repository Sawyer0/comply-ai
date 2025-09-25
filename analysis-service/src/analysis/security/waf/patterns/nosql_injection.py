"""
NoSQL injection security patterns.

This module contains patterns for detecting NoSQL injection attacks
and document database manipulation attempts.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class NoSQLInjectionPatterns(PatternCollection):
    """Collection of NoSQL injection security patterns."""

    def __init__(self):
        """Initialize NoSQL injection patterns."""
        super().__init__(AttackType.NO_SQL_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all NoSQL injection patterns."""

        # MongoDB injection
        self.add_pattern(
            name="nosql_mongodb_operators",
            pattern=r"\$[a-zA-Z]+",
            severity=ViolationSeverity.HIGH,
            description="NoSQL Injection - MongoDB operators",
        )

        self.add_pattern(
            name="nosql_mongodb_where",
            pattern=r"\$where",
            severity=ViolationSeverity.CRITICAL,
            description="NoSQL Injection - MongoDB $where operator",
        )

        self.add_pattern(
            name="nosql_mongodb_regex",
            pattern=r"\$regex",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - MongoDB $regex operator",
        )

        # JavaScript injection in NoSQL
        self.add_pattern(
            name="nosql_javascript_injection",
            pattern=r"function\s*\(",
            severity=ViolationSeverity.HIGH,
            description="NoSQL Injection - JavaScript function injection",
        )

        # Boolean injection
        self.add_pattern(
            name="nosql_boolean_injection",
            pattern=r"true|false",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Boolean injection",
        )

        # Array injection
        self.add_pattern(
            name="nosql_array_injection",
            pattern=r"\[[^\]]*\]",
            severity=ViolationSeverity.MEDIUM,
            description="NoSQL Injection - Array injection",
        )
