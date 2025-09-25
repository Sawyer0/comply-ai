"""
SSI (Server-Side Includes) injection security patterns.

This module contains patterns for detecting SSI injection attacks
and server-side include manipulation attempts.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class SSIInjectionPatterns(PatternCollection):
    """Collection of SSI injection security patterns."""

    def __init__(self):
        """Initialize SSI injection patterns."""
        super().__init__(AttackType.SSI_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all SSI injection patterns."""

        # SSI directives
        self.add_pattern(
            name="ssi_include",
            pattern=r"<!--#include[^>]*-->",
            severity=ViolationSeverity.HIGH,
            description="SSI Injection - Include directive",
        )

        self.add_pattern(
            name="ssi_exec",
            pattern=r"<!--#exec[^>]*-->",
            severity=ViolationSeverity.CRITICAL,
            description="SSI Injection - Exec directive",
        )

        self.add_pattern(
            name="ssi_echo",
            pattern=r"<!--#echo[^>]*-->",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - Echo directive",
        )

        self.add_pattern(
            name="ssi_set",
            pattern=r"<!--#set[^>]*-->",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - Set directive",
        )

        self.add_pattern(
            name="ssi_if",
            pattern=r"<!--#if[^>]*-->",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - If directive",
        )
