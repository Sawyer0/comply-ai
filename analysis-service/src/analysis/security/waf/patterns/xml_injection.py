"""
XML injection security patterns.

This module contains patterns for detecting XML injection attacks
and XML-based attacks like XXE.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class XMLInjectionPatterns(PatternCollection):
    """Collection of XML injection security patterns."""

    def __init__(self):
        """Initialize XML injection patterns."""
        super().__init__(AttackType.XML_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all XML injection patterns."""

        # XXE (XML External Entity) patterns
        self.add_pattern(
            name="xml_xxe_doctype",
            pattern=r"<!DOCTYPE[^>]*>",
            severity=ViolationSeverity.HIGH,
            description="XML Injection - DOCTYPE declaration",
        )

        self.add_pattern(
            name="xml_xxe_entity",
            pattern=r"<!ENTITY[^>]*>",
            severity=ViolationSeverity.HIGH,
            description="XML Injection - Entity declaration",
        )

        self.add_pattern(
            name="xml_xxe_system",
            pattern=r"SYSTEM\s+['\"][^'\"]*['\"]",
            severity=ViolationSeverity.CRITICAL,
            description="XML Injection - SYSTEM entity",
        )

        self.add_pattern(
            name="xml_xxe_public",
            pattern=r"PUBLIC\s+['\"][^'\"]*['\"]",
            severity=ViolationSeverity.HIGH,
            description="XML Injection - PUBLIC entity",
        )

        # XML processing instructions
        self.add_pattern(
            name="xml_pi",
            pattern=r"<\?[^>]*\?>",
            severity=ViolationSeverity.MEDIUM,
            description="XML Injection - Processing instruction",
        )

        # CDATA sections
        self.add_pattern(
            name="xml_cdata",
            pattern=r"<!\[CDATA\[.*?\]\]>",
            severity=ViolationSeverity.LOW,
            description="XML Injection - CDATA section",
        )
