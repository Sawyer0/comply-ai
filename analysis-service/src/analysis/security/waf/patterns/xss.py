"""
XSS (Cross-Site Scripting) security patterns.

This module contains comprehensive patterns for detecting
XSS attacks across different vectors and techniques.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class XSSPatterns(PatternCollection):
    """Collection of XSS security patterns."""

    def __init__(self):
        """Initialize XSS patterns."""
        super().__init__(AttackType.XSS)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all XSS patterns."""

        # Script tag patterns
        self.add_pattern(
            name="xss_script_tag",
            pattern=r"<script[^>]*>.*?</script>",
            severity=ViolationSeverity.HIGH,
            description="XSS - Script tags",
        )

        self.add_pattern(
            name="xss_script_src",
            pattern=r"<script[^>]*src\s*=\s*['\"][^'\"]*['\"]",
            severity=ViolationSeverity.HIGH,
            description="XSS - Script with external source",
        )

        # JavaScript protocol
        self.add_pattern(
            name="xss_javascript_protocol",
            pattern=r"javascript:",
            severity=ViolationSeverity.HIGH,
            description="XSS - JavaScript protocol",
        )

        self.add_pattern(
            name="xss_javascript_encoded",
            pattern=r"javascript\s*:",
            severity=ViolationSeverity.HIGH,
            description="XSS - JavaScript protocol (encoded)",
        )

        # Event handlers
        self.add_pattern(
            name="xss_onload_event",
            pattern=r"onload\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - onload event handler",
        )

        self.add_pattern(
            name="xss_onclick_event",
            pattern=r"onclick\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - onclick event handler",
        )

        self.add_pattern(
            name="xss_onerror_event",
            pattern=r"onerror\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - onerror event handler",
        )

        # Common XSS vectors
        self.add_pattern(
            name="xss_alert_function",
            pattern=r"alert\s*\(",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - alert() function",
        )

        self.add_pattern(
            name="xss_document_cookie",
            pattern=r"document\.cookie",
            severity=ViolationSeverity.HIGH,
            description="XSS - document.cookie access",
        )

        self.add_pattern(
            name="xss_document_write",
            pattern=r"document\.write",
            severity=ViolationSeverity.HIGH,
            description="XSS - document.write",
        )

        # Encoded XSS attempts
        self.add_pattern(
            name="xss_hex_encoded",
            pattern=r"\\x3cscript",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Hex encoded script tag",
        )

        self.add_pattern(
            name="xss_url_encoded",
            pattern=r"%3cscript",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - URL encoded script tag",
        )

        # Advanced evasion techniques
        self.add_pattern(
            name="xss_case_insensitive",
            pattern=r"<ScRiPt",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Case insensitive script tag",
        )

        self.add_pattern(
            name="xss_whitespace_obfuscation",
            pattern=r"<script\s+",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Whitespace obfuscation",
        )

        # DOM-based XSS
        self.add_pattern(
            name="xss_dom_location",
            pattern=r"location\.",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - DOM location manipulation",
        )

        self.add_pattern(
            name="xss_dom_innerhtml",
            pattern=r"innerHTML\s*=",
            severity=ViolationSeverity.HIGH,
            description="XSS - innerHTML manipulation",
        )

        # Filter bypass attempts
        self.add_pattern(
            name="xss_script_bypass",
            pattern=r"<scr<script>ipt>",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Script tag bypass attempt",
        )

        self.add_pattern(
            name="xss_double_encoding",
            pattern=r"%253cscript",
            severity=ViolationSeverity.MEDIUM,
            description="XSS - Double URL encoding",
        )
