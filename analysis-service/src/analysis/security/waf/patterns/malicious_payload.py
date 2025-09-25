"""
Malicious payload security patterns.

This module contains patterns for detecting various malicious payloads
and suspicious content that could indicate attacks.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class MaliciousPayloadPatterns(PatternCollection):
    """Collection of malicious payload security patterns."""

    def __init__(self):
        """Initialize malicious payload patterns."""
        super().__init__(AttackType.MALICIOUS_PAYLOAD)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all malicious payload patterns."""

        # Base64 encoded content
        self.add_pattern(
            name="payload_base64",
            pattern=r"[A-Za-z0-9+/]{20,}={0,2}",
            severity=ViolationSeverity.LOW,
            description="Malicious Payload - Base64 encoded content",
        )

        # Hex encoded content
        self.add_pattern(
            name="payload_hex",
            pattern=r"\\x[0-9a-fA-F]{2}",
            severity=ViolationSeverity.MEDIUM,
            description="Malicious Payload - Hex encoded content",
        )

        # URL encoded content
        self.add_pattern(
            name="payload_url_encoded",
            pattern=r"%[0-9a-fA-F]{2}",
            severity=ViolationSeverity.LOW,
            description="Malicious Payload - URL encoded content",
        )

        # Suspicious file extensions
        self.add_pattern(
            name="payload_executable_files",
            pattern=r"\.(exe|bat|cmd|com|scr|pif|vbs|js|jar|war|ear)\b",
            severity=ViolationSeverity.HIGH,
            description="Malicious Payload - Executable file extensions",
        )

        # Suspicious protocols
        self.add_pattern(
            name="payload_suspicious_protocols",
            pattern=r"(file|ftp|gopher|jar|netdoc|nntp|sftp|tftp|ldap)://",
            severity=ViolationSeverity.MEDIUM,
            description="Malicious Payload - Suspicious protocols",
        )

        # Data URIs
        self.add_pattern(
            name="payload_data_uri",
            pattern=r"data:[^;]+;base64,",
            severity=ViolationSeverity.MEDIUM,
            description="Malicious Payload - Data URI with base64",
        )
