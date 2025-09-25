"""
Path traversal security patterns.

This module contains patterns for detecting path traversal attacks
and directory traversal attempts.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class PathTraversalPatterns(PatternCollection):
    """Collection of path traversal security patterns."""

    def __init__(self):
        """Initialize path traversal patterns."""
        super().__init__(AttackType.PATH_TRAVERSAL)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all path traversal patterns."""

        # Basic directory traversal
        self.add_pattern(
            name="path_traversal_dots",
            pattern=r"\.\./",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Directory traversal (../)",
        )

        self.add_pattern(
            name="path_traversal_backslashes",
            pattern=r"\.\.\\",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Windows directory traversal",
        )

        # Encoded traversal
        self.add_pattern(
            name="path_traversal_url_encoded",
            pattern=r"%2e%2e%2f",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - URL encoded traversal",
        )

        self.add_pattern(
            name="path_traversal_double_encoded",
            pattern=r"%252e%252e%252f",
            severity=ViolationSeverity.MEDIUM,
            description="Path Traversal - Double URL encoded",
        )

        # Null byte injection
        self.add_pattern(
            name="path_traversal_null_byte",
            pattern=r"%00",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Null byte injection",
        )

        # Windows-specific patterns
        self.add_pattern(
            name="path_traversal_windows_drive",
            pattern=r"[A-Za-z]:\\",
            severity=ViolationSeverity.MEDIUM,
            description="Path Traversal - Windows drive access",
        )

        # Unix-specific patterns
        self.add_pattern(
            name="path_traversal_unix_root",
            pattern=r"/etc/",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Unix system directory access",
        )

        self.add_pattern(
            name="path_traversal_unix_home",
            pattern=r"/home/",
            severity=ViolationSeverity.MEDIUM,
            description="Path Traversal - Unix home directory access",
        )
