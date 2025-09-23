"""
Path traversal security patterns.

This module contains patterns for detecting path traversal attacks.
"""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


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
            name="path_traversal_basic",
            pattern=r"\.\./",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Basic directory traversal"
        )
        
        self.add_pattern(
            name="path_traversal_windows",
            pattern=r"\.\.\\",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Windows directory traversal"
        )
        
        # URL encoded variations
        self.add_pattern(
            name="path_traversal_url_encoded",
            pattern=r"%2e%2e%2f",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - URL encoded"
        )
        
        self.add_pattern(
            name="path_traversal_url_encoded_windows",
            pattern=r"%2e%2e%5c",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - URL encoded Windows"
        )
        
        # Double URL encoded
        self.add_pattern(
            name="path_traversal_double_encoded",
            pattern=r"%252e%252e%252f",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Double URL encoded"
        )
        
        # System files
        self.add_pattern(
            name="path_traversal_etc_passwd",
            pattern=r"/etc/passwd",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - /etc/passwd"
        )
        
        self.add_pattern(
            name="path_traversal_etc_shadow",
            pattern=r"/etc/shadow",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - /etc/shadow"
        )
        
        self.add_pattern(
            name="path_traversal_windows_system32",
            pattern=r"/windows/system32",
            severity=ViolationSeverity.HIGH,
            description="Path Traversal - Windows system32"
        )
