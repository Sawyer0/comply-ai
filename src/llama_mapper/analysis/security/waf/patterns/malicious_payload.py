"""Malicious payload security patterns."""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


class MaliciousPayloadPatterns(PatternCollection):
    """Collection of malicious payload security patterns."""
    
    def __init__(self):
        super().__init__(AttackType.MALICIOUS_PAYLOAD)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        # Base64 decode
        self.add_pattern(
            name="mal_base64_decode",
            pattern=r"base64_decode",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Base64 decode"
        )
        
        # Eval function
        self.add_pattern(
            name="mal_eval",
            pattern=r"eval\s*\(",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Eval function"
        )
        
        # Assert function
        self.add_pattern(
            name="mal_assert",
            pattern=r"assert\s*\(",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Assert function"
        )
        
        # Preg replace with eval
        self.add_pattern(
            name="mal_preg_replace",
            pattern=r"preg_replace.*\/e",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Preg replace with eval"
        )
        
        # Create function
        self.add_pattern(
            name="mal_create_function",
            pattern=r"create_function",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Create function"
        )
        
        # Call user func
        self.add_pattern(
            name="mal_call_user_func",
            pattern=r"call_user_func",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Call user func"
        )
        
        # Remote file inclusion
        self.add_pattern(
            name="mal_file_get_contents_http",
            pattern=r"file_get_contents.*http",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Remote file inclusion"
        )
        
        self.add_pattern(
            name="mal_fopen_http",
            pattern=r"fopen.*http",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Remote file inclusion"
        )
        
        self.add_pattern(
            name="mal_include_http",
            pattern=r"include.*http",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Remote file inclusion"
        )
        
        self.add_pattern(
            name="mal_require_http",
            pattern=r"require.*http",
            severity=ViolationSeverity.CRITICAL,
            description="Malicious Payload - Remote file inclusion"
        )
