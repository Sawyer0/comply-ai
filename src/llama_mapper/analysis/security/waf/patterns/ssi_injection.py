"""SSI injection security patterns."""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


class SSIInjectionPatterns(PatternCollection):
    """Collection of SSI injection security patterns."""
    
    def __init__(self):
        super().__init__(AttackType.SSI_INJECTION)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        # SSI directives
        self.add_pattern(
            name="ssi_directive",
            pattern=r"<!--#",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - SSI directive"
        )
        
        # Include directive
        self.add_pattern(
            name="ssi_include",
            pattern=r"#include",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - Include"
        )
        
        # Exec directive
        self.add_pattern(
            name="ssi_exec",
            pattern=r"#exec",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - Exec"
        )
        
        # Echo directive
        self.add_pattern(
            name="ssi_echo",
            pattern=r"#echo",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - Echo"
        )
        
        # Set directive
        self.add_pattern(
            name="ssi_set",
            pattern=r"#set",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - Set"
        )
        
        # If directive
        self.add_pattern(
            name="ssi_if",
            pattern=r"#if",
            severity=ViolationSeverity.MEDIUM,
            description="SSI Injection - If"
        )
