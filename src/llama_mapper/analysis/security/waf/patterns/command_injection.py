"""Command injection security patterns."""

from .base import PatternCollection
from ..interfaces import AttackType, ViolationSeverity


class CommandInjectionPatterns(PatternCollection):
    """Collection of command injection security patterns."""
    
    def __init__(self):
        super().__init__(AttackType.COMMAND_INJECTION)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        # Basic metacharacters
        self.add_pattern(
            name="cmd_metacharacters",
            pattern=r"[;&|`$(){}\\[\\]\\\\]",
            severity=ViolationSeverity.CRITICAL,
            description="Command Injection - Basic metacharacters"
        )
        
        # File commands
        self.add_pattern(
            name="cmd_file_commands",
            pattern=r"\b(cat|ls|dir|type|more|less|head|tail|grep|find|locate)\b",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - File commands"
        )
        
        # Process commands
        self.add_pattern(
            name="cmd_process_commands",
            pattern=r"\b(ps|top|kill|killall|pkill)\b",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Process commands"
        )
        
        # Network commands
        self.add_pattern(
            name="cmd_network_commands",
            pattern=r"\b(netstat|ss|lsof|netcat|nc)\b",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Network commands"
        )
        
        # Download commands
        self.add_pattern(
            name="cmd_download_commands",
            pattern=r"\b(wget|curl|ftp|scp|rsync)\b",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Download commands"
        )
        
        # Execution functions
        self.add_pattern(
            name="cmd_execution_functions",
            pattern=r"\b(eval|exec|system|shell_exec|passthru)\b",
            severity=ViolationSeverity.CRITICAL,
            description="Command Injection - Execution functions"
        )
