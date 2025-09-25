"""
Command injection security patterns.

This module contains patterns for detecting command injection attacks
and system command execution attempts.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class CommandInjectionPatterns(PatternCollection):
    """Collection of command injection security patterns."""

    def __init__(self):
        """Initialize command injection patterns."""
        super().__init__(AttackType.COMMAND_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all command injection patterns."""

        # Basic command injection
        self.add_pattern(
            name="cmd_semicolon",
            pattern=r";\s*[a-zA-Z]",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Semicolon command chaining",
        )

        self.add_pattern(
            name="cmd_pipe",
            pattern=r"\|\s*[a-zA-Z]",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Pipe command chaining",
        )

        self.add_pattern(
            name="cmd_ampersand",
            pattern=r"&\s*[a-zA-Z]",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Ampersand command chaining",
        )

        # System commands
        self.add_pattern(
            name="cmd_system_commands",
            pattern=r"\b(cat|ls|dir|type|more|less|head|tail|grep|find|ps|netstat|whoami|id|uname|ifconfig|ipconfig)\b",
            severity=ViolationSeverity.MEDIUM,
            description="Command Injection - System commands",
        )

        # Dangerous commands
        self.add_pattern(
            name="cmd_dangerous_commands",
            pattern=r"\b(rm|del|format|fdisk|mkfs|shutdown|reboot|halt|kill|killall)\b",
            severity=ViolationSeverity.CRITICAL,
            description="Command Injection - Dangerous system commands",
        )

        # Command substitution
        self.add_pattern(
            name="cmd_backtick_substitution",
            pattern=r"`[^`]+`",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Backtick command substitution",
        )

        self.add_pattern(
            name="cmd_dollar_substitution",
            pattern=r"\$\([^)]+\)",
            severity=ViolationSeverity.HIGH,
            description="Command Injection - Dollar command substitution",
        )

        # Redirection
        self.add_pattern(
            name="cmd_redirection",
            pattern=r"[><]\s*[a-zA-Z]",
            severity=ViolationSeverity.MEDIUM,
            description="Command Injection - Output redirection",
        )

        # Environment variables
        self.add_pattern(
            name="cmd_env_variables",
            pattern=r"\$[A-Za-z_][A-Za-z0-9_]*",
            severity=ViolationSeverity.LOW,
            description="Command Injection - Environment variables",
        )
