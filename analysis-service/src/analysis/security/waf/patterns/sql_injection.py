"""
SQL injection security patterns.

This module contains comprehensive patterns for detecting
SQL injection attacks across different databases and techniques.
"""

from ..interfaces import AttackType, ViolationSeverity
from .base import PatternCollection


class SQLInjectionPatterns(PatternCollection):
    """Collection of SQL injection security patterns."""

    def __init__(self):
        """Initialize SQL injection patterns."""
        super().__init__(AttackType.SQL_INJECTION)
        self._initialize_patterns()

    def _initialize_patterns(self):
        """Initialize all SQL injection patterns."""

        # Basic SQL injection patterns
        self.add_pattern(
            name="sql_basic_select",
            pattern=r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b.*\b(FROM|INTO|SET|WHERE|VALUES)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Basic SQL commands",
        )

        self.add_pattern(
            name="sql_boolean_1_equals_1",
            pattern=r"\b(OR|AND)\b.*\b(1=1|2=2|'1'='1'|'a'='a')\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Boolean-based (1=1)",
        )

        self.add_pattern(
            name="sql_boolean_always_true",
            pattern=r"\b(OR|AND)\b.*\b(true|TRUE|1|'1')\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Boolean-based (always true)",
        )

        # Subquery patterns
        self.add_pattern(
            name="sql_subquery_exists",
            pattern=r"\b(OR|AND)\b.*\b(EXISTS|IN)\b.*\b(SELECT|\(.*SELECT.*\))",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Subquery with EXISTS/IN",
        )

        # Union-based injection
        self.add_pattern(
            name="sql_union_select",
            pattern=r"\bUNION\b.*\bSELECT\b",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - UNION-based",
        )

        # Time-based blind injection
        self.add_pattern(
            name="sql_time_delay",
            pattern=r"\b(OR|AND)\b.*\b(SLEEP|WAITFOR|DELAY|BENCHMARK)\b.*\b\(\d+\)",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Time-based blind",
        )

        # Error-based injection
        self.add_pattern(
            name="sql_error_division",
            pattern=r"\b(OR|AND)\b.*\b(\d+/\d+|\d+\s*/\s*0)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Error-based (division by zero)",
        )

        # Comment patterns
        self.add_pattern(
            name="sql_comment_dash",
            pattern=r"--\s*$",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Comment (--)",
        )

        self.add_pattern(
            name="sql_comment_hash",
            pattern=r"#\s*$",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Comment (#)",
        )

        # Database-specific patterns
        self.add_pattern(
            name="sql_mysql_functions",
            pattern=r"\b(LOAD_FILE|INTO\s+OUTFILE|INTO\s+DUMPFILE)\b",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - MySQL file operations",
        )

        self.add_pattern(
            name="sql_mssql_functions",
            pattern=r"\b(xp_cmdshell|sp_executesql|OPENROWSET)\b",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - MSSQL system functions",
        )

        self.add_pattern(
            name="sql_oracle_functions",
            pattern=r"\b(UTL_FILE|UTL_HTTP|DBMS_LOB)\b",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - Oracle system functions",
        )

        # Information gathering
        self.add_pattern(
            name="sql_information_schema",
            pattern=r"\bINFORMATION_SCHEMA\b",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Information schema access",
        )

        self.add_pattern(
            name="sql_version_info",
            pattern=r"\b(VERSION|@@VERSION|@@VERSION_COMMENT)\b",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Version information gathering",
        )

        # Advanced evasion techniques
        self.add_pattern(
            name="sql_hex_encoding",
            pattern=r"0x[0-9a-fA-F]+",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Hex encoding",
        )

        self.add_pattern(
            name="sql_char_function",
            pattern=r"\bCHAR\s*\([^)]+\)",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - CHAR function encoding",
        )

        # Stacked queries
        self.add_pattern(
            name="sql_stacked_semicolon",
            pattern=r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - Stacked queries",
        )

        # Conditional statements
        self.add_pattern(
            name="sql_if_statement",
            pattern=r"\bIF\s*\([^)]+\)\s*BEGIN",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Conditional IF statement",
        )

        self.add_pattern(
            name="sql_case_statement",
            pattern=r"\bCASE\s+WHEN\s+[^THEN]+\s+THEN",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - CASE statement",
        )
