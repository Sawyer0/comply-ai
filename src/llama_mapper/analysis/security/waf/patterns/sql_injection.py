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

        self.add_pattern(
            name="sql_subquery_nested",
            pattern=r"\bSELECT\b.*\bSELECT\b.*\bFROM\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Nested subqueries",
        )

        # UNION-based attacks
        self.add_pattern(
            name="sql_union_basic",
            pattern=r"\b(UNION|UNION ALL)\b.*\b(SELECT|\(.*SELECT.*\))",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - UNION-based attack",
        )

        self.add_pattern(
            name="sql_union_with_select",
            pattern=r"\bUNION\b.*\bSELECT\b.*\bFROM\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - UNION with SELECT FROM",
        )

        # Time-based attacks
        self.add_pattern(
            name="sql_time_delay",
            pattern=r"\b(WAITFOR|DELAY|BENCHMARK|SLEEP)\b.*\b(\d+|\'[^\']+\')",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Time-based delay",
        )

        self.add_pattern(
            name="sql_time_sleep",
            pattern=r"\bSLEEP\s*\(\s*\d+\s*\)",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - SLEEP function",
        )

        self.add_pattern(
            name="sql_time_benchmark",
            pattern=r"\bBENCHMARK\s*\(\s*\d+\s*,\s*[^)]+\s*\)",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - BENCHMARK function",
        )

        # File operations
        self.add_pattern(
            name="sql_file_load",
            pattern=r"\b(LOAD_FILE|INTO OUTFILE|INTO DUMPFILE)\b",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - File operations",
        )

        self.add_pattern(
            name="sql_file_into_outfile",
            pattern=r"\bINTO\s+OUTFILE\s+['\"][^'\"]+['\"]",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - INTO OUTFILE",
        )

        self.add_pattern(
            name="sql_file_load_file",
            pattern=r"\bLOAD_FILE\s*\(\s*['\"][^'\"]+['\"]\s*\)",
            severity=ViolationSeverity.CRITICAL,
            description="SQL Injection - LOAD_FILE function",
        )

        # Information gathering
        self.add_pattern(
            name="sql_info_schema",
            pattern=r"\b(INFORMATION_SCHEMA|MYSQL\.USER|SYS\.DUAL|PG_TABLES|ALL_TABLES)\b",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Information schema access",
        )

        self.add_pattern(
            name="sql_info_version",
            pattern=r"\b(VERSION|@@VERSION|@@GLOBAL\.VERSION)\b",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Version information",
        )

        self.add_pattern(
            name="sql_info_database",
            pattern=r"\b(DATABASE|@@DATABASE|CURRENT_DATABASE)\b",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Database information",
        )

        # Encoding and obfuscation
        self.add_pattern(
            name="sql_encoding_char",
            pattern=r"\b(CHAR|ASCII|ORD|HEX|UNHEX)\b.*\b\(.*\)",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Character encoding",
        )

        self.add_pattern(
            name="sql_encoding_hex",
            pattern=r"0x[0-9a-fA-F]+",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Hexadecimal encoding",
        )

        self.add_pattern(
            name="sql_encoding_unicode",
            pattern=r"\\u[0-9a-fA-F]{4}",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Unicode encoding",
        )

        # String functions
        self.add_pattern(
            name="sql_string_concat",
            pattern=r"\b(CONCAT|SUBSTRING|MID|LEFT|RIGHT|LENGTH)\b.*\b\(.*\)",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - String manipulation functions",
        )

        self.add_pattern(
            name="sql_string_substring",
            pattern=r"\bSUBSTRING\s*\(\s*[^,]+,\s*\d+\s*,\s*\d+\s*\)",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - SUBSTRING function",
        )

        # Type conversion
        self.add_pattern(
            name="sql_type_cast",
            pattern=r"\b(CAST|CONVERT)\b.*\b\(.*\)",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - Type conversion",
        )

        self.add_pattern(
            name="sql_type_convert",
            pattern=r"\bCONVERT\s*\(\s*[^,]+,\s*[^)]+\s*\)",
            severity=ViolationSeverity.MEDIUM,
            description="SQL Injection - CONVERT function",
        )

        # Comment-based obfuscation
        self.add_pattern(
            name="sql_comment_dash",
            pattern=r"--\s*[^\r\n]*",
            severity=ViolationSeverity.LOW,
            description="SQL Injection - Comment obfuscation (--)",
        )

        self.add_pattern(
            name="sql_comment_hash",
            pattern=r"#\s*[^\r\n]*",
            severity=ViolationSeverity.LOW,
            description="SQL Injection - Comment obfuscation (#)",
        )

        self.add_pattern(
            name="sql_comment_block",
            pattern=r"/\*.*?\*/",
            severity=ViolationSeverity.LOW,
            description="SQL Injection - Block comment obfuscation",
        )

        # Database-specific patterns
        self.add_pattern(
            name="sql_mysql_specific",
            pattern=r"\b(LOAD_DATA|INTO DUMPFILE|INTO OUTFILE|BENCHMARK|SLEEP|WAITFOR)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - MySQL-specific functions",
        )

        self.add_pattern(
            name="sql_postgres_specific",
            pattern=r"\b(PG_SLEEP|COPY|\\copy|LO_IMPORT|LO_EXPORT)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - PostgreSQL-specific functions",
        )

        self.add_pattern(
            name="sql_mssql_specific",
            pattern=r"\b(WAITFOR|OPENROWSET|OPENDATASOURCE|BULK INSERT)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - SQL Server-specific functions",
        )

        self.add_pattern(
            name="sql_oracle_specific",
            pattern=r"\b(UTL_FILE|UTL_HTTP|DBMS_LOB|DBMS_XMLGEN)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Oracle-specific functions",
        )

        # Advanced techniques
        self.add_pattern(
            name="sql_blind_boolean",
            pattern=r"\b(OR|AND)\b.*\b(ASCII|LENGTH|SUBSTRING)\b.*\b(>|<|=)\b.*\b\d+\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Blind boolean-based",
        )

        self.add_pattern(
            name="sql_error_based",
            pattern=r"\b(EXTRACTVALUE|UPDATEXML|GTID_SUBSET|GTID_SUBTRACT)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Error-based extraction",
        )

        self.add_pattern(
            name="sql_second_order",
            pattern=r"\b(INSERT|UPDATE)\b.*\b(SELECT|UNION)\b",
            severity=ViolationSeverity.HIGH,
            description="SQL Injection - Second-order injection",
        )
