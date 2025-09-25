"""
WAF attack patterns for common security threats.

This module provides predefined patterns for detecting
various types of attacks and security threats.
"""

from typing import List
from .rule import WAFRule
from .interfaces import AttackType, ViolationSeverity


class SQLInjectionPatterns:
    """SQL injection attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get SQL injection patterns."""
        return [
            WAFRule(
                name="sql_union_select",
                pattern=r"(?i)(union\s+select|union\s+all\s+select)",
                attack_type=AttackType.SQL_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="SQL UNION SELECT injection"
            ),
            WAFRule(
                name="sql_comment",
                pattern=r"(?i)(--|\#|\/\*|\*\/)",
                attack_type=AttackType.SQL_INJECTION,
                severity=ViolationSeverity.MEDIUM,
                description="SQL comment injection"
            ),
            WAFRule(
                name="sql_function",
                pattern=r"(?i)(concat|substring|ascii|char|length|count|sum|avg|max|min)",
                attack_type=AttackType.SQL_INJECTION,
                severity=ViolationSeverity.MEDIUM,
                description="SQL function injection"
            ),
        ]


class XSSPatterns:
    """XSS attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get XSS patterns."""
        return [
            WAFRule(
                name="xss_script",
                pattern=r"(?i)(<script[^>]*>|<\/script>)",
                attack_type=AttackType.XSS,
                severity=ViolationSeverity.HIGH,
                description="Script tag XSS"
            ),
            WAFRule(
                name="xss_javascript",
                pattern=r"(?i)(javascript:|vbscript:|onload=|onerror=|onclick=)",
                attack_type=AttackType.XSS,
                severity=ViolationSeverity.HIGH,
                description="JavaScript XSS"
            ),
            WAFRule(
                name="xss_iframe",
                pattern=r"(?i)(<iframe[^>]*>|<\/iframe>)",
                attack_type=AttackType.XSS,
                severity=ViolationSeverity.MEDIUM,
                description="Iframe XSS"
            ),
        ]


class PathTraversalPatterns:
    """Path traversal attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get path traversal patterns."""
        return [
            WAFRule(
                name="path_traversal_dots",
                pattern=r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
                attack_type=AttackType.PATH_TRAVERSAL,
                severity=ViolationSeverity.HIGH,
                description="Directory traversal with dots"
            ),
            WAFRule(
                name="path_traversal_encoded",
                pattern=r"(?i)(%2e%2e%2f|%2e%2e%5c|%252e%252e%252f)",
                attack_type=AttackType.PATH_TRAVERSAL,
                severity=ViolationSeverity.HIGH,
                description="Encoded directory traversal"
            ),
        ]


class CommandInjectionPatterns:
    """Command injection attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get command injection patterns."""
        return [
            WAFRule(
                name="cmd_pipe",
                pattern=r"(\||\|\|)",
                attack_type=AttackType.COMMAND_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="Command pipe injection"
            ),
            WAFRule(
                name="cmd_semicolon",
                pattern=r"(;|&&)",
                attack_type=AttackType.COMMAND_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="Command separator injection"
            ),
            WAFRule(
                name="cmd_backtick",
                pattern=r"(`|\$\()",
                attack_type=AttackType.COMMAND_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="Command substitution injection"
            ),
        ]


class LDAPInjectionPatterns:
    """LDAP injection attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get LDAP injection patterns."""
        return [
            WAFRule(
                name="ldap_wildcard",
                pattern=r"(\*|\(|\)|\\|/)",
                attack_type=AttackType.LDAP_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="LDAP wildcard injection"
            ),
        ]


class NoSQLInjectionPatterns:
    """NoSQL injection attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get NoSQL injection patterns."""
        return [
            WAFRule(
                name="nosql_operator",
                pattern=r"(\$where|\$ne|\$gt|\$lt|\$regex)",
                attack_type=AttackType.NO_SQL_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="NoSQL operator injection"
            ),
        ]


class XMLInjectionPatterns:
    """XML injection attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get XML injection patterns."""
        return [
            WAFRule(
                name="xml_entity",
                pattern=r"(&lt;|&gt;|&amp;|&quot;|&#x?[0-9a-fA-F]+;)",
                attack_type=AttackType.XML_INJECTION,
                severity=ViolationSeverity.MEDIUM,
                description="XML entity injection"
            ),
        ]


class SSIInjectionPatterns:
    """SSI injection attack patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get SSI injection patterns."""
        return [
            WAFRule(
                name="ssi_directive",
                pattern=r"(<!--#|#exec|#include|#echo|#set)",
                attack_type=AttackType.SSI_INJECTION,
                severity=ViolationSeverity.HIGH,
                description="SSI directive injection"
            ),
        ]


class MaliciousPayloadPatterns:
    """Malicious payload patterns."""
    
    @staticmethod
    def get_patterns() -> List[WAFRule]:
        """Get malicious payload patterns."""
        return [
            WAFRule(
                name="malicious_payload",
                pattern=r"(?i)(eval\(|exec\(|system\(|shell_exec\()",
                attack_type=AttackType.MALICIOUS_PAYLOAD,
                severity=ViolationSeverity.CRITICAL,
                description="Malicious function execution"
            ),
        ]
