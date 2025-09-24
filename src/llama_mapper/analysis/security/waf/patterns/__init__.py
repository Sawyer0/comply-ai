"""
WAF security patterns package.

This package contains organized security patterns for different
types of attacks and malicious payloads.
"""

from .command_injection import CommandInjectionPatterns
from .ldap_injection import LDAPInjectionPatterns
from .malicious_payload import MaliciousPayloadPatterns
from .nosql_injection import NoSQLInjectionPatterns
from .path_traversal import PathTraversalPatterns
from .sql_injection import SQLInjectionPatterns
from .ssi_injection import SSIInjectionPatterns
from .xml_injection import XMLInjectionPatterns
from .xss import XSSPatterns

__all__ = [
    "SQLInjectionPatterns",
    "XSSPatterns",
    "PathTraversalPatterns",
    "CommandInjectionPatterns",
    "LDAPInjectionPatterns",
    "NoSQLInjectionPatterns",
    "XMLInjectionPatterns",
    "SSIInjectionPatterns",
    "MaliciousPayloadPatterns",
]
