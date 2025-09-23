"""
WAF security patterns package.

This package contains organized security patterns for different
types of attacks and malicious payloads.
"""

from .sql_injection import SQLInjectionPatterns
from .xss import XSSPatterns
from .path_traversal import PathTraversalPatterns
from .command_injection import CommandInjectionPatterns
from .ldap_injection import LDAPInjectionPatterns
from .nosql_injection import NoSQLInjectionPatterns
from .xml_injection import XMLInjectionPatterns
from .ssi_injection import SSIInjectionPatterns
from .malicious_payload import MaliciousPayloadPatterns

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
