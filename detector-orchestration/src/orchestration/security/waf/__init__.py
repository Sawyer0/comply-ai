"""Web Application Firewall functionality following SRP.

This module provides WAF capabilities:
- Attack Detection: Identify malicious input patterns
- SQL Injection Protection: Detect SQL injection attempts (to be implemented)
- XSS Protection: Detect XSS attempts (to be implemented)
- Command Injection Protection: Detect command injection (to be implemented)
- Path Traversal Protection: Detect path traversal attempts (to be implemented)
"""

from .attack_detector import (
    AttackDetector,
    AttackPattern,
    AttackDetection,
    AttackType,
    AttackSeverity,
)

__all__ = [
    "AttackDetector",
    "AttackPattern",
    "AttackDetection",
    "AttackType",
    "AttackSeverity",
]
