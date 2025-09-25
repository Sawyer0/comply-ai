"""
Content scanning module for the Analysis Service.

Detects malicious or inappropriate content using pattern matching.
"""

import re
from datetime import datetime
from typing import Any, Dict, List

import structlog

from .config import SecurityConfig
from .exceptions import ContentSecurityError

logger = structlog.get_logger(__name__)


class ContentScanner:
    """
    Advanced content scanner for detecting malicious or inappropriate content.

    Features:
    - Pattern-based detection
    - Custom rule engine
    - Threat severity assessment
    """

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="content_scanner")

        # Predefined security patterns
        self.security_patterns = {
            "xss": [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>",
            ],
            "sql_injection": [
                r"(union|select|insert|update|delete|drop)\s+",
                r"(--|#|/\*|\*/)",
                r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
                r"(\bUNION\s+SELECT\b)",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"\\windows\\system32",
            ],
            "command_injection": [r"[;&|`]", r"\$\([^)]*\)", r"`[^`]*`"],
        }

    async def scan_content(self, content: str) -> Dict[str, Any]:
        """
        Scan content for security threats.

        Args:
            content: Content to scan

        Returns:
            Scan results with threat information

        Raises:
            ContentSecurityError: If malicious content is detected
        """
        scan_result = {
            "safe": True,
            "threats": [],
            "risk_score": 0.0,
            "scan_timestamp": datetime.now().isoformat(),
        }

        # Scan for each threat category
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    threat = {
                        "category": category,
                        "pattern": pattern,
                        "matches": matches,
                        "severity": self._get_threat_severity(category),
                    }
                    scan_result["threats"].append(threat)
                    scan_result["safe"] = False

        # Check for blocked patterns from config
        for blocked_pattern in self.config.blocked_patterns:
            if blocked_pattern.lower() in content.lower():
                threat = {
                    "category": "custom_blocked",
                    "pattern": blocked_pattern,
                    "matches": [blocked_pattern],
                    "severity": "medium",
                }
                scan_result["threats"].append(threat)
                scan_result["safe"] = False

        # Calculate risk score
        if scan_result["threats"]:
            severity_scores = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
            total_score = sum(
                severity_scores.get(threat["severity"], 0.5)
                for threat in scan_result["threats"]
            )
            scan_result["risk_score"] = min(
                1.0, total_score / len(scan_result["threats"])
            )

            # Log security threat
            self.logger.warning(
                "Security threats detected in content",
                threat_count=len(scan_result["threats"]),
                risk_score=scan_result["risk_score"],
                categories=[t["category"] for t in scan_result["threats"]],
            )

            # Raise exception for high-risk content
            if scan_result["risk_score"] >= 0.8:
                raise ContentSecurityError(
                    f"High-risk content detected (score: {scan_result['risk_score']})"
                )

        return scan_result

    async def validate_input(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Validate and sanitize input content.

        Args:
            content: Input content to validate
            content_type: Content type header

        Returns:
            Validation result with sanitized content

        Raises:
            ContentSecurityError: If validation fails
        """
        validation_result = {
            "valid": True,
            "sanitized_content": content,
            "warnings": [],
            "blocked_patterns": [],
        }

        # Check content type
        if content_type not in self.config.allowed_content_types:
            raise ContentSecurityError(f"Content type {content_type} not allowed")

        # Check content size
        if len(content.encode("utf-8")) > self.config.max_input_size_bytes:
            raise ContentSecurityError("Input content too large")

        # Scan for security threats if enabled
        if self.config.enable_content_scanning:
            scan_result = await self.scan_content(content)
            if not scan_result["safe"]:
                validation_result["blocked_patterns"] = [
                    {
                        "pattern": threat["pattern"],
                        "category": threat["category"],
                        "matches": threat["matches"],
                    }
                    for threat in scan_result["threats"]
                ]

        # Sanitize content (basic HTML escaping)
        sanitized = content.replace("<", "&lt;").replace(">", "&gt;")
        if sanitized != content:
            validation_result["sanitized_content"] = sanitized
            validation_result["warnings"].append("HTML characters escaped")

        return validation_result

    def _get_threat_severity(self, category: str) -> str:
        """Get severity level for threat category."""
        severity_map = {
            "xss": "high",
            "sql_injection": "critical",
            "path_traversal": "high",
            "command_injection": "critical",
            "custom_blocked": "medium",
        }

        return severity_map.get(category, "medium")

    def add_custom_pattern(self, category: str, pattern: str) -> None:
        """
        Add custom security pattern.

        Args:
            category: Threat category
            pattern: Regex pattern to detect
        """
        if category not in self.security_patterns:
            self.security_patterns[category] = []

        self.security_patterns[category].append(pattern)
        self.logger.info(
            "Custom security pattern added", category=category, pattern=pattern
        )

    def remove_custom_pattern(self, category: str, pattern: str) -> bool:
        """
        Remove custom security pattern.

        Args:
            category: Threat category
            pattern: Regex pattern to remove

        Returns:
            True if pattern was removed, False if not found
        """
        if category in self.security_patterns:
            try:
                self.security_patterns[category].remove(pattern)
                self.logger.info(
                    "Custom security pattern removed",
                    category=category,
                    pattern=pattern,
                )
                return True
            except ValueError:
                pass

        return False
