"""Attack pattern detection functionality following SRP.

This module provides ONLY attack pattern detection - identifying malicious input patterns.
Single Responsibility: Detect various attack patterns in input data.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class AttackType(str, Enum):
    """Types of attacks that can be detected."""

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    COMMAND_INJECTION = "command_injection"
    PATH_TRAVERSAL = "path_traversal"
    LDAP_INJECTION = "ldap_injection"
    XML_INJECTION = "xml_injection"
    SCRIPT_INJECTION = "script_injection"
    HEADER_INJECTION = "header_injection"


class AttackSeverity(str, Enum):
    """Severity levels for detected attacks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AttackPattern:
    """Attack pattern definition."""

    name: str
    pattern: str
    attack_type: AttackType
    severity: AttackSeverity
    description: str
    flags: int = re.IGNORECASE


@dataclass
class AttackDetection:
    """Result of attack detection."""

    attack_type: AttackType
    severity: AttackSeverity
    pattern_name: str
    matched_text: str
    position: int
    description: str
    confidence: float


class AttackDetector:
    """Detects various attack patterns in input data.

    Single Responsibility: Identify malicious patterns in input strings.
    Does NOT handle: input sanitization, blocking, logging attacks.
    """

    def __init__(self):
        """Initialize attack detector with predefined patterns."""
        self._patterns = self._initialize_attack_patterns()
        self._compiled_patterns = self._compile_patterns()

    def _initialize_attack_patterns(self) -> List[AttackPattern]:
        """Initialize predefined attack patterns."""
        return [
            # SQL Injection patterns
            AttackPattern(
                name="sql_union_select",
                pattern=r"\b(union\s+select|union\s+all\s+select)\b",
                attack_type=AttackType.SQL_INJECTION,
                severity=AttackSeverity.HIGH,
                description="SQL UNION SELECT injection attempt",
            ),
            AttackPattern(
                name="sql_or_injection",
                pattern=r"\b(or\s+\d+\s*=\s*\d+|or\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)\b",
                attack_type=AttackType.SQL_INJECTION,
                severity=AttackSeverity.HIGH,
                description="SQL OR condition injection",
            ),
            AttackPattern(
                name="sql_comment_injection",
                pattern=r"(--|#|/\*|\*/)",
                attack_type=AttackType.SQL_INJECTION,
                severity=AttackSeverity.MEDIUM,
                description="SQL comment injection",
            ),
            AttackPattern(
                name="sql_stacked_queries",
                pattern=r";\s*(drop|delete|insert|update|create|alter|exec|execute)\s+",
                attack_type=AttackType.SQL_INJECTION,
                severity=AttackSeverity.CRITICAL,
                description="SQL stacked queries injection",
            ),
            # XSS patterns
            AttackPattern(
                name="xss_script_tag",
                pattern=r"<script[^>]*>.*?</script>",
                attack_type=AttackType.XSS,
                severity=AttackSeverity.HIGH,
                description="XSS script tag injection",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            AttackPattern(
                name="xss_javascript_protocol",
                pattern=r"javascript\s*:",
                attack_type=AttackType.XSS,
                severity=AttackSeverity.HIGH,
                description="XSS javascript protocol injection",
            ),
            AttackPattern(
                name="xss_event_handler",
                pattern=r"on\w+\s*=\s*['\"]?[^'\"]*['\"]?",
                attack_type=AttackType.XSS,
                severity=AttackSeverity.MEDIUM,
                description="XSS event handler injection",
            ),
            AttackPattern(
                name="xss_iframe_tag",
                pattern=r"<iframe[^>]*>.*?</iframe>",
                attack_type=AttackType.XSS,
                severity=AttackSeverity.HIGH,
                description="XSS iframe injection",
                flags=re.IGNORECASE | re.DOTALL,
            ),
            # Command Injection patterns
            AttackPattern(
                name="cmd_pipe_operators",
                pattern=r"[;&|`$(){}[\]\\]",
                attack_type=AttackType.COMMAND_INJECTION,
                severity=AttackSeverity.HIGH,
                description="Command injection special characters",
            ),
            AttackPattern(
                name="cmd_system_commands",
                pattern=r"\b(cat|ls|dir|type|echo|whoami|id|pwd|cd|rm|del|mv|cp|chmod|chown)\b",
                attack_type=AttackType.COMMAND_INJECTION,
                severity=AttackSeverity.MEDIUM,
                description="System command injection",
            ),
            # Path Traversal patterns
            AttackPattern(
                name="path_traversal_unix",
                pattern=r"\.\./",
                attack_type=AttackType.PATH_TRAVERSAL,
                severity=AttackSeverity.HIGH,
                description="Unix path traversal attempt",
            ),
            AttackPattern(
                name="path_traversal_windows",
                pattern=r"\.\.\\",
                attack_type=AttackType.PATH_TRAVERSAL,
                severity=AttackSeverity.HIGH,
                description="Windows path traversal attempt",
            ),
            AttackPattern(
                name="path_traversal_encoded",
                pattern=r"(%2e%2e%2f|%2e%2e%5c|%252e%252e%252f)",
                attack_type=AttackType.PATH_TRAVERSAL,
                severity=AttackSeverity.HIGH,
                description="Encoded path traversal attempt",
            ),
            # LDAP Injection patterns
            AttackPattern(
                name="ldap_injection_chars",
                pattern=r"[()&|!*]",
                attack_type=AttackType.LDAP_INJECTION,
                severity=AttackSeverity.MEDIUM,
                description="LDAP injection special characters",
            ),
            # XML Injection patterns
            AttackPattern(
                name="xml_entity_injection",
                pattern=r"<!ENTITY[^>]*>",
                attack_type=AttackType.XML_INJECTION,
                severity=AttackSeverity.HIGH,
                description="XML entity injection",
            ),
            # Header Injection patterns
            AttackPattern(
                name="header_injection_crlf",
                pattern=r"(\r\n|\n|\r)",
                attack_type=AttackType.HEADER_INJECTION,
                severity=AttackSeverity.MEDIUM,
                description="HTTP header injection (CRLF)",
            ),
        ]

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for better performance."""
        compiled = {}
        for pattern in self._patterns:
            try:
                compiled[pattern.name] = re.compile(pattern.pattern, pattern.flags)
            except re.error as e:
                logger.error("Failed to compile pattern %s: %s", pattern.name, str(e))
        return compiled

    def detect_attacks(self, input_text: str) -> List[AttackDetection]:
        """Detect attack patterns in input text.

        Args:
            input_text: Text to analyze for attack patterns

        Returns:
            List of detected attacks
        """
        correlation_id = get_correlation_id()
        detections = []

        if not input_text:
            return detections

        logger.debug(
            "Analyzing input for attack patterns (length: %d)",
            len(input_text),
            extra={"correlation_id": correlation_id},
        )

        for pattern_def in self._patterns:
            compiled_pattern = self._compiled_patterns.get(pattern_def.name)
            if not compiled_pattern:
                continue

            try:
                matches = compiled_pattern.finditer(input_text)
                for match in matches:
                    detection = AttackDetection(
                        attack_type=pattern_def.attack_type,
                        severity=pattern_def.severity,
                        pattern_name=pattern_def.name,
                        matched_text=match.group(0),
                        position=match.start(),
                        description=pattern_def.description,
                        confidence=self._calculate_confidence(
                            pattern_def, match.group(0)
                        ),
                    )
                    detections.append(detection)

            except Exception as e:
                logger.error(
                    "Error processing pattern %s: %s",
                    pattern_def.name,
                    str(e),
                    extra={"correlation_id": correlation_id},
                )

        if detections:
            logger.warning(
                "Detected %d potential attacks in input",
                len(detections),
                extra={
                    "correlation_id": correlation_id,
                    "attack_types": list(set(d.attack_type.value for d in detections)),
                    "severities": list(set(d.severity.value for d in detections)),
                },
            )

        return detections

    def detect_attack_types(self, input_text: str) -> Set[AttackType]:
        """Detect which types of attacks are present in input.

        Args:
            input_text: Text to analyze

        Returns:
            Set of detected attack types
        """
        detections = self.detect_attacks(input_text)
        return set(detection.attack_type for detection in detections)

    def has_high_severity_attacks(self, input_text: str) -> bool:
        """Check if input contains high or critical severity attacks.

        Args:
            input_text: Text to analyze

        Returns:
            True if high/critical severity attacks detected
        """
        detections = self.detect_attacks(input_text)
        return any(
            detection.severity in [AttackSeverity.HIGH, AttackSeverity.CRITICAL]
            for detection in detections
        )

    def get_attack_summary(self, input_text: str) -> Dict[str, Any]:
        """Get a summary of detected attacks.

        Args:
            input_text: Text to analyze

        Returns:
            Dictionary with attack summary
        """
        detections = self.detect_attacks(input_text)

        if not detections:
            return {
                "total_attacks": 0,
                "attack_types": [],
                "max_severity": None,
                "high_confidence_attacks": 0,
            }

        attack_types = list(set(d.attack_type.value for d in detections))
        severities = [d.severity for d in detections]

        # Determine max severity
        severity_order = [
            AttackSeverity.LOW,
            AttackSeverity.MEDIUM,
            AttackSeverity.HIGH,
            AttackSeverity.CRITICAL,
        ]
        max_severity = max(severities, key=lambda s: severity_order.index(s))

        # Count high confidence attacks
        high_confidence_count = sum(1 for d in detections if d.confidence > 0.8)

        return {
            "total_attacks": len(detections),
            "attack_types": attack_types,
            "max_severity": max_severity.value,
            "high_confidence_attacks": high_confidence_count,
            "detections": [
                {
                    "type": d.attack_type.value,
                    "severity": d.severity.value,
                    "pattern": d.pattern_name,
                    "position": d.position,
                    "confidence": d.confidence,
                }
                for d in detections
            ],
        }

    def _calculate_confidence(self, pattern: AttackPattern, matched_text: str) -> float:
        """Calculate confidence score for a detection.

        Args:
            pattern: Pattern that matched
            matched_text: Text that was matched

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence based on pattern specificity
        base_confidence = {
            AttackSeverity.CRITICAL: 0.95,
            AttackSeverity.HIGH: 0.85,
            AttackSeverity.MEDIUM: 0.70,
            AttackSeverity.LOW: 0.60,
        }.get(pattern.severity, 0.50)

        # Adjust based on match length (longer matches are more confident)
        length_factor = min(len(matched_text) / 20.0, 0.2)

        # Adjust based on pattern type (some patterns are more reliable)
        type_factor = {
            AttackType.SQL_INJECTION: 0.1,
            AttackType.XSS: 0.05,
            AttackType.COMMAND_INJECTION: 0.15,
            AttackType.PATH_TRAVERSAL: 0.1,
        }.get(pattern.attack_type, 0.0)

        confidence = base_confidence + length_factor + type_factor
        return min(confidence, 1.0)

    def add_custom_pattern(self, pattern: AttackPattern) -> bool:
        """Add a custom attack pattern.

        Args:
            pattern: Custom pattern to add

        Returns:
            True if pattern added successfully
        """
        correlation_id = get_correlation_id()

        try:
            # Validate pattern by compiling it
            compiled = re.compile(pattern.pattern, pattern.flags)

            # Add to patterns
            self._patterns.append(pattern)
            self._compiled_patterns[pattern.name] = compiled

            logger.info(
                "Added custom attack pattern: %s",
                pattern.name,
                extra={
                    "correlation_id": correlation_id,
                    "pattern_name": pattern.name,
                    "attack_type": pattern.attack_type.value,
                },
            )

            return True

        except re.error as e:
            logger.error(
                "Failed to add custom pattern %s: %s",
                pattern.name,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "pattern_name": pattern.name,
                    "error": str(e),
                },
            )
            return False


# Export only the attack detection functionality
__all__ = [
    "AttackDetector",
    "AttackPattern",
    "AttackDetection",
    "AttackType",
    "AttackSeverity",
]
