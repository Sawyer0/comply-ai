"""
WAF rule engine implementation.

This module provides the core WAF rule engine that orchestrates
security pattern matching and violation detection.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from ..interfaces import (
    AttackType,
    IWAFRule,
    IWAFRuleEngine,
    ViolationSeverity,
    WAFViolation,
)
from ..patterns import (
    CommandInjectionPatterns,
    LDAPInjectionPatterns,
    MaliciousPayloadPatterns,
    NoSQLInjectionPatterns,
    PathTraversalPatterns,
    SQLInjectionPatterns,
    SSIInjectionPatterns,
    XMLInjectionPatterns,
    XSSPatterns,
)

logger = logging.getLogger(__name__)


class WAFRule(IWAFRule):
    """Implementation of WAF rule interface."""

    def __init__(
        self,
        name: str,
        pattern: str,
        attack_type: AttackType,
        severity: ViolationSeverity,
        description: str = "",
        case_sensitive: bool = False,
    ):
        """
        Initialize WAF rule.

        Args:
            name: Rule name
            pattern: Regex pattern to match
            attack_type: Type of attack this rule detects
            severity: Rule severity
            description: Human-readable description
            case_sensitive: Whether pattern matching is case sensitive
        """
        self._name = name
        self._pattern = re.compile(pattern, re.IGNORECASE if not case_sensitive else 0)
        self._attack_type = attack_type
        self._severity = severity
        self._description = description
        self._case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        """Rule name."""
        return self._name

    @property
    def attack_type(self) -> AttackType:
        """Attack type this rule detects."""
        return self._attack_type

    @property
    def severity(self) -> ViolationSeverity:
        """Rule severity."""
        return self._severity

    @property
    def description(self) -> str:
        """Rule description."""
        return self._description

    @property
    def case_sensitive(self) -> bool:
        """Whether pattern matching is case sensitive."""
        return self._case_sensitive

    @property
    def pattern(self) -> re.Pattern:
        """Compiled regex pattern."""
        return self._pattern

    def matches(self, text: str) -> bool:
        """Check if rule matches the given text."""
        return bool(self._pattern.search(text))

    def create_violation(self, target: str, client_ip: str = "unknown") -> WAFViolation:
        """Create a violation for this rule."""
        return WAFViolation(
            violation_type=self._attack_type,
            severity=self._severity,
            rule_name=self._name,
            message=f"Detected {self._description} in {target}",
            target=target,
            pattern=self._pattern.pattern,
            client_ip=client_ip,
        )


class WAFRuleEngine(IWAFRuleEngine):
    """
    WAF rule engine implementation.

    Provides comprehensive security filtering for various attack patterns
    commonly used against web applications.
    """

    def __init__(self):
        """Initialize WAF rule engine with predefined security rules."""
        self.rules: List[WAFRule] = []
        self.blocked_ips: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = {}

        self._initialize_patterns()
        logger.info("WAF Rule Engine initialized with %s rules", len(self.rules))

    def _initialize_patterns(self):
        """Initialize comprehensive set of WAF security rules."""
        pattern_collections = [
            SQLInjectionPatterns(),
            XSSPatterns(),
            PathTraversalPatterns(),
            CommandInjectionPatterns(),
            LDAPInjectionPatterns(),
            NoSQLInjectionPatterns(),
            XMLInjectionPatterns(),
            SSIInjectionPatterns(),
            MaliciousPayloadPatterns(),
        ]

        for collection in pattern_collections:
            for pattern in collection.get_patterns():
                rule = WAFRule(
                    name=f"{collection.attack_type.value}_{pattern.name}",
                    pattern=pattern.pattern,
                    attack_type=pattern.attack_type,
                    severity=pattern.severity,
                    description=pattern.description,
                    case_sensitive=pattern.case_sensitive,
                )
                self.rules.append(rule)

    def scan_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        body: Optional[str] = None,
        client_ip: str = "unknown",
    ) -> Tuple[bool, List[WAFViolation]]:
        """Scan request for malicious patterns."""
        violations = []

        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            violation = WAFViolation(
                violation_type=AttackType.BLOCKED_IP,
                severity=ViolationSeverity.CRITICAL,
                rule_name="ip_blocklist",
                message=f"IP {client_ip} is blocked",
                target="client_ip",
                pattern="blocked_ip",
                client_ip=client_ip,
            )
            violations.append(violation)
            return False, violations

        # Scan different parts of the request
        scan_targets = [
            ("path", path),
            ("method", method),
            ("body", body or ""),
        ]

        # Add query parameters
        for key, value in query_params.items():
            scan_targets.append((f"query_{key}", value))

        # Add headers
        for key, value in headers.items():
            scan_targets.append((f"header_{key}", value))

        # Apply rules to each scan target
        for target_name, target_value in scan_targets:
            if not target_value:
                continue

            for rule in self.rules:
                if rule.matches(str(target_value)):
                    violation = rule.create_violation(target_name, client_ip)
                    violations.append(violation)

                    logger.warning(
                        f"WAF violation detected: {rule.description} "
                        f"from IP {client_ip} in {target_name}"
                    )

        # Track suspicious IPs
        if violations:
            self.suspicious_ips[client_ip] = self.suspicious_ips.get(client_ip, 0) + 1

            # Block IP if too many violations
            if self.suspicious_ips[client_ip] >= 5:
                self.blocked_ips.add(client_ip)
                logger.warning("IP %s blocked due to repeated violations", client_ip)

        return len(violations) == 0, violations

    def is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked."""
        return client_ip in self.blocked_ips

    def unblock_ip(self, client_ip: str) -> bool:
        """Unblock an IP address."""
        if client_ip in self.blocked_ips:
            self.blocked_ips.remove(client_ip)
            if client_ip in self.suspicious_ips:
                del self.suspicious_ips[client_ip]
            logger.info("IP %s unblocked", client_ip)
            return True
        return False

    def get_blocked_ips(self) -> Set[str]:
        """Get list of blocked IP addresses."""
        return self.blocked_ips.copy()

    def get_suspicious_ips(self) -> Dict[str, int]:
        """Get list of suspicious IP addresses with violation counts."""
        return self.suspicious_ips.copy()

    def add_custom_rule(
        self,
        name: str,
        pattern: str,
        attack_type: AttackType,
        severity: ViolationSeverity = ViolationSeverity.MEDIUM,
        description: str = "",
    ) -> bool:
        """Add custom WAF rule."""
        try:
            rule = WAFRule(name, pattern, attack_type, severity, description)
            self.rules.append(rule)
            logger.info("Custom WAF rule added: %s", name)
            return True
        except re.error as e:
            logger.error("Invalid regex pattern for custom rule %s: %s", name, e)
            return False

    def remove_rule(self, rule_name: str) -> bool:
        """Remove WAF rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info("WAF rule removed: %s", rule_name)
                return True
        return False

    def get_rule_statistics(self) -> Dict[str, any]:
        """Get statistics about loaded rules."""
        stats = {
            "total_rules": len(self.rules),
            "rules_by_attack_type": {},
            "rules_by_severity": {},
            "blocked_ips_count": len(self.blocked_ips),
            "suspicious_ips_count": len(self.suspicious_ips),
        }

        for rule in self.rules:
            attack_type = rule.attack_type.value
            severity = rule.severity.value

            stats["rules_by_attack_type"][attack_type] = (
                stats["rules_by_attack_type"].get(attack_type, 0) + 1
            )
            stats["rules_by_severity"][severity] = (
                stats["rules_by_severity"].get(severity, 0) + 1
            )

        return stats
