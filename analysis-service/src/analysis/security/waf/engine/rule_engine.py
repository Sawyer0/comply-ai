"""
WAF rule engine implementation.

This module provides the core WAF rule engine that orchestrates
security pattern matching and violation detection.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any

from ..interfaces import (
    AttackType,
    IWAFRule,
    IWAFRuleEngine,
    ViolationSeverity,
    WAFViolation,
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

    def get_matches(self, text: str) -> List[Tuple[str, str]]:
        """Get all matches in the text with their positions."""
        matches = []
        for match in self._pattern.finditer(text):
            matches.append((match.group(), f"{match.start()}-{match.end()}"))
        return matches

    def create_violation(self, target: str, client_ip: str = "unknown") -> WAFViolation:
        """Create a violation for this rule."""
        return WAFViolation(
            violation_type=self._attack_type,
            severity=self._severity,
            rule_name=self._name,
            message=f"Detected {self._attack_type.value} pattern",
            target=target,
            pattern=self._pattern.pattern,
            client_ip=client_ip,
        )


class WAFRuleEngine(IWAFRuleEngine):
    """Implementation of WAF rule engine."""

    def __init__(self):
        self._rules: Dict[str, IWAFRule] = {}
        self._metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "violations_by_type": {},
            "violations_by_severity": {},
        }

    def add_rule(self, rule: IWAFRule) -> None:
        """Add a rule to the engine."""
        self._rules[rule.name] = rule
        logger.info("Added WAF rule", rule_name=rule.name, attack_type=rule.attack_type.value)

    def remove_rule(self, rule_name: str) -> None:
        """Remove a rule by name."""
        if rule_name in self._rules:
            del self._rules[rule_name]
            logger.info("Removed WAF rule", rule_name=rule_name)
        else:
            logger.warning("Rule not found", rule_name=rule_name)

    def get_rules(self) -> List[IWAFRule]:
        """Get all rules."""
        return list(self._rules.values())

    def scan(self, text: str, client_ip: str = "unknown") -> List[WAFViolation]:
        """Scan text for violations."""
        violations = []
        
        for rule in self._rules.values():
            if rule.matches(text):
                violation = rule.create_violation(text, client_ip)
                violations.append(violation)
                
                # Update metrics
                self._update_violation_metrics(violation)
        
        # Update request metrics
        self._metrics["total_requests"] += 1
        if violations:
            self._metrics["blocked_requests"] += 1
        
        return violations

    def _update_violation_metrics(self, violation: WAFViolation) -> None:
        """Update metrics for a violation."""
        # Count by attack type
        attack_type = violation.violation_type.value
        self._metrics["violations_by_type"][attack_type] = \
            self._metrics["violations_by_type"].get(attack_type, 0) + 1
        
        # Count by severity
        severity = violation.severity.value
        self._metrics["violations_by_severity"][severity] = \
            self._metrics["violations_by_severity"].get(severity, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        metrics = self._metrics.copy()
        
        # Calculate additional metrics
        if metrics["total_requests"] > 0:
            metrics["block_rate"] = metrics["blocked_requests"] / metrics["total_requests"]
        else:
            metrics["block_rate"] = 0.0
        
        metrics["rule_count"] = len(self._rules)
        
        return metrics

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "violations_by_type": {},
            "violations_by_severity": {},
        }
        logger.info("WAF metrics reset")
