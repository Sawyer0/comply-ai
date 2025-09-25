"""
Interfaces for WAF (Web Application Firewall) components.

This module defines the core interfaces and abstract base classes
for WAF functionality, ensuring consistent implementation patterns.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class AttackType(Enum):
    """Types of attacks that WAF rules can detect."""

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    LDAP_INJECTION = "ldap_injection"
    NO_SQL_INJECTION = "nosql_injection"
    XML_INJECTION = "xml_injection"
    SSI_INJECTION = "ssi_injection"
    MALICIOUS_PAYLOAD = "malicious_payload"
    SUSPICIOUS_HEADERS = "suspicious_headers"
    RATE_LIMIT_ABUSE = "rate_limit_abuse"


class ViolationSeverity(Enum):
    """Severity levels for WAF violations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WAFViolation:
    """Represents a WAF violation."""

    def __init__(
        self,
        violation_type: AttackType,
        severity: ViolationSeverity,
        rule_name: str,
        message: str,
        target: str,
        pattern: str,
        client_ip: str = "unknown",
    ):
        self.violation_type = violation_type
        self.severity = severity
        self.rule_name = rule_name
        self.message = message
        self.target = target
        self.pattern = pattern
        self.client_ip = client_ip

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "type": self.violation_type.value,
            "severity": self.severity.value,
            "rule": self.rule_name,
            "message": self.message,
            "target": self.target,
            "pattern": self.pattern,
            "client_ip": self.client_ip,
        }


class IWAFRule(ABC):
    """Interface for WAF rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Rule name."""
        pass

    @property
    @abstractmethod
    def attack_type(self) -> AttackType:
        """Attack type this rule detects."""
        pass

    @property
    @abstractmethod
    def severity(self) -> ViolationSeverity:
        """Rule severity."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Rule description."""
        pass

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Check if rule matches the given text."""
        pass

    @abstractmethod
    def get_matches(self, text: str) -> List[Tuple[str, str]]:
        """Get all matches in the text with their positions."""
        pass


class IWAFRuleEngine(ABC):
    """Interface for WAF rule engine."""

    @abstractmethod
    def add_rule(self, rule: IWAFRule) -> None:
        """Add a rule to the engine."""
        pass

    @abstractmethod
    def remove_rule(self, rule_name: str) -> None:
        """Remove a rule by name."""
        pass

    @abstractmethod
    def get_rules(self) -> List[IWAFRule]:
        """Get all rules."""
        pass

    @abstractmethod
    def scan(self, text: str, client_ip: str = "unknown") -> List[WAFViolation]:
        """Scan text for violations."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics."""
        pass


class IWAFMiddleware(ABC):
    """Interface for WAF middleware."""

    @abstractmethod
    async def process_request(self, request: Any) -> Tuple[bool, List[WAFViolation]]:
        """Process incoming request."""
        pass

    @abstractmethod
    async def process_response(self, response: Any) -> Tuple[bool, List[WAFViolation]]:
        """Process outgoing response."""
        pass


class IWAFMetricsCollector(ABC):
    """Interface for WAF metrics collection."""

    @abstractmethod
    def record_violation(self, violation: WAFViolation) -> None:
        """Record a violation."""
        pass

    @abstractmethod
    def record_request(self, client_ip: str, blocked: bool) -> None:
        """Record a request."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        pass

    @abstractmethod
    def reset_metrics(self) -> None:
        """Reset metrics."""
        pass
