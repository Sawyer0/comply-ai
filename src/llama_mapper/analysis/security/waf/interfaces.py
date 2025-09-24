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
        """
        Check if rule matches the given text.

        Args:
            text: Text to check

        Returns:
            True if rule matches, False otherwise
        """
        pass

    @abstractmethod
    def create_violation(self, target: str, client_ip: str = "unknown") -> WAFViolation:
        """
        Create a violation for this rule.

        Args:
            target: Target where violation occurred
            client_ip: Client IP address

        Returns:
            WAF violation
        """
        pass


class IWAFRuleEngine(ABC):
    """Interface for WAF rule engine."""

    @abstractmethod
    def scan_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        body: Optional[str] = None,
        client_ip: str = "unknown",
    ) -> Tuple[bool, List[WAFViolation]]:
        """
        Scan request for malicious patterns.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers
            query_params: Query parameters
            body: Request body
            client_ip: Client IP address

        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        pass

    @abstractmethod
    def is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked."""
        pass

    @abstractmethod
    def unblock_ip(self, client_ip: str) -> bool:
        """Unblock an IP address."""
        pass

    @abstractmethod
    def get_blocked_ips(self) -> Set[str]:
        """Get list of blocked IP addresses."""
        pass

    @abstractmethod
    def get_suspicious_ips(self) -> Dict[str, int]:
        """Get list of suspicious IP addresses with violation counts."""
        pass

    @abstractmethod
    def add_custom_rule(
        self,
        name: str,
        pattern: str,
        attack_type: AttackType,
        severity: ViolationSeverity = ViolationSeverity.MEDIUM,
        description: str = "",
    ) -> bool:
        """Add custom WAF rule."""
        pass

    @abstractmethod
    def remove_rule(self, rule_name: str) -> bool:
        """Remove WAF rule by name."""
        pass


class IWAFMiddleware(ABC):
    """Interface for WAF middleware."""

    @abstractmethod
    async def process_request(self, request: Any) -> Tuple[bool, List[WAFViolation]]:
        """
        Process request through WAF.

        Args:
            request: HTTP request object

        Returns:
            Tuple of (is_safe, list_of_violations)
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get WAF statistics."""
        pass

    @abstractmethod
    def unblock_ip(self, client_ip: str) -> bool:
        """Unblock an IP address."""
        pass


class IWAFMetricsCollector(ABC):
    """Interface for WAF metrics collection."""

    @abstractmethod
    def record_waf_scan(
        self,
        is_safe: bool,
        violations_count: int,
        processing_time_ms: float,
        client_ip: str = "unknown",
    ) -> None:
        """Record WAF scan metrics."""
        pass

    @abstractmethod
    def record_waf_violation(
        self,
        violation_type: AttackType,
        severity: ViolationSeverity,
        rule_name: str,
        client_ip: str = "unknown",
    ) -> None:
        """Record WAF violation metrics."""
        pass
