"""
Unit tests for WAF rules and security patterns.

This module provides comprehensive tests for the WAF rule engine,
covering all attack patterns and security scenarios.
"""

from unittest.mock import Mock, patch

import pytest

from ...security.waf import (
    AttackType,
    ViolationSeverity,
    WAFRule,
    WAFRuleEngine,
)


class TestWAFRule:
    """Test cases for individual WAF rules."""

    def test_waf_rule_creation(self):
        """Test WAF rule creation with valid parameters."""
        rule = WAFRule(
            name="test_rule",
            pattern=r"test.*pattern",
            attack_type=AttackType.SQL_INJECTION,
            severity="high",
            description="Test rule",
        )

        assert rule.name == "test_rule"
        assert rule.attack_type == AttackType.SQL_INJECTION
        assert rule.severity == "high"
        assert rule.description == "Test rule"
        assert rule.case_sensitive is False

    def test_waf_rule_case_sensitive(self):
        """Test WAF rule with case sensitive matching."""
        rule = WAFRule(
            name="case_sensitive_rule",
            pattern=r"TEST",
            attack_type=AttackType.XSS,
            severity=ViolationSeverity.HIGH,
            case_sensitive=True,
        )

        assert rule.case_sensitive is True
        assert rule.pattern.search("TEST") is not None
        assert rule.pattern.search("test") is None

    def test_waf_rule_case_insensitive(self):
        """Test WAF rule with case insensitive matching."""
        rule = WAFRule(
            name="case_insensitive_rule",
            pattern=r"test",
            attack_type=AttackType.XSS,
            severity=ViolationSeverity.MEDIUM,
            case_sensitive=False,
        )

        assert rule.case_sensitive is False
        assert rule.pattern.search("TEST") is not None
        assert rule.pattern.search("test") is not None
        assert rule.pattern.search("Test") is not None


class TestWAFRuleEngine:
    """Test cases for WAF rule engine."""

    @pytest.fixture
    def waf_engine(self):
        """Create WAF rule engine for testing."""
        return WAFRuleEngine()

    def test_waf_engine_initialization(self, waf_engine):
        """Test WAF engine initialization."""
        assert len(waf_engine.rules) > 0
        assert len(waf_engine.blocked_ips) == 0
        assert len(waf_engine.suspicious_ips) == 0

    def test_sql_injection_detection(self, waf_engine):
        """Test SQL injection pattern detection."""
        # Test basic SQL injection
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="SELECT * FROM users WHERE id = 1",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert len(violations) > 0
        assert any(v["type"] == "sql_injection" for v in violations)

    def test_sql_injection_union_attack(self, waf_engine):
        """Test SQL injection with UNION attack."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="SELECT * FROM users UNION SELECT * FROM passwords",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("union" in v["message"].lower() for v in violations)

    def test_sql_injection_boolean_attack(self, waf_engine):
        """Test SQL injection with boolean-based attack."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="SELECT * FROM users WHERE id = 1 OR 1=1",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("boolean" in v["message"].lower() for v in violations)

    def test_xss_detection(self, waf_engine):
        """Test XSS pattern detection."""
        # Test script tag XSS
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="<script>alert('XSS')</script>",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "xss" for v in violations)

    def test_xss_javascript_protocol(self, waf_engine):
        """Test XSS with javascript: protocol."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="javascript:alert('XSS')",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("javascript" in v["message"].lower() for v in violations)

    def test_xss_event_handlers(self, waf_engine):
        """Test XSS with event handlers."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="<img src=x onerror=alert('XSS')>",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("event" in v["message"].lower() for v in violations)

    def test_path_traversal_detection(self, waf_engine):
        """Test path traversal pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/analyze/../../../etc/passwd",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "path_traversal" for v in violations)

    def test_path_traversal_url_encoded(self, waf_engine):
        """Test path traversal with URL encoding."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/analyze/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("url encoded" in v["message"].lower() for v in violations)

    def test_command_injection_detection(self, waf_engine):
        """Test command injection pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="; cat /etc/passwd",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "command_injection" for v in violations)

    def test_command_injection_pipe(self, waf_engine):
        """Test command injection with pipe operator."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="ls | grep password",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("metacharacters" in v["message"].lower() for v in violations)

    def test_ldap_injection_detection(self, waf_engine):
        """Test LDAP injection pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="uid=admin)(&(objectClass=*))",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "ldap_injection" for v in violations)

    def test_nosql_injection_detection(self, waf_engine):
        """Test NoSQL injection pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body='{"$where": "this.password == this.username"}',
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "nosql_injection" for v in violations)

    def test_xml_injection_detection(self, waf_engine):
        """Test XML injection pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/xml"},
            query_params={},
            body="<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "xml_injection" for v in violations)

    def test_ssi_injection_detection(self, waf_engine):
        """Test SSI injection pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "text/html"},
            query_params={},
            body="<!--#exec cmd='ls'-->",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "ssi_injection" for v in violations)

    def test_malicious_payload_detection(self, waf_engine):
        """Test malicious payload pattern detection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="eval(base64_decode('cGhwaW5mbygpOw=='))",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["type"] == "malicious_payload" for v in violations)

    def test_safe_request_passes(self, waf_engine):
        """Test that safe requests pass WAF scan."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body='{"period": "2023-01-01T00:00:00Z/2023-01-01T23:59:59Z", "tenant": "test"}',
            client_ip="192.168.1.1",
        )

        assert is_safe
        assert len(violations) == 0

    def test_multiple_violations(self, waf_engine):
        """Test request with multiple violation types."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="SELECT * FROM users; <script>alert('XSS')</script>",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert len(violations) >= 2
        violation_types = [v["type"] for v in violations]
        assert "sql_injection" in violation_types
        assert "xss" in violation_types

    def test_ip_blocking_after_violations(self, waf_engine):
        """Test IP blocking after repeated violations."""
        client_ip = "192.168.1.100"

        # Make multiple requests with violations
        for _ in range(6):  # More than the threshold
            waf_engine.scan_request(
                method="POST",
                path="/api/analyze",
                headers={"Content-Type": "application/json"},
                query_params={},
                body="SELECT * FROM users",
                client_ip=client_ip,
            )

        # Check if IP is blocked
        assert waf_engine.is_ip_blocked(client_ip)
        assert client_ip in waf_engine.get_blocked_ips()

    def test_blocked_ip_request(self, waf_engine):
        """Test that requests from blocked IPs are rejected."""
        client_ip = "192.168.1.200"

        # Block the IP
        waf_engine.blocked_ips.add(client_ip)

        # Try to make a request
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body='{"period": "2023-01-01T00:00:00Z/2023-01-01T23:59:59Z"}',
            client_ip=client_ip,
        )

        assert not is_safe
        assert any(v["type"] == "blocked_ip" for v in violations)

    def test_ip_unblocking(self, waf_engine):
        """Test IP unblocking functionality."""
        client_ip = "192.168.1.300"

        # Block the IP
        waf_engine.blocked_ips.add(client_ip)
        assert waf_engine.is_ip_blocked(client_ip)

        # Unblock the IP
        result = waf_engine.unblock_ip(client_ip)
        assert result is True
        assert not waf_engine.is_ip_blocked(client_ip)
        assert client_ip not in waf_engine.get_blocked_ips()

    def test_custom_rule_addition(self, waf_engine):
        """Test adding custom WAF rules."""
        initial_rule_count = len(waf_engine.rules)

        # Add custom rule
        result = waf_engine.add_custom_rule(
            name="custom_test_rule",
            pattern=r"custom.*pattern",
            attack_type=AttackType.MALICIOUS_PAYLOAD,
            severity="medium",
            description="Custom test rule",
        )

        assert result is True
        assert len(waf_engine.rules) == initial_rule_count + 1

        # Test the custom rule
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="This is a custom pattern test",
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any(v["rule"] == "custom_test_rule" for v in violations)

    def test_custom_rule_invalid_regex(self, waf_engine):
        """Test adding custom rule with invalid regex."""
        result = waf_engine.add_custom_rule(
            name="invalid_rule",
            pattern=r"[invalid regex",  # Invalid regex
            attack_type=AttackType.MALICIOUS_PAYLOAD,
        )

        assert result is False

    def test_rule_removal(self, waf_engine):
        """Test removing WAF rules."""
        # Find a rule to remove
        rule_to_remove = waf_engine.rules[0]
        initial_rule_count = len(waf_engine.rules)

        # Remove the rule
        result = waf_engine.remove_rule(rule_to_remove.name)
        assert result is True
        assert len(waf_engine.rules) == initial_rule_count - 1

        # Try to remove non-existent rule
        result = waf_engine.remove_rule("non_existent_rule")
        assert result is False

    def test_query_parameter_scanning(self, waf_engine):
        """Test scanning of query parameters."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/analyze",
            headers={},
            query_params={"q": "SELECT * FROM users"},
            body=None,
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("query_q" in v["target"] for v in violations)

    def test_header_scanning(self, waf_engine):
        """Test scanning of request headers."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/analyze",
            headers={"User-Agent": "<script>alert('XSS')</script>"},
            query_params={},
            body=None,
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("header_User-Agent" in v["target"] for v in violations)

    def test_path_scanning(self, waf_engine):
        """Test scanning of request path."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/analyze/../../../etc/passwd",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("path" in v["target"] for v in violations)

    def test_method_scanning(self, waf_engine):
        """Test scanning of HTTP method."""
        is_safe, violations = waf_engine.scan_request(
            method="SELECT * FROM users",
            path="/api/analyze",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.1",
        )

        assert not is_safe
        assert any("method" in v["target"] for v in violations)

    def test_empty_request_components(self, waf_engine):
        """Test handling of empty request components."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/analyze",
            headers={},
            query_params={},
            body=None,
            client_ip="192.168.1.1",
        )

        assert is_safe
        assert len(violations) == 0

    def test_suspicious_ip_tracking(self, waf_engine):
        """Test tracking of suspicious IPs."""
        client_ip = "192.168.1.400"

        # Make a request with violation
        waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="SELECT * FROM users",
            client_ip=client_ip,
        )

        # Check if IP is tracked as suspicious
        assert client_ip in waf_engine.get_suspicious_ips()
        assert waf_engine.get_suspicious_ips()[client_ip] == 1

        # Make another violation
        waf_engine.scan_request(
            method="POST",
            path="/api/analyze",
            headers={"Content-Type": "application/json"},
            query_params={},
            body="<script>alert('XSS')</script>",
            client_ip=client_ip,
        )

        # Check if violation count increased
        assert waf_engine.get_suspicious_ips()[client_ip] == 2
