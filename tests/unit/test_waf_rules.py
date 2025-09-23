"""
Unit tests for WAF rules and security patterns.

Tests the WAF rule engine, pattern matching, and security filtering
functionality for request filtering and abuse prevention.
"""

import pytest
import re
from unittest.mock import Mock, patch
from typing import Dict, List

from src.llama_mapper.analysis.security.waf.engine.rule_engine import WAFRuleEngine, WAFRule
from src.llama_mapper.analysis.security.waf.interfaces import (
    AttackType, ViolationSeverity, WAFViolation
)
from src.llama_mapper.analysis.security.waf.patterns import (
    SQLInjectionPatterns, XSSPatterns, PathTraversalPatterns,
    CommandInjectionPatterns, LDAPInjectionPatterns
)


class TestWAFRule:
    """Test WAF rule functionality."""
    
    def test_waf_rule_creation(self):
        """Test WAF rule creation and properties."""
        rule = WAFRule(
            name="test_rule",
            pattern=r"test.*pattern",
            attack_type=AttackType.SQL_INJECTION,
            severity=ViolationSeverity.HIGH,
            description="Test rule",
            case_sensitive=False
        )
        
        assert rule.name == "test_rule"
        assert rule.attack_type == AttackType.SQL_INJECTION
        assert rule.severity == ViolationSeverity.HIGH
        assert rule.description == "Test rule"
    
    def test_waf_rule_matching(self):
        """Test WAF rule pattern matching."""
        rule = WAFRule(
            name="test_rule",
            pattern=r"SELECT.*FROM",
            attack_type=AttackType.SQL_INJECTION,
            severity=ViolationSeverity.HIGH,
            description="Test SQL pattern"
        )
        
        # Test positive match
        assert rule.matches("SELECT * FROM users") is True
        assert rule.matches("select * from users") is True  # Case insensitive
        
        # Test negative match
        assert rule.matches("INSERT INTO users") is False
        assert rule.matches("normal text") is False
    
    def test_waf_rule_case_sensitive(self):
        """Test case-sensitive pattern matching."""
        rule = WAFRule(
            name="case_sensitive_rule",
            pattern=r"SELECT.*FROM",
            attack_type=AttackType.SQL_INJECTION,
            severity=ViolationSeverity.HIGH,
            description="Case sensitive rule",
            case_sensitive=True
        )
        
        # Test case sensitive matching
        assert rule.matches("SELECT * FROM users") is True
        assert rule.matches("select * from users") is False  # Case sensitive
    
    def test_waf_rule_violation_creation(self):
        """Test WAF violation creation."""
        rule = WAFRule(
            name="test_rule",
            pattern=r"SELECT.*FROM",
            attack_type=AttackType.SQL_INJECTION,
            severity=ViolationSeverity.HIGH,
            description="Test rule"
        )
        
        violation = rule.create_violation("request_body", "192.168.1.1")
        
        assert violation.violation_type == AttackType.SQL_INJECTION
        assert violation.severity == ViolationSeverity.HIGH
        assert violation.rule_name == "test_rule"
        assert violation.target == "request_body"
        assert violation.client_ip == "192.168.1.1"
        assert "SELECT.*FROM" in violation.message


class TestWAFRuleEngine:
    """Test WAF rule engine functionality."""
    
    @pytest.fixture
    def waf_engine(self):
        """Create WAF rule engine for testing."""
        return WAFRuleEngine()
    
    def test_waf_engine_initialization(self, waf_engine):
        """Test WAF engine initialization."""
        assert len(waf_engine.rules) > 0
        assert isinstance(waf_engine.blocked_ips, set)
        assert isinstance(waf_engine.suspicious_ips, dict)
    
    def test_scan_safe_request(self, waf_engine):
        """Test scanning a safe request."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/users",
            headers={"Content-Type": "application/json"},
            query_params={"page": "1"},
            body='{"name": "John"}',
            client_ip="192.168.1.1"
        )
        
        assert is_safe is True
        assert len(violations) == 0
    
    def test_scan_sql_injection_request(self, waf_engine):
        """Test scanning a request with SQL injection."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/users",
            headers={"Content-Type": "application/json"},
            query_params={"id": "1' OR '1'='1"},
            body=None,
            client_ip="192.168.1.1"
        )
        
        assert is_safe is False
        assert len(violations) > 0
        
        # Check that we found SQL injection violations
        sql_violations = [v for v in violations if v.violation_type == AttackType.SQL_INJECTION]
        assert len(sql_violations) > 0
    
    def test_scan_xss_request(self, waf_engine):
        """Test scanning a request with XSS."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/comments",
            headers={"Content-Type": "application/json"},
            query_params={},
            body='{"comment": "<script>alert(\'xss\')</script>"}',
            client_ip="192.168.1.1"
        )
        
        assert is_safe is False
        assert len(violations) > 0
        
        # Check that we found XSS violations
        xss_violations = [v for v in violations if v.violation_type == AttackType.XSS]
        assert len(xss_violations) > 0
    
    def test_scan_path_traversal_request(self, waf_engine):
        """Test scanning a request with path traversal."""
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/files/../../../etc/passwd",
            headers={"Content-Type": "application/json"},
            query_params={},
            body=None,
            client_ip="192.168.1.1"
        )
        
        assert is_safe is False
        assert len(violations) > 0
        
        # Check that we found path traversal violations
        path_violations = [v for v in violations if v.violation_type == AttackType.PATH_TRAVERSAL]
        assert len(path_violations) > 0
    
    def test_scan_command_injection_request(self, waf_engine):
        """Test scanning a request with command injection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/execute",
            headers={"Content-Type": "application/json"},
            query_params={},
            body='{"command": "ls; cat /etc/passwd"}',
            client_ip="192.168.1.1"
        )
        
        assert is_safe is False
        assert len(violations) > 0
        
        # Check that we found command injection violations
        cmd_violations = [v for v in violations if v.violation_type == AttackType.COMMAND_INJECTION]
        assert len(cmd_violations) > 0
    
    def test_scan_ldap_injection_request(self, waf_engine):
        """Test scanning a request with LDAP injection."""
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/search",
            headers={"Content-Type": "application/json"},
            query_params={},
            body='{"filter": "(uid=*)(|(password=*))"}',
            client_ip="192.168.1.1"
        )
        
        assert is_safe is False
        assert len(violations) > 0
        
        # Check that we found LDAP injection violations
        ldap_violations = [v for v in violations if v.violation_type == AttackType.LDAP_INJECTION]
        assert len(ldap_violations) > 0
    
    def test_blocked_ip_handling(self, waf_engine):
        """Test handling of blocked IPs."""
        # Block an IP
        waf_engine.blocked_ips.add("192.168.1.100")
        
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/users",
            headers={"Content-Type": "application/json"},
            query_params={},
            body=None,
            client_ip="192.168.1.100"
        )
        
        assert is_safe is False
        assert len(violations) == 1
        assert violations[0].rule_name == "ip_blocklist"
        assert violations[0].client_ip == "192.168.1.100"
    
    def test_suspicious_ip_tracking(self, waf_engine):
        """Test tracking of suspicious IPs."""
        # Make multiple malicious requests from same IP
        for _ in range(3):
            waf_engine.scan_request(
                method="GET",
                path="/api/users",
                headers={"Content-Type": "application/json"},
                query_params={"id": "1' OR '1'='1"},
                body=None,
                client_ip="192.168.1.200"
            )
        
        # Check that IP is tracked as suspicious
        assert "192.168.1.200" in waf_engine.suspicious_ips
        assert waf_engine.suspicious_ips["192.168.1.200"] >= 3
    
    def test_add_custom_rule(self, waf_engine):
        """Test adding custom WAF rules."""
        initial_rule_count = len(waf_engine.rules)
        
        success = waf_engine.add_custom_rule(
            name="custom_test_rule",
            pattern=r"custom.*pattern",
            attack_type=AttackType.SQL_INJECTION,
            severity=ViolationSeverity.MEDIUM,
            description="Custom test rule"
        )
        
        assert success is True
        assert len(waf_engine.rules) == initial_rule_count + 1
        
        # Test that the custom rule works
        is_safe, violations = waf_engine.scan_request(
            method="GET",
            path="/api/test",
            headers={"Content-Type": "application/json"},
            query_params={"data": "custom test pattern"},
            body=None,
            client_ip="192.168.1.1"
        )
        
        assert is_safe is False
        custom_violations = [v for v in violations if v.rule_name == "custom_test_rule"]
        assert len(custom_violations) > 0
    
    def test_remove_rule(self, waf_engine):
        """Test removing WAF rules."""
        initial_rule_count = len(waf_engine.rules)
        
        # Try to remove a non-existent rule
        success = waf_engine.remove_rule("non_existent_rule")
        assert success is False
        assert len(waf_engine.rules) == initial_rule_count
        
        # Remove an existing rule
        if waf_engine.rules:
            rule_name = waf_engine.rules[0].name
            success = waf_engine.remove_rule(rule_name)
            assert success is True
            assert len(waf_engine.rules) == initial_rule_count - 1
    
    def test_get_blocked_ips(self, waf_engine):
        """Test getting blocked IPs."""
        waf_engine.blocked_ips.add("192.168.1.1")
        waf_engine.blocked_ips.add("192.168.1.2")
        
        blocked_ips = waf_engine.get_blocked_ips()
        assert "192.168.1.1" in blocked_ips
        assert "192.168.1.2" in blocked_ips
    
    def test_get_suspicious_ips(self, waf_engine):
        """Test getting suspicious IPs."""
        waf_engine.suspicious_ips["192.168.1.1"] = 5
        waf_engine.suspicious_ips["192.168.1.2"] = 3
        
        suspicious_ips = waf_engine.get_suspicious_ips()
        assert suspicious_ips["192.168.1.1"] == 5
        assert suspicious_ips["192.168.1.2"] == 3
    
    def test_unblock_ip(self, waf_engine):
        """Test unblocking IPs."""
        waf_engine.blocked_ips.add("192.168.1.1")
        
        success = waf_engine.unblock_ip("192.168.1.1")
        assert success is True
        assert "192.168.1.1" not in waf_engine.blocked_ips
        
        # Try to unblock non-blocked IP
        success = waf_engine.unblock_ip("192.168.1.2")
        assert success is False


class TestSecurityPatterns:
    """Test individual security pattern collections."""
    
    def test_sql_injection_patterns(self):
        """Test SQL injection pattern collection."""
        patterns = SQLInjectionPatterns()
        pattern_list = patterns.get_patterns()
        
        assert len(pattern_list) > 0
        
        # Test that patterns have required attributes
        for pattern in pattern_list:
            assert hasattr(pattern, 'name')
            assert hasattr(pattern, 'pattern')
            assert hasattr(pattern, 'attack_type')
            assert hasattr(pattern, 'severity')
            assert hasattr(pattern, 'description')
            assert pattern.attack_type == AttackType.SQL_INJECTION
    
    def test_xss_patterns(self):
        """Test XSS pattern collection."""
        patterns = XSSPatterns()
        pattern_list = patterns.get_patterns()
        
        assert len(pattern_list) > 0
        
        for pattern in pattern_list:
            assert pattern.attack_type == AttackType.XSS
    
    def test_path_traversal_patterns(self):
        """Test path traversal pattern collection."""
        patterns = PathTraversalPatterns()
        pattern_list = patterns.get_patterns()
        
        assert len(pattern_list) > 0
        
        for pattern in pattern_list:
            assert pattern.attack_type == AttackType.PATH_TRAVERSAL
    
    def test_command_injection_patterns(self):
        """Test command injection pattern collection."""
        patterns = CommandInjectionPatterns()
        pattern_list = patterns.get_patterns()
        
        assert len(pattern_list) > 0
        
        for pattern in pattern_list:
            assert pattern.attack_type == AttackType.COMMAND_INJECTION
    
    def test_ldap_injection_patterns(self):
        """Test LDAP injection pattern collection."""
        patterns = LDAPInjectionPatterns()
        pattern_list = patterns.get_patterns()
        
        assert len(pattern_list) > 0
        
        for pattern in pattern_list:
            assert pattern.attack_type == AttackType.LDAP_INJECTION


class TestWAFIntegration:
    """Integration tests for WAF functionality."""
    
    def test_comprehensive_attack_detection(self):
        """Test detection of various attack types in a single request."""
        waf_engine = WAFRuleEngine()
        
        # Create a request with multiple attack patterns
        malicious_request = {
            "method": "POST",
            "path": "/api/users/../../../etc/passwd",
            "headers": {
                "Content-Type": "application/json",
                "User-Agent": "<script>alert('xss')</script>"
            },
            "query_params": {
                "id": "1' OR '1'='1",
                "filter": "(uid=*)(|(password=*))"
            },
            "body": '{"command": "ls; cat /etc/passwd", "sql": "SELECT * FROM users"}',
            "client_ip": "192.168.1.1"
        }
        
        is_safe, violations = waf_engine.scan_request(**malicious_request)
        
        assert is_safe is False
        assert len(violations) > 0
        
        # Check that we detected multiple attack types
        attack_types = set(v.violation_type for v in violations)
        expected_attacks = {
            AttackType.SQL_INJECTION,
            AttackType.XSS,
            AttackType.PATH_TRAVERSAL,
            AttackType.COMMAND_INJECTION,
            AttackType.LDAP_INJECTION
        }
        
        # We should detect at least some of these attack types
        assert len(attack_types.intersection(expected_attacks)) > 0
    
    def test_false_positive_prevention(self):
        """Test that legitimate requests don't trigger false positives."""
        waf_engine = WAFRuleEngine()
        
        # Create legitimate requests that might trigger false positives
        legitimate_requests = [
            {
                "method": "GET",
                "path": "/api/users",
                "headers": {"Content-Type": "application/json"},
                "query_params": {"page": "1", "limit": "10"},
                "body": None,
                "client_ip": "192.168.1.1"
            },
            {
                "method": "POST",
                "path": "/api/comments",
                "headers": {"Content-Type": "application/json"},
                "query_params": {},
                "body": '{"comment": "This is a normal comment with no malicious content"}',
                "client_ip": "192.168.1.1"
            },
            {
                "method": "GET",
                "path": "/api/files/document.pdf",
                "headers": {"Content-Type": "application/json"},
                "query_params": {},
                "body": None,
                "client_ip": "192.168.1.1"
            }
        ]
        
        for request in legitimate_requests:
            is_safe, violations = waf_engine.scan_request(**request)
            # Most legitimate requests should be safe
            # (Some might trigger low-severity violations which is acceptable)
            high_severity_violations = [v for v in violations if v.severity in [ViolationSeverity.HIGH, ViolationSeverity.CRITICAL]]
            assert len(high_severity_violations) == 0, f"High severity violation in legitimate request: {violations}"
    
    def test_performance_with_large_request(self):
        """Test WAF performance with large requests."""
        waf_engine = WAFRuleEngine()
        
        # Create a large request body
        large_body = '{"data": "' + "x" * 10000 + '"}'
        
        import time
        start_time = time.time()
        
        is_safe, violations = waf_engine.scan_request(
            method="POST",
            path="/api/data",
            headers={"Content-Type": "application/json"},
            query_params={},
            body=large_body,
            client_ip="192.168.1.1"
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process large requests quickly (less than 1 second)
        assert processing_time < 1.0
        assert is_safe is True  # Large legitimate request should be safe
