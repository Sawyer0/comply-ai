"""
Unit tests for evidence reference validation edge cases.

Tests the evidence reference validation system with comprehensive edge cases
as specified in task 2.2.
"""

from typing import Any, Dict, List

import pytest

from src.llama_mapper.analysis.infrastructure.validator import AnalysisValidator
from src.llama_mapper.analysis.models import ALLOWED_EVIDENCE_REFS, AnalysisRequest


class TestEvidenceReferenceEdgeCases:
    """Test evidence reference validation with comprehensive edge cases."""

    def test_all_allowed_evidence_refs(self):
        """Test that all allowed evidence references pass validation."""
        validator = AnalysisValidator()

        # Test each allowed evidence ref individually
        for ref in ALLOWED_EVIDENCE_REFS:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [ref],
                "opa_diff": "",
            }

            # Should not raise exception
            validator._validate_evidence_refs(output)

    def test_multiple_valid_evidence_refs(self):
        """Test validation with multiple valid evidence references."""
        validator = AnalysisValidator()

        output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": [
                "observed_coverage",
                "required_coverage",
                "detector_errors",
                "high_sev_hits",
                "false_positive_bands",
            ],
            "opa_diff": "",
        }

        # Should not raise exception
        validator._validate_evidence_refs(output)

    def test_case_sensitivity_validation(self):
        """Test that evidence reference validation is case-sensitive."""
        validator = AnalysisValidator()

        # Test various case variations
        invalid_cases = [
            "Observed_Coverage",  # Wrong case
            "OBSERVED_COVERAGE",  # All caps
            "observed-coverage",  # Hyphen instead of underscore
            "observed.coverage",  # Dot instead of underscore
            "observedCoverage",  # CamelCase
        ]

        for invalid_ref in invalid_cases:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_whitespace_handling(self):
        """Test evidence reference validation with whitespace variations."""
        validator = AnalysisValidator()

        # Test with leading/trailing whitespace
        invalid_cases = [
            " observed_coverage",  # Leading space
            "observed_coverage ",  # Trailing space
            " observed_coverage ",  # Both leading and trailing
            "\tobserved_coverage",  # Leading tab
            "observed_coverage\n",  # Trailing newline
        ]

        for invalid_ref in invalid_cases:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)

    def test_empty_string_evidence_ref(self):
        """Test validation with empty string evidence reference."""
        validator = AnalysisValidator()

        output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": [""],  # Empty string
            "opa_diff": "",
        }

        with pytest.raises(Exception) as exc_info:
            validator._validate_evidence_refs(output)

        assert "Invalid evidence reference" in str(exc_info.value)

    def test_none_evidence_ref(self):
        """Test validation with None evidence reference."""
        validator = AnalysisValidator()

        output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": [None],  # None value
            "opa_diff": "",
        }

        with pytest.raises(Exception) as exc_info:
            validator._validate_evidence_refs(output)

        assert "Invalid evidence reference" in str(exc_info.value)

    def test_non_string_evidence_ref(self):
        """Test validation with non-string evidence reference."""
        validator = AnalysisValidator()

        # Test with various non-string types
        invalid_types = [123, 0.5, True, [], {}, None]

        for invalid_ref in invalid_types:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)

    def test_mixed_valid_invalid_evidence_refs(self):
        """Test validation with mix of valid and invalid evidence references."""
        validator = AnalysisValidator()

        output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": [
                "observed_coverage",  # Valid
                "invalid_ref",  # Invalid
                "required_coverage",  # Valid
                "another_invalid",  # Invalid
            ],
            "opa_diff": "",
        }

        with pytest.raises(Exception) as exc_info:
            validator._validate_evidence_refs(output)

        assert "Invalid evidence reference" in str(exc_info.value)
        assert "invalid_ref" in str(exc_info.value)

    def test_duplicate_evidence_refs(self):
        """Test validation with duplicate evidence references."""
        validator = AnalysisValidator()

        output = {
            "reason": "coverage gap detected",
            "remediation": "add secondary detector",
            "confidence": 0.8,
            "evidence_refs": [
                "observed_coverage",
                "observed_coverage",  # Duplicate
                "required_coverage",
            ],
            "opa_diff": "",
        }

        # Duplicates should be allowed (validation only checks if refs are valid)
        validator._validate_evidence_refs(output)

    def test_evidence_refs_with_special_characters(self):
        """Test evidence reference validation with special characters."""
        validator = AnalysisValidator()

        # Test various special character combinations
        invalid_refs = [
            "observed_coverage!",  # Exclamation mark
            "observed_coverage@",  # At symbol
            "observed_coverage#",  # Hash
            "observed_coverage$",  # Dollar sign
            "observed_coverage%",  # Percent
            "observed_coverage^",  # Caret
            "observed_coverage&",  # Ampersand
            "observed_coverage*",  # Asterisk
            "observed_coverage(",  # Parenthesis
            "observed_coverage)",  # Parenthesis
            "observed_coverage+",  # Plus
            "observed_coverage=",  # Equals
            "observed_coverage[",  # Square bracket
            "observed_coverage]",  # Square bracket
            "observed_coverage{",  # Curly brace
            "observed_coverage}",  # Curly brace
            "observed_coverage|",  # Pipe
            "observed_coverage\\",  # Backslash
            "observed_coverage:",  # Colon
            "observed_coverage;",  # Semicolon
            'observed_coverage"',  # Quote
            "observed_coverage'",  # Apostrophe
            "observed_coverage<",  # Less than
            "observed_coverage>",  # Greater than
            "observed_coverage,",  # Comma
            "observed_coverage.",  # Period
            "observed_coverage?",  # Question mark
            "observed_coverage/",  # Forward slash
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_unicode(self):
        """Test evidence reference validation with unicode characters."""
        validator = AnalysisValidator()

        # Test various unicode characters
        invalid_refs = [
            "observed_coverage🚨",  # Emoji
            "observed_coverageα",  # Greek letter
            "observed_coverage中",  # Chinese character
            "observed_coverageñ",  # Accented character
            "observed_coverage™",  # Trademark symbol
            "observed_coverage©",  # Copyright symbol
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_numbers(self):
        """Test evidence reference validation with numeric characters."""
        validator = AnalysisValidator()

        # Test various numeric combinations
        invalid_refs = [
            "observed_coverage123",  # Numbers at end
            "123observed_coverage",  # Numbers at start
            "observed123coverage",  # Numbers in middle
            "observed_coverage_123",  # Numbers with underscore
            "observed_coverage-123",  # Numbers with hyphen
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_underscores_and_hyphens(self):
        """Test evidence reference validation with underscores and hyphens."""
        validator = AnalysisValidator()

        # Test underscore variations (should be valid for some fields)
        valid_underscore_refs = [
            "observed_coverage",
            "required_coverage",
            "detector_errors",
            "high_sev_hits",
            "false_positive_bands",
        ]

        for valid_ref in valid_underscore_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [valid_ref],
                "opa_diff": "",
            }

            # Should not raise exception
            validator._validate_evidence_refs(output)

        # Test hyphen variations (should be invalid)
        invalid_hyphen_refs = [
            "observed-coverage",
            "required-coverage",
            "detector-errors",
            "high-sev-hits",
            "false-positive-bands",
        ]

        for invalid_ref in invalid_hyphen_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_spaces(self):
        """Test evidence reference validation with spaces."""
        validator = AnalysisValidator()

        # Test various space combinations
        invalid_refs = [
            "observed coverage",  # Space in middle
            " observed_coverage",  # Leading space
            "observed_coverage ",  # Trailing space
            "observed  coverage",  # Multiple spaces
            "observed\tcoverage",  # Tab character
            "observed\ncoverage",  # Newline character
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_sql_injection_patterns(self):
        """Test evidence reference validation with SQL injection patterns."""
        validator = AnalysisValidator()

        # Test various SQL injection patterns
        invalid_refs = [
            "observed_coverage'; DROP TABLE users; --",
            "observed_coverage' OR '1'='1",
            "observed_coverage UNION SELECT * FROM users",
            "observed_coverage'; INSERT INTO users VALUES ('hacker'); --",
            "observed_coverage' AND 1=1 --",
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_xss_patterns(self):
        """Test evidence reference validation with XSS patterns."""
        validator = AnalysisValidator()

        # Test various XSS patterns
        invalid_refs = [
            "observed_coverage<script>alert('xss')</script>",
            'observed_coverage" onmouseover="alert(\'xss\')"',
            "observed_coverage'><img src=x onerror=alert('xss')>",
            "observed_coveragejavascript:alert('xss')",
            "observed_coverage<iframe src=javascript:alert('xss')></iframe>",
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_path_traversal_patterns(self):
        """Test evidence reference validation with path traversal patterns."""
        validator = AnalysisValidator()

        # Test various path traversal patterns
        invalid_refs = [
            "observed_coverage../../../etc/passwd",
            "observed_coverage..\\..\\..\\windows\\system32\\config\\sam",
            "observed_coverage%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "observed_coverage....//....//....//etc/passwd",
            "observed_coverage%252e%252e%252f%252e%252e%252f%252e%252e%252fetc%252fpasswd",
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_command_injection_patterns(self):
        """Test evidence reference validation with command injection patterns."""
        validator = AnalysisValidator()

        # Test various command injection patterns
        invalid_refs = [
            "observed_coverage; rm -rf /",
            "observed_coverage| cat /etc/passwd",
            "observed_coverage&& whoami",
            "observed_coverage|| id",
            "observed_coverage`whoami`",
            "observed_coverage$(whoami)",
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)

    def test_evidence_refs_with_very_long_strings(self):
        """Test evidence reference validation with very long strings."""
        validator = AnalysisValidator()

        # Test with very long strings
        long_strings = [
            "observed_coverage" + "a" * 1000,  # 1000 character string
            "observed_coverage" + "b" * 10000,  # 10000 character string
            "observed_coverage" + "c" * 100000,  # 100000 character string
        ]

        for long_ref in long_strings:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [long_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert long_ref in str(exc_info.value)

    def test_evidence_refs_with_null_bytes(self):
        """Test evidence reference validation with null bytes."""
        validator = AnalysisValidator()

        # Test with null bytes
        invalid_refs = [
            "observed_coverage\x00",
            "\x00observed_coverage",
            "observed\x00coverage",
            "observed_coverage\x00\x00",
        ]

        for invalid_ref in invalid_refs:
            output = {
                "reason": "coverage gap detected",
                "remediation": "add secondary detector",
                "confidence": 0.8,
                "evidence_refs": [invalid_ref],
                "opa_diff": "",
            }

            with pytest.raises(Exception) as exc_info:
                validator._validate_evidence_refs(output)

            assert "Invalid evidence reference" in str(exc_info.value)
            assert invalid_ref in str(exc_info.value)
