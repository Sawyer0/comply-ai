"""Tests for error handling utilities."""

import pytest

from detector_orchestration.errors import (
    http_status_for,
    is_retryable,
    build_error_body,
    ERROR_HTTP_MAP,
    RETRYABLE,
)
from detector_orchestration.models import ErrorBody


class TestErrorHandling:
    def test_http_status_for_known_codes(self):
        """Test HTTP status mapping for known error codes."""
        assert http_status_for("INVALID_REQUEST") == 400
        assert http_status_for("POLICY_NOT_FOUND") == 400
        assert http_status_for("UNAUTHORIZED") == 401
        assert http_status_for("INSUFFICIENT_RBAC") == 403
        assert http_status_for("RATE_LIMITED") == 403
        assert http_status_for("PARTIAL_COVERAGE") == 206
        assert http_status_for("REQUEST_TIMEOUT") == 408
        assert http_status_for("DETECTOR_OVERLOADED") == 429
        assert http_status_for("ALL_DETECTORS_UNAVAILABLE") == 502
        assert http_status_for("DETECTOR_COMMUNICATION_FAILED") == 502
        assert http_status_for("AGGREGATION_FAILED") == 500
        assert http_status_for("INTERNAL_ERROR") == 500

    def test_http_status_for_unknown_code(self):
        """Test HTTP status mapping for unknown error codes."""
        assert http_status_for("UNKNOWN_ERROR") == 500
        assert http_status_for("SOME_NEW_ERROR") == 500

    def test_is_retryable_known_codes(self):
        """Test retry guidance for known error codes."""
        assert is_retryable("RATE_LIMITED") is True
        assert is_retryable("REQUEST_TIMEOUT") is True
        assert is_retryable("DETECTOR_OVERLOADED") is True
        assert is_retryable("ALL_DETECTORS_UNAVAILABLE") is True
        assert is_retryable("DETECTOR_COMMUNICATION_FAILED") is True

    def test_is_retryable_non_retryable_codes(self):
        """Test retry guidance for non-retryable error codes."""
        assert is_retryable("INVALID_REQUEST") is False
        assert is_retryable("POLICY_NOT_FOUND") is False
        assert is_retryable("UNAUTHORIZED") is False
        assert is_retryable("INSUFFICIENT_RBAC") is False
        assert is_retryable("PARTIAL_COVERAGE") is False
        assert is_retryable("AGGREGATION_FAILED") is False
        assert is_retryable("INTERNAL_ERROR") is False

    def test_is_retryable_unknown_code(self):
        """Test retry guidance for unknown error codes."""
        assert is_retryable("UNKNOWN_ERROR") is False
        assert is_retryable("SOME_NEW_ERROR") is False

    def test_build_error_body_with_message(self):
        """Test building error body with custom message."""
        error_body = build_error_body(
            request_id="test-request-123",
            code="INVALID_REQUEST",
            message="Custom error message"
        )

        assert isinstance(error_body, ErrorBody)
        assert error_body.error_code == "INVALID_REQUEST"
        assert error_body.message == "Custom error message"
        assert error_body.request_id == "test-request-123"
        assert error_body.retryable is False

    def test_build_error_body_without_message(self):
        """Test building error body without custom message."""
        error_body = build_error_body(
            request_id="test-request-456",
            code="REQUEST_TIMEOUT",
        )

        assert isinstance(error_body, ErrorBody)
        assert error_body.error_code == "REQUEST_TIMEOUT"
        assert error_body.message is None
        assert error_body.request_id == "test-request-456"
        assert error_body.retryable is True

    def test_build_error_body_with_none_request_id(self):
        """Test building error body with None request ID."""
        error_body = build_error_body(
            request_id=None,
            code="INTERNAL_ERROR",
            message="Something went wrong"
        )

        assert isinstance(error_body, ErrorBody)
        assert error_body.error_code == "INTERNAL_ERROR"
        assert error_body.message == "Something went wrong"
        assert error_body.request_id is None
        assert error_body.retryable is False

    def test_error_mapping_completeness(self):
        """Test that error mappings are complete and consistent."""
        # All codes in ERROR_HTTP_MAP should have retry guidance
        for error_code in ERROR_HTTP_MAP.keys():
            assert error_code in RETRYABLE, f"Missing retry guidance for {error_code}"

        # All retryable codes should have HTTP mappings
        for error_code in RETRYABLE.keys():
            assert error_code in ERROR_HTTP_MAP, f"Missing HTTP mapping for {error_code}"

    def test_canonical_error_codes(self):
        """Test that canonical error codes follow expected patterns."""
        expected_patterns = {
            "INVALID_REQUEST": 400,
            "POLICY_NOT_FOUND": 400,
            "UNAUTHORIZED": 401,
            "INSUFFICIENT_RBAC": 403,
            "RATE_LIMITED": 403,
            "PARTIAL_COVERAGE": 206,
            "REQUEST_TIMEOUT": 408,
            "DETECTOR_OVERLOADED": 429,
            "ALL_DETECTORS_UNAVAILABLE": 502,
            "DETECTOR_COMMUNICATION_FAILED": 502,
            "AGGREGATION_FAILED": 500,
            "INTERNAL_ERROR": 500,
        }

        for code, expected_status in expected_patterns.items():
            assert code in ERROR_HTTP_MAP, f"Missing error code: {code}"
            assert ERROR_HTTP_MAP[code] == expected_status, f"Wrong status for {code}"
            assert code in RETRYABLE, f"Missing retry guidance for {code}"

    def test_error_body_creation_with_all_fields(self):
        """Test creating error body with all possible fields."""
        error_body = ErrorBody(
            error_code="RATE_LIMITED",
            message="Rate limit exceeded",
            request_id="req-123",
            retryable=True,
        )

        assert error_body.error_code == "RATE_LIMITED"
        assert error_body.message == "Rate limit exceeded"
        assert error_body.request_id == "req-123"
        assert error_body.retryable is True

    def test_error_body_creation_minimal(self):
        """Test creating error body with minimal fields."""
        error_body = ErrorBody(
            error_code="INVALID_REQUEST",
        )

        assert error_body.error_code == "INVALID_REQUEST"
        assert error_body.message is None
        assert error_body.request_id is None
        assert error_body.retryable is False

    def test_error_body_validation(self):
        """Test error body validation."""
        # Valid error body
        error_body = ErrorBody(
            error_code="TEST_ERROR",
            message="Test message",
            request_id="test-req",
            retryable=False,
        )

        # Should be able to serialize/deserialize
        json_str = error_body.model_dump_json()
        assert isinstance(json_str, str)
        assert "TEST_ERROR" in json_str

        # Should be able to deserialize
        loaded = ErrorBody.model_validate_json(json_str)
        assert loaded.error_code == error_body.error_code
        assert loaded.message == error_body.message
        assert loaded.request_id == error_body.request_id
        assert loaded.retryable == error_body.retryable

    def test_retry_guidance_accuracy(self):
        """Test that retry guidance is accurate based on HTTP semantics."""
        # 5xx errors should generally be retryable
        five_hundred_codes = [
            code for code in ERROR_HTTP_MAP.keys()
            if ERROR_HTTP_MAP[code] >= 500
        ]
        for code in five_hundred_codes:
            if code not in ["AGGREGATION_FAILED", "INTERNAL_ERROR"]:
                assert RETRYABLE.get(code, False), f"5xx error {code} should be retryable"

        # 4xx errors should generally not be retryable
        four_hundred_codes = [
            code for code in ERROR_HTTP_MAP.keys()
            if 400 <= ERROR_HTTP_MAP[code] < 500
        ]
        for code in four_hundred_codes:
            if code not in ["RATE_LIMITED", "REQUEST_TIMEOUT", "DETECTOR_OVERLOADED"]:
                assert not RETRYABLE.get(code, False), f"4xx error {code} should not be retryable"

    def test_error_code_naming_convention(self):
        """Test that error codes follow naming conventions."""
        for code in ERROR_HTTP_MAP.keys():
            # Should be uppercase with underscores
            assert code.isupper(), f"Error code {code} should be uppercase"
            assert "_" in code or len(code) <= 10, f"Error code {code} should use underscores or be short"

    def test_http_status_ranges(self):
        """Test that HTTP status codes are in valid ranges."""
        for code, status in ERROR_HTTP_MAP.items():
            assert 200 <= status <= 599, f"Invalid HTTP status {status} for {code}"

            # Specific range expectations
            if code in ["INVALID_REQUEST", "POLICY_NOT_FOUND"]:
                assert status == 400, f"{code} should be 400"
            elif code in ["UNAUTHORIZED"]:
                assert status == 401, f"{code} should be 401"
            elif code in ["INSUFFICIENT_RBAC", "RATE_LIMITED"]:
                assert status == 403, f"{code} should be 403"
            elif code in ["PARTIAL_COVERAGE"]:
                assert status == 206, f"{code} should be 206"
            elif code in ["REQUEST_TIMEOUT"]:
                assert status == 408, f"{code} should be 408"
            elif code in ["DETECTOR_OVERLOADED"]:
                assert status == 429, f"{code} should be 429"
            elif code in ["ALL_DETECTORS_UNAVAILABLE", "DETECTOR_COMMUNICATION_FAILED"]:
                assert status == 502, f"{code} should be 502"
            elif code in ["AGGREGATION_FAILED", "INTERNAL_ERROR"]:
                assert status == 500, f"{code} should be 500"
