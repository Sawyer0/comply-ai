"""
Unit tests for log scrubbing functionality.

Tests the LogScrubber and RequestResponseLogger classes to ensure
proper PII redaction and sensitive data scrubbing.
"""

import logging
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import Request, Response
from starlette.responses import JSONResponse

from src.llama_mapper.analysis.api.log_scrubbing import (
    LogScrubber,
    LogScrubbingMiddleware,
    RequestResponseLogger,
)


class TestLogScrubber:
    """Test LogScrubber functionality."""

    def test_scrub_text_email(self):
        """Test email redaction in text."""
        scrubber = LogScrubber()

        text = "Contact us at john.doe@example.com for support"
        result = scrubber.scrub_text(text)

        assert "[EMAIL_REDACTED]" in result
        assert "john.doe@example.com" not in result

    def test_scrub_text_phone(self):
        """Test phone number redaction in text."""
        scrubber = LogScrubber()

        text = "Call us at (555) 123-4567 or 555-123-4567"
        result = scrubber.scrub_text(text)

        assert "[PHONE_REDACTED]" in result
        assert "(555) 123-4567" not in result
        assert "555-123-4567" not in result

    def test_scrub_text_ssn(self):
        """Test SSN redaction in text."""
        scrubber = LogScrubber()

        text = "SSN: 123-45-6789"
        result = scrubber.scrub_text(text)

        assert "[SSN_REDACTED]" in result
        assert "123-45-6789" not in result

    def test_scrub_text_credit_card(self):
        """Test credit card redaction in text."""
        scrubber = LogScrubber()

        text = "Card: 1234-5678-9012-3456"
        result = scrubber.scrub_text(text)

        assert "[CARD_REDACTED]" in result
        assert "1234-5678-9012-3456" not in result

    def test_scrub_text_ip_address(self):
        """Test IP address redaction in text."""
        scrubber = LogScrubber()

        text = "IP: 192.168.1.1"
        result = scrubber.scrub_text(text)

        assert "[IP_REDACTED]" in result
        assert "192.168.1.1" not in result

    def test_scrub_text_api_key(self):
        """Test API key redaction in text."""
        scrubber = LogScrubber()

        text = "API Key: abc123def456ghi789jkl012mno345pqr678"
        result = scrubber.scrub_text(text)

        assert "[API_KEY_REDACTED]" in result
        assert "abc123def456ghi789jkl012mno345pqr678" not in result

    def test_scrub_text_password(self):
        """Test password redaction in text."""
        scrubber = LogScrubber()

        text = 'password: "secret123"'
        result = scrubber.scrub_text(text)

        assert "[PASSWORD_REDACTED]" in result
        assert "secret123" not in result

    def test_scrub_dict_sensitive_fields(self):
        """Test complete redaction of sensitive fields in dictionary."""
        scrubber = LogScrubber()

        data = {
            "username": "john_doe",
            "password": "secret123",
            "api_key": "abc123def456",
            "email": "john@example.com",
            "normal_field": "normal_value",
        }

        result = scrubber.scrub_dict(data)

        assert result["username"] == "john_doe"  # Not sensitive
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["email"] == "[EMAIL_REDACTED]"
        assert result["normal_field"] == "normal_value"

    def test_scrub_dict_partial_redaction(self):
        """Test partial redaction of certain fields."""
        scrubber = LogScrubber()

        data = {
            "request_id": "req_1234567890abcdef",
            "correlation_id": "corr_abcdef1234567890",
        }

        result = scrubber.scrub_dict(data)

        assert result["request_id"] == "req_...cdef"
        assert result["correlation_id"] == "corr...7890"

    def test_scrub_dict_nested_structure(self):
        """Test scrubbing of nested dictionary structures."""
        scrubber = LogScrubber()

        data = {
            "user": {
                "name": "John Doe",
                "email": "john@example.com",
                "credentials": {"password": "secret123", "api_key": "abc123def456"},
            },
            "metadata": {"ip": "192.168.1.1", "normal_field": "normal_value"},
        }

        result = scrubber.scrub_dict(data)

        assert result["user"]["name"] == "John Doe"
        assert result["user"]["email"] == "[EMAIL_REDACTED]"
        assert result["user"]["credentials"]["password"] == "[REDACTED]"
        assert result["user"]["credentials"]["api_key"] == "[REDACTED]"
        assert result["metadata"]["ip"] == "[IP_REDACTED]"
        assert result["metadata"]["normal_field"] == "normal_value"

    def test_scrub_list(self):
        """Test scrubbing of list structures."""
        scrubber = LogScrubber()

        data = [
            {"email": "user1@example.com", "name": "User 1"},
            {"email": "user2@example.com", "name": "User 2"},
            "normal_string",
            {"password": "secret123"},
        ]

        result = scrubber.scrub_list(data)

        assert result[0]["email"] == "[EMAIL_REDACTED]"
        assert result[0]["name"] == "User 1"
        assert result[1]["email"] == "[EMAIL_REDACTED]"
        assert result[1]["name"] == "User 2"
        assert result[2] == "normal_string"
        assert result[3]["password"] == "[REDACTED]"

    def test_scrub_headers(self):
        """Test scrubbing of HTTP headers."""
        scrubber = LogScrubber()

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer abc123def456",
            "X-API-Key": "secret_key_123",
            "User-Agent": "Mozilla/5.0",
            "X-Request-ID": "req_1234567890abcdef",
        }

        result = scrubber.scrub_headers(headers)

        assert result["Content-Type"] == "application/json"
        assert result["Authorization"] == "[REDACTED]"
        assert result["X-API-Key"] == "[REDACTED]"
        assert result["User-Agent"] == "Mozilla/5.0"
        assert result["X-Request-ID"] == "req_...cdef"


class TestRequestResponseLogger:
    """Test RequestResponseLogger functionality."""

    def test_log_request(self, caplog):
        """Test request logging with scrubbing."""
        with caplog.at_level(logging.INFO):
            logger = RequestResponseLogger()

            request_data = {
                "username": "john_doe",
                "password": "secret123",
                "email": "john@example.com",
            }

            headers = {
                "Authorization": "Bearer abc123def456",
                "Content-Type": "application/json",
            }

            logger.log_request(
                request_id="req_123",
                method="POST",
                path="/api/analyze",
                data=request_data,
                headers=headers,
            )

        # Check that logs were generated
        assert len(caplog.records) > 0

        # Check that sensitive data was scrubbed in logs
        log_messages = [record.getMessage() for record in caplog.records]
        combined_logs = " ".join(log_messages)

        assert "secret123" not in combined_logs
        assert "john@example.com" not in combined_logs
        assert "abc123def456" not in combined_logs
        assert "[PASSWORD_REDACTED]" in combined_logs
        assert "[EMAIL_REDACTED]" in combined_logs
        assert "[REDACTED]" in combined_logs

    def test_log_response(self, caplog):
        """Test response logging with scrubbing."""
        with caplog.at_level(logging.INFO):
            logger = RequestResponseLogger()

            response_data = {
                "result": "success",
                "user_info": {"email": "user@example.com", "api_key": "secret_key_123"},
            }

            headers = {
                "Content-Type": "application/json",
                "X-API-Key": "secret_header_key",
            }

            logger.log_response(
                request_id="req_123",
                status_code=200,
                data=response_data,
                headers=headers,
            )

        # Check that logs were generated
        assert len(caplog.records) > 0

        # Check that sensitive data was scrubbed in logs
        log_messages = [record.getMessage() for record in caplog.records]
        combined_logs = " ".join(log_messages)

        assert "user@example.com" not in combined_logs
        assert "secret_key_123" not in combined_logs
        assert "secret_header_key" not in combined_logs
        assert "[EMAIL_REDACTED]" in combined_logs
        assert "[REDACTED]" in combined_logs

    def test_log_error(self, caplog):
        """Test error logging with scrubbing."""
        with caplog.at_level(logging.ERROR):
            logger = RequestResponseLogger()

            context = {
                "user_data": {"email": "user@example.com", "password": "secret123"},
                "request_info": {"ip": "192.168.1.1", "api_key": "abc123def456"},
            }

            error = ValueError("Test error")

            logger.log_error(request_id="req_123", error=error, context=context)

        # Check that logs were generated
        assert len(caplog.records) > 0

        # Check that sensitive data was scrubbed in logs
        log_messages = [record.getMessage() for record in caplog.records]
        combined_logs = " ".join(log_messages)

        assert "user@example.com" not in combined_logs
        assert "secret123" not in combined_logs
        assert "192.168.1.1" not in combined_logs
        assert "abc123def456" not in combined_logs
        assert "[EMAIL_REDACTED]" in combined_logs
        assert "[PASSWORD_REDACTED]" in combined_logs
        assert "[IP_REDACTED]" in combined_logs
        assert "[REDACTED]" in combined_logs


class TestLogScrubbingMiddleware:
    """Test LogScrubbingMiddleware functionality."""

    @pytest.mark.asyncio
    async def test_middleware_request_logging(self, caplog):
        """Test that middleware logs requests with scrubbing."""
        with caplog.at_level(logging.INFO):
            # Create mock request
            request = Mock(spec=Request)
            request.method = "POST"
            request.url.path = "/api/analyze"
            request.headers = {
                "X-Request-ID": "req_123",
                "Authorization": "Bearer secret123",
            }
            request.query_params = {}

            # Create mock response
            response = Mock(spec=Response)
            response.status_code = 200
            response.headers = {"Content-Type": "application/json"}

            # Create mock call_next
            async def mock_call_next(req):
                return response

            # Create middleware
            middleware = LogScrubbingMiddleware(Mock())

            # Call middleware
            result = await middleware.dispatch(request, mock_call_next)

        # Check that logs were generated
        assert len(caplog.records) > 0

        # Check that sensitive data was scrubbed
        log_messages = [record.getMessage() for record in caplog.records]
        combined_logs = " ".join(log_messages)

        assert "secret123" not in combined_logs
        assert "[REDACTED]" in combined_logs

    @pytest.mark.asyncio
    async def test_middleware_error_handling(self, caplog):
        """Test that middleware handles errors with scrubbed logging."""
        with caplog.at_level(logging.ERROR):
            # Create mock request
            request = Mock(spec=Request)
            request.method = "POST"
            request.url.path = "/api/analyze"
            request.headers = {
                "X-Request-ID": "req_123",
                "Authorization": "Bearer secret123",
            }
            request.query_params = {}

            # Create mock call_next that raises an exception
            async def mock_call_next(req):
                raise ValueError("Test error")

            # Create middleware
            middleware = LogScrubbingMiddleware(Mock())

            # Call middleware and expect exception
            with pytest.raises(ValueError):
                await middleware.dispatch(request, mock_call_next)

        # Check that error logs were generated
        assert len(caplog.records) > 0

        # Check that sensitive data was scrubbed in error logs
        log_messages = [record.getMessage() for record in caplog.records]
        combined_logs = " ".join(log_messages)

        assert "secret123" not in combined_logs
        assert "[REDACTED]" in combined_logs


class TestLogScrubbingIntegration:
    """Integration tests for log scrubbing functionality."""

    def test_comprehensive_pii_redaction(self):
        """Test comprehensive PII redaction across all patterns."""
        scrubber = LogScrubber()

        # Comprehensive test data with various PII types
        test_data = {
            "user_profile": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "(555) 123-4567",
                "ssn": "123-45-6789",
                "address": "123 Main St, Anytown, USA",
            },
            "payment_info": {
                "credit_card": "1234-5678-9012-3456",
                "billing_address": "456 Oak Ave, Somewhere, USA",
            },
            "technical_info": {
                "ip_address": "192.168.1.100",
                "mac_address": "00:1B:44:11:3A:B7",
                "api_key": "sk_live_abc123def456ghi789jkl012mno345pqr678",
                "jwt_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
            },
            "credentials": {
                "username": "johndoe",
                "password": "MySecretPassword123!",
                "secret": "super_secret_value",
                "key": "encryption_key_abc123",
            },
            "normal_data": {
                "preferences": ["option1", "option2"],
                "settings": {"theme": "dark", "language": "en"},
                "metadata": {"version": "1.0", "build": "123"},
            },
        }

        result = scrubber.scrub_dict(test_data)

        # Verify all PII was redacted
        assert result["user_profile"]["email"] == "[EMAIL_REDACTED]"
        assert result["user_profile"]["phone"] == "[PHONE_REDACTED]"
        assert result["user_profile"]["ssn"] == "[SSN_REDACTED]"

        assert result["payment_info"]["credit_card"] == "[CARD_REDACTED]"

        assert result["technical_info"]["ip_address"] == "[IP_REDACTED]"
        assert result["technical_info"]["mac_address"] == "[MAC_REDACTED]"
        assert result["technical_info"]["api_key"] == "[API_KEY_REDACTED]"
        assert result["technical_info"]["jwt_token"] == "[TOKEN_REDACTED]"

        assert result["credentials"]["password"] == "[REDACTED]"
        assert result["credentials"]["secret"] == "[REDACTED]"
        assert result["credentials"]["key"] == "[REDACTED]"

        # Verify normal data was preserved
        assert result["normal_data"]["preferences"] == ["option1", "option2"]
        assert result["normal_data"]["settings"]["theme"] == "dark"
        assert result["normal_data"]["metadata"]["version"] == "1.0"

        # Verify original sensitive data is not present
        original_json = str(test_data)
        result_json = str(result)

        assert "john.doe@example.com" not in result_json
        assert "(555) 123-4567" not in result_json
        assert "123-45-6789" not in result_json
        assert "1234-5678-9012-3456" not in result_json
        assert "192.168.1.100" not in result_json
        assert "MySecretPassword123!" not in result_json
        assert "super_secret_value" not in result_json
