import pytest

from src.llama_mapper.api.errors import build_error_body, http_status_for, is_retryable


def test_error_helper_http_status_and_retryable():
    # Known codes
    assert http_status_for("INVALID_REQUEST") == 400
    assert http_status_for("REQUEST_TIMEOUT") == 408
    assert http_status_for("INTERNAL_ERROR") == 500

    assert is_retryable("REQUEST_TIMEOUT") is True
    assert is_retryable("INVALID_REQUEST") is False


def test_build_error_body_contents():
    eb = build_error_body("req-1", "REQUEST_TIMEOUT", "Timed out")
    assert eb.error_code == "REQUEST_TIMEOUT"
    assert eb.request_id == "req-1"
    assert eb.retryable is True
    assert eb.message == "Timed out"
