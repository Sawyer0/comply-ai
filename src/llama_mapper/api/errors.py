"""
Canonical error utilities for Mapper API.
"""
from __future__ import annotations


from .models import ErrorBody

# Canonical mapping per service-contracts Section 8
ERROR_HTTP_MAP = {
    "INVALID_REQUEST": 400,
    "POLICY_NOT_FOUND": 400,
    "UNAUTHORIZED": 401,
    "INSUFFICIENT_RBAC": 403,
    "RATE_LIMITED": 429,
    "PARTIAL_COVERAGE": 206,
    "REQUEST_TIMEOUT": 408,
    "DETECTOR_OVERLOADED": 429,
    "ALL_DETECTORS_UNAVAILABLE": 502,
    "DETECTOR_COMMUNICATION_FAILED": 502,
    "AGGREGATION_FAILED": 500,
    "INTERNAL_ERROR": 500,
}

# Retry guidance (true means client may retry safely)
RETRYABLE = {
    "RATE_LIMITED": True,
    "REQUEST_TIMEOUT": True,
    "DETECTOR_OVERLOADED": True,
    "ALL_DETECTORS_UNAVAILABLE": True,
    "DETECTOR_COMMUNICATION_FAILED": True,
}


def http_status_for(code: str) -> int:
    return int(ERROR_HTTP_MAP.get(code, 500))


def is_retryable(code: str) -> bool:
    return bool(RETRYABLE.get(code, False))


def build_error_body(request_id: str | None, code: str, message: str | None = None) -> ErrorBody:
    return ErrorBody(
        error_code=code,
        message=message,
        request_id=request_id,
        retryable=is_retryable(code),
    )
