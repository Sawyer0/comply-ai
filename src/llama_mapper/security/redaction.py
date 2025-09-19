"""
PII redaction utilities for request payloads and logs.

Implements conservative regex-based redaction for common PII and secrets.
Designed to be used both for sanitizing request payloads before logging
and as a last-mile safeguard around any free-form strings.
"""

from __future__ import annotations

import re
from dataclasses import is_dataclass, replace
from typing import Any, Dict, Mapping, Sequence, cast

# Regex patterns for common PII and secrets
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b"
)
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CC_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
# JWT-like tokens: header.payload.signature (base64-ish)
JWT_RE = re.compile(r"\b[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\b")
# API keys / tokens (hex/base64-ish 20+ chars)
API_KEY_RE = re.compile(r"\b(?:[A-Fa-f0-9]{32,64}|[A-Za-z0-9-_]{24,})\b")

SENSITIVE_KEYS = {
    "raw_input",
    "detector_input",
    "content",
    "text",
    "message_content",
    "user_input",
    "prompt",
    "response_text",
    "raw_output",
    "original_text",
    "secret",
    "token",
    "api_key",
    "authorization",
    "password",
    "key",
}

REPLACERS = [
    (EMAIL_RE, "[EMAIL_REDACTED]"),
    (PHONE_RE, "[PHONE_REDACTED]"),
    (IP_RE, "[IP_REDACTED]"),
    (SSN_RE, "[SSN_REDACTED]"),
    (CC_RE, "[CC_REDACTED]"),
    (JWT_RE, "[TOKEN_REDACTED]"),
    (API_KEY_RE, "[KEY_REDACTED]"),
]


def redact_text(text: str) -> str:
    """Redact common PII and secrets from a string.

    Args:
        text: Input text
    Returns:
        Redacted text
    """
    redacted = text
    for pattern, repl in REPLACERS:
        redacted = pattern.sub(repl, redacted)
    return redacted


def _redact_value(value: Any) -> Any:
    """Redact a value recursively if it is a container or string."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return redact_dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_value(v) for v in value]  # type: ignore[func-returns-value]
    if is_dataclass(value):
        try:
            # Create a redacted copy of dataclass fields
            fields = {f.name: _redact_value(getattr(value, f.name)) for f in value.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            # Cast to Any to satisfy type checker; runtime type remains the same dataclass
            return cast(Any, replace(value, **fields))  # type: ignore[type-var]
        except Exception:
            return value
    return value


def redact_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Redact values in a dict conservatively.

    Keys that are suspicious are replaced with placeholders; otherwise values are redacted.
    """
    redacted: Dict[str, Any] = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if k_lower in SENSITIVE_KEYS:
            redacted[k] = "[REDACTED]"
            continue
        redacted[k] = _redact_value(v)
    return redacted


# Optional helper for FastAPI request models without importing pydantic models at import time


def redact_request_model(request: Any) -> Any:
    """Return a shallow redacted copy of a DetectorRequest-like object.

    The object is expected to have attributes: detector, output, metadata, tenant_id.
    """
    try:
        obj = {
            "detector": getattr(request, "detector", None),
            "output": "[REDACTED]" if hasattr(request, "output") else None,
            "metadata": redact_dict(getattr(request, "metadata", {}) or {}),
            "tenant_id": getattr(request, "tenant_id", None),
        }
        return obj
    except Exception:
        return {
            "detector": None,
            "output": "[REDACTED]",
            "metadata": {},
            "tenant_id": None,
        }
