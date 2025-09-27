"""Correlation ID utilities for request tracing."""

import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable for correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str:
    """Get the current correlation ID or generate a new one."""
    current_id = _correlation_id.get()
    if current_id is None:
        current_id = str(uuid.uuid4())
        _correlation_id.set(current_id)
    return current_id


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    if not correlation_id or not isinstance(correlation_id, str):
        raise ValueError("correlation_id must be a non-empty string")
    _correlation_id.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return str(uuid.uuid4())


def clear_correlation_id() -> None:
    """Clear the correlation ID from the current context."""
    _correlation_id.set(None)
