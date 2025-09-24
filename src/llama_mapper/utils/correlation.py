"""
Correlation ID management for distributed tracing.

Provides context-aware correlation ID tracking across all services and requests.
"""

import uuid
from contextvars import ContextVar
from typing import Optional

# Context variable for request correlation
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def get_correlation_id() -> str:
    """
    Get current correlation ID or generate a new one.
    
    Returns:
        str: Current correlation ID
    """
    current_id = correlation_id.get()
    if current_id is None:
        current_id = str(uuid.uuid4())
        correlation_id.set(current_id)
    return current_id


def set_correlation_id(corr_id: str) -> None:
    """
    Set correlation ID for current context.
    
    Args:
        corr_id: Correlation ID to set
    """
    correlation_id.set(corr_id)


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID and set it in context.
    
    Returns:
        str: New correlation ID
    """
    new_id = str(uuid.uuid4())
    correlation_id.set(new_id)
    return new_id


def clear_correlation_id() -> None:
    """Clear correlation ID from current context."""
    correlation_id.set(None)