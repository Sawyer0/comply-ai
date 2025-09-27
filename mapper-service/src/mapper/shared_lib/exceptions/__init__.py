"""Shared exceptions for microservice communication."""

from .base import (
    BaseServiceException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    ConfigurationError,
)

__all__ = [
    "BaseServiceException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "ServiceUnavailableError",
    "TimeoutError",
    "ConfigurationError",
]
