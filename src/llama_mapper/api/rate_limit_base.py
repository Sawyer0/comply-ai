"""Shared types for rate limiting backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IdentityKind = Literal["api_key", "tenant", "ip"]


@dataclass
class RateLimitRequest:
    """Configuration for a rate limit request.

    Groups related rate limiting parameters to reduce function argument count.
    """
    endpoint: str
    identity: str
    limit: int
    window: int
    cost: int = 1


@dataclass
class AllowResult:
    """Result of a rate limit check operation.

    Attributes:
        allowed: Whether the request is allowed based on current rate limits.
        remaining: Number of requests remaining in the current window.
        limit: Total limit for the current window.
        reset_seconds: Seconds until the rate limit window resets.
        identity_kind: Type of identity being rate limited.
    """
    allowed: bool
    remaining: int
    limit: int
    reset_seconds: float
    identity_kind: IdentityKind


class RateLimiterBackend:
    """Abstract base class for rate limiting backends.

    This interface defines the contract for implementing rate limiting functionality.
    Concrete implementations should inherit from this class and provide the actual
    rate limiting logic.
    """

    async def allow(self, request: RateLimitRequest) -> AllowResult:
        """Check if a request is allowed based on rate limits.

        Args:
            request: Configuration object containing rate limit parameters.

        Returns:
            AllowResult containing the rate limit check result.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented.
        """
        raise NotImplementedError

    async def is_healthy(self) -> bool:
        """Check if the rate limiter backend is healthy and operational.

        Returns:
            True if the backend is healthy, False otherwise.

        Raises:
            NotImplementedError: This is an abstract method that must be implemented.
        """
        raise NotImplementedError
