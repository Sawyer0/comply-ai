"""
Security module for the Analysis Service.

Provides comprehensive security features through specialized components:
- Authentication: API key and JWT token management
- Authorization: Permission checking and access control
- Rate Limiting: Token bucket rate limiting
- Content Scanning: Malicious content detection
- Audit Logging: Security event tracking
"""

from .audit_logger import AuditLogger
from .authentication import AuthenticationManager
from .authorization import AuthorizationManager
from .config import SecurityConfig
from .content_scanner import ContentScanner
from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    ContentSecurityError,
    RateLimitError,
    SecurityError,
    ValidationError,
)
from .manager import SecurityManager
from .rate_limiter import RateLimiter

__all__ = [
    # Main components
    "SecurityManager",
    "SecurityConfig",
    # Specialized managers
    "AuthenticationManager",
    "AuthorizationManager",
    "RateLimiter",
    "ContentScanner",
    "AuditLogger",
    # Exceptions
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "ValidationError",
    "ContentSecurityError",
]
