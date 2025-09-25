"""
Security-related exceptions for the Analysis Service.
"""


class SecurityError(Exception):
    """Base security-related error."""

    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.error_code = error_code


class AuthenticationError(SecurityError):
    """Authentication failed."""

    pass


class AuthorizationError(SecurityError):
    """Authorization failed."""

    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded."""

    pass


class ValidationError(SecurityError):
    """Input validation failed."""

    pass


class ContentSecurityError(SecurityError):
    """Content security violation."""

    pass
