"""
Custom exceptions for the Risk Scoring Framework.
"""


class RiskCalculationError(Exception):
    """Exception raised when risk calculation encounters an error."""
    pass


class ConfigurationError(Exception):
    """Exception raised when configuration is invalid."""
    pass


class ValidationError(Exception):
    """Exception raised when input validation fails."""
    pass


class CacheError(Exception):
    """Exception raised when cache operations fail."""
    pass
