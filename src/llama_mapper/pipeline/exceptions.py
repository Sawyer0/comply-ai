"""
Pipeline Exception Hierarchy

Clean exception hierarchy for pipeline operations.
"""


class PipelineError(Exception):
    """Base exception for pipeline operations."""
    pass


class StageError(PipelineError):
    """Exception raised during stage execution."""
    pass


class ConfigurationError(PipelineError):
    """Exception raised for configuration issues."""
    pass


class DependencyError(PipelineError):
    """Exception raised for dependency resolution issues."""
    pass


class ValidationError(PipelineError):
    """Exception raised for validation failures."""
    pass


class RegistryError(PipelineError):
    """Exception raised for registry operations."""
    pass


class MonitoringError(PipelineError):
    """Exception raised for monitoring operations."""
    pass