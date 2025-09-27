"""Common enums and types shared across all services.

This module contains ONLY shared enums and basic types that are used across multiple services.
Single Responsibility: Provide common type definitions to avoid duplication.
"""

from enum import Enum


class ProcessingMode(str, Enum):
    """Processing modes for requests."""

    STANDARD = "standard"
    FAST = "fast"
    THOROUGH = "thorough"
    COMPREHENSIVE = "comprehensive"


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(str, Enum):
    """Job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelStatus(str, Enum):
    """Model status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ComplianceStatus(str, Enum):
    """Compliance status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"


class AlertStatus(str, Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    """Alert severity."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Export all common types
__all__ = [
    "ProcessingMode",
    "HealthStatus",
    "Severity",
    "RiskLevel",
    "JobStatus",
    "ModelStatus",
    "ComplianceStatus",
    "AlertStatus",
    "AlertSeverity",
]
