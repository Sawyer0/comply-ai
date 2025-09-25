"""
Multi-tenancy models for the Analysis Service.

This module defines tenant-related data models, configurations, and resource quotas.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TenantStatus(Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TRIAL = "trial"


class ResourceType(Enum):
    """Resource type enumeration for quotas."""

    ANALYSIS_REQUESTS = "analysis_requests"
    BATCH_REQUESTS = "batch_requests"
    STORAGE_MB = "storage_mb"
    CPU_MINUTES = "cpu_minutes"
    ML_INFERENCE_CALLS = "ml_inference_calls"


@dataclass
class ResourceQuota:
    """Resource quota configuration for a tenant."""

    resource_type: ResourceType
    limit: int
    period_hours: int = 24  # Quota period in hours
    current_usage: int = 0
    reset_at: Optional[datetime] = None

    def __post_init__(self):
        if self.reset_at is None:
            self.reset_at = datetime.now(timezone.utc)

    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.current_usage >= self.limit

    def remaining(self) -> int:
        """Get remaining quota."""
        return max(0, self.limit - self.current_usage)

    def reset_if_expired(self) -> bool:
        """Reset quota if period has expired."""
        if datetime.now(timezone.utc) >= self.reset_at:
            self.current_usage = 0
            self.reset_at = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return True
        return False


@dataclass
class TenantConfiguration:
    """Tenant-specific configuration."""

    tenant_id: str
    name: str
    status: TenantStatus = TenantStatus.ACTIVE

    # Analysis configuration
    default_confidence_threshold: float = 0.8
    enable_ml_analysis: bool = True
    enable_statistical_analysis: bool = True
    enable_pattern_recognition: bool = True

    # Quality settings
    quality_alert_threshold: float = 0.7
    enable_quality_monitoring: bool = True

    # Privacy settings
    enable_content_scrubbing: bool = True
    log_level: str = "INFO"

    # Custom analysis engines
    custom_engines: List[str] = field(default_factory=list)

    # Framework preferences
    preferred_frameworks: List[str] = field(
        default_factory=lambda: ["SOC2", "ISO27001"]
    )

    # Resource quotas
    quotas: Dict[ResourceType, ResourceQuota] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "status": self.status.value,
            "default_confidence_threshold": self.default_confidence_threshold,
            "enable_ml_analysis": self.enable_ml_analysis,
            "enable_statistical_analysis": self.enable_statistical_analysis,
            "enable_pattern_recognition": self.enable_pattern_recognition,
            "quality_alert_threshold": self.quality_alert_threshold,
            "enable_quality_monitoring": self.enable_quality_monitoring,
            "enable_content_scrubbing": self.enable_content_scrubbing,
            "log_level": self.log_level,
            "custom_engines": self.custom_engines,
            "preferred_frameworks": self.preferred_frameworks,
            "quotas": {
                rt.value: {
                    "limit": quota.limit,
                    "period_hours": quota.period_hours,
                    "current_usage": quota.current_usage,
                    "reset_at": quota.reset_at.isoformat() if quota.reset_at else None,
                }
                for rt, quota in self.quotas.items()
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenantConfiguration":
        """Create from dictionary."""
        quotas = {}
        for rt_str, quota_data in data.get("quotas", {}).items():
            rt = ResourceType(rt_str)
            quota = ResourceQuota(
                resource_type=rt,
                limit=quota_data["limit"],
                period_hours=quota_data["period_hours"],
                current_usage=quota_data["current_usage"],
                reset_at=(
                    datetime.fromisoformat(quota_data["reset_at"])
                    if quota_data.get("reset_at")
                    else None
                ),
            )
            quotas[rt] = quota

        return cls(
            tenant_id=data["tenant_id"],
            name=data["name"],
            status=TenantStatus(data.get("status", "active")),
            default_confidence_threshold=data.get("default_confidence_threshold", 0.8),
            enable_ml_analysis=data.get("enable_ml_analysis", True),
            enable_statistical_analysis=data.get("enable_statistical_analysis", True),
            enable_pattern_recognition=data.get("enable_pattern_recognition", True),
            quality_alert_threshold=data.get("quality_alert_threshold", 0.7),
            enable_quality_monitoring=data.get("enable_quality_monitoring", True),
            enable_content_scrubbing=data.get("enable_content_scrubbing", True),
            log_level=data.get("log_level", "INFO"),
            custom_engines=data.get("custom_engines", []),
            preferred_frameworks=data.get("preferred_frameworks", ["SOC2", "ISO27001"]),
            quotas=quotas,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class TenantAnalytics(BaseModel):
    """Tenant analytics data model."""

    tenant_id: str = Field(..., description="Tenant identifier")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")

    # Request metrics
    total_requests: int = Field(0, description="Total analysis requests")
    successful_requests: int = Field(0, description="Successful requests")
    failed_requests: int = Field(0, description="Failed requests")

    # Performance metrics
    avg_response_time_ms: float = Field(0.0, description="Average response time")
    p95_response_time_ms: float = Field(
        0.0, description="95th percentile response time"
    )

    # Quality metrics
    avg_confidence_score: float = Field(0.0, description="Average confidence score")
    low_confidence_count: int = Field(0, description="Low confidence results count")

    # Resource usage
    cpu_minutes_used: float = Field(0.0, description="CPU minutes consumed")
    storage_mb_used: float = Field(0.0, description="Storage MB used")
    ml_inference_calls: int = Field(0, description="ML inference calls made")

    # Analysis breakdown
    pattern_recognition_count: int = Field(
        0, description="Pattern recognition analyses"
    )
    risk_scoring_count: int = Field(0, description="Risk scoring analyses")
    compliance_mapping_count: int = Field(0, description="Compliance mapping analyses")

    # Framework usage
    framework_usage: Dict[str, int] = Field(
        default_factory=dict, description="Usage by framework"
    )

    # Error breakdown
    error_types: Dict[str, int] = Field(
        default_factory=dict, description="Error counts by type"
    )


class TenantRequest(BaseModel):
    """Request model for tenant operations."""

    name: str = Field(..., description="Tenant name")
    status: str = Field("active", description="Tenant status")
    default_confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
    enable_ml_analysis: bool = Field(True)
    enable_statistical_analysis: bool = Field(True)
    enable_pattern_recognition: bool = Field(True)
    quality_alert_threshold: float = Field(0.7, ge=0.0, le=1.0)
    enable_quality_monitoring: bool = Field(True)
    enable_content_scrubbing: bool = Field(True)
    log_level: str = Field("INFO")
    custom_engines: List[str] = Field(default_factory=list)
    preferred_frameworks: List[str] = Field(
        default_factory=lambda: ["SOC2", "ISO27001"]
    )


class QuotaRequest(BaseModel):
    """Request model for setting resource quotas."""

    resource_type: str = Field(..., description="Resource type")
    limit: int = Field(..., ge=0, description="Quota limit")
    period_hours: int = Field(24, ge=1, description="Quota period in hours")
