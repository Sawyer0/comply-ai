"""
Quality Alert Models

This module defines data models for quality alerts.
Follows SRP by focusing solely on data structure definitions.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class QualityAlert:
    """Quality alert representation."""

    id: str
    alert_type: str
    severity: AlertSeverity
    title: str
    description: str
    source_component: str
    metrics: Dict[str, Any]
    threshold_violated: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    notification_sent: bool = False


@dataclass
class ThresholdConfig:
    """Threshold configuration for alerts."""

    metric_name: str
    component: str
    threshold_type: str  # greater_than, less_than, equals, etc.
    threshold_value: float
    severity: AlertSeverity
    description: Optional[str] = None


@dataclass
class AlertRule:
    """Composite alert rule definition."""

    rule_name: str
    component: str
    description: str
    severity: AlertSeverity
    conditions: Dict[str, Any]
