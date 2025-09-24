"""Interfaces and abstractions for cost monitoring system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

# Import types using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .metrics_collector import CostBreakdown, CostMetrics, ResourceUsage


class ResourceMonitor(ABC):
    """Abstract interface for resource monitoring."""

    @abstractmethod
    async def get_current_usage(self) -> "ResourceUsage":
        """Get current resource usage metrics."""
        pass

    @abstractmethod
    async def get_historical_usage(
        self, start_time: datetime, end_time: datetime
    ) -> List["ResourceUsage"]:
        """Get historical resource usage for a time period."""
        pass


class CostCalculator(ABC):
    """Abstract interface for cost calculation."""

    @abstractmethod
    def calculate_cost(self, usage: "ResourceUsage") -> "CostMetrics":
        """Calculate cost metrics for given resource usage."""
        pass

    @abstractmethod
    def calculate_breakdown(self, usage_list: List["ResourceUsage"]) -> "CostBreakdown":
        """Calculate cost breakdown for a list of usage metrics."""
        pass


class MetricsStorage(ABC):
    """Abstract interface for metrics storage."""

    @abstractmethod
    async def store_metrics(self, metrics: "CostMetrics") -> None:
        """Store cost metrics."""
        pass

    @abstractmethod
    async def get_metrics(
        self, start_time: datetime, end_time: datetime, tenant_id: Optional[str] = None
    ) -> List["CostMetrics"]:
        """Get metrics for a time period."""
        pass

    @abstractmethod
    async def cleanup_old_metrics(self, retention_days: int) -> None:
        """Clean up old metrics based on retention policy."""
        pass

    @abstractmethod
    def get_recent_metrics(self, count: int) -> List["CostMetrics"]:
        """Get recent metrics for internal use."""
        pass


class AlertManager(ABC):
    """Abstract interface for alert management."""

    @abstractmethod
    async def create_alert(
        self, alert_type: str, message: str, severity: str, metadata: Dict[str, Any]
    ) -> None:
        """Create and send an alert."""
        pass

    @abstractmethod
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts."""
        pass


class ScalingExecutor(ABC):
    """Abstract interface for scaling operations."""

    @abstractmethod
    async def scale_up(
        self, resource_type: str, target_instances: int, metadata: Dict[str, Any]
    ) -> bool:
        """Scale up resources."""
        pass

    @abstractmethod
    async def scale_down(
        self, resource_type: str, target_instances: int, metadata: Dict[str, Any]
    ) -> bool:
        """Scale down resources."""
        pass

    @abstractmethod
    async def get_current_instances(self, resource_type: str) -> int:
        """Get current number of instances for a resource type."""
        pass
