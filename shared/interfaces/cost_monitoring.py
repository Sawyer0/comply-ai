"""
Shared Cost Monitoring Interfaces

This module provides shared interfaces and models for cost monitoring
across all microservices, based on the existing llama_mapper implementation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from decimal import Decimal
from enum import Enum


class CostCategory(str, Enum):
    """Cost category enumeration"""

    COMPUTE = "compute"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    MODEL_INFERENCE = "model_inference"
    TRAINING = "training"
    API_CALLS = "api_calls"


@dataclass
class ResourceUsage:
    """Resource usage metrics."""

    timestamp: datetime
    tenant_id: Optional[str] = None
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 0.0
    network_gb: float = 0.0
    api_calls: int = 0
    custom_metrics: Optional[Dict[str, float]] = None


@dataclass
class CostMetrics:
    """Cost calculation results."""

    resource_type: str
    usage: ResourceUsage
    cost_per_unit: Dict[str, float]
    total_cost: float
    currency: str = "USD"
    breakdown: Optional[Dict[str, float]] = None


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""

    compute_cost: float = 0.0
    memory_cost: float = 0.0
    storage_cost: float = 0.0
    network_cost: float = 0.0
    api_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


@dataclass
class CostEvent:
    """Individual cost tracking event"""

    tenant_id: str
    event_type: str
    resource_id: str
    cost_amount: Decimal
    currency: str
    timestamp: datetime
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "event_type": self.event_type,
            "resource_id": self.resource_id,
            "cost_amount": str(self.cost_amount),
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
        }


class ICostCalculator(ABC):
    """Interface for cost calculation."""

    @abstractmethod
    def calculate_cost(self, usage: ResourceUsage) -> CostMetrics:
        """Calculate cost metrics for given resource usage."""
        pass

    @abstractmethod
    def calculate_breakdown(self, usage_list: List[ResourceUsage]) -> CostBreakdown:
        """Calculate cost breakdown for a list of usage metrics."""
        pass


class ICostMonitor(ABC):
    """Interface for cost monitoring."""

    @abstractmethod
    async def track_inference_cost(
        self,
        tenant_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        inference_time_ms: float,
        gpu_type: str = "v100",
    ) -> CostEvent:
        """Track cost for model inference."""
        pass

    @abstractmethod
    async def track_training_cost(
        self,
        tenant_id: str,
        training_job_id: str,
        training_type: str,
        duration_hours: float,
        gpu_count: int = 1,
        gpu_type: str = "v100",
    ) -> CostEvent:
        """Track cost for model training."""
        pass

    @abstractmethod
    async def track_storage_cost(
        self,
        tenant_id: str,
        storage_type: str,
        size_gb: float,
        duration_hours: float = 1.0,
    ) -> CostEvent:
        """Track storage costs."""
        pass

    @abstractmethod
    async def track_api_cost(
        self,
        tenant_id: str,
        endpoint: str,
        request_count: int = 1,
        validation_requests: int = 0,
    ) -> CostEvent:
        """Track API request costs."""
        pass

    @abstractmethod
    async def get_cost_analytics(
        self, tenant_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive cost analytics."""
        pass

    @abstractmethod
    async def get_performance_analytics(
        self, tenant_id: str, days: int = 7
    ) -> Dict[str, Any]:
        """Get performance analytics and trends."""
        pass


class ICostMonitoringSystem(ABC):
    """Interface for the complete cost monitoring system."""

    @abstractmethod
    async def start(self) -> None:
        """Start the cost monitoring system."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the cost monitoring system."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the system is running."""
        pass

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of the cost monitoring system."""
        pass

    @abstractmethod
    def get_cost_breakdown(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None,
    ) -> Any:
        """Get cost breakdown for a time period."""
        pass

    @abstractmethod
    def get_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get cost trends."""
        pass

    @abstractmethod
    def get_optimization_recommendations(
        self,
        category: Optional[str] = None,
        priority_min: int = 1,
        tenant_id: Optional[str] = None,
    ) -> List[Any]:
        """Get optimization recommendations."""
        pass
