"""Cost monitoring metrics collector for tracking resource usage and costs."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from ...logging import get_logger
from .interfaces import ResourceMonitor, CostCalculator, MetricsStorage, AlertManager
from .constants import DEFAULT_RECENT_METRICS_COUNT


@dataclass
class ResourceUsage:
    """Resource usage metrics for cost calculation."""
    
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    storage_gb: float = 0.0
    network_gb: float = 0.0
    api_calls: int = 0
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CostMetrics:
    """Cost metrics for a specific resource or service."""
    
    resource_type: str
    usage: ResourceUsage
    cost_per_unit: Dict[str, float]  # e.g., {"cpu": 0.05, "memory": 0.01}
    total_cost: float
    currency: str = "USD"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CostBreakdown(BaseModel):
    """Detailed cost breakdown by component."""
    
    compute_cost: float = Field(description="Cost for compute resources (CPU, GPU)")
    memory_cost: float = Field(description="Cost for memory usage")
    storage_cost: float = Field(description="Cost for storage usage")
    network_cost: float = Field(description="Cost for network usage")
    api_cost: float = Field(description="Cost for API calls")
    total_cost: float = Field(description="Total cost")
    currency: str = Field(default="USD", description="Currency code")
    period_start: datetime = Field(description="Start of the cost period")
    period_end: datetime = Field(description="End of the cost period")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID for multi-tenant costs")


class CostAlert(BaseModel):
    """Cost alert configuration and status."""
    
    alert_id: str = Field(description="Unique alert identifier")
    alert_type: str = Field(description="Type of cost alert")
    threshold: float = Field(description="Cost threshold that triggers the alert")
    current_cost: float = Field(description="Current cost value")
    currency: str = Field(default="USD", description="Currency code")
    severity: str = Field(description="Alert severity (low, medium, high, critical)")
    message: str = Field(description="Alert message")
    triggered_at: datetime = Field(description="When the alert was triggered")
    tenant_id: Optional[str] = Field(default=None, description="Tenant ID")
    resolved: bool = Field(default=False, description="Whether the alert is resolved")


class CostMonitoringConfig(BaseModel):
    """Configuration for cost monitoring."""
    
    enabled: bool = Field(default=True, description="Enable cost monitoring")
    collection_interval_seconds: int = Field(default=60, description="Metrics collection interval")
    retention_days: int = Field(default=90, description="Data retention period")
    cost_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "daily_budget": 100.0,
            "hourly_budget": 10.0,
            "api_call_cost": 0.001,
            "compute_cost_per_hour": 0.50,
        },
        description="Cost thresholds for alerts"
    )
    currency: str = Field(default="USD", description="Default currency")
    cost_per_unit: Dict[str, float] = Field(
        default_factory=lambda: {
            "cpu_per_hour": 0.05,
            "memory_per_gb_hour": 0.01,
            "gpu_per_hour": 2.00,
            "storage_per_gb_month": 0.10,
            "network_per_gb": 0.05,
            "api_call": 0.001,
        },
        description="Cost per unit for different resources"
    )


class CostMetricsCollector:
    """Collects and processes cost-related metrics."""
    
    def __init__(
        self, 
        config: CostMonitoringConfig,
        resource_monitor: Optional[ResourceMonitor] = None,
        cost_calculator: Optional[CostCalculator] = None,
        metrics_storage: Optional[MetricsStorage] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        self.config = config
        self.logger = get_logger(__name__)
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # Dependency injection
        self.resource_monitor = resource_monitor
        self.cost_calculator = cost_calculator
        self.metrics_storage = metrics_storage
        self.alert_manager = alert_manager
        
        # Fallback to in-memory storage if not provided
        if not self.metrics_storage:
            self.metrics_storage = InMemoryMetricsStorage()
        
        # Fallback to mock implementations if not provided
        if not self.resource_monitor:
            from .resource_monitor import MockResourceMonitor
            self.resource_monitor = MockResourceMonitor()
        
        if not self.cost_calculator:
            from .cost_calculator import StandardCostCalculator
            self.cost_calculator = StandardCostCalculator(config.cost_per_unit)
    
    async def start(self) -> None:
        """Start the cost metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Cost metrics collector started")
    
    async def stop(self) -> None:
        """Stop the cost metrics collection."""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            finally:
                self._collection_task = None
        self.logger.info("Cost metrics collector stopped")
    
    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.collection_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in cost metrics collection", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _collect_metrics(self) -> None:
        """Collect current cost metrics."""
        try:
            # Collect resource usage
            usage = await self.resource_monitor.get_current_usage()
            
            # Calculate costs
            cost_metrics = self.cost_calculator.calculate_cost(usage)
            
            # Store metrics
            await self.metrics_storage.store_metrics(cost_metrics)
            
            # Check for alerts
            await self._check_cost_alerts(cost_metrics)
            
            # Cleanup old metrics
            await self.metrics_storage.cleanup_old_metrics(self.config.retention_days)
            
        except Exception as e:
            self.logger.error("Failed to collect cost metrics", error=str(e))
    
    
    async def _check_cost_alerts(self, cost_metrics: CostMetrics) -> None:
        """Check if cost metrics trigger any alerts."""
        thresholds = self.config.cost_thresholds
        
        # Check daily budget
        daily_cost = await self._get_daily_cost()
        if daily_cost > thresholds["daily_budget"]:
            await self._create_alert(
                alert_type="daily_budget_exceeded",
                threshold=thresholds["daily_budget"],
                current_cost=daily_cost,
                severity="high",
                message=f"Daily budget exceeded: ${daily_cost:.2f} > ${thresholds['daily_budget']:.2f}",
            )
        
        # Check hourly budget
        hourly_cost = await self._get_hourly_cost()
        if hourly_cost > thresholds["hourly_budget"]:
            await self._create_alert(
                alert_type="hourly_budget_exceeded",
                threshold=thresholds["hourly_budget"],
                current_cost=hourly_cost,
                severity="medium",
                message=f"Hourly budget exceeded: ${hourly_cost:.2f} > ${thresholds['hourly_budget']:.2f}",
            )
        
        # Check API call costs
        if cost_metrics.usage.api_calls * cost_metrics.cost_per_unit["api_call"] > thresholds["api_call_cost"]:
            await self._create_alert(
                alert_type="api_cost_high",
                threshold=thresholds["api_call_cost"],
                current_cost=cost_metrics.usage.api_calls * cost_metrics.cost_per_unit["api_call"],
                severity="low",
                message=f"API call costs high: ${cost_metrics.usage.api_calls * cost_metrics.cost_per_unit['api_call']:.2f}",
            )
    
    async def _create_alert(
        self,
        alert_type: str,
        threshold: float,
        current_cost: float,
        severity: str,
        message: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        """Create a new cost alert."""
        alert = CostAlert(
            alert_id=f"{alert_type}_{int(time.time())}",
            alert_type=alert_type,
            threshold=threshold,
            current_cost=current_cost,
            currency=self.config.currency,
            severity=severity,
            message=message,
            triggered_at=datetime.now(timezone.utc),
            tenant_id=tenant_id,
        )
        
        self._alerts.append(alert)
        self.logger.warning("Cost alert triggered", alert=alert.model_dump())
    
    async def _get_daily_cost(self) -> float:
        """Get total cost for the current day."""
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_metrics = await self.metrics_storage.get_metrics(start_of_day, now)
        return sum(m.total_cost for m in daily_metrics)
    
    async def _get_hourly_cost(self) -> float:
        """Get total cost for the current hour."""
        now = datetime.now(timezone.utc)
        start_of_hour = now.replace(minute=0, second=0, microsecond=0)
        
        hourly_metrics = await self.metrics_storage.get_metrics(start_of_hour, now)
        return sum(m.total_cost for m in hourly_metrics)
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics based on retention policy."""
        # This method is now handled by the metrics storage
        pass
    
    async def get_cost_breakdown(
        self,
        start_time: datetime,
        end_time: datetime,
        tenant_id: Optional[str] = None,
    ) -> CostBreakdown:
        """Get cost breakdown for a specific time period."""
        # Get metrics from storage
        period_metrics = await self.metrics_storage.get_metrics(start_time, end_time, tenant_id)
        
        if not period_metrics:
            return CostBreakdown(
                period_start=start_time,
                period_end=end_time,
                tenant_id=tenant_id,
            )
        
        # Calculate breakdown
        compute_cost = sum(
            m.usage.cpu_cores * m.cost_per_unit["cpu_per_hour"] / 3600 +
            m.usage.gpu_count * m.cost_per_unit["gpu_per_hour"] / 3600
            for m in period_metrics
        )
        
        memory_cost = sum(
            m.usage.memory_gb * m.cost_per_unit["memory_per_gb_hour"] / 3600 +
            m.usage.gpu_memory_gb * m.cost_per_unit["gpu_per_hour"] / 3600
            for m in period_metrics
        )
        
        storage_cost = sum(
            m.usage.storage_gb * m.cost_per_unit["storage_per_gb_month"] / (30 * 24 * 3600)
            for m in period_metrics
        )
        
        network_cost = sum(
            m.usage.network_gb * m.cost_per_unit["network_per_gb"]
            for m in period_metrics
        )
        
        api_cost = sum(
            m.usage.api_calls * m.cost_per_unit["api_call"]
            for m in period_metrics
        )
        
        total_cost = compute_cost + memory_cost + storage_cost + network_cost + api_cost
        
        return CostBreakdown(
            compute_cost=compute_cost,
            memory_cost=memory_cost,
            storage_cost=storage_cost,
            network_cost=network_cost,
            api_cost=api_cost,
            total_cost=total_cost,
            currency=self.config.currency,
            period_start=start_time,
            period_end=end_time,
            tenant_id=tenant_id,
        )
    
    def get_active_alerts(self, tenant_id: Optional[str] = None) -> List[CostAlert]:
        """Get active (unresolved) alerts."""
        return [
            alert for alert in self._alerts
            if not alert.resolved and (tenant_id is None or alert.tenant_id == tenant_id)
        ]
    
    async def get_cost_trends(self, days: int = 7) -> Dict[str, List[float]]:
        """Get cost trends over the specified number of days."""
        end_time = datetime.now(timezone.utc)
        start_time = end_time.replace(day=end_time.day - days)
        
        # Get metrics from storage
        metrics = await self.metrics_storage.get_metrics(start_time, end_time)
        
        # Group metrics by day
        daily_costs = {}
        for metric in metrics:
            day_key = metric.timestamp.date().isoformat()
            if day_key not in daily_costs:
                daily_costs[day_key] = 0.0
            daily_costs[day_key] += metric.total_cost
        
        # Convert to lists
        dates = sorted(daily_costs.keys())
        costs = [daily_costs[date] for date in dates]
        
        return {
            "dates": dates,
            "costs": costs,
            "total_cost": sum(costs),
            "average_daily_cost": sum(costs) / len(costs) if costs else 0.0,
        }


class InMemoryMetricsStorage(MetricsStorage):
    """In-memory implementation of metrics storage."""
    
    def __init__(self):
        self._metrics: List[CostMetrics] = []
    
    async def store_metrics(self, metrics: CostMetrics) -> None:
        """Store cost metrics in memory."""
        self._metrics.append(metrics)
    
    async def get_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime,
        tenant_id: Optional[str] = None
    ) -> List[CostMetrics]:
        """Get metrics for a time period."""
        filtered_metrics = []
        for metric in self._metrics:
            if start_time <= metric.timestamp <= end_time:
                if tenant_id is None or getattr(metric, 'tenant_id', None) == tenant_id:
                    filtered_metrics.append(metric)
        return filtered_metrics
    
    async def cleanup_old_metrics(self, retention_days: int) -> None:
        """Clean up old metrics based on retention policy."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=retention_days)
        self._metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
    
    def get_recent_metrics(self, count: int = DEFAULT_RECENT_METRICS_COUNT) -> List[CostMetrics]:
        """Get recent metrics for internal use."""
        return self._metrics[-count:] if self._metrics else []
