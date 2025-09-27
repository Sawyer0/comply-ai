"""
Shared cost monitoring for mapper service.

This module implements cost monitoring using shared interfaces
for consistent cost tracking across all microservices.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from decimal import Decimal

from ..shared_integration import (
    get_shared_logger,
    CostEvent,
    CostCategory,
    ResourceUsage,
)

logger = get_shared_logger(__name__)


class MapperCostMonitor:
    """Cost monitor for mapper service operations."""

    def __init__(self):
        """Initialize the cost monitor."""
        self._cost_events: List[CostEvent] = []
        self._total_costs: Dict[CostCategory, Decimal] = {}
        self._tenant_costs: Dict[str, Dict[CostCategory, Decimal]] = {}
        
        logger.info("Mapper cost monitor initialized")

    def record_cost_event(
        self,
        tenant_id: str,
        category: CostCategory,
        amount: Decimal,
        resource_usage: Optional[ResourceUsage] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a cost event."""
        try:
            # Create cost event
            cost_event = CostEvent(
                tenant_id=tenant_id,
                category=category,
                amount=amount,
                resource_usage=resource_usage,
                metadata=metadata or {},
                timestamp=datetime.utcnow(),
            )

            # Store the event
            self._cost_events.append(cost_event)

            # Update totals
            if category not in self._total_costs:
                self._total_costs[category] = Decimal('0')
            self._total_costs[category] += amount

            # Update tenant costs
            if tenant_id not in self._tenant_costs:
                self._tenant_costs[tenant_id] = {}
            if category not in self._tenant_costs[tenant_id]:
                self._tenant_costs[tenant_id][category] = Decimal('0')
            self._tenant_costs[tenant_id][category] += amount

            logger.info(
                "Recorded cost event",
                tenant_id=tenant_id,
                category=category.value,
                amount=float(amount),
                metadata=metadata
            )

        except Exception as e:
            logger.error("Failed to record cost event", error=str(e))

    def record_mapping_cost(
        self,
        tenant_id: str,
        mapping_type: str,
        processing_time_ms: float,
        model_used: Optional[str] = None,
        confidence_score: Optional[float] = None
    ) -> None:
        """Record cost for mapping operations."""
        try:
            # Calculate cost based on processing time and model
            base_cost = Decimal(str(processing_time_ms / 1000.0 * 0.002))  # $0.002 per second for mapping
            
            # Add model-specific costs
            if model_used:
                if "llama" in model_used.lower():
                    base_cost *= Decimal('1.2')  # 1.2x cost for Llama models
                elif "gpt" in model_used.lower():
                    base_cost *= Decimal('1.5')  # 1.5x cost for GPT models

            # Create resource usage
            resource_usage = ResourceUsage(
                cpu_seconds=processing_time_ms / 1000.0,
                memory_mb=1024,  # Higher memory usage for mapping
                gpu_seconds=0.0,
                api_calls=1,
                tokens_processed=2000,  # Higher token count for mapping
            )

            # Record the cost event
            self.record_cost_event(
                tenant_id=tenant_id,
                category=CostCategory.MODEL_INFERENCE,
                amount=base_cost,
                resource_usage=resource_usage,
                metadata={
                    "mapping_type": mapping_type,
                    "model_used": model_used,
                    "confidence_score": confidence_score,
                    "processing_time_ms": processing_time_ms,
                }
            )

        except Exception as e:
            logger.error("Failed to record mapping cost", error=str(e))

    def record_api_cost(
        self,
        tenant_id: str,
        endpoint: str,
        request_size_bytes: int,
        response_size_bytes: int
    ) -> None:
        """Record cost for API operations."""
        try:
            # Calculate cost based on data transfer
            total_bytes = request_size_bytes + response_size_bytes
            cost_per_mb = Decimal('0.0001')  # $0.0001 per MB
            cost = Decimal(str(total_bytes / (1024 * 1024))) * cost_per_mb

            # Create resource usage
            resource_usage = ResourceUsage(
                cpu_seconds=0.1,  # Minimal CPU for API calls
                memory_mb=64,     # Minimal memory for API calls
                gpu_seconds=0.0,
                api_calls=1,
                tokens_processed=0,
            )

            # Record the cost event
            self.record_cost_event(
                tenant_id=tenant_id,
                category=CostCategory.API_CALLS,
                amount=cost,
                resource_usage=resource_usage,
                metadata={
                    "endpoint": endpoint,
                    "request_size_bytes": request_size_bytes,
                    "response_size_bytes": response_size_bytes,
                }
            )

        except Exception as e:
            logger.error("Failed to record API cost", error=str(e))

    def get_tenant_costs(self, tenant_id: str) -> Dict[CostCategory, Decimal]:
        """Get total costs for a tenant."""
        try:
            return self._tenant_costs.get(tenant_id, {})
        except Exception as e:
            logger.error("Failed to get tenant costs", error=str(e))
            return {}

    def get_total_costs(self) -> Dict[CostCategory, Decimal]:
        """Get total costs across all tenants."""
        try:
            return self._total_costs.copy()
        except Exception as e:
            logger.error("Failed to get total costs", error=str(e))
            return {}

    def get_cost_summary(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get a cost summary for a tenant or all tenants."""
        try:
            if tenant_id:
                costs = self.get_tenant_costs(tenant_id)
                total = sum(costs.values())
            else:
                costs = self.get_total_costs()
                total = sum(costs.values())

            return {
                "tenant_id": tenant_id,
                "costs_by_category": {k.value: float(v) for k, v in costs.items()},
                "total_cost": float(total),
                "event_count": len(self._cost_events),
            }

        except Exception as e:
            logger.error("Failed to get cost summary", error=str(e))
            return {}


# Global cost monitor instance
_cost_monitor: Optional[MapperCostMonitor] = None


def get_shared_cost_monitor() -> MapperCostMonitor:
    """Get the global cost monitor instance."""
    global _cost_monitor
    if _cost_monitor is None:
        _cost_monitor = MapperCostMonitor()
    return _cost_monitor


__all__ = [
    "MapperCostMonitor",
    "get_shared_cost_monitor",
]
