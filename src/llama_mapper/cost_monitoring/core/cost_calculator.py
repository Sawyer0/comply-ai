"""Cost calculation implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List

from .constants import (
    DEFAULT_COST_PER_UNIT,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MONTH,
)
from .interfaces import CostCalculator
from .metrics_collector import CostBreakdown, CostMetrics, ResourceUsage


class StandardCostCalculator(CostCalculator):
    """Standard cost calculator with configurable pricing."""

    def __init__(self, cost_per_unit: Dict[str, float] = None):
        self.cost_per_unit = cost_per_unit or DEFAULT_COST_PER_UNIT.copy()

    def calculate_cost(self, usage: ResourceUsage) -> CostMetrics:
        """Calculate cost metrics for given resource usage."""
        # Calculate individual costs
        cpu_cost = self._calculate_cpu_cost(usage.cpu_cores)
        memory_cost = self._calculate_memory_cost(usage.memory_gb)
        gpu_cost = self._calculate_gpu_cost(usage.gpu_count, usage.gpu_memory_gb)
        storage_cost = self._calculate_storage_cost(usage.storage_gb)
        network_cost = self._calculate_network_cost(usage.network_gb)
        api_cost = self._calculate_api_cost(usage.api_calls)

        total_cost = (
            cpu_cost + memory_cost + gpu_cost + storage_cost + network_cost + api_cost
        )

        return CostMetrics(
            resource_type="llama_mapper",
            usage=usage,
            cost_per_unit=self.cost_per_unit,
            total_cost=total_cost,
            currency="USD",
        )

    def _calculate_cpu_cost(self, cpu_cores: float) -> float:
        """Calculate CPU cost per second."""
        return cpu_cores * self.cost_per_unit["cpu_per_hour"] / SECONDS_PER_HOUR

    def _calculate_memory_cost(self, memory_gb: float) -> float:
        """Calculate memory cost per second."""
        return memory_gb * self.cost_per_unit["memory_per_gb_hour"] / SECONDS_PER_HOUR

    def _calculate_gpu_cost(self, gpu_count: int, gpu_memory_gb: float) -> float:
        """Calculate GPU cost per second."""
        gpu_instance_cost = (
            gpu_count * self.cost_per_unit["gpu_per_hour"] / SECONDS_PER_HOUR
        )
        gpu_memory_cost = (
            gpu_memory_gb * self.cost_per_unit["memory_per_gb_hour"] / SECONDS_PER_HOUR
        )
        return gpu_instance_cost + gpu_memory_cost

    def _calculate_storage_cost(self, storage_gb: float) -> float:
        """Calculate storage cost per second."""
        return (
            storage_gb * self.cost_per_unit["storage_per_gb_month"] / SECONDS_PER_MONTH
        )

    def _calculate_network_cost(self, network_gb: float) -> float:
        """Calculate network cost."""
        return network_gb * self.cost_per_unit["network_per_gb"]

    def _calculate_api_cost(self, api_calls: int) -> float:
        """Calculate API call cost."""
        return api_calls * self.cost_per_unit["api_call"]

    def calculate_breakdown(self, usage_list: List[ResourceUsage]) -> CostBreakdown:
        """Calculate cost breakdown for a list of usage metrics."""
        if not usage_list:
            return CostBreakdown(
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
            )

        # Calculate totals
        total_compute_cost = 0.0
        total_memory_cost = 0.0
        total_storage_cost = 0.0
        total_network_cost = 0.0
        total_api_cost = 0.0

        for usage in usage_list:
            total_compute_cost += self._calculate_cpu_cost(usage.cpu_cores)
            total_compute_cost += self._calculate_gpu_cost(
                usage.gpu_count, usage.gpu_memory_gb
            )
            total_memory_cost += self._calculate_memory_cost(usage.memory_gb)
            total_storage_cost += self._calculate_storage_cost(usage.storage_gb)
            total_network_cost += self._calculate_network_cost(usage.network_gb)
            total_api_cost += self._calculate_api_cost(usage.api_calls)

        total_cost = (
            total_compute_cost
            + total_memory_cost
            + total_storage_cost
            + total_network_cost
            + total_api_cost
        )

        return CostBreakdown(
            compute_cost=total_compute_cost,
            memory_cost=total_memory_cost,
            storage_cost=total_storage_cost,
            network_cost=total_network_cost,
            api_cost=total_api_cost,
            total_cost=total_cost,
            currency="USD",
            period_start=usage_list[0].timestamp,
            period_end=usage_list[-1].timestamp,
        )


class TieredCostCalculator(CostCalculator):
    """Cost calculator with tiered pricing."""

    def __init__(self, cost_per_unit: Dict[str, float] = None):
        self.cost_per_unit = cost_per_unit or DEFAULT_COST_PER_UNIT.copy()
        self.tiers = {
            "cpu": [
                {"threshold": 0, "rate": self.cost_per_unit["cpu_per_hour"]},
                {
                    "threshold": 4,
                    "rate": self.cost_per_unit["cpu_per_hour"] * 0.8,
                },  # 20% discount
                {
                    "threshold": 8,
                    "rate": self.cost_per_unit["cpu_per_hour"] * 0.6,
                },  # 40% discount
            ],
            "memory": [
                {"threshold": 0, "rate": self.cost_per_unit["memory_per_gb_hour"]},
                {
                    "threshold": 16,
                    "rate": self.cost_per_unit["memory_per_gb_hour"] * 0.8,
                },
                {
                    "threshold": 32,
                    "rate": self.cost_per_unit["memory_per_gb_hour"] * 0.6,
                },
            ],
        }

    def calculate_cost(self, usage: ResourceUsage) -> CostMetrics:
        """Calculate cost with tiered pricing."""
        # Calculate individual costs with tiered pricing
        cpu_cost = self._calculate_tiered_cost("cpu", usage.cpu_cores)
        memory_cost = self._calculate_tiered_cost("memory", usage.memory_gb)
        gpu_cost = self._calculate_gpu_cost(usage.gpu_count, usage.gpu_memory_gb)
        storage_cost = self._calculate_storage_cost(usage.storage_gb)
        network_cost = self._calculate_network_cost(usage.network_gb)
        api_cost = self._calculate_api_cost(usage.api_calls)

        total_cost = (
            cpu_cost + memory_cost + gpu_cost + storage_cost + network_cost + api_cost
        )

        return CostMetrics(
            resource_type="llama_mapper",
            usage=usage,
            cost_per_unit=self.cost_per_unit,
            total_cost=total_cost,
            currency="USD",
        )

    def _calculate_tiered_cost(self, resource_type: str, amount: float) -> float:
        """Calculate cost using tiered pricing."""
        if resource_type not in self.tiers:
            return 0.0

        tiers = self.tiers[resource_type]
        total_cost = 0.0
        remaining_amount = amount

        for i, tier in enumerate(tiers):
            if remaining_amount <= 0:
                break

            # Determine the amount for this tier
            if i == len(tiers) - 1:  # Last tier
                tier_amount = remaining_amount
            else:
                next_tier_threshold = tiers[i + 1]["threshold"]
                tier_amount = min(
                    remaining_amount, next_tier_threshold - tier["threshold"]
                )

            if tier_amount > 0:
                tier_cost = tier_amount * tier["rate"] / SECONDS_PER_HOUR
                total_cost += tier_cost
                remaining_amount -= tier_amount

        return total_cost

    def _calculate_gpu_cost(self, gpu_count: int, gpu_memory_gb: float) -> float:
        """Calculate GPU cost (no tiering for GPU)."""
        gpu_instance_cost = (
            gpu_count * self.cost_per_unit["gpu_per_hour"] / SECONDS_PER_HOUR
        )
        gpu_memory_cost = (
            gpu_memory_gb * self.cost_per_unit["memory_per_gb_hour"] / SECONDS_PER_HOUR
        )
        return gpu_instance_cost + gpu_memory_cost

    def _calculate_storage_cost(self, storage_gb: float) -> float:
        """Calculate storage cost (no tiering for storage)."""
        return (
            storage_gb * self.cost_per_unit["storage_per_gb_month"] / SECONDS_PER_MONTH
        )

    def _calculate_network_cost(self, network_gb: float) -> float:
        """Calculate network cost (no tiering for network)."""
        return network_gb * self.cost_per_unit["network_per_gb"]

    def _calculate_api_cost(self, api_calls: int) -> float:
        """Calculate API call cost (no tiering for API calls)."""
        return api_calls * self.cost_per_unit["api_call"]

    def calculate_breakdown(self, usage_list: List[ResourceUsage]) -> CostBreakdown:
        """Calculate cost breakdown with tiered pricing."""
        if not usage_list:
            return CostBreakdown(
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
            )

        # Calculate totals with tiered pricing
        total_compute_cost = 0.0
        total_memory_cost = 0.0
        total_storage_cost = 0.0
        total_network_cost = 0.0
        total_api_cost = 0.0

        for usage in usage_list:
            total_compute_cost += self._calculate_tiered_cost("cpu", usage.cpu_cores)
            total_compute_cost += self._calculate_gpu_cost(
                usage.gpu_count, usage.gpu_memory_gb
            )
            total_memory_cost += self._calculate_tiered_cost("memory", usage.memory_gb)
            total_storage_cost += self._calculate_storage_cost(usage.storage_gb)
            total_network_cost += self._calculate_network_cost(usage.network_gb)
            total_api_cost += self._calculate_api_cost(usage.api_calls)

        total_cost = (
            total_compute_cost
            + total_memory_cost
            + total_storage_cost
            + total_network_cost
            + total_api_cost
        )

        return CostBreakdown(
            compute_cost=total_compute_cost,
            memory_cost=total_memory_cost,
            storage_cost=total_storage_cost,
            network_cost=total_network_cost,
            api_cost=total_api_cost,
            total_cost=total_cost,
            currency="USD",
            period_start=usage_list[0].timestamp,
            period_end=usage_list[-1].timestamp,
        )
