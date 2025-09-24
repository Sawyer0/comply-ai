"""Factory for creating cost monitoring components."""

from __future__ import annotations

from typing import Optional

from ..config.cost_config import CostMonitoringConfig
from .cost_calculator import StandardCostCalculator, TieredCostCalculator
from .interfaces import (
    AlertManager,
    CostCalculator,
    MetricsStorage,
    ResourceMonitor,
    ScalingExecutor,
)
from .metrics_collector import InMemoryMetricsStorage
from .resource_monitor import (
    ApplicationMetricsMonitor,
    MockResourceMonitor,
    SystemResourceMonitor,
)


class CostMonitoringFactory:
    """Factory for creating cost monitoring components."""

    @staticmethod
    def create_resource_monitor(
        monitor_type: str = "system",
        include_gpu: bool = False,
        mock_data: Optional[dict] = None,
    ) -> ResourceMonitor:
        """Create a resource monitor."""
        if monitor_type == "system":
            return SystemResourceMonitor(include_gpu=include_gpu)
        elif monitor_type == "mock":
            if mock_data:
                from .metrics_collector import ResourceUsage

                mock_usage = ResourceUsage(**mock_data)
                return MockResourceMonitor(mock_usage)
            return MockResourceMonitor()
        elif monitor_type == "application":
            base_monitor = CostMonitoringFactory.create_resource_monitor(
                "system", include_gpu
            )
            return ApplicationMetricsMonitor(base_monitor)
        else:
            raise ValueError(f"Unknown resource monitor type: {monitor_type}")

    @staticmethod
    def create_cost_calculator(
        calculator_type: str = "standard", cost_per_unit: Optional[dict] = None
    ) -> CostCalculator:
        """Create a cost calculator."""
        if calculator_type == "standard":
            return StandardCostCalculator(cost_per_unit)
        elif calculator_type == "tiered":
            return TieredCostCalculator(cost_per_unit)
        else:
            raise ValueError(f"Unknown cost calculator type: {calculator_type}")

    @staticmethod
    def create_metrics_storage(storage_type: str = "memory") -> MetricsStorage:
        """Create a metrics storage."""
        if storage_type == "memory":
            return InMemoryMetricsStorage()
        else:
            raise ValueError(f"Unknown metrics storage type: {storage_type}")

    @staticmethod
    def create_alert_manager(manager_type: str = "mock") -> AlertManager:
        """Create an alert manager."""
        if manager_type == "mock":
            return MockAlertManager()
        else:
            raise ValueError(f"Unknown alert manager type: {manager_type}")

    @staticmethod
    def create_scaling_executor(executor_type: str = "mock") -> ScalingExecutor:
        """Create a scaling executor."""
        if executor_type == "mock":
            return MockScalingExecutor()
        else:
            raise ValueError(f"Unknown scaling executor type: {executor_type}")

    @staticmethod
    def create_development_config() -> CostMonitoringConfig:
        """Create a development configuration."""
        from ..config.cost_config import CostMonitoringConfig

        return CostMonitoringConfig(
            collection_interval_seconds=10,  # Fast for development
            retention_days=7,  # Short retention for development
            cost_thresholds={
                "daily_budget": 50.0,  # Lower budget for development
                "hourly_budget": 5.0,
            },
        )

    @staticmethod
    def create_production_config() -> CostMonitoringConfig:
        """Create a production configuration."""
        from ..config.cost_config import CostMonitoringConfig

        return CostMonitoringConfig(
            collection_interval_seconds=60,  # Standard interval
            retention_days=90,  # Long retention for production
            cost_thresholds={
                "daily_budget": 1000.0,  # Higher budget for production
                "hourly_budget": 100.0,
            },
        )

    @staticmethod
    def create_testing_config() -> CostMonitoringConfig:
        """Create a testing configuration."""
        from ..config.cost_config import CostMonitoringConfig

        return CostMonitoringConfig(
            collection_interval_seconds=1,  # Very fast for testing
            retention_days=1,  # Minimal retention for testing
            cost_thresholds={
                "daily_budget": 10.0,  # Low budget for testing
                "hourly_budget": 1.0,
            },
        )


class MockAlertManager(AlertManager):
    """Mock alert manager for testing."""

    def __init__(self):
        self.alerts = []

    async def create_alert(
        self, alert_type: str, message: str, severity: str, metadata: dict
    ) -> None:
        """Create a mock alert."""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "metadata": metadata,
            "created_at": "2024-01-01T00:00:00Z",
        }
        self.alerts.append(alert)

    async def get_active_alerts(self) -> list:
        """Get active mock alerts."""
        return self.alerts.copy()


class MockScalingExecutor(ScalingExecutor):
    """Mock scaling executor for testing."""

    def __init__(self):
        self.current_instances = {}
        self.scaling_history = []

    async def scale_up(
        self, resource_type: str, target_instances: int, metadata: dict
    ) -> bool:
        """Mock scale up operation."""
        current = self.current_instances.get(resource_type, 1)
        self.current_instances[resource_type] = target_instances
        self.scaling_history.append(
            {
                "action": "scale_up",
                "resource_type": resource_type,
                "from": current,
                "to": target_instances,
                "metadata": metadata,
            }
        )
        return True

    async def scale_down(
        self, resource_type: str, target_instances: int, metadata: dict
    ) -> bool:
        """Mock scale down operation."""
        current = self.current_instances.get(resource_type, 1)
        self.current_instances[resource_type] = target_instances
        self.scaling_history.append(
            {
                "action": "scale_down",
                "resource_type": resource_type,
                "from": current,
                "to": target_instances,
                "metadata": metadata,
            }
        )
        return True

    async def get_current_instances(self, resource_type: str) -> int:
        """Get current number of instances."""
        return self.current_instances.get(resource_type, 1)
