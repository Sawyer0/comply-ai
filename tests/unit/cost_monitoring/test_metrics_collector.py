"""Unit tests for cost monitoring metrics collector."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llama_mapper.cost_monitoring.core.metrics_collector import (
    CostBreakdown,
    CostMetrics,
    CostMetricsCollector,
    CostMonitoringConfig,
    ResourceUsage,
)


class TestCostMonitoringConfig:
    """Test cost monitoring configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CostMonitoringConfig()

        assert config.enabled is True
        assert config.collection_interval_seconds == 60
        assert config.retention_days == 90
        assert config.currency == "USD"
        assert "daily_budget" in config.cost_thresholds
        assert "cpu_per_hour" in config.cost_per_unit

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CostMonitoringConfig(
            enabled=False,
            collection_interval_seconds=30,
            retention_days=30,
            currency="EUR",
            cost_thresholds={"custom_threshold": 100.0},
            cost_per_unit={"custom_unit": 0.5},
        )

        assert config.enabled is False
        assert config.collection_interval_seconds == 30
        assert config.retention_days == 30
        assert config.currency == "EUR"
        assert config.cost_thresholds["custom_threshold"] == 100.0
        assert config.cost_per_unit["custom_unit"] == 0.5


class TestResourceUsage:
    """Test resource usage data structure."""

    def test_default_resource_usage(self):
        """Test default resource usage values."""
        usage = ResourceUsage()

        assert usage.cpu_cores == 0.0
        assert usage.memory_gb == 0.0
        assert usage.gpu_count == 0
        assert usage.gpu_memory_gb == 0.0
        assert usage.storage_gb == 0.0
        assert usage.network_gb == 0.0
        assert usage.api_calls == 0
        assert usage.processing_time_ms == 0.0
        assert isinstance(usage.timestamp, datetime)

    def test_custom_resource_usage(self):
        """Test custom resource usage values."""
        timestamp = datetime.now(timezone.utc)
        usage = ResourceUsage(
            cpu_cores=2.5,
            memory_gb=8.0,
            gpu_count=1,
            gpu_memory_gb=16.0,
            storage_gb=100.0,
            network_gb=5.0,
            api_calls=1000,
            processing_time_ms=150.0,
            timestamp=timestamp,
        )

        assert usage.cpu_cores == 2.5
        assert usage.memory_gb == 8.0
        assert usage.gpu_count == 1
        assert usage.gpu_memory_gb == 16.0
        assert usage.storage_gb == 100.0
        assert usage.network_gb == 5.0
        assert usage.api_calls == 1000
        assert usage.processing_time_ms == 150.0
        assert usage.timestamp == timestamp


class TestCostMetrics:
    """Test cost metrics data structure."""

    def test_cost_metrics_creation(self):
        """Test cost metrics creation."""
        usage = ResourceUsage(cpu_cores=2.0, memory_gb=4.0)
        cost_per_unit = {"cpu_per_hour": 0.05, "memory_per_gb_hour": 0.01}

        metrics = CostMetrics(
            resource_type="test",
            usage=usage,
            cost_per_unit=cost_per_unit,
            total_cost=0.5,
            currency="USD",
        )

        assert metrics.resource_type == "test"
        assert metrics.usage == usage
        assert metrics.cost_per_unit == cost_per_unit
        assert metrics.total_cost == 0.5
        assert metrics.currency == "USD"
        assert isinstance(metrics.timestamp, datetime)


class TestCostBreakdown:
    """Test cost breakdown data structure."""

    def test_cost_breakdown_creation(self):
        """Test cost breakdown creation."""
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        end_time = datetime.now(timezone.utc)

        breakdown = CostBreakdown(
            compute_cost=100.0,
            memory_cost=50.0,
            storage_cost=25.0,
            network_cost=10.0,
            api_cost=5.0,
            total_cost=190.0,
            currency="USD",
            period_start=start_time,
            period_end=end_time,
            tenant_id="test-tenant",
        )

        assert breakdown.compute_cost == 100.0
        assert breakdown.memory_cost == 50.0
        assert breakdown.storage_cost == 25.0
        assert breakdown.network_cost == 10.0
        assert breakdown.api_cost == 5.0
        assert breakdown.total_cost == 190.0
        assert breakdown.currency == "USD"
        assert breakdown.period_start == start_time
        assert breakdown.period_end == end_time
        assert breakdown.tenant_id == "test-tenant"


class TestCostMetricsCollector:
    """Test cost metrics collector."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CostMonitoringConfig(
            collection_interval_seconds=1,  # Fast for testing
            retention_days=1,
        )

    @pytest.fixture
    def collector(self, config):
        """Create test collector."""
        return CostMetricsCollector(config)

    def test_collector_initialization(self, collector):
        """Test collector initialization."""
        assert collector.config is not None
        assert collector.resource_monitor is not None
        assert collector.cost_calculator is not None
        assert collector.metrics_storage is not None
        assert collector._running is False
        assert collector._collection_task is None

    @pytest.mark.asyncio
    async def test_start_stop(self, collector):
        """Test starting and stopping the collector."""
        # Start collector
        await collector.start()
        assert collector._running is True
        assert collector._collection_task is not None

        # Stop collector
        await collector.stop()
        assert collector._running is False
        assert collector._collection_task is None

    @pytest.mark.asyncio
    async def test_double_start(self, collector):
        """Test starting collector twice."""
        await collector.start()
        assert collector._running is True

        # Starting again should not create new task
        await collector.start()
        assert collector._running is True

    @pytest.mark.asyncio
    async def test_get_current_resource_usage(self, collector):
        """Test getting current resource usage."""
        usage = await collector.resource_monitor.get_current_usage()

        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_cores >= 0
        assert usage.memory_gb >= 0
        assert usage.gpu_count >= 0
        assert usage.gpu_memory_gb >= 0
        assert usage.storage_gb >= 0
        assert usage.network_gb >= 0
        assert usage.api_calls >= 0
        assert usage.processing_time_ms >= 0
        assert isinstance(usage.timestamp, datetime)

    def test_calculate_costs(self, collector):
        """Test cost calculation."""
        usage = ResourceUsage(
            cpu_cores=2.0,
            memory_gb=4.0,
            gpu_count=1,
            gpu_memory_gb=8.0,
            storage_gb=50.0,
            network_gb=1.0,
            api_calls=100,
        )

        cost_metrics = collector.cost_calculator.calculate_cost(usage)

        assert isinstance(cost_metrics, CostMetrics)
        assert cost_metrics.resource_type == "llama_mapper"
        assert cost_metrics.usage == usage
        assert cost_metrics.total_cost >= 0
        assert cost_metrics.currency == collector.config.currency

    @pytest.mark.asyncio
    async def test_get_daily_cost(self, collector):
        """Test getting daily cost."""
        # Add some test metrics
        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Add metrics from today
        test_metrics = [
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(cpu_cores=1.0),
                cost_per_unit={"cpu_per_hour": 0.05},
                total_cost=1.0,
                timestamp=start_of_day + timedelta(hours=1),
            ),
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(cpu_cores=2.0),
                cost_per_unit={"cpu_per_hour": 0.05},
                total_cost=2.0,
                timestamp=start_of_day + timedelta(hours=2),
            ),
        ]

        # Store metrics in storage
        for metric in test_metrics:
            await collector.metrics_storage.store_metrics(metric)

        daily_cost = await collector._get_daily_cost()
        assert daily_cost == 3.0

    @pytest.mark.asyncio
    async def test_get_hourly_cost(self, collector):
        """Test getting hourly cost."""
        # Add some test metrics
        now = datetime.now(timezone.utc)
        start_of_hour = now.replace(minute=0, second=0, microsecond=0)

        # Add metrics from current hour
        test_metrics = [
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(cpu_cores=1.0),
                cost_per_unit={"cpu_per_hour": 0.05},
                total_cost=1.0,
                timestamp=start_of_hour + timedelta(minutes=10),
            ),
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(cpu_cores=2.0),
                cost_per_unit={"cpu_per_hour": 0.05},
                total_cost=2.0,
                timestamp=start_of_hour + timedelta(minutes=20),
            ),
        ]

        # Store metrics in storage
        for metric in test_metrics:
            await collector.metrics_storage.store_metrics(metric)

        hourly_cost = await collector._get_hourly_cost()
        assert hourly_cost == 3.0

    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, collector):
        """Test cleanup of old metrics."""
        now = datetime.now(timezone.utc)

        # Add old and new metrics
        test_metrics = [
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(),
                cost_per_unit={},
                total_cost=1.0,
                timestamp=now - timedelta(days=2),  # Old
            ),
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(),
                cost_per_unit={},
                total_cost=2.0,
                timestamp=now - timedelta(hours=1),  # Recent
            ),
        ]

        # Store metrics
        for metric in test_metrics:
            await collector.metrics_storage.store_metrics(metric)

        # Cleanup old metrics
        await collector.metrics_storage.cleanup_old_metrics(1)  # 1 day retention

        # Check that only recent metrics remain
        remaining_metrics = await collector.metrics_storage.get_metrics(
            now - timedelta(days=2), now
        )
        assert len(remaining_metrics) == 1
        assert remaining_metrics[0].total_cost == 2.0

    @pytest.mark.asyncio
    async def test_get_cost_breakdown(self, collector):
        """Test getting cost breakdown."""
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        end_time = datetime.now(timezone.utc)

        # Add test metrics
        test_metric = CostMetrics(
            resource_type="test",
            usage=ResourceUsage(
                cpu_cores=2.0,
                memory_gb=4.0,
                storage_gb=10.0,
                network_gb=1.0,
                api_calls=100,
            ),
            cost_per_unit={
                "cpu_per_hour": 0.05,
                "memory_per_gb_hour": 0.01,
                "gpu_per_hour": 2.00,
                "storage_per_gb_month": 0.10,
                "network_per_gb": 0.05,
                "api_call": 0.001,
            },
            total_cost=1.0,
            timestamp=start_time + timedelta(hours=1),
        )

        # Store metric
        await collector.metrics_storage.store_metrics(test_metric)

        breakdown = await collector.get_cost_breakdown(start_time, end_time)

        assert isinstance(breakdown, CostBreakdown)
        assert breakdown.period_start == start_time
        assert breakdown.period_end == end_time
        assert breakdown.total_cost >= 0

    @pytest.mark.asyncio
    async def test_get_cost_trends(self, collector):
        """Test getting cost trends."""
        # Add test metrics
        now = datetime.now(timezone.utc)
        for i in range(5):
            metric = CostMetrics(
                resource_type="test",
                usage=ResourceUsage(),
                cost_per_unit={},
                total_cost=float(i + 1),
                timestamp=now - timedelta(days=i),
            )
            await collector.metrics_storage.store_metrics(metric)

        trends = await collector.get_cost_trends(days=5)

        assert "dates" in trends
        assert "costs" in trends
        assert "total_cost" in trends
        assert "average_daily_cost" in trends
        assert len(trends["costs"]) == 5
        assert trends["total_cost"] == 15.0  # 1+2+3+4+5
        assert trends["average_daily_cost"] == 3.0  # 15/5
