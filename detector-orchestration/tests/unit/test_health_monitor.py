"""Tests for health monitoring."""

import asyncio
import time
import pytest
from unittest.mock import Mock, AsyncMock

from detector_orchestration.health_monitor import HealthMonitor, HealthStatus
from detector_orchestration.clients import DetectorClient
from detector_orchestration.metrics import OrchestrationMetricsCollector


class TestHealthMonitor:
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        clients = {"toxicity": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)

        monitor = HealthMonitor(
            clients=clients,
            interval_seconds=30,
            metrics=metrics,
            unhealthy_threshold=3,
        )

        assert monitor.clients == clients
        assert monitor.interval_seconds == 30
        assert monitor.unhealthy_threshold == 3
        assert monitor.metrics == metrics
        assert "toxicity" in monitor._status
        assert monitor._task is None
        assert monitor._stop is None

    def test_health_status_initialization(self):
        """Test initial health status for all clients."""
        clients = {"det1": Mock(spec=DetectorClient), "det2": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        for detector_name in clients:
            status = monitor._status[detector_name]
            assert status.is_healthy is True
            assert status.last_check > 0
            assert status.response_time_ms is None
            assert status.consecutive_failures == 0

    def test_is_healthy(self):
        """Test health status checking."""
        clients = {"det1": Mock(spec=DetectorClient), "det2": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # All detectors should initially be healthy
        assert monitor.is_healthy("det1") is True
        assert monitor.is_healthy("det2") is True

        # Unknown detector should be unhealthy
        assert monitor.is_healthy("unknown") is False

    def test_get_health_status(self):
        """Test getting health status details."""
        clients = {"det1": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        status = monitor.get_health_status("det1")
        assert isinstance(status, HealthStatus)
        assert status.is_healthy is True

        # Unknown detector
        status = monitor.get_health_status("unknown")
        assert status is None

    async def test_record_success(self):
        """Test recording successful health check."""
        clients = {"det1": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Record success
        await monitor._record_success("det1", 150)

        status = monitor._status["det1"]
        assert status.is_healthy is True
        assert status.response_time_ms == 150
        assert status.consecutive_failures == 0

    async def test_record_failure(self):
        """Test recording failed health check."""
        clients = {"det1": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Record failure
        await monitor._record_failure("det1", 3000)

        status = monitor._status["det1"]
        assert status.is_healthy is False
        assert status.response_time_ms == 3000
        assert status.consecutive_failures == 1

        # Record another failure
        await monitor._record_failure("det1", 4000)
        status = monitor._status["det1"]
        assert status.consecutive_failures == 2

    async def test_health_check_success(self):
        """Test successful health check execution."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.health_check = AsyncMock(return_value=True)

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Perform health check
        await monitor._check_detector("det1")

        # Should record success
        status = monitor._status["det1"]
        assert status.is_healthy is True
        assert status.consecutive_failures == 0
        assert status.response_time_ms is not None

        # Client should have been called
        mock_client.health_check.assert_called_once()

    async def test_health_check_failure(self):
        """Test failed health check execution."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.health_check = AsyncMock(side_effect=Exception("Connection failed"))

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Perform health check
        await monitor._check_detector("det1")

        # Should record failure
        status = monitor._status["det1"]
        assert status.is_healthy is False
        assert status.consecutive_failures == 1
        assert status.response_time_ms is not None

    async def test_health_check_timeout(self):
        """Test health check timeout."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.health_check = AsyncMock(side_effect=asyncio.TimeoutError())

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Perform health check
        await monitor._check_detector("det1")

        # Should record failure due to timeout
        status = monitor._status["det1"]
        assert status.is_healthy is False
        assert status.consecutive_failures == 1

    async def test_unhealthy_threshold(self):
        """Test unhealthy threshold behavior."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.health_check = AsyncMock(side_effect=Exception("Always fails"))

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(
            clients=clients,
            interval_seconds=30,
            metrics=metrics,
            unhealthy_threshold=2  # Lower threshold for faster testing
        )

        # First failure
        await monitor._check_detector("det1")
        assert monitor._status["det1"].is_healthy is False
        assert monitor._status["det1"].consecutive_failures == 1

        # Second failure - should remain unhealthy but not change state
        await monitor._check_detector("det1")
        assert monitor._status["det1"].is_healthy is False
        assert monitor._status["det1"].consecutive_failures == 2

    async def test_recovery_after_failure(self):
        """Test recovery after consecutive failures."""
        mock_client = Mock(spec=DetectorClient)

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(
            clients=clients,
            interval_seconds=30,
            metrics=metrics,
            unhealthy_threshold=2
        )

        # Make client fail first
        mock_client.health_check = AsyncMock(side_effect=Exception("Fails"))

        # First failure
        await monitor._check_detector("det1")
        assert monitor._status["det1"].is_healthy is False
        assert monitor._status["det1"].consecutive_failures == 1

        # Second failure
        await monitor._check_detector("det1")
        assert monitor._status["det1"].is_healthy is False
        assert monitor._status["det1"].consecutive_failures == 2

        # Now make client succeed
        mock_client.health_check = AsyncMock(return_value=True)

        # Successful check should reset failure count
        await monitor._check_detector("det1")
        assert monitor._status["det1"].is_healthy is True
        assert monitor._status["det1"].consecutive_failures == 0

    async def test_start_stop_monitoring(self):
        """Test starting and stopping health monitoring."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.health_check = AsyncMock(return_value=True)

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Start monitoring
        await monitor.start()

        # Should have created a task
        assert monitor._task is not None
        assert not monitor._task.done()

        # Stop monitoring
        await monitor.stop()

        # Task should be cancelled/done
        assert monitor._task.done()

    async def test_monitoring_loop_execution(self):
        """Test that monitoring loop executes health checks."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.health_check = AsyncMock(return_value=True)

        clients = {"det1": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=1, metrics=metrics)  # 1 second interval

        # Start monitoring
        await monitor.start()

        # Wait a bit for the monitoring loop to run
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop()

        # Client should have been called at least once
        assert mock_client.health_check.call_count >= 1

    def test_get_all_health_status(self):
        """Test getting health status for all detectors."""
        clients = {"det1": Mock(spec=DetectorClient), "det2": Mock(spec=DetectorClient)}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        all_status = monitor.get_all_health_status()

        assert len(all_status) == 2
        assert "det1" in all_status
        assert "det2" in all_status
        assert all(isinstance(status, HealthStatus) for status in all_status.values())

    def test_get_unhealthy_detectors(self):
        """Test getting list of unhealthy detectors."""
        mock_client = Mock(spec=DetectorClient)

        clients = {"det1": mock_client, "det2": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        monitor = HealthMonitor(clients=clients, interval_seconds=30, metrics=metrics)

        # Initially all should be healthy
        unhealthy = monitor.get_unhealthy_detectors()
        assert len(unhealthy) == 0

        # Mark one as unhealthy
        monitor._status["det1"].is_healthy = False
        monitor._status["det1"].consecutive_failures = 3

        unhealthy = monitor.get_unhealthy_detectors()
        assert len(unhealthy) == 1
        assert "det1" in unhealthy
        assert "det2" not in unhealthy
