"""Tests for detector coordinator."""

import asyncio
from typing import cast, Dict
from unittest.mock import Mock, AsyncMock, patch

import pytest

from detector_orchestration.models import (
    DetectorResult,
    RoutingPlan,
    DetectorStatus,
)
from detector_orchestration.coordinator import DetectorCoordinator
from detector_orchestration.clients import DetectorClient
from detector_orchestration.circuit_breaker import CircuitBreakerManager
from detector_orchestration.metrics import OrchestrationMetricsCollector


class TestDetectorCoordinator:
    def test_coordinator_initialization(self):
        """Test detector coordinator initialization."""
        clients = cast(
            Dict[str, DetectorClient], {"toxicity": Mock(spec=DetectorClient)}
        )
        breakers = Mock(spec=CircuitBreakerManager)
        metrics = Mock(spec=OrchestrationMetricsCollector)

        coordinator = DetectorCoordinator(
            clients=clients,
            breakers=breakers,
            metrics=metrics,
            retry_on_timeouts=True,
            retry_on_failures=False,
        )

        assert coordinator.clients == clients
        assert coordinator.breakers == breakers
        assert coordinator.metrics == metrics
        assert coordinator._retry_on_timeouts is True
        assert coordinator._retry_on_failures is False

    def test_coordinator_initialization_defaults(self):
        """Test detector coordinator initialization with defaults."""
        clients = cast(
            Dict[str, DetectorClient], {"toxicity": Mock(spec=DetectorClient)}
        )

        coordinator = DetectorCoordinator(clients=clients)

        assert coordinator.clients == clients
        assert coordinator.breakers is None
        assert coordinator.metrics is None
        assert coordinator._retry_on_timeouts is True
        assert coordinator._retry_on_failures is True

    async def test_execute_routing_plan_single_group(self):
        """Test executing a routing plan with single parallel group."""
        # Mock detector client
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="toxicity",
                status=DetectorStatus.SUCCESS,
                output="clean",
                confidence=0.9,
                processing_time_ms=1500,
            )
        )

        clients = cast(Dict[str, DetectorClient], {"toxicity": mock_client})
        coordinator = DetectorCoordinator(clients=clients)

        routing_plan = RoutingPlan(
            primary_detectors=["toxicity"],
            parallel_groups=[["toxicity"]],
            timeout_config={"toxicity": 3000},
            retry_config={"toxicity": 0},
        )

        results = await coordinator.execute_routing_plan(
            content="Test content",
            routing_plan=routing_plan,
            request_id="test-request-123",
        )

        assert len(results) == 1
        assert results[0].detector == "toxicity"
        assert results[0].status == DetectorStatus.SUCCESS
        assert results[0].output == "clean"

        # Verify client was called
        mock_client.analyze.assert_called_once_with("Test content", {})

    async def test_execute_routing_plan_multiple_groups(self):
        """Test executing a routing plan with multiple parallel groups."""
        # Mock detector clients
        mock_client1 = Mock(spec=DetectorClient)
        mock_client1.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="toxicity",
                status=DetectorStatus.SUCCESS,
                output="clean",
                confidence=0.9,
                processing_time_ms=1500,
            )
        )

        mock_client2 = Mock(spec=DetectorClient)
        mock_client2.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="regex-pii",
                status=DetectorStatus.SUCCESS,
                output="clean",
                confidence=0.8,
                processing_time_ms=500,
            )
        )

        clients = cast(
            Dict[str, DetectorClient],
            {"toxicity": mock_client1, "regex-pii": mock_client2},
        )
        coordinator = DetectorCoordinator(clients=clients)

        routing_plan = RoutingPlan(
            primary_detectors=["toxicity", "regex-pii"],
            parallel_groups=[["toxicity"], ["regex-pii"]],  # Two separate groups
            timeout_config={"toxicity": 3000, "regex-pii": 2000},
            retry_config={"toxicity": 0, "regex-pii": 0},
        )

        results = await coordinator.execute_routing_plan(
            content="Test content",
            routing_plan=routing_plan,
            request_id="test-request-123",
        )

        assert len(results) == 2
        detectors = {r.detector for r in results}
        assert detectors == {"toxicity", "regex-pii"}

        # Verify both clients were called
        mock_client1.analyze.assert_called_once()
        mock_client2.analyze.assert_called_once()

    async def test_execute_routing_plan_with_fallback(self):
        """Test executing a routing plan with fallback secondary detectors."""
        # Mock detector clients
        mock_client1 = Mock(spec=DetectorClient)
        mock_client1.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="toxicity",
                status=DetectorStatus.FAILED,
                error="timeout",
                processing_time_ms=3000,
            )
        )

        mock_client2 = Mock(spec=DetectorClient)
        mock_client2.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="echo",
                status=DetectorStatus.SUCCESS,
                output="clean",
                confidence=0.7,
                processing_time_ms=100,
            )
        )

        clients = cast(
            Dict[str, DetectorClient], {"toxicity": mock_client1, "echo": mock_client2}
        )
        coordinator = DetectorCoordinator(clients=clients)

        routing_plan = RoutingPlan(
            primary_detectors=["toxicity"],
            secondary_detectors=["echo"],  # Fallback detector
            parallel_groups=[["toxicity"]],
            timeout_config={"toxicity": 3000, "echo": 2000},
            retry_config={"toxicity": 0, "echo": 0},
        )

        results = await coordinator.execute_routing_plan(
            content="Test content",
            routing_plan=routing_plan,
            request_id="test-request-123",
        )

        # Should include both primary (failed) and secondary (success) results
        assert len(results) == 2
        result_statuses = {r.status for r in results}
        assert result_statuses == {DetectorStatus.FAILED, DetectorStatus.SUCCESS}

    async def test_execute_single_with_timeout_success(self):
        """Test single detector execution with successful result."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="toxicity",
                status=DetectorStatus.SUCCESS,
                output="clean",
                confidence=0.9,
                processing_time_ms=1500,
            )
        )

        clients = {"toxicity": mock_client}
        coordinator = DetectorCoordinator(clients=clients)

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.SUCCESS
        assert result.output == "clean"

    async def test_execute_single_with_timeout_timeout(self):
        """Test single detector execution with timeout."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(side_effect=asyncio.TimeoutError())

        clients = {"toxicity": mock_client}
        coordinator = DetectorCoordinator(clients=clients)

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.TIMEOUT
        assert result.error == "timeout"

    async def test_execute_single_with_timeout_exception(self):
        """Test single detector execution with general exception."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(side_effect=Exception("Connection failed"))

        clients = {"toxicity": mock_client}
        coordinator = DetectorCoordinator(clients=clients)

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.FAILED
        assert result.error == "Connection failed"

    async def test_execute_single_with_retries_on_timeout(self):
        """Test single detector execution with retries on timeout."""
        mock_client = Mock(spec=DetectorClient)
        # First call times out, second call succeeds
        mock_client.analyze = AsyncMock(
            side_effect=[
                asyncio.TimeoutError(),
                DetectorResult(
                    detector="toxicity",
                    status=DetectorStatus.SUCCESS,
                    output="clean",
                    confidence=0.9,
                    processing_time_ms=1500,
                ),
            ]
        )

        clients = {"toxicity": mock_client}
        coordinator = DetectorCoordinator(clients=clients, retry_on_timeouts=True)

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=1,  # Allow 1 retry
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.SUCCESS
        assert result.output == "clean"

        # Verify client was called twice (initial + 1 retry)
        assert mock_client.analyze.call_count == 2

    async def test_execute_single_with_retries_on_failure(self):
        """Test single detector execution with retries on failure."""
        mock_client = Mock(spec=DetectorClient)
        # First call fails, second call succeeds
        mock_client.analyze = AsyncMock(
            side_effect=[
                Exception("Temporary error"),
                DetectorResult(
                    detector="toxicity",
                    status=DetectorStatus.SUCCESS,
                    output="clean",
                    confidence=0.9,
                    processing_time_ms=1500,
                ),
            ]
        )

        clients = {"toxicity": mock_client}
        coordinator = DetectorCoordinator(clients=clients, retry_on_failures=True)

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=1,  # Allow 1 retry
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.SUCCESS
        assert result.output == "clean"

        # Verify client was called twice (initial + 1 retry)
        assert mock_client.analyze.call_count == 2

    async def test_execute_single_with_retries_disabled(self):
        """Test single detector execution with retries disabled."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(side_effect=Exception("Connection failed"))

        clients = {"toxicity": mock_client}
        coordinator = DetectorCoordinator(
            clients=clients, retry_on_timeouts=False, retry_on_failures=False
        )

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=3,  # Even though retries=3, should not retry
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.FAILED
        assert result.error == "Connection failed"

        # Verify client was called only once (no retries)
        assert mock_client.analyze.call_count == 1

    async def test_execute_single_unavailable_detector(self):
        """Test single detector execution when detector is not registered."""
        clients = {}  # No clients registered
        coordinator = DetectorCoordinator(clients=clients)

        result = await coordinator._execute_single_with_timeout(
            detector="nonexistent",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "nonexistent"
        assert result.status == DetectorStatus.UNAVAILABLE
        assert result.error == "not_registered"

    async def test_execute_single_with_circuit_breaker_open(self):
        """Test single detector execution with circuit breaker open."""
        mock_client = Mock(spec=DetectorClient)

        # Mock circuit breaker manager and breaker
        mock_breaker = Mock()
        mock_breaker.allow_request.return_value = False
        mock_breaker.state.value = "open"

        mock_breaker_manager = Mock(spec=CircuitBreakerManager)
        mock_breaker_manager.get.return_value = mock_breaker

        clients = {"toxicity": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        coordinator = DetectorCoordinator(
            clients=clients, breakers=mock_breaker_manager, metrics=metrics
        )

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.UNAVAILABLE
        assert result.error == "circuit_open"

        # Verify circuit breaker was checked
        mock_breaker.allow_request.assert_called_once()
        mock_breaker_manager.get.assert_called_once_with("toxicity")
        # Client should not be called when circuit is open
        mock_client.analyze.assert_not_called()

    async def test_execute_single_with_circuit_breaker_success(self):
        """Test single detector execution with circuit breaker success."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(
            return_value=DetectorResult(
                detector="toxicity",
                status=DetectorStatus.SUCCESS,
                output="clean",
                confidence=0.9,
                processing_time_ms=1500,
            )
        )

        # Mock circuit breaker manager and breaker
        mock_breaker = Mock()
        mock_breaker.allow_request.return_value = True
        mock_breaker.state.value = "closed"

        mock_breaker_manager = Mock(spec=CircuitBreakerManager)
        mock_breaker_manager.get.return_value = mock_breaker

        clients = {"toxicity": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        coordinator = DetectorCoordinator(
            clients=clients, breakers=mock_breaker_manager, metrics=metrics
        )

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.SUCCESS

        # Verify circuit breaker success was recorded
        mock_breaker.record_success.assert_called_once()
        mock_breaker_manager.get.assert_called_once_with("toxicity")

    async def test_execute_single_with_circuit_breaker_failure(self):
        """Test single detector execution with circuit breaker failure."""
        mock_client = Mock(spec=DetectorClient)
        mock_client.analyze = AsyncMock(side_effect=Exception("Connection failed"))

        # Mock circuit breaker manager and breaker
        mock_breaker = Mock()
        mock_breaker.allow_request.return_value = True
        mock_breaker.state.value = "closed"

        mock_breaker_manager = Mock(spec=CircuitBreakerManager)
        mock_breaker_manager.get.return_value = mock_breaker

        clients = {"toxicity": mock_client}
        metrics = Mock(spec=OrchestrationMetricsCollector)
        coordinator = DetectorCoordinator(
            clients=clients, breakers=mock_breaker_manager, metrics=metrics
        )

        result = await coordinator._execute_single_with_timeout(
            detector="toxicity",
            content="Test content",
            timeout_ms=3000,
            metadata={},
            retries=0,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.FAILED

        # Verify circuit breaker failure was recorded
        mock_breaker.record_failure.assert_called_once()
        mock_breaker_manager.get.assert_called_once_with("toxicity")
