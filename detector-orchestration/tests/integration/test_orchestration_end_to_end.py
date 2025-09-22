"""End-to-end integration tests for detector orchestration pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Iterable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from detector_orchestration.api.main import app
from detector_orchestration.models import (
    ContentType,
    OrchestrationRequest,
    OrchestrationResponse,
    Priority,
    ProcessingMode,
    DetectorStatus,
    DetectorResult,
    RoutingDecision,
    RoutingPlan,
)


class MockDetector:
    """Mock detector for testing."""

    def __init__(self, name: str, fail: bool = False, timeout: bool = False):
        self.name = name
        self.fail = fail
        self.timeout = timeout

    async def mock_detect(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> dict:
        """Mock detector response."""
        if self.timeout:
            await asyncio.sleep(5)  # Simulate timeout
            return {"error": "timeout"}

        if self.fail:
            return {"error": "detector failed"}

        return {
            "detector": self.name,
            "status": "success",
            "output": f"Mock result from {self.name}",
            "confidence": 0.8,
            "processing_time_ms": 100,
        }


@pytest.fixture
def test_client():
    """Provide test client for FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_detectors():
    """Provide mock detectors for testing."""
    return {
        "mock-detector-1": MockDetector("mock-detector-1"),
        "mock-detector-2": MockDetector("mock-detector-2"),
        "mock-detector-3": MockDetector("mock-detector-3"),
        "failing-detector": MockDetector("failing-detector", fail=True),
        "timeout-detector": MockDetector("timeout-detector", timeout=True),
    }


@pytest.fixture
def sample_orchestration_request() -> OrchestrationRequest:
    """Provide a sample orchestration request."""
    return OrchestrationRequest(
        content="This is test content for detector analysis",
        content_type=ContentType.TEXT,
        tenant_id="test-tenant",
        policy_bundle="default",
        environment="dev",
        processing_mode=ProcessingMode.SYNC,
        priority=Priority.NORMAL,
        metadata={"test": True},
    )


def build_routing_decision(
    selected_detectors: list[str],
    *,
    policy_bundle: str = "default",
    tenant_id: str = "test-tenant",
    coverage_requirements: dict[str, Any] | None = None,
    routing_reason: str = "integration-test",
    health_status: Iterable[str] | None = None,
) -> RoutingDecision:
    healthy_set = set(health_status or selected_detectors)
    return RoutingDecision(
        selected_detectors=selected_detectors,
        routing_reason=routing_reason,
        policy_applied=policy_bundle,
        coverage_requirements=dict(
            coverage_requirements or {"min_success_fraction": 1.0}
        ),
        health_status={
            detector: detector in healthy_set for detector in selected_detectors
        },
    )


class TestOrchestrationEndToEnd:
    """End-to-end tests for the full orchestration pipeline."""

    def test_full_orchestration_pipeline_success(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
        mock_detectors: dict,
    ):
        """Test successful end-to-end orchestration pipeline."""
        # Mock the detector clients
        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.detector_clients = mock_detectors

            # Mock the router and coordinator
            mock_routing_plan = RoutingPlan(
                primary_detectors=["mock-detector-1", "mock-detector-2"],
                parallel_groups=[["mock-detector-1", "mock-detector-2"]],
                timeout_config={"mock-detector-1": 3000, "mock-detector-2": 3000},
                retry_config={"mock-detector-1": 1, "mock-detector-2": 1},
                coverage_method="weighted",
                weights={"mock-detector-1": 1.0, "mock-detector-2": 1.0},
                required_taxonomy_categories=[],
            )

            mock_decision = build_routing_decision(
                selected_detectors=["mock-detector-1", "mock-detector-2"],
                coverage_requirements={
                    "coverage_threshold": 0.9,
                    "coverage_method": "weighted",
                },
            )

            with patch(
                "detector_orchestration.api.main.ContentRouter"
            ) as mock_router_class:
                mock_router = MagicMock()
                mock_router.route_request = AsyncMock(
                    return_value=(mock_routing_plan, mock_decision)
                )
                mock_router_class.return_value = mock_router

                with patch(
                    "detector_orchestration.api.main.DetectorCoordinator"
                ) as mock_coord_class:
                    mock_coord = MagicMock()
                    mock_coord.execute_detector_group = AsyncMock(
                        return_value=[
                            DetectorResult(
                                detector="mock-detector-1",
                                status=DetectorStatus.SUCCESS,
                                output="Mock result 1",
                                confidence=0.8,
                                processing_time_ms=100,
                            ),
                            DetectorResult(
                                detector="mock-detector-2",
                                status=DetectorStatus.SUCCESS,
                                output="Mock result 2",
                                confidence=0.9,
                                processing_time_ms=120,
                            ),
                        ]
                    )
                    mock_coord_class.return_value = mock_coord

                    with patch(
                        "detector_orchestration.api.main.ResponseAggregator"
                    ) as mock_agg_class:
                        mock_agg = MagicMock()
                        mock_agg.aggregate = MagicMock(
                            return_value=(
                                MagicMock(),  # payload
                                0.95,  # coverage
                            )
                        )
                        mock_agg_class.return_value = mock_agg

                        response = test_client.post(
                            "/orchestrate",
                            json=sample_orchestration_request.model_dump(),
                            headers={
                                "Authorization": "Bearer test-token",
                                "X-Tenant-ID": "test-tenant",
                            },
                        )

                        assert response.status_code == 200
                        result = OrchestrationResponse(**response.json())

                        assert result.detectors_attempted == 2
                        assert result.detectors_succeeded == 2
                        assert result.detectors_failed == 0
                        assert result.coverage_achieved == 0.95
                        assert result.fallback_used is False
                        assert result.error_code is None
                        assert len(result.detector_results) == 2

    def test_orchestration_with_detector_failures(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
    ):
        """Test orchestration pipeline with detector failures."""
        # Update request to include failing detector
        sample_orchestration_request.content = "Test content with failures"

        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.detector_clients = {
                "failing-detector-1": MockDetector("failing-detector-1", fail=True),
                "failing-detector-2": MockDetector("failing-detector-2", fail=True),
            }

            mock_routing_plan = RoutingPlan(
                primary_detectors=["failing-detector-1", "failing-detector-2"],
                parallel_groups=[["failing-detector-1", "failing-detector-2"]],
                timeout_config={"failing-detector-1": 3000, "failing-detector-2": 3000},
                retry_config={"failing-detector-1": 1, "failing-detector-2": 1},
                coverage_method="weighted",
                weights={"failing-detector-1": 1.0, "failing-detector-2": 1.0},
                required_taxonomy_categories=[],
            )

            mock_decision = build_routing_decision(
                selected_detectors=["failing-detector-1", "failing-detector-2"],
                coverage_requirements={
                    "coverage_threshold": 1.0,
                    "coverage_method": "weighted",
                },
                routing_reason="failure-path",
                health_status=[],
            )

            with patch(
                "detector_orchestration.api.main.ContentRouter"
            ) as mock_router_class:
                mock_router = MagicMock()
                mock_router.route_request = AsyncMock(
                    return_value=(mock_routing_plan, mock_decision)
                )
                mock_router_class.return_value = mock_router

                with patch(
                    "detector_orchestration.api.main.DetectorCoordinator"
                ) as mock_coord_class:
                    mock_coord = MagicMock()
                    mock_coord.execute_detector_group = AsyncMock(
                        return_value=[
                            DetectorResult(
                                detector="failing-detector-1",
                                status=DetectorStatus.FAILED,
                                output="Failed",
                                confidence=0.0,
                                processing_time_ms=100,
                            ),
                            DetectorResult(
                                detector="failing-detector-2",
                                status=DetectorStatus.FAILED,
                                output="Failed",
                                confidence=0.0,
                                processing_time_ms=100,
                            ),
                        ]
                    )
                    mock_coord_class.return_value = mock_coord

                    with patch(
                        "detector_orchestration.api.main.ResponseAggregator"
                    ) as mock_agg_class:
                        mock_agg = MagicMock()
                        mock_agg.aggregate = MagicMock(
                            return_value=(
                                MagicMock(),  # payload
                                0.0,  # coverage
                            )
                        )
                        mock_agg_class.return_value = mock_agg

                        response = test_client.post(
                            "/orchestrate",
                            json=sample_orchestration_request.model_dump(),
                            headers={
                                "Authorization": "Bearer test-token",
                                "X-Tenant-ID": "test-tenant",
                            },
                        )

                        assert response.status_code == 502  # Communication failed
                        result = OrchestrationResponse(**response.json())

                        assert result.detectors_attempted == 2
                        assert result.detectors_succeeded == 0
                        assert result.detectors_failed == 2
                        assert result.coverage_achieved == 0.0
                        assert result.fallback_used is True
                        assert result.error_code == "DETECTOR_COMMUNICATION_FAILED"

    def test_orchestration_with_partial_coverage(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
    ):
        """Test orchestration with partial coverage (206 response)."""
        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.detector_clients = {
                "partial-detector-1": MockDetector("partial-detector-1")
            }

            mock_routing_plan = RoutingPlan(
                primary_detectors=["partial-detector-1"],
                parallel_groups=[["partial-detector-1"]],
                timeout_config={"partial-detector-1": 3000},
                retry_config={"partial-detector-1": 1},
                coverage_method="weighted",
                weights={"partial-detector-1": 1.0},
                required_taxonomy_categories=[],
            )

            mock_decision = build_routing_decision(
                selected_detectors=["partial-detector-1"],
                coverage_requirements={
                    "coverage_threshold": 0.8,
                    "coverage_method": "weighted",
                },
                routing_reason="partial-coverage",
            )

            with patch(
                "detector_orchestration.api.main.ContentRouter"
            ) as mock_router_class:
                mock_router = MagicMock()
                mock_router.route_request = AsyncMock(
                    return_value=(mock_routing_plan, mock_decision)
                )
                mock_router_class.return_value = mock_router

                with patch(
                    "detector_orchestration.api.main.DetectorCoordinator"
                ) as mock_coord_class:
                    mock_coord = MagicMock()
                    mock_coord.execute_detector_group = AsyncMock(
                        return_value=[
                            DetectorResult(
                                detector="partial-detector-1",
                                status=DetectorStatus.SUCCESS,
                                output="Partial result",
                                confidence=0.6,
                                processing_time_ms=100,
                            ),
                        ]
                    )
                    mock_coord_class.return_value = mock_coord

                    with patch(
                        "detector_orchestration.api.main.ResponseAggregator"
                    ) as mock_agg_class:
                        mock_agg = MagicMock()
                        mock_agg.aggregate = MagicMock(
                            return_value=(
                                MagicMock(),  # payload
                                0.6,  # coverage below 0.8 threshold
                            )
                        )
                        mock_agg_class.return_value = mock_agg

                        response = test_client.post(
                            "/orchestrate",
                            json=sample_orchestration_request.model_dump(),
                            headers={
                                "Authorization": "Bearer test-token",
                                "X-Tenant-ID": "test-tenant",
                            },
                        )

                        assert response.status_code == 206  # Partial content
                        result = OrchestrationResponse(**response.json())

                        assert result.coverage_achieved == 0.6
                        assert result.error_code == "PARTIAL_COVERAGE"
                        assert result.fallback_used is True

    def test_async_job_processing(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
    ):
        """Test async job processing and status tracking."""
        # Set request to async mode
        sample_orchestration_request.processing_mode = ProcessingMode.ASYNC

        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.job_manager = MagicMock()
            mock_factory.job_manager.enqueue = AsyncMock(return_value="test-job-123")
            mock_factory.job_manager.get_status = MagicMock(return_value=MagicMock())

            response = test_client.post(
                "/orchestrate",
                json=sample_orchestration_request.model_dump(),
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Tenant-ID": "test-tenant",
                },
            )

            assert response.status_code == 202  # Accepted
            result = OrchestrationResponse(**response.json())

            assert result.job_id == "test-job-123"
            assert result.processing_mode == ProcessingMode.ASYNC
            mock_factory.job_manager.enqueue.assert_called_once()

    def test_batch_orchestration(
        self,
        test_client: TestClient,
    ):
        """Test batch orchestration processing."""
        batch_requests = [
            OrchestrationRequest(
                content=f"Test content {i}",
                content_type=ContentType.TEXT,
                tenant_id="test-tenant",
                policy_bundle="default",
                environment="dev",
                priority=Priority.NORMAL,
            )
            for i in range(3)
        ]

        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.coordinator = MagicMock()
            mock_response = OrchestrationResponse(
                request_id="req",
                processing_mode=ProcessingMode.SYNC,
                detector_results=[],
                aggregated_payload=None,
                mapping_result=None,
                total_processing_time_ms=100,
                detectors_attempted=1,
                detectors_succeeded=1,
                detectors_failed=0,
                coverage_achieved=1.0,
                routing_decision=build_routing_decision(
                    ["mock-detector"], routing_reason="batch"
                ),
                fallback_used=False,
                timestamp=datetime.now(timezone.utc),
            )
            mock_factory.coordinator.return_value = mock_response

            response = test_client.post(
                "/orchestrate/batch",
                json={"requests": [req.model_dump() for req in batch_requests]},
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Tenant-ID": "test-tenant",
                },
            )

            assert response.status_code == 200
            result = response.json()

            assert "results" in result
            assert len(result["results"]) == 3
            assert mock_factory.coordinator.call_count == 3  # Called once per request

    def test_idempotency_handling(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
    ):
        """Test idempotency key handling."""
        idempotency_key = "test-idempotency-key-123"

        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.idempotency_cache = MagicMock()
            mock_factory.idempotency_cache.get = MagicMock(return_value="cached-response")
            mock_factory.idempotency_cache.get_entry = MagicMock(return_value=None)

            response = test_client.post(
                "/orchestrate",
                json=sample_orchestration_request.model_dump(),
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Tenant-ID": "test-tenant",
                    "Idempotency-Key": idempotency_key,
                },
            )

            # Should return cached response
            assert response.status_code == 200
            mock_factory.idempotency_cache.get.assert_called_with(idempotency_key)

    def test_response_caching(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
    ):
        """Test response caching functionality."""
        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.response_cache = MagicMock()
            mock_factory.response_cache.get = MagicMock(return_value="cached-response")
            mock_factory.response_cache.build_key = MagicMock(return_value="test-cache-key")

            with patch("detector_orchestration.api.main.settings") as mock_settings:
                mock_settings.config.cache_enabled = True

                response = test_client.post(
                    "/orchestrate",
                    json=sample_orchestration_request.model_dump(),
                    headers={
                        "Authorization": "Bearer test-token",
                        "X-Tenant-ID": "test-tenant",
                    },
                )

                # Should return cached response
                assert response.status_code == 200
                mock_factory.response_cache.get.assert_called_once()

    def test_error_handling_and_retry_logic(
        self,
        test_client: TestClient,
        sample_orchestration_request: OrchestrationRequest,
    ):
        """Test error handling and retry logic in orchestration."""
        with patch("detector_orchestration.api.main.factory") as mock_factory:
            mock_factory.detector_clients = {
                "retry-detector-1": MockDetector("retry-detector-1")
            }

            mock_routing_plan = RoutingPlan(
                primary_detectors=["retry-detector-1"],
                parallel_groups=[["retry-detector-1"]],
                timeout_config={"retry-detector-1": 3000},
                retry_config={"retry-detector-1": 3},  # Multiple retries
                coverage_method="weighted",
                weights={"retry-detector-1": 1.0},
                required_taxonomy_categories=[],
            )

            mock_decision = build_routing_decision(
                selected_detectors=["retry-detector-1"],
                coverage_requirements={
                    "coverage_threshold": 1.0,
                    "coverage_method": "weighted",
                },
                routing_reason="retry",
            )

            with patch(
                "detector_orchestration.api.main.ContentRouter"
            ) as mock_router_class:
                mock_router = MagicMock()
                mock_router.route_request = AsyncMock(
                    return_value=(mock_routing_plan, mock_decision)
                )
                mock_router_class.return_value = mock_router

                with patch(
                    "detector_orchestration.api.main.DetectorCoordinator"
                ) as mock_coord_class:
                    mock_coord = MagicMock()
                    # First call fails, retry succeeds
                    mock_coord.execute_detector_group = AsyncMock(
                        side_effect=[
                            [
                                DetectorResult(
                                    detector="retry-detector-1",
                                    status=DetectorStatus.FAILED,
                                    output="Retry failed",
                                    confidence=0.0,
                                    processing_time_ms=100,
                                )
                            ],
                            [
                                DetectorResult(
                                    detector="retry-detector-1",
                                    status=DetectorStatus.SUCCESS,
                                    output="Retry succeeded",
                                    confidence=0.8,
                                    processing_time_ms=100,
                                )
                            ],
                        ]
                    )
                    mock_coord_class.return_value = mock_coord

                    with patch(
                        "detector_orchestration.api.main.ResponseAggregator"
                    ) as mock_agg_class:
                        mock_agg = MagicMock()
                        mock_agg.aggregate = MagicMock(
                            return_value=(
                                MagicMock(),  # payload
                                1.0,  # coverage
                            )
                        )
                        mock_agg_class.return_value = mock_agg

                        response = test_client.post(
                            "/orchestrate",
                            json=sample_orchestration_request.model_dump(),
                            headers={
                                "Authorization": "Bearer test-token",
                                "X-Tenant-ID": "test-tenant",
                            },
                        )

                        assert response.status_code == 200
                        result = OrchestrationResponse(**response.json())

                        assert result.detectors_attempted == 1
                        assert result.detectors_succeeded == 1
                        assert result.detectors_failed == 0
                        # Should have attempted retries
                        assert mock_coord.execute_detector_group.call_count == 2
