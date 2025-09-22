"""
Comprehensive integration tests for the Detector Orchestration service.

This test suite validates all major functionality of the orchestration service
including orchestration, batch processing, job management, and health monitoring.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

# Import orchestration components
import sys
from pathlib import Path
orch_src = Path(__file__).parent.parent.parent / "detector-orchestration" / "src"
if str(orch_src) not in sys.path:
    sys.path.insert(0, str(orch_src))

from detector_orchestration.api.main import app
from detector_orchestration.models import (
    OrchestrationRequest,
    OrchestrationResponse,
    ContentType,
    ProcessingMode,
    Priority,
    DetectorResult,
    DetectorStatus,
    RoutingDecision,
    RoutingPlan
)


class MockDetectorClient:
    """Mock detector client for testing."""
    
    def __init__(self, name: str, success: bool = True, delay: float = 0.1):
        self.name = name
        self.success = success
        self.delay = delay
    
    async def detect(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock detection method."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        if not self.success:
            return {
                "detector": self.name,
                "status": "error",
                "error": "Mock detector failure",
                "processing_time_ms": 100
            }
        
        return {
            "detector": self.name,
            "status": "success",
            "output": f"Mock result from {self.name}",
            "confidence": 0.8,
            "processing_time_ms": 100,
            "metadata": metadata or {}
        }


@pytest.fixture
def test_client():
    """Create test client for the orchestration service."""
    return TestClient(app)


@pytest.fixture
def sample_orchestration_request():
    """Create a sample orchestration request."""
    return OrchestrationRequest(
        content="This is test content for orchestration",
        content_type=ContentType.TEXT,
        tenant_id="test-tenant",
        policy_bundle="default",
        environment="test",
        priority=Priority.NORMAL,
        processing_mode=ProcessingMode.SYNC
    )


@pytest.fixture
def mock_detector_clients():
    """Create mock detector clients."""
    return {
        "mock-detector-1": MockDetectorClient("mock-detector-1"),
        "mock-detector-2": MockDetectorClient("mock-detector-2"),
        "failing-detector": MockDetectorClient("failing-detector", success=False),
        "slow-detector": MockDetectorClient("slow-detector", delay=0.5)
    }


class TestOrchestrationEndpoints:
    """Test the main orchestration endpoints."""
    
    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data
        assert "detectors_total" in data
    
    def test_readiness_endpoint(self, test_client):
        """Test the readiness check endpoint."""
        response = test_client.get("/health/ready")
        
        # Should return 200 if ready, 503 if not ready
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
        assert data["status"] in ["ready", "not_ready"]
        assert "version" in data
        assert "dependencies" in data
    
    def test_metrics_endpoint(self, test_client):
        """Test the metrics endpoint."""
        response = test_client.get("/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    @patch("detector_orchestration.api.main.factory")
    def test_orchestrate_endpoint_success(self, mock_factory, test_client, sample_orchestration_request):
        """Test successful orchestration endpoint."""
        # Mock the factory components
        mock_factory.router.route_request = AsyncMock(return_value=(
            RoutingPlan(
                primary_detectors=["mock-detector-1", "mock-detector-2"],
                parallel_groups=[["mock-detector-1", "mock-detector-2"]],
                timeout_config={"mock-detector-1": 3000, "mock-detector-2": 3000},
                retry_config={"mock-detector-1": 1, "mock-detector-2": 1},
                coverage_method="weighted",
                weights={"mock-detector-1": 1.0, "mock-detector-2": 1.0},
                required_taxonomy_categories=[]
            ),
            RoutingDecision(
                selected_detectors=["mock-detector-1", "mock-detector-2"],
                policy_bundle="default",
                tenant_id="test-tenant",
                coverage_requirements={"coverage_threshold": 0.9},
                routing_reason="test",
                health_status={"mock-detector-1": "healthy", "mock-detector-2": "healthy"}
            )
        ))
        
        mock_factory.run_pipeline = AsyncMock(return_value=OrchestrationResponse(
            request_id="test-123",
            processing_mode=ProcessingMode.SYNC,
            detector_results=[
                DetectorResult(
                    detector="mock-detector-1",
                    status=DetectorStatus.SUCCESS,
                    output="Mock result 1",
                    confidence=0.8,
                    processing_time_ms=100
                ),
                DetectorResult(
                    detector="mock-detector-2",
                    status=DetectorStatus.SUCCESS,
                    output="Mock result 2",
                    confidence=0.9,
                    processing_time_ms=120
                )
            ],
            aggregated_payload=None,
            mapping_result=None,
            total_processing_time_ms=220,
            detectors_attempted=2,
            detectors_succeeded=2,
            detectors_failed=0,
            coverage_achieved=0.85,
            routing_decision=RoutingDecision(
                selected_detectors=["mock-detector-1", "mock-detector-2"],
                policy_bundle="default",
                tenant_id="test-tenant",
                coverage_requirements={"coverage_threshold": 0.9},
                routing_reason="test",
                health_status={"mock-detector-1": "healthy", "mock-detector-2": "healthy"}
            ),
            fallback_used=False,
            timestamp=datetime.now(timezone.utc)
        ))
        
        response = test_client.post(
            "/orchestrate",
            json=sample_orchestration_request.model_dump(),
            headers={"X-API-Key": "test-key", "X-Tenant-ID": "test-tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["request_id"] == "test-123"
        assert data["processing_mode"] == "sync"
        assert len(data["detector_results"]) == 2
        assert data["detectors_attempted"] == 2
        assert data["detectors_succeeded"] == 2
        assert data["detectors_failed"] == 0
    
    def test_orchestrate_endpoint_validation_error(self, test_client):
        """Test orchestration endpoint with invalid request."""
        invalid_request = {
            "content": "",  # Empty content should fail validation
            "content_type": "text",
            "tenant_id": "test-tenant"
        }
        
        response = test_client.post(
            "/orchestrate",
            json=invalid_request,
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_orchestrate_endpoint_missing_tenant(self, test_client):
        """Test orchestration endpoint without tenant information."""
        request_data = {
            "content": "Test content",
            "content_type": "text",
            "policy_bundle": "default"
        }
        
        response = test_client.post(
            "/orchestrate",
            json=request_data,
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 400  # Bad request - missing tenant


class TestBatchOrchestration:
    """Test batch orchestration functionality."""
    
    @patch("detector_orchestration.api.main.factory")
    def test_batch_orchestration_success(self, mock_factory, test_client):
        """Test successful batch orchestration."""
        batch_requests = [
            OrchestrationRequest(
                content=f"Test content {i}",
                content_type=ContentType.TEXT,
                tenant_id="test-tenant",
                policy_bundle="default",
                environment="test",
                priority=Priority.NORMAL
            ).model_dump()
            for i in range(3)
        ]
        
        # Mock the routing and pipeline execution
        mock_factory.router.route_request = AsyncMock(return_value=(
            RoutingPlan(
                primary_detectors=["mock-detector-1"],
                parallel_groups=[["mock-detector-1"]],
                timeout_config={"mock-detector-1": 3000},
                retry_config={"mock-detector-1": 1},
                coverage_method="weighted",
                weights={"mock-detector-1": 1.0},
                required_taxonomy_categories=[]
            ),
            RoutingDecision(
                selected_detectors=["mock-detector-1"],
                policy_bundle="default",
                tenant_id="test-tenant",
                coverage_requirements={"coverage_threshold": 0.9},
                routing_reason="batch-test",
                health_status={"mock-detector-1": "healthy"}
            )
        ))
        
        mock_factory.run_pipeline = AsyncMock(return_value=OrchestrationResponse(
            request_id="batch-test-123",
            processing_mode=ProcessingMode.SYNC,
            detector_results=[
                DetectorResult(
                    detector="mock-detector-1",
                    status=DetectorStatus.SUCCESS,
                    output="Mock batch result",
                    confidence=0.8,
                    processing_time_ms=100
                )
            ],
            aggregated_payload=None,
            mapping_result=None,
            total_processing_time_ms=100,
            detectors_attempted=1,
            detectors_succeeded=1,
            detectors_failed=0,
            coverage_achieved=0.8,
            routing_decision=RoutingDecision(
                selected_detectors=["mock-detector-1"],
                policy_bundle="default",
                tenant_id="test-tenant",
                coverage_requirements={"coverage_threshold": 0.9},
                routing_reason="batch-test",
                health_status={"mock-detector-1": "healthy"}
            ),
            fallback_used=False,
            timestamp=datetime.now(timezone.utc)
        ))
        
        response = test_client.post(
            "/orchestrate/batch",
            json={"requests": batch_requests},
            headers={"X-API-Key": "test-key", "X-Tenant-ID": "test-tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "results" in data
        assert len(data["results"]) == 3
        assert "errors" not in data or data["errors"] is None
    
    def test_batch_orchestration_empty_request(self, test_client):
        """Test batch orchestration with empty request."""
        response = test_client.post(
            "/orchestrate/batch",
            json={"requests": []},
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "empty_batch" in str(data)


class TestJobManagement:
    """Test async job management functionality."""
    
    @patch("detector_orchestration.api.main.factory")
    def test_async_job_creation(self, mock_factory, test_client, sample_orchestration_request):
        """Test creating an async job."""
        # Set processing mode to async
        sample_orchestration_request.processing_mode = ProcessingMode.ASYNC
        
        # Mock job manager
        mock_job_manager = Mock()
        mock_job_manager.enqueue = AsyncMock(return_value="job-123")
        mock_factory.job_manager = mock_job_manager
        
        # Mock routing
        mock_factory.router.route_request = AsyncMock(return_value=(
            RoutingPlan(
                primary_detectors=["mock-detector-1"],
                parallel_groups=[["mock-detector-1"]],
                timeout_config={"mock-detector-1": 3000},
                retry_config={"mock-detector-1": 1},
                coverage_method="weighted",
                weights={"mock-detector-1": 1.0},
                required_taxonomy_categories=[]
            ),
            RoutingDecision(
                selected_detectors=["mock-detector-1"],
                policy_bundle="default",
                tenant_id="test-tenant",
                coverage_requirements={"coverage_threshold": 0.9},
                routing_reason="async-test",
                health_status={"mock-detector-1": "healthy"}
            )
        ))
        
        response = test_client.post(
            "/orchestrate",
            json=sample_orchestration_request.model_dump(),
            headers={"X-API-Key": "test-key", "X-Tenant-ID": "test-tenant"}
        )
        
        assert response.status_code == 202  # Accepted
        data = response.json()
        
        assert data["job_id"] == "job-123"
        assert data["status"] == "pending"
        assert "estimated_processing_time_ms" in data
    
    @patch("detector_orchestration.api.main.factory")
    def test_job_status_check(self, mock_factory, test_client):
        """Test checking job status."""
        # Mock job manager
        mock_job_manager = Mock()
        mock_job_manager.get_status.return_value = {
            "job_id": "job-123",
            "status": "completed",
            "result": {"detector_results": []},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        mock_factory.job_manager = mock_job_manager
        
        response = test_client.get(
            "/orchestrate/status/job-123",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == "job-123"
        assert data["status"] == "completed"
    
    @patch("detector_orchestration.api.main.factory")
    def test_job_cancellation(self, mock_factory, test_client):
        """Test job cancellation."""
        # Mock job manager
        mock_job_manager = Mock()
        mock_job_manager.cancel.return_value = True
        mock_factory.job_manager = mock_job_manager
        
        response = test_client.delete(
            "/orchestrate/status/job-123",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["job_id"] == "job-123"
        assert data["status"] == "cancelled"


class TestDetectorManagement:
    """Test detector registry and management endpoints."""
    
    @patch("detector_orchestration.api.main.factory")
    def test_list_detectors(self, mock_factory, test_client):
        """Test listing available detectors."""
        # Mock registry
        mock_registry = Mock()
        mock_registry.list.return_value = ["detector-1", "detector-2", "detector-3"]
        mock_factory.registry = mock_registry
        
        # Mock settings
        mock_factory.settings.detectors = {
            "detector-1": Mock(name="detector-1", model_dump=Mock(return_value={"name": "detector-1", "endpoint": "http://detector-1"})),
            "detector-2": Mock(name="detector-2", model_dump=Mock(return_value={"name": "detector-2", "endpoint": "http://detector-2"})),
            "detector-3": Mock(name="detector-3", model_dump=Mock(return_value={"name": "detector-3", "endpoint": "http://detector-3"}))
        }
        
        response = test_client.get(
            "/detectors",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "detectors" in data
        assert len(data["detectors"]) == 3
    
    @patch("detector_orchestration.api.main.factory")
    def test_register_detector(self, mock_factory, test_client):
        """Test registering a new detector."""
        # Mock registry
        mock_registry = Mock()
        mock_factory.registry = mock_registry
        
        detector_data = {
            "name": "new-detector",
            "endpoint": "http://new-detector:8080",
            "capabilities": ["text"],
            "timeout_ms": 5000
        }
        
        response = test_client.post(
            "/detectors",
            json=detector_data,
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert data["detector"] == "new-detector"
        mock_registry.register.assert_called_once()


class TestPolicyManagement:
    """Test policy management endpoints."""
    
    @patch("detector_orchestration.api.main.factory")
    def test_list_policies(self, mock_factory, test_client):
        """Test listing policies for a tenant."""
        # Mock policy store
        mock_policy_store = Mock()
        mock_policy_store.list_policies.return_value = ["policy-1", "policy-2"]
        mock_factory.policy_store = mock_policy_store
        
        response = test_client.get(
            "/policies/test-tenant",
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["tenant"] == "test-tenant"
        assert "bundles" in data
        assert len(data["bundles"]) == 2
    
    @patch("detector_orchestration.api.main.factory")
    def test_submit_policy(self, mock_factory, test_client):
        """Test submitting a new policy."""
        # Mock policy store
        mock_policy_store = Mock()
        mock_policy_store.submit_policy.return_value = Mock(
            model_dump=Mock(return_value={
                "version_id": "v1",
                "status": "approved",
                "created_at": datetime.now(timezone.utc).isoformat()
            })
        )
        mock_factory.policy_store = mock_policy_store
        mock_factory.settings.detectors = {"detector-1": Mock()}
        
        policy_data = {
            "policy": {
                "rules": [
                    {
                        "detector": "detector-1",
                        "threshold": 0.8,
                        "action": "flag"
                    }
                ]
            },
            "description": "Test policy",
            "requires_approval": False
        }
        
        response = test_client.post(
            "/policies/test-tenant/test-bundle",
            json=policy_data,
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "version" in data
        assert data["version"]["version_id"] == "v1"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_unauthorized_access(self, test_client):
        """Test access without proper authentication."""
        response = test_client.post(
            "/orchestrate",
            json={"content": "test", "content_type": "text", "tenant_id": "test"}
        )
        
        assert response.status_code == 401
    
    def test_invalid_tenant_access(self, test_client):
        """Test access with invalid tenant."""
        response = test_client.post(
            "/orchestrate",
            json={"content": "test", "content_type": "text", "tenant_id": "invalid-tenant"},
            headers={"X-API-Key": "test-key"}
        )
        
        # Should either succeed (if tenant validation is disabled) or fail with 403
        assert response.status_code in [200, 403]
    
    @patch("detector_orchestration.api.main.factory")
    def test_service_unavailable(self, mock_factory, test_client):
        """Test behavior when services are unavailable."""
        # Mock unavailable router
        mock_factory.router = None
        
        response = test_client.post(
            "/orchestrate",
            json={
                "content": "test",
                "content_type": "text",
                "tenant_id": "test-tenant",
                "policy_bundle": "default"
            },
            headers={"X-API-Key": "test-key"}
        )
        
        assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
