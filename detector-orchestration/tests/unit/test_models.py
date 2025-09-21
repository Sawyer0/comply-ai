"""Tests for data models."""

import pytest
from datetime import datetime

from detector_orchestration.models import (
    OrchestrationRequest,
    OrchestrationResponse,
    DetectorResult,
    MapperPayload,
    RoutingDecision,
    RoutingPlan,
    ContentType,
    ProcessingMode,
    Priority,
    DetectorStatus,
    Provenance,
    PolicyContext,
    MappingResponse,
    DetectorCapabilities,
    JobStatus,
    JobStatusResponse,
    ErrorBody,
)


class TestOrchestrationRequest:
    def test_valid_request_creation(self):
        """Test creating a valid orchestration request."""
        request = OrchestrationRequest(
            content="Test content for analysis",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
        )

        assert request.content == "Test content for analysis"
        assert request.content_type == ContentType.TEXT
        assert request.tenant_id == "test-tenant"
        assert request.policy_bundle == "default"
        assert request.processing_mode == ProcessingMode.SYNC
        assert request.priority == Priority.NORMAL
        assert request.metadata is None
        assert request.required_detectors is None
        assert request.excluded_detectors is None

    def test_request_with_optional_fields(self):
        """Test request with optional fields populated."""
        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="custom",
            processing_mode=ProcessingMode.ASYNC,
            priority=Priority.HIGH,
            metadata={"key": "value"},
            required_detectors=["toxicity", "regex-pii"],
            excluded_detectors=["echo"],
        )

        assert request.processing_mode == ProcessingMode.ASYNC
        assert request.priority == Priority.HIGH
        assert request.metadata == {"key": "value"}
        assert request.required_detectors == ["toxicity", "regex-pii"]
        assert request.excluded_detectors == ["echo"]

    def test_request_content_length_validation(self):
        """Test content length validation."""
        # Test with content too long (max_length is 50000)
        long_content = "a" * 50001
        with pytest.raises(ValueError, match="max_length"):
            OrchestrationRequest(
                content=long_content,
                content_type=ContentType.TEXT,
                tenant_id="test-tenant",
                policy_bundle="default",
            )

    def test_request_tenant_id_validation(self):
        """Test tenant ID validation."""
        # Test empty tenant ID
        with pytest.raises(ValueError, match="min_length"):
            OrchestrationRequest(
                content="test",
                content_type=ContentType.TEXT,
                tenant_id="",
                policy_bundle="default",
            )

        # Test tenant ID too long
        long_tenant_id = "a" * 65
        with pytest.raises(ValueError, match="max_length"):
            OrchestrationRequest(
                content="test",
                content_type=ContentType.TEXT,
                tenant_id=long_tenant_id,
                policy_bundle="default",
            )


class TestDetectorResult:
    def test_detector_result_creation(self):
        """Test creating a detector result."""
        result = DetectorResult(
            detector="toxicity",
            status=DetectorStatus.SUCCESS,
            output="clean",
            confidence=0.9,
            processing_time_ms=1500,
        )

        assert result.detector == "toxicity"
        assert result.status == DetectorStatus.SUCCESS
        assert result.output == "clean"
        assert result.confidence == 0.9
        assert result.processing_time_ms == 1500
        assert result.metadata is None
        assert result.error is None

    def test_detector_result_with_error(self):
        """Test detector result with error information."""
        result = DetectorResult(
            detector="toxicity",
            status=DetectorStatus.FAILED,
            error="Connection timeout",
            processing_time_ms=3000,
        )

        assert result.status == DetectorStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.output is None
        assert result.confidence is None


class TestMapperPayload:
    def test_mapper_payload_creation(self):
        """Test creating a mapper payload."""
        metadata = {"key": "value", "confidence": 0.8}
        payload = MapperPayload(
            detector="orchestrated-multi",
            output="clean",
            tenant_id="test-tenant",
            metadata=metadata,
        )

        assert payload.detector == "orchestrated-multi"
        assert payload.output == "clean"
        assert payload.tenant_id == "test-tenant"
        assert payload.metadata == metadata


class TestRoutingDecision:
    def test_routing_decision_creation(self):
        """Test creating a routing decision."""
        decision = RoutingDecision(
            selected_detectors=["toxicity", "regex-pii"],
            routing_reason="policy+default",
            policy_applied="default",
            coverage_requirements={"min_success_fraction": 1.0},
            health_status={"toxicity": True, "regex-pii": True},
        )

        assert decision.selected_detectors == ["toxicity", "regex-pii"]
        assert decision.routing_reason == "policy+default"
        assert decision.policy_applied == "default"
        assert decision.coverage_requirements == {"min_success_fraction": 1.0}
        assert decision.health_status == {"toxicity": True, "regex-pii": True}


class TestRoutingPlan:
    def test_routing_plan_creation(self):
        """Test creating a routing plan."""
        plan = RoutingPlan(
            primary_detectors=["toxicity", "regex-pii"],
            secondary_detectors=["echo"],
            parallel_groups=[["toxicity", "regex-pii"]],
            timeout_config={"toxicity": 3000, "regex-pii": 2000},
            retry_config={"toxicity": 1, "regex-pii": 2},
            coverage_method="required_set",
            weights={"toxicity": 0.7, "regex-pii": 0.3},
            required_taxonomy_categories=["security", "privacy"],
        )

        assert plan.primary_detectors == ["toxicity", "regex-pii"]
        assert plan.secondary_detectors == ["echo"]
        assert plan.parallel_groups == [["toxicity", "regex-pii"]]
        assert plan.timeout_config == {"toxicity": 3000, "regex-pii": 2000}
        assert plan.retry_config == {"toxicity": 1, "regex-pii": 2}
        assert plan.coverage_method == "required_set"
        assert plan.weights == {"toxicity": 0.7, "regex-pii": 0.3}
        assert plan.required_taxonomy_categories == ["security", "privacy"]


class TestOrchestrationResponse:
    def test_orchestration_response_creation(self):
        """Test creating an orchestration response."""
        routing_decision = RoutingDecision(
            selected_detectors=["toxicity"],
            routing_reason="policy+default",
            policy_applied="default",
            coverage_requirements={"min_success_fraction": 1.0},
            health_status={"toxicity": True},
        )

        response = OrchestrationResponse(
            request_id="test-request-123",
            processing_mode=ProcessingMode.SYNC,
            detector_results=[
                DetectorResult(
                    detector="toxicity",
                    status=DetectorStatus.SUCCESS,
                    output="clean",
                    confidence=0.9,
                    processing_time_ms=1500,
                )
            ],
            total_processing_time_ms=1500,
            detectors_attempted=1,
            detectors_succeeded=1,
            detectors_failed=0,
            coverage_achieved=1.0,
            routing_decision=routing_decision,
            fallback_used=False,
            timestamp=datetime.now(),
        )

        assert response.request_id == "test-request-123"
        assert response.processing_mode == ProcessingMode.SYNC
        assert len(response.detector_results) == 1
        assert response.detectors_attempted == 1
        assert response.detectors_succeeded == 1
        assert response.detectors_failed == 0
        assert response.coverage_achieved == 1.0
        assert response.fallback_used is False


class TestProvenance:
    def test_provenance_creation(self):
        """Test creating provenance information."""
        provenance = Provenance(
            vendor="test-vendor",
            detector="toxicity",
            detector_version="1.0.0",
            raw_ref="test-ref",
            route="test-route",
            model="test-model",
            tenant_id="test-tenant",
            ts=datetime.now(),
        )

        assert provenance.vendor == "test-vendor"
        assert provenance.detector == "toxicity"
        assert provenance.detector_version == "1.0.0"
        assert provenance.raw_ref == "test-ref"
        assert provenance.route == "test-route"
        assert provenance.model == "test-model"
        assert provenance.tenant_id == "test-tenant"


class TestMappingResponse:
    def test_mapping_response_creation(self):
        """Test creating a mapping response."""
        provenance = Provenance(
            detector="toxicity",
            detector_version="1.0.0",
            tenant_id="test-tenant",
        )

        policy_context = PolicyContext(
            expected_detectors=["toxicity", "regex-pii"],
            environment="dev",
        )

        response = MappingResponse(
            taxonomy=["security", "privacy"],
            scores={"security": 0.8, "privacy": 0.6},
            confidence=0.9,
            notes="Test mapping response",
            provenance=provenance,
            policy_context=policy_context,
        )

        assert response.taxonomy == ["security", "privacy"]
        assert response.scores == {"security": 0.8, "privacy": 0.6}
        assert response.confidence == 0.9
        assert response.notes == "Test mapping response"
        assert response.provenance == provenance
        assert response.policy_context == policy_context


class TestDetectorCapabilities:
    def test_detector_capabilities_creation(self):
        """Test creating detector capabilities."""
        capabilities = DetectorCapabilities(
            supported_content_types=[ContentType.TEXT, ContentType.DOCUMENT],
            max_content_length=50000,
            average_processing_time_ms=1500,
            confidence_calibrated=True,
            batch_supported=False,
        )

        assert capabilities.supported_content_types == [ContentType.TEXT, ContentType.DOCUMENT]
        assert capabilities.max_content_length == 50000
        assert capabilities.average_processing_time_ms == 1500
        assert capabilities.confidence_calibrated is True
        assert capabilities.batch_supported is False


class TestJobStatusResponse:
    def test_job_status_response_creation(self):
        """Test creating a job status response."""
        response = JobStatusResponse(
            job_id="test-job-123",
            status=JobStatus.COMPLETED,
            progress=1.0,
            result=None,
            error=None,
        )

        assert response.job_id == "test-job-123"
        assert response.status == JobStatus.COMPLETED
        assert response.progress == 1.0
        assert response.result is None
        assert response.error is None


class TestErrorBody:
    def test_error_body_creation(self):
        """Test creating an error body."""
        error = ErrorBody(
            error_code="ORCH_001",
            message="Test error message",
            request_id="test-request-123",
            retryable=True,
        )

        assert error.error_code == "ORCH_001"
        assert error.message == "Test error message"
        assert error.request_id == "test-request-123"
        assert error.retryable is True
