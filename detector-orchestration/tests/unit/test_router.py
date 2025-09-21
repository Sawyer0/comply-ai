"""Tests for content router."""

import pytest
from unittest.mock import Mock, AsyncMock

from detector_orchestration.models import (
    OrchestrationRequest,
    RoutingPlan,
    RoutingDecision,
    ContentType,
    Priority,
    ProcessingMode,
)
from detector_orchestration.config import Settings, OrchestrationConfig, DetectorEndpoint
from detector_orchestration.router import ContentRouter
from detector_orchestration.policy import PolicyManager, PolicyDecision, CoverageMethod


class TestContentRouter:
    def test_router_initialization(self):
        """Test content router initialization."""
        settings = Settings()
        health_monitor = Mock()
        policy_manager = Mock()

        router = ContentRouter(settings, health_monitor, policy_manager)

        assert router.settings == settings
        assert router.health_monitor == health_monitor
        assert router.policy_manager == policy_manager

    def test_router_initialization_defaults(self):
        """Test content router initialization with default dependencies."""
        settings = Settings()

        router = ContentRouter(settings)

        assert router.settings == settings
        assert router.health_monitor is None
        assert router.policy_manager is None

    async def test_route_request_with_required_detectors(self):
        """Test routing with explicitly required detectors."""
        settings = Settings()
        router = ContentRouter(settings)

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
            required_detectors=["toxicity", "nonexistent-detector"],
        )

        plan, decision = await router.route_request(request)

        # Should only include detectors that exist in settings
        assert "toxicity" in plan.primary_detectors
        assert "nonexistent-detector" not in plan.primary_detectors
        assert len(plan.primary_detectors) == 1

    async def test_route_request_with_excluded_detectors(self):
        """Test routing with excluded detectors."""
        settings = Settings()
        router = ContentRouter(settings)

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
            excluded_detectors=["echo"],  # Exclude echo detector
        )

        plan, decision = await router.route_request(request)

        # Should include toxicity and regex-pii but not echo
        assert "toxicity" in plan.primary_detectors
        assert "regex-pii" in plan.primary_detectors
        assert "echo" not in plan.primary_detectors
        assert len(plan.primary_detectors) == 2

    async def test_route_request_content_type_filtering(self):
        """Test routing with content type filtering."""
        # Create settings with mixed content type support
        detectors = {
            "text-only": DetectorEndpoint(
                name="text-only",
                endpoint="builtin:text-only",
                supported_content_types=["text"]
            ),
            "multi-type": DetectorEndpoint(
                name="multi-type",
                endpoint="builtin:multi-type",
                supported_content_types=["text", "document", "image"]
            ),
        }
        settings = Settings(detectors=detectors)
        router = ContentRouter(settings)

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.IMAGE,  # Only multi-type detector supports images
            tenant_id="test-tenant",
            policy_bundle="default",
        )

        plan, decision = await router.route_request(request)

        # Should only include multi-type detector
        assert plan.primary_detectors == ["multi-type"]
        assert len(plan.primary_detectors) == 1

    async def test_route_request_with_policy_manager(self):
        """Test routing with policy manager decisions."""
        settings = Settings()
        policy_manager = Mock(spec=PolicyManager)
        router = ContentRouter(settings, policy_manager=policy_manager)

        # Mock policy decision
        policy_decision = PolicyDecision(
            selected_detectors=["toxicity"],  # Only toxicity, not regex-pii
            coverage_method=CoverageMethod.REQUIRED_SET,
            coverage_requirements={"min_success_fraction": 1.0},
            routing_reason="policy-based",
        )
        policy_manager.decide = AsyncMock(return_value=policy_decision)

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="test-bundle",
        )

        plan, decision = await router.route_request(request)

        # Should use policy manager's decision
        assert plan.primary_detectors == ["toxicity"]
        assert len(plan.primary_detectors) == 1
        assert decision.routing_reason == "policy+policy-based"
        assert decision.policy_applied == "test-bundle"

        # Verify policy manager was called with correct parameters
        policy_manager.decide.assert_called_once_with(
            tenant_id="test-tenant",
            bundle="test-bundle",
            content_type=ContentType.TEXT,
            candidate_detectors=["toxicity", "regex-pii", "echo"],  # Default candidates
        )

    async def test_route_request_without_policy_manager(self):
        """Test routing without policy manager (fallback to defaults)."""
        settings = Settings()
        router = ContentRouter(settings)

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
        )

        plan, decision = await router.route_request(request)

        # Should use all available detectors
        assert "toxicity" in plan.primary_detectors
        assert "regex-pii" in plan.primary_detectors
        assert "echo" in plan.primary_detectors
        assert len(plan.primary_detectors) == 3
        assert decision.routing_reason == "policy+default"

    async def test_route_request_health_filtering(self):
        """Test routing with health status filtering."""
        settings = Settings()
        health_monitor = Mock()
        router = ContentRouter(settings, health_monitor=health_monitor)

        # Mock health status: toxicity healthy, regex-pii unhealthy, echo healthy
        health_monitor.is_healthy = Mock(side_effect=lambda d: d in ["toxicity", "echo"])

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
        )

        plan, decision = await router.route_request(request)

        # Should only include healthy detectors
        assert "toxicity" in plan.primary_detectors
        assert "echo" in plan.primary_detectors
        assert "regex-pii" not in plan.primary_detectors
        assert len(plan.primary_detectors) == 2

        # Check health status in decision
        assert decision.health_status["toxicity"] is True
        assert decision.health_status["regex-pii"] is False
        assert decision.health_status["echo"] is True

    async def test_route_request_health_filtering_disabled(self):
        """Test routing without health monitor."""
        settings = Settings()
        router = ContentRouter(settings)  # No health monitor

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
        )

        plan, decision = await router.route_request(request)

        # Should include all detectors when health monitor is disabled
        assert "toxicity" in plan.primary_detectors
        assert "regex-pii" in plan.primary_detectors
        assert "echo" in plan.primary_detectors
        assert len(plan.primary_detectors) == 3

        # Health status should be empty when no health monitor
        assert decision.health_status == {}

    def test_routing_plan_structure(self):
        """Test that routing plan has correct structure."""
        settings = Settings()
        router = ContentRouter(settings)

        # Test with single detector
        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
            required_detectors=["toxicity"],
        )

        async def run_test():
            plan, decision = await router.route_request(request)

            # Check routing plan structure
            assert plan.primary_detectors == ["toxicity"]
            assert plan.parallel_groups == [["toxicity"]]  # Single detector in its own group
            assert "toxicity" in plan.timeout_config
            assert "toxicity" in plan.retry_config
            assert plan.coverage_method == "required_set"
            assert plan.weights == {}  # Default empty weights
            assert plan.required_taxonomy_categories == []  # Default empty

        # Run the async test
        import asyncio
        asyncio.run(run_test())

    async def test_routing_plan_with_multiple_detectors(self):
        """Test routing plan structure with multiple detectors."""
        settings = Settings()
        router = ContentRouter(settings)

        request = OrchestrationRequest(
            content="Test content",
            content_type=ContentType.TEXT,
            tenant_id="test-tenant",
            policy_bundle="default",
        )

        plan, decision = await router.route_request(request)

        # All primary detectors should be in a single parallel group
        assert len(plan.parallel_groups) == 1
        assert set(plan.parallel_groups[0]) == set(plan.primary_detectors)

        # Timeout and retry configs should be set for all detectors
        for detector in plan.primary_detectors:
            assert detector in plan.timeout_config
            assert detector in plan.retry_config
            assert plan.timeout_config[detector] == settings.detectors[detector].timeout_ms
            assert plan.retry_config[detector] == settings.detectors[detector].max_retries
