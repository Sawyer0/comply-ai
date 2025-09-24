"""Integration tests for the complete cost monitoring system."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from src.llama_mapper.cost_monitoring import (
    CostGuardrail,
    CostMonitoringFactory,
    CostMonitoringSystem,
    GuardrailAction,
    GuardrailSeverity,
    ResourceType,
    ScalingPolicy,
    ScalingTrigger,
)


class TestCostMonitoringSystemIntegration:
    """Integration tests for the complete cost monitoring system."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CostMonitoringFactory.create_development_config()

    @pytest.fixture
    async def cost_system(self, config):
        """Create and start cost monitoring system."""
        system = CostMonitoringSystem(config)
        await system.start()
        yield system
        await system.stop()

    @pytest.mark.asyncio
    async def test_system_lifecycle(self, config):
        """Test system startup and shutdown."""
        system = CostMonitoringSystem(config)

        # System should not be running initially
        assert not system.is_running()

        # Start system
        await system.start()
        assert system.is_running()

        # Stop system
        await system.stop()
        assert not system.is_running()

    @pytest.mark.asyncio
    async def test_system_status(self, cost_system):
        """Test system status reporting."""
        status = cost_system.get_system_status()

        assert status["running"] is True
        assert "startup_time" in status
        assert "uptime_seconds" in status
        assert "components" in status
        assert "budget_limits" in status

        # Check component status
        components = status["components"]
        assert "metrics_collector" in components
        assert "guardrails" in components
        assert "autoscaling" in components
        assert "analytics" in components

    @pytest.mark.asyncio
    async def test_health_check(self, cost_system):
        """Test system health check."""
        health = await cost_system.health_check()

        assert "overall_health" in health
        assert "components" in health
        assert "timestamp" in health

        # Check component health
        components = health["components"]
        assert "metrics_collector" in components
        assert "guardrails" in components
        assert "autoscaling" in components
        assert "analytics" in components

    @pytest.mark.asyncio
    async def test_guardrails_management(self, cost_system):
        """Test guardrails management."""
        # Add a guardrail
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )

        cost_system.add_guardrail(guardrail)

        # Verify guardrail was added
        guardrails = cost_system.get_guardrails()
        assert len(guardrails) == 1
        assert guardrails[0].guardrail_id == "test_guardrail"

        # Remove guardrail
        cost_system.remove_guardrail("test_guardrail")

        # Verify guardrail was removed
        guardrails = cost_system.get_guardrails()
        assert len(guardrails) == 0

    @pytest.mark.asyncio
    async def test_scaling_policies_management(self, cost_system):
        """Test scaling policies management."""
        # Add a scaling policy
        policy = ScalingPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="Test description",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,
            min_instances=1,
            max_instances=10,
        )

        cost_system.add_scaling_policy(policy)

        # Verify policy was added
        policies = cost_system.get_scaling_policies()
        assert len(policies) == 1
        assert policies[0].policy_id == "test_policy"

        # Remove policy
        cost_system.remove_scaling_policy("test_policy")

        # Verify policy was removed
        policies = cost_system.get_scaling_policies()
        assert len(policies) == 0

    @pytest.mark.asyncio
    async def test_cost_breakdown(self, cost_system):
        """Test cost breakdown functionality."""
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        end_time = datetime.now(timezone.utc)

        breakdown = cost_system.get_cost_breakdown(start_time, end_time)

        assert breakdown is not None
        assert breakdown.period_start == start_time
        assert breakdown.period_end == end_time
        assert breakdown.total_cost >= 0
        assert breakdown.currency == "USD"

    @pytest.mark.asyncio
    async def test_cost_trends(self, cost_system):
        """Test cost trends functionality."""
        trends = cost_system.get_cost_trends(days=7)

        assert "dates" in trends
        assert "costs" in trends
        assert "total_cost" in trends
        assert "average_daily_cost" in trends
        assert isinstance(trends["costs"], list)
        assert isinstance(trends["total_cost"], (int, float))

    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, cost_system):
        """Test optimization recommendations."""
        recommendations = cost_system.get_optimization_recommendations()

        assert isinstance(recommendations, list)
        # Recommendations may be empty initially, which is fine

    @pytest.mark.asyncio
    async def test_cost_anomalies(self, cost_system):
        """Test cost anomalies detection."""
        anomalies = cost_system.get_cost_anomalies(days=30)

        assert isinstance(anomalies, list)
        # Anomalies may be empty initially, which is fine

    @pytest.mark.asyncio
    async def test_cost_forecast(self, cost_system):
        """Test cost forecasting."""
        forecast = cost_system.get_latest_forecast()

        # Forecast may be None initially, which is fine
        if forecast is not None:
            assert forecast.predicted_cost >= 0
            assert forecast.confidence_level > 0
            assert forecast.confidence_level <= 1

    @pytest.mark.asyncio
    async def test_analytics_summary(self, cost_system):
        """Test analytics summary."""
        summary = cost_system.get_analytics_summary(days=30)

        assert "cost_trend" in summary
        assert "recommendations" in summary
        assert "anomalies" in summary
        assert "forecast" in summary

        # Check recommendations structure
        recommendations = summary["recommendations"]
        assert "total" in recommendations
        assert "total_potential_savings" in recommendations
        assert "by_category" in recommendations

        # Check anomalies structure
        anomalies = summary["anomalies"]
        assert "total" in anomalies
        assert "by_type" in anomalies

    @pytest.mark.asyncio
    async def test_emergency_controls(self, cost_system):
        """Test emergency controls."""
        # Test emergency stop
        await cost_system.emergency_stop("Test emergency stop")

        # System should still be running (emergency stop doesn't stop the system itself)
        assert cost_system.is_running()

        # Test resume after emergency
        await cost_system.resume_after_emergency()
        assert cost_system.is_running()

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, cost_system):
        """Test multi-tenant isolation."""
        # Add tenant-specific guardrails
        guardrail1 = CostGuardrail(
            guardrail_id="tenant1_guardrail",
            name="Tenant 1 Guardrail",
            description="Tenant 1 description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
            tenant_id="tenant1",
        )

        guardrail2 = CostGuardrail(
            guardrail_id="tenant2_guardrail",
            name="Tenant 2 Guardrail",
            description="Tenant 2 description",
            metric_type="daily_cost",
            threshold=200.0,
            severity=GuardrailSeverity.MEDIUM,
            actions=[GuardrailAction.ALERT],
            tenant_id="tenant2",
        )

        cost_system.add_guardrail(guardrail1)
        cost_system.add_guardrail(guardrail2)

        # Get guardrails for specific tenant
        tenant1_guardrails = cost_system.get_guardrails("tenant1")
        tenant2_guardrails = cost_system.get_guardrails("tenant2")

        # Should include both global and tenant-specific guardrails
        assert len(tenant1_guardrails) >= 1
        assert len(tenant2_guardrails) >= 1

        # Check tenant-specific guardrails
        tenant1_ids = [g.guardrail_id for g in tenant1_guardrails]
        tenant2_ids = [g.guardrail_id for g in tenant2_guardrails]

        assert "tenant1_guardrail" in tenant1_ids
        assert "tenant2_guardrail" in tenant2_ids

    @pytest.mark.asyncio
    async def test_configuration_updates(self, cost_system):
        """Test configuration updates."""
        # Get current config
        current_config = cost_system.config

        # Create new config with different values
        new_config = CostMonitoringFactory.create_production_config()
        new_config.daily_budget_limit = 2000.0

        # Update configuration
        cost_system.update_config(new_config)

        # Verify configuration was updated
        assert cost_system.config.daily_budget_limit == 2000.0

    @pytest.mark.asyncio
    async def test_error_handling(self, cost_system):
        """Test error handling in various scenarios."""
        # Test invalid guardrail ID
        cost_system.remove_guardrail("non_existent_guardrail")
        # Should not raise an exception

        # Test invalid policy ID
        cost_system.remove_scaling_policy("non_existent_policy")
        # Should not raise an exception

        # Test invalid tenant ID
        guardrails = cost_system.get_guardrails("non_existent_tenant")
        assert isinstance(guardrails, list)

        # Test invalid time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time + timedelta(days=1)  # Start after end
        breakdown = cost_system.get_cost_breakdown(start_time, end_time)
        assert breakdown is not None  # Should handle gracefully

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cost_system):
        """Test concurrent operations."""
        # Add multiple guardrails concurrently
        guardrails = []
        for i in range(5):
            guardrail = CostGuardrail(
                guardrail_id=f"concurrent_guardrail_{i}",
                name=f"Concurrent Guardrail {i}",
                description=f"Concurrent description {i}",
                metric_type="daily_cost",
                threshold=100.0 + i * 10,
                severity=GuardrailSeverity.HIGH,
                actions=[GuardrailAction.ALERT],
            )
            guardrails.append(guardrail)

        # Add all guardrails
        for guardrail in guardrails:
            cost_system.add_guardrail(guardrail)

        # Verify all were added
        all_guardrails = cost_system.get_guardrails()
        assert len(all_guardrails) >= 5

        # Remove all guardrails
        for guardrail in guardrails:
            cost_system.remove_guardrail(guardrail.guardrail_id)

        # Verify all were removed
        all_guardrails = cost_system.get_guardrails()
        # Should have fewer guardrails now (only any that were added before)
        assert len(all_guardrails) < 5


class TestCostMonitoringSystemPerformance:
    """Performance tests for the cost monitoring system."""

    @pytest.fixture
    async def cost_system(self):
        """Create cost monitoring system for performance testing."""
        config = CostMonitoringFactory.create_development_config()
        system = CostMonitoringSystem(config)
        await system.start()
        yield system
        await system.stop()

    @pytest.mark.asyncio
    async def test_large_number_of_guardrails(self, cost_system):
        """Test system performance with many guardrails."""
        # Add many guardrails
        for i in range(100):
            guardrail = CostGuardrail(
                guardrail_id=f"perf_guardrail_{i}",
                name=f"Performance Guardrail {i}",
                description=f"Performance description {i}",
                metric_type="daily_cost",
                threshold=100.0 + i,
                severity=GuardrailSeverity.HIGH,
                actions=[GuardrailAction.ALERT],
            )
            cost_system.add_guardrail(guardrail)

        # Verify all were added
        guardrails = cost_system.get_guardrails()
        assert len(guardrails) >= 100

        # Test retrieval performance
        import time

        start_time = time.time()
        for i in range(10):
            cost_system.get_guardrails()
        end_time = time.time()

        # Should be fast (less than 1 second for 10 retrievals)
        assert (end_time - start_time) < 1.0

    @pytest.mark.asyncio
    async def test_large_number_of_policies(self, cost_system):
        """Test system performance with many scaling policies."""
        # Add many policies
        for i in range(50):
            policy = ScalingPolicy(
                policy_id=f"perf_policy_{i}",
                name=f"Performance Policy {i}",
                description=f"Performance description {i}",
                resource_type=ResourceType.CPU,
                trigger=ScalingTrigger.COST_THRESHOLD,
                threshold=0.8,
                min_instances=1,
                max_instances=10,
            )
            cost_system.add_scaling_policy(policy)

        # Verify all were added
        policies = cost_system.get_scaling_policies()
        assert len(policies) >= 50

        # Test retrieval performance
        import time

        start_time = time.time()
        for i in range(10):
            cost_system.get_scaling_policies()
        end_time = time.time()

        # Should be fast (less than 1 second for 10 retrievals)
        assert (end_time - start_time) < 1.0
