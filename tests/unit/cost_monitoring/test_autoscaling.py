"""Unit tests for cost-aware autoscaling system."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from src.llama_mapper.cost_monitoring.autoscaling.cost_aware_scaler import (
    CostAwareScaler,
    CostAwareScalingConfig,
    ResourceType,
    ScalingAction,
    ScalingDecision,
    ScalingPolicy,
    ScalingTrigger,
)


class TestCostAwareScalingConfig:
    """Test cost-aware scaling configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CostAwareScalingConfig()

        assert config.enabled is True
        assert config.evaluation_interval_seconds == 60
        assert config.cost_threshold_percent == 80.0
        assert config.performance_threshold_percent == 70.0
        assert config.max_cost_increase_percent == 50.0
        assert config.min_performance_improvement_percent == 20.0
        assert config.prediction_horizon_hours == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CostAwareScalingConfig(
            enabled=False,
            evaluation_interval_seconds=30,
            cost_threshold_percent=90.0,
            performance_threshold_percent=80.0,
            max_cost_increase_percent=25.0,
            min_performance_improvement_percent=30.0,
            prediction_horizon_hours=2,
        )

        assert config.enabled is False
        assert config.evaluation_interval_seconds == 30
        assert config.cost_threshold_percent == 90.0
        assert config.performance_threshold_percent == 80.0
        assert config.max_cost_increase_percent == 25.0
        assert config.min_performance_improvement_percent == 30.0
        assert config.prediction_horizon_hours == 2


class TestScalingPolicy:
    """Test scaling policy data structure."""

    def test_policy_creation(self):
        """Test policy creation."""
        policy = ScalingPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="Test description",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,
            min_instances=1,
            max_instances=10,
            scale_up_cooldown_minutes=5,
            scale_down_cooldown_minutes=15,
            cost_weight=0.6,
            performance_weight=0.4,
            enabled=True,
            tenant_id="test-tenant",
        )

        assert policy.policy_id == "test_policy"
        assert policy.name == "Test Policy"
        assert policy.description == "Test description"
        assert policy.resource_type == ResourceType.CPU
        assert policy.trigger == ScalingTrigger.COST_THRESHOLD
        assert policy.threshold == 0.8
        assert policy.min_instances == 1
        assert policy.max_instances == 10
        assert policy.scale_up_cooldown_minutes == 5
        assert policy.scale_down_cooldown_minutes == 15
        assert policy.cost_weight == 0.6
        assert policy.performance_weight == 0.4
        assert policy.enabled is True
        assert policy.tenant_id == "test-tenant"


class TestScalingDecision:
    """Test scaling decision data structure."""

    def test_decision_creation(self):
        """Test decision creation."""
        timestamp = datetime.now(timezone.utc)
        decision = ScalingDecision(
            decision_id="test_decision",
            policy_id="test_policy",
            action=ScalingAction.SCALE_UP,
            resource_type=ResourceType.CPU,
            current_instances=2,
            target_instances=3,
            reason="High CPU usage detected",
            cost_impact=10.0,
            performance_impact=25.0,
            triggered_at=timestamp,
            tenant_id="test-tenant",
        )

        assert decision.decision_id == "test_decision"
        assert decision.policy_id == "test_policy"
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.resource_type == ResourceType.CPU
        assert decision.current_instances == 2
        assert decision.target_instances == 3
        assert decision.reason == "High CPU usage detected"
        assert decision.cost_impact == 10.0
        assert decision.performance_impact == 25.0
        assert decision.triggered_at == timestamp
        assert decision.tenant_id == "test-tenant"
        assert decision.executed_at is None  # Default value


class TestCostAwareScaler:
    """Test cost-aware scaler system."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CostAwareScalingConfig(
            evaluation_interval_seconds=1,  # Fast for testing
        )

    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector._metrics_history = []
        return collector

    @pytest.fixture
    def scaler(self, config, mock_metrics_collector):
        """Create test scaler system."""
        return CostAwareScaler(config, mock_metrics_collector)

    def test_scaler_initialization(self, scaler):
        """Test scaler initialization."""
        assert scaler.config is not None
        assert scaler.metrics_collector is not None
        assert scaler._policies == {}
        assert scaler._decisions == []
        assert scaler._running is False
        assert scaler._scaling_task is None
        assert scaler._last_scaling_times == {}
        assert scaler._current_instances == {}

    def test_add_policy(self, scaler):
        """Test adding a scaling policy."""
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

        scaler.add_policy(policy)

        assert "test_policy" in scaler._policies
        assert scaler._policies["test_policy"] == policy
        assert scaler._current_instances["test_policy"] == 1  # min_instances

    def test_remove_policy(self, scaler):
        """Test removing a scaling policy."""
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

        scaler.add_policy(policy)
        assert "test_policy" in scaler._policies
        assert "test_policy" in scaler._current_instances

        scaler.remove_policy("test_policy")
        assert "test_policy" not in scaler._policies
        assert "test_policy" not in scaler._current_instances

    def test_get_policies(self, scaler):
        """Test getting policies."""
        # Add policies with and without tenant
        policy1 = ScalingPolicy(
            policy_id="global_policy",
            name="Global Policy",
            description="Global description",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,
            min_instances=1,
            max_instances=10,
        )

        policy2 = ScalingPolicy(
            policy_id="tenant_policy",
            name="Tenant Policy",
            description="Tenant description",
            resource_type=ResourceType.GPU,
            trigger=ScalingTrigger.PERFORMANCE_DEGRADATION,
            threshold=0.7,
            min_instances=0,
            max_instances=5,
            tenant_id="test-tenant",
        )

        scaler.add_policy(policy1)
        scaler.add_policy(policy2)

        # Get all policies
        all_policies = scaler.get_policies()
        assert len(all_policies) == 2

        # Get tenant-specific policies
        tenant_policies = scaler.get_policies("test-tenant")
        assert len(tenant_policies) == 2  # Global + tenant-specific

        # Get policies for different tenant
        other_policies = scaler.get_policies("other-tenant")
        assert len(other_policies) == 1  # Only global

    def test_is_in_cooldown(self, scaler):
        """Test cooldown checking."""
        policy = ScalingPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="Test description",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,
            min_instances=1,
            max_instances=10,
            scale_up_cooldown_minutes=60,
        )

        # No previous scaling
        assert not scaler._is_in_cooldown(policy)

        # Recent scaling
        scaler._last_scaling_times["test_policy"] = datetime.now(timezone.utc)
        assert scaler._is_in_cooldown(policy)

        # Old scaling
        scaler._last_scaling_times["test_policy"] = datetime.now(
            timezone.utc
        ) - timedelta(hours=2)
        assert not scaler._is_in_cooldown(policy)

    @pytest.mark.asyncio
    async def test_get_current_cost(self, scaler):
        """Test getting current cost."""
        # Mock the metrics collector method
        scaler.metrics_collector._get_hourly_cost = AsyncMock(return_value=25.0)

        cost = await scaler._get_current_cost()
        assert cost == 25.0

    @pytest.mark.asyncio
    async def test_get_current_performance(self, scaler):
        """Test getting current performance."""
        performance = await scaler._get_current_performance()
        assert performance == 75.0  # Mock value from implementation

    @pytest.mark.asyncio
    async def test_predict_cost(self, scaler):
        """Test cost prediction."""
        policy = ScalingPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="Test description",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,
            min_instances=2,
            max_instances=10,
        )

        scaler._current_instances["test_policy"] = 2

        # Mock current cost
        scaler._get_current_cost = AsyncMock(return_value=20.0)

        predicted_cost = await scaler._predict_cost(policy, 4)
        assert predicted_cost == 40.0  # 20.0 * (4/2)

    @pytest.mark.asyncio
    async def test_predict_performance(self, scaler):
        """Test performance prediction."""
        policy = ScalingPolicy(
            policy_id="test_policy",
            name="Test Policy",
            description="Test description",
            resource_type=ResourceType.CPU,
            trigger=ScalingTrigger.COST_THRESHOLD,
            threshold=0.8,
            min_instances=2,
            max_instances=10,
        )

        scaler._current_instances["test_policy"] = 2

        # Mock current performance
        scaler._get_current_performance = AsyncMock(return_value=60.0)

        predicted_performance = await scaler._predict_performance(policy, 4)
        assert predicted_performance == 100.0  # 60.0 * (4/2), capped at 100

    def test_calculate_decision_score(self, scaler):
        """Test decision scoring."""
        decision = ScalingDecision(
            decision_id="test_decision",
            policy_id="test_policy",
            action=ScalingAction.SCALE_UP,
            resource_type=ResourceType.CPU,
            current_instances=2,
            target_instances=3,
            reason="Test reason",
            cost_impact=10.0,
            performance_impact=25.0,
            triggered_at=datetime.now(timezone.utc),
        )

        current_cost = 100.0
        current_performance = 50.0

        score = scaler._calculate_decision_score(
            decision, current_cost, current_performance
        )
        assert isinstance(score, float)

    def test_get_scaling_decisions(self, scaler):
        """Test getting scaling decisions with filtering."""
        now = datetime.now(timezone.utc)

        # Create test decisions
        decision1 = ScalingDecision(
            decision_id="decision_1",
            policy_id="policy_1",
            action=ScalingAction.SCALE_UP,
            resource_type=ResourceType.CPU,
            current_instances=2,
            target_instances=3,
            reason="High CPU usage",
            cost_impact=10.0,
            performance_impact=25.0,
            triggered_at=now - timedelta(hours=1),
            tenant_id="tenant_1",
        )

        decision2 = ScalingDecision(
            decision_id="decision_2",
            policy_id="policy_2",
            action=ScalingAction.SCALE_DOWN,
            resource_type=ResourceType.GPU,
            current_instances=3,
            target_instances=2,
            reason="Low GPU usage",
            cost_impact=-5.0,
            performance_impact=-10.0,
            triggered_at=now - timedelta(hours=2),
            tenant_id="tenant_2",
        )

        scaler._decisions = [decision1, decision2]

        # Get all decisions
        all_decisions = scaler.get_scaling_decisions()
        assert len(all_decisions) == 2

        # Filter by tenant
        tenant_decisions = scaler.get_scaling_decisions(tenant_id="tenant_1")
        assert len(tenant_decisions) == 1
        assert tenant_decisions[0].decision_id == "decision_1"

        # Filter by time range
        recent_decisions = scaler.get_scaling_decisions(
            start_time=now - timedelta(hours=1, minutes=30)
        )
        assert len(recent_decisions) == 1
        assert recent_decisions[0].decision_id == "decision_1"

    def test_get_scaling_summary(self, scaler):
        """Test getting scaling summary."""
        now = datetime.now(timezone.utc)

        # Create test decisions
        decisions = []
        for i in range(3):
            decision = ScalingDecision(
                decision_id=f"decision_{i}",
                policy_id=f"policy_{i}",
                action=ScalingAction.SCALE_UP if i < 2 else ScalingAction.SCALE_DOWN,
                resource_type=ResourceType.CPU if i < 2 else ResourceType.GPU,
                current_instances=2 + i,
                target_instances=3 + i,
                reason=f"Reason {i}",
                cost_impact=10.0 + i * 5,
                performance_impact=25.0 + i * 10,
                triggered_at=now - timedelta(hours=i),
                tenant_id="test-tenant",
            )
            decisions.append(decision)

        scaler._decisions = decisions

        summary = scaler.get_scaling_summary(days=7)

        assert summary["total_decisions"] == 3
        assert summary["by_action"]["scale_up"] == 2
        assert summary["by_action"]["scale_down"] == 1
        assert summary["by_resource_type"]["cpu"] == 2
        assert summary["by_resource_type"]["gpu"] == 1
        assert summary["total_cost_impact"] == 45.0  # 10 + 15 + 20
        assert summary["total_performance_impact"] == 90.0  # 25 + 35 + 45
        assert "test-tenant" in summary["by_tenant"]
        assert summary["by_tenant"]["test-tenant"] == 3
