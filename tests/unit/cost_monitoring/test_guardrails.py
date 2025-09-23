"""Unit tests for cost guardrails system."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock

from src.llama_mapper.cost_monitoring.guardrails.cost_guardrails import (
    CostGuardrails,
    CostGuardrailsConfig,
    CostGuardrail,
    GuardrailViolation,
    GuardrailAction,
    GuardrailSeverity,
)


class TestCostGuardrailsConfig:
    """Test cost guardrails configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CostGuardrailsConfig()
        
        assert config.enabled is True
        assert GuardrailAction.ALERT in config.default_actions
        assert GuardrailAction.NOTIFY_ADMIN in config.default_actions
        assert config.escalation_delay_minutes == 30
        assert config.max_violations_per_hour == 10
        assert GuardrailAction.PAUSE_SERVICE in config.emergency_actions
        assert GuardrailAction.BLOCK_REQUESTS in config.emergency_actions
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CostGuardrailsConfig(
            enabled=False,
            default_actions=[GuardrailAction.ALERT],
            escalation_delay_minutes=60,
            max_violations_per_hour=5,
            emergency_actions=[GuardrailAction.PAUSE_SERVICE],
        )
        
        assert config.enabled is False
        assert config.default_actions == [GuardrailAction.ALERT]
        assert config.escalation_delay_minutes == 60
        assert config.max_violations_per_hour == 5
        assert config.emergency_actions == [GuardrailAction.PAUSE_SERVICE]


class TestCostGuardrail:
    """Test cost guardrail data structure."""
    
    def test_guardrail_creation(self):
        """Test guardrail creation."""
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT, GuardrailAction.THROTTLE],
            cooldown_minutes=60,
            enabled=True,
            tenant_id="test-tenant",
        )
        
        assert guardrail.guardrail_id == "test_guardrail"
        assert guardrail.name == "Test Guardrail"
        assert guardrail.description == "Test description"
        assert guardrail.metric_type == "daily_cost"
        assert guardrail.threshold == 100.0
        assert guardrail.severity == GuardrailSeverity.HIGH
        assert guardrail.actions == [GuardrailAction.ALERT, GuardrailAction.THROTTLE]
        assert guardrail.cooldown_minutes == 60
        assert guardrail.enabled is True
        assert guardrail.tenant_id == "test-tenant"
        assert guardrail.currency == "USD"  # Default value


class TestGuardrailViolation:
    """Test guardrail violation data structure."""
    
    def test_violation_creation(self):
        """Test violation creation."""
        timestamp = datetime.now(timezone.utc)
        violation = GuardrailViolation(
            violation_id="test_violation",
            guardrail_id="test_guardrail",
            metric_value=150.0,
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions_taken=[GuardrailAction.ALERT],
            triggered_at=timestamp,
            tenant_id="test-tenant",
            metadata={"test": "data"},
        )
        
        assert violation.violation_id == "test_violation"
        assert violation.guardrail_id == "test_guardrail"
        assert violation.metric_value == 150.0
        assert violation.threshold == 100.0
        assert violation.severity == GuardrailSeverity.HIGH
        assert violation.actions_taken == [GuardrailAction.ALERT]
        assert violation.triggered_at == timestamp
        assert violation.tenant_id == "test-tenant"
        assert violation.metadata == {"test": "data"}
        assert violation.resolved is False  # Default value
        assert violation.resolved_at is None  # Default value


class TestCostGuardrails:
    """Test cost guardrails system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CostGuardrailsConfig(
            max_violations_per_hour=5,  # Lower for testing
        )
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create mock metrics collector."""
        collector = Mock()
        collector._metrics_history = []
        return collector
    
    @pytest.fixture
    def guardrails(self, config, mock_metrics_collector):
        """Create test guardrails system."""
        return CostGuardrails(config, mock_metrics_collector)
    
    def test_guardrails_initialization(self, guardrails):
        """Test guardrails initialization."""
        assert guardrails.config is not None
        assert guardrails.metrics_collector is not None
        assert guardrails._guardrails == {}
        assert guardrails._violations == []
        assert guardrails._running is False
        assert guardrails._monitoring_task is None
        assert guardrails._last_violation_times == {}
    
    def test_add_guardrail(self, guardrails):
        """Test adding a guardrail."""
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )
        
        guardrails.add_guardrail(guardrail)
        
        assert "test_guardrail" in guardrails._guardrails
        assert guardrails._guardrails["test_guardrail"] == guardrail
    
    def test_remove_guardrail(self, guardrails):
        """Test removing a guardrail."""
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )
        
        guardrails.add_guardrail(guardrail)
        assert "test_guardrail" in guardrails._guardrails
        
        guardrails.remove_guardrail("test_guardrail")
        assert "test_guardrail" not in guardrails._guardrails
    
    def test_get_guardrails(self, guardrails):
        """Test getting guardrails."""
        # Add guardrails with and without tenant
        guardrail1 = CostGuardrail(
            guardrail_id="global_guardrail",
            name="Global Guardrail",
            description="Global description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )
        
        guardrail2 = CostGuardrail(
            guardrail_id="tenant_guardrail",
            name="Tenant Guardrail",
            description="Tenant description",
            metric_type="daily_cost",
            threshold=50.0,
            severity=GuardrailSeverity.MEDIUM,
            actions=[GuardrailAction.ALERT],
            tenant_id="test-tenant",
        )
        
        guardrails.add_guardrail(guardrail1)
        guardrails.add_guardrail(guardrail2)
        
        # Get all guardrails
        all_guardrails = guardrails.get_guardrails()
        assert len(all_guardrails) == 2
        
        # Get tenant-specific guardrails
        tenant_guardrails = guardrails.get_guardrails("test-tenant")
        assert len(tenant_guardrails) == 2  # Global + tenant-specific
        
        # Get guardrails for different tenant
        other_guardrails = guardrails.get_guardrails("other-tenant")
        assert len(other_guardrails) == 1  # Only global
    
    def test_is_in_cooldown(self, guardrails):
        """Test cooldown checking."""
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
            cooldown_minutes=60,
        )
        
        # No previous violation
        assert not guardrails._is_in_cooldown(guardrail)
        
        # Recent violation
        guardrails._last_violation_times["test_guardrail"] = datetime.now(timezone.utc)
        assert guardrails._is_in_cooldown(guardrail)
        
        # Old violation
        guardrails._last_violation_times["test_guardrail"] = datetime.now(timezone.utc) - timedelta(hours=2)
        assert not guardrails._is_in_cooldown(guardrail)
    
    @pytest.mark.asyncio
    async def test_get_current_metric_value_daily_cost(self, guardrails):
        """Test getting current metric value for daily cost."""
        # Mock the metrics collector method
        guardrails.metrics_collector._get_daily_cost = AsyncMock(return_value=150.0)
        
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="daily_cost",
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )
        
        value = await guardrails._get_current_metric_value(guardrail)
        assert value == 150.0
    
    @pytest.mark.asyncio
    async def test_get_current_metric_value_hourly_cost(self, guardrails):
        """Test getting current metric value for hourly cost."""
        # Mock the metrics collector method
        guardrails.metrics_collector._get_hourly_cost = AsyncMock(return_value=25.0)
        
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="hourly_cost",
            threshold=20.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )
        
        value = await guardrails._get_current_metric_value(guardrail)
        assert value == 25.0
    
    @pytest.mark.asyncio
    async def test_get_current_metric_value_api_calls(self, guardrails):
        """Test getting current metric value for API calls."""
        # Mock metrics history
        from src.llama_mapper.cost_monitoring.core.metrics_collector import CostMetrics, ResourceUsage
        
        guardrails.metrics_collector._metrics_history = [
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(api_calls=100),
                cost_per_unit={},
                total_cost=1.0,
            ),
            CostMetrics(
                resource_type="test",
                usage=ResourceUsage(api_calls=200),
                cost_per_unit={},
                total_cost=2.0,
            ),
        ]
        
        guardrail = CostGuardrail(
            guardrail_id="test_guardrail",
            name="Test Guardrail",
            description="Test description",
            metric_type="api_calls",
            threshold=250.0,
            severity=GuardrailSeverity.HIGH,
            actions=[GuardrailAction.ALERT],
        )
        
        value = await guardrails._get_current_metric_value(guardrail)
        assert value == 300.0  # 100 + 200
    
    def test_should_trigger_emergency_actions(self, guardrails):
        """Test emergency action triggering."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        # No recent violations
        assert not guardrails._should_trigger_emergency_actions()
        
        # Add violations within the hour
        for i in range(6):  # More than max_violations_per_hour (5)
            violation = GuardrailViolation(
                violation_id=f"violation_{i}",
                guardrail_id="test_guardrail",
                metric_value=150.0,
                threshold=100.0,
                severity=GuardrailSeverity.HIGH,
                actions_taken=[GuardrailAction.ALERT],
                triggered_at=hour_ago + timedelta(minutes=i * 10),
            )
            guardrails._violations.append(violation)
        
        assert guardrails._should_trigger_emergency_actions()
    
    def test_resolve_violation(self, guardrails):
        """Test resolving a violation."""
        violation = GuardrailViolation(
            violation_id="test_violation",
            guardrail_id="test_guardrail",
            metric_value=150.0,
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions_taken=[GuardrailAction.ALERT],
            triggered_at=datetime.now(timezone.utc),
        )
        
        guardrails._violations.append(violation)
        
        # Resolve the violation
        result = guardrails.resolve_violation("test_violation")
        assert result is True
        assert violation.resolved_at is not None
        
        # Try to resolve non-existent violation
        result = guardrails.resolve_violation("non_existent")
        assert result is False
    
    def test_get_violations(self, guardrails):
        """Test getting violations with filtering."""
        now = datetime.now(timezone.utc)
        
        # Create test violations
        violation1 = GuardrailViolation(
            violation_id="violation_1",
            guardrail_id="guardrail_1",
            metric_value=150.0,
            threshold=100.0,
            severity=GuardrailSeverity.HIGH,
            actions_taken=[GuardrailAction.ALERT],
            triggered_at=now - timedelta(hours=1),
            tenant_id="tenant_1",
        )
        
        violation2 = GuardrailViolation(
            violation_id="violation_2",
            guardrail_id="guardrail_2",
            metric_value=200.0,
            threshold=150.0,
            severity=GuardrailSeverity.CRITICAL,
            actions_taken=[GuardrailAction.ALERT],
            triggered_at=now - timedelta(hours=2),
            tenant_id="tenant_2",
        )
        
        guardrails._violations = [violation1, violation2]
        
        # Get all violations
        all_violations = guardrails.get_violations()
        assert len(all_violations) == 2
        
        # Filter by tenant
        tenant_violations = guardrails.get_violations(tenant_id="tenant_1")
        assert len(tenant_violations) == 1
        assert tenant_violations[0].violation_id == "violation_1"
        
        # Filter by severity
        high_violations = guardrails.get_violations(severity=GuardrailSeverity.HIGH)
        assert len(high_violations) == 1
        assert high_violations[0].violation_id == "violation_1"
        
        # Filter by time range
        recent_violations = guardrails.get_violations(
            start_time=now - timedelta(hours=1, minutes=30)
        )
        assert len(recent_violations) == 1
        assert recent_violations[0].violation_id == "violation_1"
    
    def test_get_violation_summary(self, guardrails):
        """Test getting violation summary."""
        now = datetime.now(timezone.utc)
        
        # Create test violations
        violations = []
        for i in range(3):
            violation = GuardrailViolation(
                violation_id=f"violation_{i}",
                guardrail_id=f"guardrail_{i}",
                metric_value=150.0 + i * 10,
                threshold=100.0,
                severity=GuardrailSeverity.HIGH if i < 2 else GuardrailSeverity.CRITICAL,
                actions_taken=[GuardrailAction.ALERT],
                triggered_at=now - timedelta(hours=i),
                tenant_id="test-tenant",
            )
            violations.append(violation)
        
        # Resolve one violation
        violations[0].resolved_at = now
        
        guardrails._violations = violations
        
        summary = guardrails.get_violation_summary(days=7)
        
        assert summary["total_violations"] == 3
        assert summary["by_severity"]["high"] == 2
        assert summary["by_severity"]["critical"] == 1
        assert summary["resolved_violations"] == 1
        assert summary["unresolved_violations"] == 2
        assert "test-tenant" in summary["by_tenant"]
        assert summary["by_tenant"]["test-tenant"] == 3
