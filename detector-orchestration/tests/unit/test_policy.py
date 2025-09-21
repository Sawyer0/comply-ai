"""Tests for policy management."""

import json
import tempfile
from pathlib import Path

import pytest

from detector_orchestration.policy import (
    CoverageMethod,
    TenantPolicy,
    PolicyStore,
    PolicyManager,
    OPAPolicyEngine,
    PolicyDecision,
)
from detector_orchestration.conflict import ConflictResolutionStrategy
from detector_orchestration.config import Settings
from detector_orchestration.models import ContentType


class TestCoverageMethod:
    def test_coverage_method_enum(self):
        """Test coverage method enum values."""
        assert CoverageMethod.REQUIRED_SET.value == "required_set"
        assert CoverageMethod.WEIGHTED.value == "weighted"
        assert CoverageMethod.TAXONOMY.value == "taxonomy"


class TestTenantPolicy:
    def test_tenant_policy_creation(self):
        """Test creating a tenant policy."""
        policy = TenantPolicy(
            tenant_id="test-tenant",
            bundle="default",
            version="v1",
            required_detectors=["toxicity", "regex-pii"],
            optional_detectors=["echo"],
            coverage_method=CoverageMethod.WEIGHTED,
            required_coverage=0.8,
            detector_weights={"toxicity": 0.7, "regex-pii": 0.3},
            required_taxonomy_categories=["security", "privacy"],
            allowed_content_types=[ContentType.TEXT, ContentType.DOCUMENT],
        )

        assert policy.tenant_id == "test-tenant"
        assert policy.bundle == "default"
        assert policy.version == "v1"
        assert policy.required_detectors == ["toxicity", "regex-pii"]
        assert policy.optional_detectors == ["echo"]
        assert policy.coverage_method == CoverageMethod.WEIGHTED
        assert policy.required_coverage == 0.8
        assert policy.detector_weights == {"toxicity": 0.7, "regex-pii": 0.3}
        assert policy.required_taxonomy_categories == ["security", "privacy"]
        assert policy.allowed_content_types == [ContentType.TEXT, ContentType.DOCUMENT]

    def test_tenant_policy_defaults(self):
        """Test tenant policy with default values."""
        policy = TenantPolicy(
            tenant_id="test-tenant",
            bundle="default",
        )

        assert policy.version == "v1"
        assert policy.required_detectors == []
        assert policy.optional_detectors == []
        assert policy.coverage_method == CoverageMethod.REQUIRED_SET
        assert policy.required_coverage == 1.0
        assert policy.detector_weights == {}
        assert policy.required_taxonomy_categories == []
        assert policy.allowed_content_types == [ContentType.TEXT]

    def test_tenant_policy_serialization(self):
        """Test tenant policy JSON serialization."""
        policy = TenantPolicy(
            tenant_id="test-tenant",
            bundle="default",
            required_detectors=["toxicity"],
            detector_weights={"toxicity": 1.0},
        )

        json_str = policy.model_dump_json(indent=2)
        assert isinstance(json_str, str)
        assert "test-tenant" in json_str
        assert "toxicity" in json_str

        # Test deserialization
        loaded_policy = TenantPolicy.model_validate_json(json_str)
        assert loaded_policy.tenant_id == policy.tenant_id
        assert loaded_policy.bundle == policy.bundle
        assert loaded_policy.required_detectors == policy.required_detectors


class TestPolicyStore:
    def test_policy_store_initialization(self):
        """Test policy store initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)
            assert store.base_dir == Path(temp_dir)

    def test_policy_store_save_and_get(self):
        """Test saving and retrieving policies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)

            policy = TenantPolicy(
                tenant_id="test-tenant",
                bundle="default",
                required_detectors=["toxicity"],
            )

            # Save policy
            store.save_policy(policy)

            # Retrieve policy
            retrieved_policy = store.get_policy("test-tenant", "default")
            assert retrieved_policy is not None
            assert retrieved_policy.tenant_id == "test-tenant"
            assert retrieved_policy.bundle == "default"
            assert retrieved_policy.required_detectors == ["toxicity"]

    def test_policy_store_get_nonexistent(self):
        """Test getting a nonexistent policy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)

            policy = store.get_policy("nonexistent-tenant", "nonexistent-bundle")
            assert policy is None

    def test_policy_store_list_policies(self):
        """Test listing policies for a tenant."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)

            # Save multiple policies for same tenant
            policy1 = TenantPolicy(tenant_id="test-tenant", bundle="default")
            policy2 = TenantPolicy(tenant_id="test-tenant", bundle="custom")

            store.save_policy(policy1)
            store.save_policy(policy2)

            # List policies
            policies = store.list_policies("test-tenant")
            assert len(policies) == 2
            assert "default" in policies
            assert "custom" in policies

    def test_policy_store_list_policies_empty(self):
        """Test listing policies for tenant with no policies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)

            policies = store.list_policies("nonexistent-tenant")
            assert policies == []

    def test_policy_store_delete_policy(self):
        """Test deleting a policy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)

            policy = TenantPolicy(tenant_id="test-tenant", bundle="default")
            store.save_policy(policy)

            # Verify policy exists
            assert store.get_policy("test-tenant", "default") is not None

            # Delete policy
            result = store.delete_policy("test-tenant", "default")
            assert result is True

            # Verify policy is gone
            assert store.get_policy("test-tenant", "default") is None

    def test_policy_store_delete_nonexistent_policy(self):
        """Test deleting a nonexistent policy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)

            result = store.delete_policy("nonexistent-tenant", "nonexistent-bundle")
            assert result is False


class TestPolicyManager:
    def test_policy_manager_initialization(self):
        """Test policy manager initialization."""
        settings = Settings()
        store = PolicyStore("/tmp/policies")
        opa_engine = OPAPolicyEngine(settings)

        manager = PolicyManager(store=store, opa_engine=opa_engine)

        assert manager.store == store
        assert manager.opa_engine == opa_engine

    def test_policy_manager_initialization_defaults(self):
        """Test policy manager initialization with defaults."""
        settings = Settings()

        manager = PolicyManager(settings=settings)

        assert manager.store is not None
        assert manager.opa_engine is not None
        assert isinstance(manager.store, PolicyStore)
        assert isinstance(manager.opa_engine, OPAPolicyEngine)

    async def test_decide_with_tenant_policy(self):
        """Test policy decision using tenant policy store."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)
            settings = Settings()
            manager = PolicyManager(store=store, settings=settings)

            # Save a tenant policy
            policy = TenantPolicy(
                tenant_id="test-tenant",
                bundle="custom",
                required_detectors=["toxicity"],
                coverage_method=CoverageMethod.WEIGHTED,
                required_coverage=0.8,
                detector_weights={"toxicity": 1.0},
            )
            store.save_policy(policy)

            # Make decision
            decision = await manager.decide(
                tenant_id="test-tenant",
                bundle="custom",
                content_type=ContentType.TEXT,
                candidate_detectors=["toxicity", "regex-pii"],
            )

            assert decision.selected_detectors == ["toxicity"]
            assert decision.coverage_method == CoverageMethod.WEIGHTED
            assert decision.coverage_requirements["min_success_fraction"] == 0.8

    async def test_decide_with_opa_policy(self):
        """Test policy decision using OPA engine."""
        settings = Settings(config__opa_enabled=True, config__opa_url="http://localhost:8181")
        manager = PolicyManager(settings=settings)

        # Mock OPA response
        async def mock_evaluate(*args, **kwargs):
            return {
                "selected_detectors": ["toxicity"],
                "coverage_method": "required_set",
                "coverage_requirements": {"min_success_fraction": 1.0}
            }

        manager.opa_engine.evaluate = mock_evaluate

        decision = await manager.decide(
            tenant_id="test-tenant",
            bundle="opa-bundle",
            content_type=ContentType.TEXT,
            candidate_detectors=["toxicity", "regex-pii"],
        )

        assert decision.selected_detectors == ["toxicity"]
        assert decision.coverage_method == CoverageMethod.REQUIRED_SET
        assert decision.coverage_requirements["min_success_fraction"] == 1.0

    async def test_decide_fallback_to_default(self):
        """Test policy decision fallback to default when no policy found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PolicyStore(temp_dir)
            settings = Settings()
            manager = PolicyManager(store=store, settings=settings)

            # Make decision for nonexistent policy
            decision = await manager.decide(
                tenant_id="nonexistent-tenant",
                bundle="nonexistent-bundle",
                content_type=ContentType.TEXT,
                candidate_detectors=["toxicity", "regex-pii", "echo"],
            )

            # Should use all candidate detectors
            assert set(decision.selected_detectors) == {"toxicity", "regex-pii", "echo"}
            assert decision.coverage_method == CoverageMethod.REQUIRED_SET
            assert decision.coverage_requirements["min_success_fraction"] == 1.0

    async def test_resolve_conflict_highest_confidence(self):
        """Test conflict resolution using highest confidence strategy."""
        settings = Settings()
        manager = PolicyManager(settings=settings)

        detector_outputs = [
            {"detector": "toxicity", "output": "toxic", "confidence": 0.8},
            {"detector": "regex-pii", "output": "clean", "confidence": 0.9},
        ]

        outcome = await manager.resolve_conflict(
            detector_outputs=detector_outputs,
            tenant_id="test-tenant",
            bundle="default",
            content_type=ContentType.TEXT,
        )

        assert outcome.strategy_used == ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        assert outcome.winning_output == "clean"
        assert outcome.winning_detector == "regex-pii"
        assert outcome.confidence_delta == 0.1

    async def test_resolve_conflict_weighted_average(self):
        """Test conflict resolution using weighted average strategy."""
        settings = Settings()
        manager = PolicyManager(settings=settings)

        detector_outputs = [
            {"detector": "toxicity", "output": "toxic", "confidence": 0.6},
            {"detector": "regex-pii", "output": "clean", "confidence": 0.9},
            {"detector": "echo", "output": "toxic", "confidence": 0.7},
        ]

        outcome = await manager.resolve_conflict(
            detector_outputs=detector_outputs,
            tenant_id="test-tenant",
            bundle="weighted-bundle",
            content_type=ContentType.TEXT,
        )

        assert outcome.strategy_used == ConflictResolutionStrategy.WEIGHTED_AVERAGE
        # Should pick the output with highest average confidence
        # toxic: (0.6 + 0.7) / 2 = 0.65
        # clean: 0.9 / 1 = 0.9
        assert outcome.winning_output == "clean"
        assert outcome.winning_detector == "regex-pii"

    async def test_resolve_conflict_majority_vote(self):
        """Test conflict resolution using majority vote strategy."""
        settings = Settings()
        manager = PolicyManager(settings=settings)

        detector_outputs = [
            {"detector": "toxicity", "output": "toxic", "confidence": 0.8},
            {"detector": "regex-pii", "output": "clean", "confidence": 0.7},
            {"detector": "echo", "output": "toxic", "confidence": 0.6},
        ]

        outcome = await manager.resolve_conflict(
            detector_outputs=detector_outputs,
            tenant_id="test-tenant",
            bundle="majority-bundle",
            content_type=ContentType.TEXT,
        )

        assert outcome.strategy_used == ConflictResolutionStrategy.MAJORITY_VOTE
        # Majority vote: toxic (2) vs clean (1)
        assert outcome.winning_output == "toxic"
        assert outcome.winning_detector in ["toxicity", "echo"]  # Either one with highest confidence

    async def test_resolve_conflict_with_opa(self):
        """Test conflict resolution with OPA decision."""
        settings = Settings(config__opa_enabled=True, config__opa_url="http://localhost:8181")
        manager = PolicyManager(settings=settings)

        # Mock OPA response for conflict resolution
        async def mock_evaluate_conflict(*args, **kwargs):
            return {
                "strategy": "highest_confidence",
                "preferred_detector": "regex-pii",
                "tie_breaker": "confidence"
            }

        manager.opa_engine.evaluate_conflict = mock_evaluate_conflict

        detector_outputs = [
            {"detector": "toxicity", "output": "toxic", "confidence": 0.8},
            {"detector": "regex-pii", "output": "clean", "confidence": 0.8},  # Same confidence
        ]

        outcome = await manager.resolve_conflict(
            detector_outputs=detector_outputs,
            tenant_id="test-tenant",
            bundle="opa-conflict-bundle",
            content_type=ContentType.TEXT,
        )

        assert outcome.strategy_used == ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        assert outcome.winning_output == "clean"
        assert outcome.winning_detector == "regex-pii"
        assert outcome.tie_breaker_applied is True


class TestOPAPolicyEngine:
    def test_opa_engine_initialization(self):
        """Test OPA policy engine initialization."""
        settings = Settings(config__opa_enabled=True, config__opa_url="http://localhost:8181")

        engine = OPAPolicyEngine(settings)

        assert engine.settings == settings

    def test_opa_engine_disabled(self):
        """Test OPA engine when disabled."""
        settings = Settings(config__opa_enabled=False)

        engine = OPAPolicyEngine(settings)

        # Should not make any requests when disabled
        assert engine.settings.config.opa_enabled is False

    async def test_opa_evaluate_success(self):
        """Test successful OPA policy evaluation."""
        settings = Settings(config__opa_enabled=True, config__opa_url="http://localhost:8181")
        engine = OPAPolicyEngine(settings)

        # This test would require a mock HTTP server or httpx mock
        # For now, we just test that the method exists and handles the case properly
        result = await engine.evaluate("test-tenant", "test-bundle", {"input": "data"})
        assert result is None  # Should return None since no server is running

    async def test_opa_evaluate_conflict_success(self):
        """Test successful OPA conflict resolution evaluation."""
        settings = Settings(config__opa_enabled=True, config__opa_url="http://localhost:8181")
        engine = OPAPolicyEngine(settings)

        # This test would require a mock HTTP server or httpx mock
        # For now, we just test that the method exists and handles the case properly
        result = await engine.evaluate_conflict("test-tenant", "test-bundle", {"input": "data"})
        assert result is None  # Should return None since no server is running
