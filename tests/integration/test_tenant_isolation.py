"""
Tests for the TenantIsolationManager class.

This module tests tenant isolation functionality including
access control, configuration management, and data filtering.
"""

import pytest

from src.llama_mapper.config.settings import Settings
from llama_mapper.storage.tenant_isolation import (
    TenantAccessLevel,
    TenantConfig,
    TenantIsolationError,
    TenantIsolationManager,
)


@pytest.fixture
def settings():
    """Create test settings."""
    return Settings()


@pytest.fixture
def tenant_manager(settings):
    """Create a TenantIsolationManager instance."""
    return TenantIsolationManager(settings)


class TestTenantIsolationManager:
    """Test cases for TenantIsolationManager."""

    def test_create_tenant_context_strict(self, tenant_manager):
        """Test creating a strict tenant context."""
        context = tenant_manager.create_tenant_context("tenant-1")

        assert context.tenant_id == "tenant-1"
        assert context.access_level == TenantAccessLevel.STRICT
        assert context.allowed_tenants is None
        assert context.configuration_overrides == {}

    def test_create_tenant_context_shared(self, tenant_manager):
        """Test creating a shared tenant context."""
        allowed_tenants = {"tenant-2", "tenant-3"}
        context = tenant_manager.create_tenant_context(
            "tenant-1",
            access_level=TenantAccessLevel.SHARED,
            allowed_tenants=allowed_tenants,
        )

        assert context.tenant_id == "tenant-1"
        assert context.access_level == TenantAccessLevel.SHARED
        assert context.allowed_tenants == allowed_tenants

    def test_create_tenant_context_admin(self, tenant_manager):
        """Test creating an admin tenant context."""
        context = tenant_manager.create_tenant_context(
            "admin-tenant", access_level=TenantAccessLevel.ADMIN
        )

        assert context.tenant_id == "admin-tenant"
        assert context.access_level == TenantAccessLevel.ADMIN

    def test_create_tenant_context_invalid_id(self, tenant_manager):
        """Test creating tenant context with invalid ID."""
        with pytest.raises(TenantIsolationError, match="Invalid tenant_id"):
            tenant_manager.create_tenant_context("")

        with pytest.raises(TenantIsolationError, match="Invalid tenant_id"):
            tenant_manager.create_tenant_context(None)

    def test_create_tenant_context_shared_without_allowed(self, tenant_manager):
        """Test creating shared context without allowed tenants."""
        with pytest.raises(
            TenantIsolationError, match="Shared access requires allowed_tenants"
        ):
            tenant_manager.create_tenant_context(
                "tenant-1", access_level=TenantAccessLevel.SHARED
            )

    def test_validate_tenant_access_same_tenant(self, tenant_manager):
        """Test access validation for same tenant."""
        tenant_manager.create_tenant_context("tenant-1")

        assert tenant_manager.validate_tenant_access("tenant-1", "tenant-1", "read")
        assert tenant_manager.validate_tenant_access("tenant-1", "tenant-1", "write")

    def test_validate_tenant_access_strict_isolation(self, tenant_manager):
        """Test strict isolation prevents cross-tenant access."""
        tenant_manager.create_tenant_context("tenant-1", TenantAccessLevel.STRICT)

        assert not tenant_manager.validate_tenant_access("tenant-1", "tenant-2", "read")
        assert not tenant_manager.validate_tenant_access(
            "tenant-1", "tenant-2", "write"
        )

    def test_validate_tenant_access_shared_allowed(self, tenant_manager):
        """Test shared access with allowed tenants."""
        tenant_manager.create_tenant_context(
            "tenant-1",
            access_level=TenantAccessLevel.SHARED,
            allowed_tenants={"tenant-2", "tenant-3"},
        )

        assert tenant_manager.validate_tenant_access("tenant-1", "tenant-2", "read")
        assert tenant_manager.validate_tenant_access("tenant-1", "tenant-3", "read")
        assert not tenant_manager.validate_tenant_access("tenant-1", "tenant-4", "read")

    def test_validate_tenant_access_admin(self, tenant_manager):
        """Test admin access allows cross-tenant operations."""
        tenant_manager.create_tenant_context("admin-tenant", TenantAccessLevel.ADMIN)

        assert tenant_manager.validate_tenant_access("admin-tenant", "tenant-1", "read")
        assert tenant_manager.validate_tenant_access(
            "admin-tenant", "tenant-2", "write"
        )
        assert tenant_manager.validate_tenant_access(
            "admin-tenant", "tenant-3", "delete"
        )

    def test_validate_tenant_access_no_context(self, tenant_manager):
        """Test access validation without context."""
        assert not tenant_manager.validate_tenant_access(
            "unknown-tenant", "tenant-1", "read"
        )

    def test_apply_tenant_filter_strict(self, tenant_manager):
        """Test applying tenant filter for strict access."""
        context = tenant_manager.create_tenant_context("tenant-1")

        query = "SELECT * FROM storage_records"
        filtered = tenant_manager.apply_tenant_filter(query, context)

        expected = "SELECT * FROM storage_records WHERE tenant_id = 'tenant-1' "
        assert filtered == expected

    def test_apply_tenant_filter_with_existing_where(self, tenant_manager):
        """Test applying tenant filter to query with existing WHERE clause."""
        context = tenant_manager.create_tenant_context("tenant-1")

        query = "SELECT * FROM storage_records WHERE timestamp > '2023-01-01'"
        filtered = tenant_manager.apply_tenant_filter(query, context)

        expected = "SELECT * FROM storage_records WHERE tenant_id = 'tenant-1' AND timestamp > '2023-01-01'"
        assert filtered == expected

    def test_apply_tenant_filter_shared(self, tenant_manager):
        """Test applying tenant filter for shared access."""
        context = tenant_manager.create_tenant_context(
            "tenant-1",
            access_level=TenantAccessLevel.SHARED,
            allowed_tenants={"tenant-2", "tenant-3"},
        )

        query = "SELECT * FROM storage_records"
        filtered = tenant_manager.apply_tenant_filter(query, context)

        # Check that all expected tenant IDs are present (order doesn't matter)
        assert "tenant_id IN (" in filtered
        assert "'tenant-1'" in filtered
        assert "'tenant-2'" in filtered
        assert "'tenant-3'" in filtered

    def test_apply_tenant_filter_admin(self, tenant_manager):
        """Test applying tenant filter for admin access (no filtering)."""
        context = tenant_manager.create_tenant_context("admin", TenantAccessLevel.ADMIN)

        query = "SELECT * FROM storage_records"
        filtered = tenant_manager.apply_tenant_filter(query, context)

        assert filtered == query  # No filtering for admin

    def test_apply_tenant_filter_with_table_alias(self, tenant_manager):
        """Test applying tenant filter with table alias."""
        context = tenant_manager.create_tenant_context("tenant-1")

        query = "SELECT * FROM storage_records sr"
        filtered = tenant_manager.apply_tenant_filter(query, context, table_alias="sr")

        expected = "SELECT * FROM storage_records sr WHERE sr.tenant_id = 'tenant-1' "
        assert filtered == expected

    def test_get_tenant_config_default(self, tenant_manager):
        """Test getting default tenant configuration."""
        config = tenant_manager.get_tenant_config("tenant-1")

        assert config.tenant_id == "tenant-1"
        assert config.confidence_threshold is None
        assert config.detector_whitelist is None
        assert config.detector_blacklist is None
        assert config.encryption_enabled is True
        assert config.audit_level == "standard"

    def test_get_tenant_config_with_overrides(self, tenant_manager):
        """Test getting tenant configuration with overrides."""
        overrides = {
            "confidence_threshold": 0.8,
            "detector_whitelist": ["detector-1", "detector-2"],
            "encryption_enabled": False,
            "audit_level": "verbose",
        }

        tenant_manager.create_tenant_context(
            "tenant-1", configuration_overrides=overrides
        )

        config = tenant_manager.get_tenant_config("tenant-1")

        assert config.confidence_threshold == 0.8
        assert config.detector_whitelist == ["detector-1", "detector-2"]
        assert config.encryption_enabled is False
        assert config.audit_level == "verbose"

    def test_update_tenant_config(self, tenant_manager):
        """Test updating tenant configuration."""
        config = TenantConfig(
            tenant_id="tenant-1",
            confidence_threshold=0.9,
            detector_blacklist=["bad-detector"],
            storage_retention_days=30,
        )

        tenant_manager.update_tenant_config("tenant-1", config)

        retrieved_config = tenant_manager.get_tenant_config("tenant-1")
        assert retrieved_config.confidence_threshold == 0.9
        assert retrieved_config.detector_blacklist == ["bad-detector"]
        assert retrieved_config.storage_retention_days == 30

    def test_update_tenant_config_mismatched_id(self, tenant_manager):
        """Test updating tenant config with mismatched ID."""
        config = TenantConfig(tenant_id="tenant-2")

        with pytest.raises(TenantIsolationError, match="Config tenant_id must match"):
            tenant_manager.update_tenant_config("tenant-1", config)

    def test_validate_detector_access_whitelist(self, tenant_manager):
        """Test detector access validation with whitelist."""
        config = TenantConfig(
            tenant_id="tenant-1", detector_whitelist=["detector-1", "detector-2"]
        )
        tenant_manager.update_tenant_config("tenant-1", config)

        assert tenant_manager.validate_detector_access("tenant-1", "detector-1")
        assert tenant_manager.validate_detector_access("tenant-1", "detector-2")
        assert not tenant_manager.validate_detector_access("tenant-1", "detector-3")

    def test_validate_detector_access_blacklist(self, tenant_manager):
        """Test detector access validation with blacklist."""
        config = TenantConfig(tenant_id="tenant-1", detector_blacklist=["bad-detector"])
        tenant_manager.update_tenant_config("tenant-1", config)

        assert not tenant_manager.validate_detector_access("tenant-1", "bad-detector")
        assert tenant_manager.validate_detector_access("tenant-1", "good-detector")

    def test_validate_detector_access_blacklist_priority(self, tenant_manager):
        """Test that blacklist takes priority over whitelist."""
        config = TenantConfig(
            tenant_id="tenant-1",
            detector_whitelist=["detector-1", "detector-2"],
            detector_blacklist=["detector-1"],  # detector-1 in both lists
        )
        tenant_manager.update_tenant_config("tenant-1", config)

        assert not tenant_manager.validate_detector_access("tenant-1", "detector-1")
        assert tenant_manager.validate_detector_access("tenant-1", "detector-2")

    def test_get_effective_confidence_threshold_default(self, tenant_manager):
        """Test getting effective confidence threshold with default."""
        threshold = tenant_manager.get_effective_confidence_threshold("tenant-1")

        # Should return global default
        assert threshold == tenant_manager.settings.confidence.default_threshold

    def test_get_effective_confidence_threshold_override(self, tenant_manager):
        """Test getting effective confidence threshold with tenant override."""
        config = TenantConfig(tenant_id="tenant-1", confidence_threshold=0.8)
        tenant_manager.update_tenant_config("tenant-1", config)

        threshold = tenant_manager.get_effective_confidence_threshold("tenant-1")
        assert threshold == 0.8

    def test_create_tenant_scoped_record_id(self, tenant_manager):
        """Test creating tenant-scoped record ID."""
        scoped_id = tenant_manager.create_tenant_scoped_record_id(
            "tenant-1", "record-123"
        )

        assert scoped_id == "tenant-1:record-123"

    def test_extract_tenant_from_record_id(self, tenant_manager):
        """Test extracting tenant from scoped record ID."""
        tenant_id, base_id = tenant_manager.extract_tenant_from_record_id(
            "tenant-1:record-123"
        )

        assert tenant_id == "tenant-1"
        assert base_id == "record-123"

    def test_extract_tenant_from_record_id_invalid(self, tenant_manager):
        """Test extracting tenant from invalid record ID."""
        with pytest.raises(
            TenantIsolationError, match="Invalid scoped record ID format"
        ):
            tenant_manager.extract_tenant_from_record_id("invalid-id")

    def test_get_tenant_access_audit(self, tenant_manager):
        """Test getting tenant access audit trail."""
        # Create context and perform some access validations
        tenant_manager.create_tenant_context("tenant-1", TenantAccessLevel.ADMIN)
        tenant_manager.validate_tenant_access("tenant-1", "tenant-2", "read")
        tenant_manager.validate_tenant_access("tenant-1", "tenant-3", "write")

        audit = tenant_manager.get_tenant_access_audit("tenant-1")

        assert len(audit) == 2
        assert audit[0]["target_tenant"] == "tenant-2"
        assert audit[0]["operation"] == "read"
        assert audit[0]["access_type"] == "admin"
        assert audit[1]["target_tenant"] == "tenant-3"
        assert audit[1]["operation"] == "write"
        assert audit[1]["access_type"] == "admin"

    def test_clear_tenant_context(self, tenant_manager):
        """Test clearing tenant context and configuration."""
        tenant_manager.create_tenant_context("tenant-1")
        config = TenantConfig(tenant_id="tenant-1", confidence_threshold=0.8)
        tenant_manager.update_tenant_config("tenant-1", config)

        # Verify they exist
        assert tenant_manager.get_tenant_context("tenant-1") is not None
        assert tenant_manager.get_tenant_config("tenant-1").confidence_threshold == 0.8

        # Clear and verify they're gone
        tenant_manager.clear_tenant_context("tenant-1")

        assert tenant_manager.get_tenant_context("tenant-1") is None
        # Config should be recreated as default
        assert tenant_manager.get_tenant_config("tenant-1").confidence_threshold is None

    def test_get_tenant_statistics(self, tenant_manager):
        """Test getting tenant statistics."""
        # Create some contexts
        tenant_manager.create_tenant_context("tenant-1", TenantAccessLevel.STRICT)
        tenant_manager.create_tenant_context(
            "tenant-2", TenantAccessLevel.SHARED, {"tenant-3"}
        )
        tenant_manager.create_tenant_context("admin", TenantAccessLevel.ADMIN)

        # Perform some access operations
        tenant_manager.validate_tenant_access("admin", "tenant-1", "read")

        stats = tenant_manager.get_tenant_statistics()

        assert stats["active_contexts"] == 3
        assert stats["tenant_access_levels"]["strict"] == 1
        assert stats["tenant_access_levels"]["shared"] == 1
        assert stats["tenant_access_levels"]["admin"] == 1
        assert stats["total_access_events"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
