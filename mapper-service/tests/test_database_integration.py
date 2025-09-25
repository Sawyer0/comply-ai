"""
Test database integration for Mapper Service.

Tests the database manager, API key management, and authentication system.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from mapper.infrastructure.database_manager import DatabaseManager, DatabaseConfig
from mapper.security.api_key_manager import APIKeyManager, APIKeyCreateRequest
from mapper.security.authentication import AuthenticationService
from mapper.security.authorization import AuthorizationService, Permission


class TestDatabaseIntegration:
    """Test database integration components."""

    @pytest.fixture
    async def database_manager(self):
        """Create test database manager."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_mapper_service",
            username="test_user",
            password="test_password",
            pool_min_size=1,
            pool_max_size=2,
        )

        db_manager = DatabaseManager(config)

        try:
            await db_manager.initialize()
            yield db_manager
        finally:
            await db_manager.close()

    @pytest.fixture
    async def api_key_manager(self, database_manager):
        """Create test API key manager."""
        return APIKeyManager(database_manager)

    @pytest.fixture
    async def auth_service(self, api_key_manager):
        """Create test authentication service."""
        return AuthenticationService(api_key_manager)

    @pytest.fixture
    def authz_service(self):
        """Create test authorization service."""
        return AuthorizationService()

    @pytest.mark.asyncio
    async def test_database_health_check(self, database_manager):
        """Test database health check."""
        health = await database_manager.health_check()

        assert health["status"] in ["healthy", "unhealthy"]
        assert "response_time_ms" in health
        assert "timestamp" in health

    @pytest.mark.asyncio
    async def test_api_key_lifecycle(self, api_key_manager):
        """Test API key creation, validation, and revocation."""
        # Create API key
        request = APIKeyCreateRequest(
            tenant_id="test_tenant",
            name="Test Key",
            description="Test API key",
            permissions=["map:canonical", "map:framework"],
            scopes=["read", "write"],
            expires_in_days=30,
        )

        result = await api_key_manager.create_api_key(request)

        assert "api_key" in result
        assert "key_id" in result
        assert result["tenant_id"] == "test_tenant"

        api_key = result["api_key"]
        key_id = result["key_id"]

        # Validate API key
        key_info = await api_key_manager.validate_api_key(api_key)

        assert key_info is not None
        assert key_info.tenant_id == "test_tenant"
        assert key_info.name == "Test Key"
        assert "map:canonical" in key_info.permissions
        assert "read" in key_info.scopes

        # List API keys
        keys = await api_key_manager.list_api_keys("test_tenant")
        assert len(keys) >= 1
        assert any(k.key_id == key_id for k in keys)

        # Revoke API key
        success = await api_key_manager.revoke_api_key(key_id, "test_tenant")
        assert success

        # Validate revoked key
        revoked_key_info = await api_key_manager.validate_api_key(api_key)
        assert revoked_key_info is None

    @pytest.mark.asyncio
    async def test_usage_tracking(self, api_key_manager):
        """Test API key usage tracking."""
        # Create API key first
        request = APIKeyCreateRequest(
            tenant_id="test_tenant",
            name="Usage Test Key",
            permissions=["map:canonical"],
            scopes=["read"],
        )

        result = await api_key_manager.create_api_key(request)
        key_id = result["key_id"]

        # Track usage
        from mapper.security.api_key_manager import APIKeyUsage

        usage = APIKeyUsage(
            api_key_id=key_id,
            tenant_id="test_tenant",
            endpoint="/api/v1/map",
            method="POST",
            status_code=200,
            response_time_ms=150,
            tokens_processed=100,
            cost_cents=0.05,
        )

        await api_key_manager.track_usage(usage)

        # Get usage statistics
        stats = await api_key_manager.get_usage_statistics("test_tenant", key_id=key_id)

        assert stats["total_requests"] >= 1
        assert stats["total_tokens"] >= 100

    def test_authorization_permissions(self, authz_service):
        """Test authorization permission checking."""
        from mapper.security.api_key_manager import APIKeyInfo

        # Create mock API key info
        api_key_info = APIKeyInfo(
            key_id="test_key",
            tenant_id="test_tenant",
            name="Test Key",
            permissions=["map:canonical", "cost:view"],
            scopes=["read", "write"],
            is_active=True,
            usage_count=0,
            rate_limit_per_minute=100,
            created_at=datetime.utcnow(),
        )

        # Test permission checking
        result = authz_service.check_permission(api_key_info, Permission.MAP_CANONICAL)
        assert result.authorized

        result = authz_service.check_permission(api_key_info, Permission.ADMIN_SYSTEM)
        assert not result.authorized

        # Test scope checking
        from mapper.security.authorization import Scope

        result = authz_service.check_scope(api_key_info, [Scope.READ])
        assert result.authorized

        result = authz_service.check_scope(api_key_info, [Scope.ADMIN])
        assert not result.authorized

        # Test tenant access
        result = authz_service.check_tenant_access(api_key_info, "test_tenant")
        assert result.authorized

        result = authz_service.check_tenant_access(api_key_info, "other_tenant")
        assert not result.authorized


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
