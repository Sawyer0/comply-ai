"""
Unit tests for Analysis Module authentication system.

Tests API key management, validation, rotation, and middleware functionality.
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.llama_mapper.analysis.infrastructure.auth import (
    APIKeyManager,
    APIKey,
    APIKeyRequest,
    APIKeyResponse,
    APIKeyScope,
    APIKeyStatus
)
from src.llama_mapper.analysis.api.auth_middleware import (
    APIKeyAuthMiddleware,
    APIKeyAuthDependency
)


class TestAPIKeyManager:
    """Test API key manager functionality."""
    
    def test_create_api_key(self):
        """Test API key creation."""
        manager = APIKeyManager()
        
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="test-key",
            description="Test API key",
            scopes=["analyze"],
            expires_in_days=30
        )
        
        response = manager.create_api_key(request)
        
        assert response is not None
        assert response.tenant_id == "test-tenant"
        assert response.name == "test-key"
        assert response.description == "Test API key"
        assert response.scopes == ["analyze"]
        assert response.status == "active"
        assert response.api_key.startswith("ak_")
        assert len(response.api_key) > 32
    
    def test_validate_api_key(self):
        """Test API key validation."""
        manager = APIKeyManager()
        
        # Create a key
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="test-key",
            scopes=["analyze"]
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        
        # Validate the key
        validated_key = manager.validate_api_key(
            response.api_key, 
            [APIKeyScope.ANALYZE]
        )
        
        assert validated_key is not None
        assert validated_key.tenant_id == "test-tenant"
        assert validated_key.name == "test-key"
        assert APIKeyScope.ANALYZE in validated_key.scopes
        assert validated_key.status == APIKeyStatus.ACTIVE
    
    def test_validate_invalid_api_key(self):
        """Test validation of invalid API key."""
        manager = APIKeyManager()
        
        # Try to validate a non-existent key
        validated_key = manager.validate_api_key(
            "invalid-key", 
            [APIKeyScope.ANALYZE]
        )
        
        assert validated_key is None
    
    def test_validate_expired_api_key(self):
        """Test validation of expired API key."""
        manager = APIKeyManager()
        
        # Create a key that expires in the past
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="test-key",
            scopes=["analyze"],
            expires_in_days=-1  # Expired
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        
        # Validate the key (should fail due to expiration)
        validated_key = manager.validate_api_key(
            response.api_key, 
            [APIKeyScope.ANALYZE]
        )
        
        assert validated_key is None
    
    def test_validate_insufficient_scopes(self):
        """Test validation with insufficient scopes."""
        manager = APIKeyManager()
        
        # Create a key with limited scopes
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="test-key",
            scopes=["read"]  # Only read scope
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        
        # Try to validate with admin scope (should fail)
        validated_key = manager.validate_api_key(
            response.api_key, 
            [APIKeyScope.ADMIN]
        )
        
        assert validated_key is None
    
    def test_revoke_api_key(self):
        """Test API key revocation."""
        manager = APIKeyManager()
        
        # Create a key
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="test-key",
            scopes=["analyze"]
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        
        # Revoke the key
        success = manager.revoke_api_key(response.key_id)
        assert success is True
        
        # Try to validate the revoked key
        validated_key = manager.validate_api_key(
            response.api_key, 
            [APIKeyScope.ANALYZE]
        )
        
        assert validated_key is None
    
    def test_rotate_api_key(self):
        """Test API key rotation."""
        manager = APIKeyManager()
        
        # Create a key
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="test-key",
            scopes=["analyze"]
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        
        # Rotate the key
        new_response = manager.rotate_api_key(response.key_id)
        assert new_response is not None
        assert new_response.key_id != response.key_id
        assert new_response.tenant_id == response.tenant_id
        assert new_response.scopes == response.scopes
        
        # Old key should be invalid
        old_validated = manager.validate_api_key(
            response.api_key, 
            [APIKeyScope.ANALYZE]
        )
        assert old_validated is None
        
        # New key should be valid
        new_validated = manager.validate_api_key(
            new_response.api_key, 
            [APIKeyScope.ANALYZE]
        )
        assert new_validated is not None
        assert new_validated.key_id == new_response.key_id
    
    def test_list_tenant_keys(self):
        """Test listing API keys for a tenant."""
        manager = APIKeyManager()
        
        # Create multiple keys for the same tenant
        for i in range(3):
            request = APIKeyRequest(
                tenant_id="test-tenant",
                name=f"test-key-{i}",
                scopes=["analyze"]
            )
            manager.create_api_key(request)
        
        # List keys
        keys = manager.list_tenant_keys("test-tenant")
        assert len(keys) == 3
        
        # Check that all keys belong to the tenant
        for key in keys:
            assert key.tenant_id == "test-tenant"
    
    def test_cleanup_expired_keys(self):
        """Test cleanup of expired keys."""
        manager = APIKeyManager()
        
        # Create an expired key
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="expired-key",
            scopes=["analyze"],
            expires_in_days=-1
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        
        # Cleanup expired keys
        cleaned_count = manager.cleanup_expired_keys()
        assert cleaned_count >= 1


class TestAPIKeyAuthMiddleware:
    """Test API key authentication middleware."""
    
    @pytest.fixture
    def mock_api_key_manager(self):
        """Create a mock API key manager."""
        manager = Mock()
        
        # Create a mock API key
        api_key = APIKey(
            key_id="test-key-id",
            key_hash="test-hash",
            tenant_id="test-tenant",
            name="test-key",
            scopes=[APIKeyScope.ANALYZE],
            status=APIKeyStatus.ACTIVE
        )
        
        manager.validate_api_key.return_value = api_key
        return manager
    
    @pytest.fixture
    def middleware(self, mock_api_key_manager):
        """Create authentication middleware."""
        return APIKeyAuthMiddleware(
            app=Mock(),
            api_key_manager=mock_api_key_manager,
            required_scopes={APIKeyScope.ANALYZE}
        )
    
    def test_should_skip_auth_health_endpoints(self, middleware):
        """Test that health endpoints skip authentication."""
        # Mock request for health endpoint
        request = Mock()
        request.url.path = "/health/live"
        
        assert middleware._should_skip_auth(request) is True
        
        request.url.path = "/health/ready"
        assert middleware._should_skip_auth(request) is True
    
    def test_should_skip_auth_docs_endpoints(self, middleware):
        """Test that docs endpoints skip authentication."""
        # Mock request for docs endpoint
        request = Mock()
        request.url.path = "/docs"
        
        assert middleware._should_skip_auth(request) is True
        
        request.url.path = "/redoc"
        assert middleware._should_skip_auth(request) is True
    
    def test_should_not_skip_auth_api_endpoints(self, middleware):
        """Test that API endpoints require authentication."""
        # Mock request for API endpoint
        request = Mock()
        request.url.path = "/api/v1/analysis/analyze"
        
        assert middleware._should_skip_auth(request) is False
    
    def test_extract_api_key_from_header(self, middleware):
        """Test API key extraction from X-API-Key header."""
        request = Mock()
        request.headers = {"X-API-Key": "test-api-key"}
        
        api_key = middleware._extract_api_key(request)
        assert api_key == "test-api-key"
    
    def test_extract_api_key_from_authorization_header(self, middleware):
        """Test API key extraction from Authorization header."""
        request = Mock()
        request.headers = {"Authorization": "Bearer test-api-key"}
        
        api_key = middleware._extract_api_key(request)
        assert api_key == "test-api-key"
    
    def test_extract_api_key_not_found(self, middleware):
        """Test API key extraction when not found."""
        request = Mock()
        request.headers = {}
        
        api_key = middleware._extract_api_key(request)
        assert api_key is None
    
    def test_check_rate_limit(self, middleware):
        """Test rate limiting functionality."""
        # Create a mock API key
        api_key = APIKey(
            key_id="test-key-id",
            key_hash="test-hash",
            tenant_id="test-tenant",
            name="test-key",
            scopes=[APIKeyScope.ANALYZE],
            status=APIKeyStatus.ACTIVE,
            rate_limit=10
        )
        
        request = Mock()
        
        # First 10 requests should pass
        for i in range(10):
            assert middleware._check_rate_limit(api_key, request) is True
        
        # 11th request should fail
        assert middleware._check_rate_limit(api_key, request) is False


class TestAPIKeyAuthDependency:
    """Test API key authentication dependency."""
    
    @pytest.fixture
    def mock_api_key_manager(self):
        """Create a mock API key manager."""
        manager = Mock()
        
        # Create a mock API key
        api_key = APIKey(
            key_id="test-key-id",
            key_hash="test-hash",
            tenant_id="test-tenant",
            name="test-key",
            scopes=[APIKeyScope.ANALYZE],
            status=APIKeyStatus.ACTIVE
        )
        
        manager.validate_api_key.return_value = api_key
        return manager
    
    @pytest.fixture
    def auth_dependency(self, mock_api_key_manager):
        """Create authentication dependency."""
        return APIKeyAuthDependency(
            api_key_manager=mock_api_key_manager,
            required_scopes={APIKeyScope.ANALYZE}
        )
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self, auth_dependency, mock_api_key_manager):
        """Test successful authentication."""
        request = Mock()
        request.headers = {"X-API-Key": "test-api-key"}
        
        result = await auth_dependency(request)
        
        assert result["tenant_id"] == "test-tenant"
        assert result["scopes"] == ["analyze"]
        assert result["api_key"].key_id == "test-key-id"
        
        # Verify that validate_api_key was called
        mock_api_key_manager.validate_api_key.assert_called_once_with(
            "test-api-key",
            [APIKeyScope.ANALYZE]
        )
    
    @pytest.mark.asyncio
    async def test_authentication_failure_no_key(self, auth_dependency):
        """Test authentication failure when no API key provided."""
        request = Mock()
        request.headers = {}
        
        with pytest.raises(Exception):  # HTTPException
            await auth_dependency(request)
    
    @pytest.mark.asyncio
    async def test_authentication_failure_invalid_key(self, auth_dependency, mock_api_key_manager):
        """Test authentication failure with invalid API key."""
        mock_api_key_manager.validate_api_key.return_value = None
        
        request = Mock()
        request.headers = {"X-API-Key": "invalid-key"}
        
        with pytest.raises(Exception):  # HTTPException
            await auth_dependency(request)


class TestAPIKeyIntegration:
    """Integration tests for API key system."""
    
    def test_full_api_key_lifecycle(self):
        """Test complete API key lifecycle."""
        manager = APIKeyManager()
        
        # 1. Create API key
        request = APIKeyRequest(
            tenant_id="test-tenant",
            name="integration-test-key",
            description="Integration test key",
            scopes=["analyze", "read"],
            expires_in_days=7
        )
        
        response = manager.create_api_key(request)
        assert response is not None
        assert response.tenant_id == "test-tenant"
        assert response.name == "integration-test-key"
        
        # 2. Validate API key
        validated_key = manager.validate_api_key(
            response.api_key,
            [APIKeyScope.ANALYZE]
        )
        assert validated_key is not None
        assert validated_key.tenant_id == "test-tenant"
        
        # 3. List tenant keys
        keys = manager.list_tenant_keys("test-tenant")
        assert len(keys) >= 1
        assert any(key.key_id == response.key_id for key in keys)
        
        # 4. Rotate API key
        new_response = manager.rotate_api_key(response.key_id)
        assert new_response is not None
        assert new_response.key_id != response.key_id
        
        # 5. Verify old key is invalid
        old_validated = manager.validate_api_key(
            response.api_key,
            [APIKeyScope.ANALYZE]
        )
        assert old_validated is None
        
        # 6. Verify new key is valid
        new_validated = manager.validate_api_key(
            new_response.api_key,
            [APIKeyScope.ANALYZE]
        )
        assert new_validated is not None
        
        # 7. Revoke new key
        success = manager.revoke_api_key(new_response.key_id)
        assert success is True
        
        # 8. Verify revoked key is invalid
        revoked_validated = manager.validate_api_key(
            new_response.api_key,
            [APIKeyScope.ANALYZE]
        )
        assert revoked_validated is None
