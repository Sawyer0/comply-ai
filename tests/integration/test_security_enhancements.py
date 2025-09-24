"""
Integration tests for security enhancements.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from src.llama_mapper.security.input_sanitization import (
    SecuritySanitizer, 
    SanitizationLevel, 
    AttackType,
    SecureDetectorRequest
)
from src.llama_mapper.security.rotation import (
    SecretsRotationManager, 
    RotationStatus
)
from src.llama_mapper.security.secrets_manager import SecretsManager
from src.llama_mapper.utils.correlation import (
    get_correlation_id, 
    set_correlation_id, 
    generate_correlation_id
)


class TestCorrelationIDs:
    """Test correlation ID functionality."""
    
    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = generate_correlation_id()
        assert corr_id is not None
        assert len(corr_id) == 36  # UUID format
        assert get_correlation_id() == corr_id
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-id"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
    
    def test_correlation_id_isolation(self):
        """Test that correlation IDs are isolated per context."""
        # This would require async context testing in a real scenario
        corr_id1 = generate_correlation_id()
        assert get_correlation_id() == corr_id1


class TestInputSanitization:
    """Test input sanitization functionality."""
    
    def test_basic_sanitization(self):
        """Test basic input sanitization."""
        sanitizer = SecuritySanitizer(SanitizationLevel.BASIC)
        
        # Test normal input
        clean_input = "This is a normal string"
        result = sanitizer.sanitize_string(clean_input)
        assert result == clean_input
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection."""
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "admin'--",
            "' OR 1=1 --"
        ]
        
        for malicious_input in malicious_inputs:
            attacks = sanitizer.detect_malicious_patterns(malicious_input)
            assert AttackType.SQL_INJECTION in attacks
    
    def test_xss_detection(self):
        """Test XSS detection."""
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<svg onload=alert('xss')>"
        ]
        
        for malicious_input in malicious_inputs:
            attacks = sanitizer.detect_malicious_patterns(malicious_input)
            assert AttackType.XSS in attacks
    
    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        
        malicious_inputs = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "\\windows\\system32\\config\\sam"
        ]
        
        for malicious_input in malicious_inputs:
            attacks = sanitizer.detect_malicious_patterns(malicious_input)
            assert AttackType.PATH_TRAVERSAL in attacks
    
    def test_length_validation(self):
        """Test input length validation."""
        sanitizer = SecuritySanitizer(SanitizationLevel.PARANOID)
        
        # Test input that exceeds paranoid limit (1000 chars)
        long_input = "A" * 1500
        
        with pytest.raises(ValueError, match="Input exceeds maximum length"):
            sanitizer.sanitize_string(long_input)
    
    def test_html_escaping(self):
        """Test HTML escaping in strict mode."""
        sanitizer = SecuritySanitizer(SanitizationLevel.STRICT)
        
        input_with_html = "<div>Hello & goodbye</div>"
        result = sanitizer.sanitize_string(input_with_html)
        
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result
    
    def test_paranoid_mode_character_removal(self):
        """Test character removal in paranoid mode."""
        sanitizer = SecuritySanitizer(SanitizationLevel.PARANOID)
        
        input_with_special_chars = "Hello<>\"';\\&|`$(){}[]*?"
        result = sanitizer.sanitize_string(input_with_special_chars)
        
        # Should remove all special characters
        assert result == "Hello"
    
    def test_secure_detector_request_validation(self):
        """Test SecureDetectorRequest validation."""
        # Valid request
        valid_request = SecureDetectorRequest(
            detector_type="presidio",
            content="This is safe content",
            metadata={"key": "value"},
            tenant_id="test-tenant"
        )
        assert valid_request.detector_type == "presidio"
        
        # Invalid detector type
        with pytest.raises(ValueError, match="Invalid detector type"):
            SecureDetectorRequest(
                detector_type="invalid",
                content="content",
                metadata={}
            )
        
        # Invalid tenant ID
        with pytest.raises(ValueError, match="Invalid tenant ID format"):
            SecureDetectorRequest(
                detector_type="presidio",
                content="content",
                metadata={},
                tenant_id="invalid@tenant"
            )
    
    def test_file_path_validation(self):
        """Test file path validation."""
        sanitizer = SecuritySanitizer()
        
        # Valid paths
        valid_paths = [
            "document.txt",
            "folder/document.txt",
            "data/reports/report.pdf"
        ]
        
        for path in valid_paths:
            result = sanitizer.validate_file_path(path)
            assert result == path
        
        # Invalid paths
        invalid_paths = [
            "../../../etc/passwd",
            "/absolute/path",
            "..\\windows\\system32"
        ]
        
        for path in invalid_paths:
            with pytest.raises(ValueError, match="path traversal detected"):
                sanitizer.validate_file_path(path)
    
    def test_email_validation(self):
        """Test email validation."""
        sanitizer = SecuritySanitizer()
        
        # Valid emails
        valid_emails = [
            "user@example.com",
            "test.email+tag@domain.co.uk",
            "user123@test-domain.org"
        ]
        
        for email in valid_emails:
            result = sanitizer.validate_email(email)
            assert result == email.lower()
        
        # Invalid emails
        invalid_emails = [
            "invalid-email",
            "@domain.com",
            "user@",
            "user@domain",
            "user space@domain.com"
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValueError, match="Invalid email format"):
                sanitizer.validate_email(email)
    
    def test_url_validation(self):
        """Test URL validation."""
        sanitizer = SecuritySanitizer()
        
        # Valid URLs
        valid_urls = [
            "https://example.com",
            "http://localhost:8080/path",
            "https://sub.domain.com/path?param=value"
        ]
        
        for url in valid_urls:
            result = sanitizer.validate_url(url)
            assert result == url
        
        # Invalid URLs
        invalid_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert(1)</script>",
            "ftp://example.com",
            "file:///etc/passwd",
            "not-a-url"
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValueError):
                sanitizer.validate_url(url)


class TestSecretsRotation:
    """Test secrets rotation functionality."""
    
    @pytest.fixture
    def mock_secrets_manager(self):
        """Mock secrets manager."""
        return Mock(spec=SecretsManager)
    
    @pytest.fixture
    def rotation_manager(self, mock_secrets_manager):
        """Create rotation manager with mocked dependencies."""
        return SecretsRotationManager(mock_secrets_manager)
    
    @pytest.mark.asyncio
    async def test_database_credential_rotation_success(self, rotation_manager):
        """Test successful database credential rotation."""
        # Mock the internal methods
        rotation_manager._get_current_credentials = AsyncMock(return_value={"version": "v1"})
        rotation_manager._generate_database_credentials = AsyncMock(return_value={
            "username": "new_user",
            "password": "new_password"
        })
        rotation_manager._update_database_user = AsyncMock()
        rotation_manager._store_credentials_in_vault = AsyncMock(return_value="v2")
        rotation_manager._verify_database_connectivity = AsyncMock(return_value=True)
        
        result = await rotation_manager.rotate_database_credentials("test_db")
        
        assert result.status == RotationStatus.COMPLETED
        assert result.secret_name == "database/test_db"
        assert result.new_version == "v2"
        assert result.rollback_available is True
    
    @pytest.mark.asyncio
    async def test_database_credential_rotation_with_rollback(self, rotation_manager):
        """Test database credential rotation with rollback on verification failure."""
        # Mock the internal methods
        rotation_manager._get_current_credentials = AsyncMock(return_value={"version": "v1"})
        rotation_manager._generate_database_credentials = AsyncMock(return_value={
            "username": "new_user",
            "password": "new_password"
        })
        rotation_manager._update_database_user = AsyncMock()
        rotation_manager._store_credentials_in_vault = AsyncMock(return_value="v2")
        rotation_manager._verify_database_connectivity = AsyncMock(return_value=False)  # Verification fails
        rotation_manager._rollback_database_credentials = AsyncMock()
        
        result = await rotation_manager.rotate_database_credentials("test_db")
        
        assert result.status == RotationStatus.ROLLED_BACK
        assert result.error_message == "Verification failed, rolled back to previous credentials"
        rotation_manager._rollback_database_credentials.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_api_key_rotation_success(self, rotation_manager):
        """Test successful API key rotation."""
        # Mock the internal methods
        rotation_manager._get_current_api_key = AsyncMock(return_value={"version": "v1"})
        rotation_manager._generate_secure_api_key = Mock(return_value="new_api_key_123")
        rotation_manager._store_api_key_in_vault = AsyncMock(return_value="v2")
        rotation_manager._update_tenant_api_key = AsyncMock()
        rotation_manager._notify_tenant_key_rotation = AsyncMock()
        
        result = await rotation_manager.rotate_api_keys("test_tenant")
        
        assert result.status == RotationStatus.COMPLETED
        assert result.secret_name == "api-keys/test_tenant"
        assert result.new_version == "v2"
    
    @pytest.mark.asyncio
    async def test_rotation_failure_handling(self, rotation_manager):
        """Test rotation failure handling."""
        # Mock method to raise exception
        rotation_manager._get_current_credentials = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await rotation_manager.rotate_database_credentials("test_db")
        
        assert result.status == RotationStatus.FAILED
        assert "Connection failed" in result.error_message
    
    def test_rotation_history_tracking(self, rotation_manager):
        """Test rotation history tracking."""
        from src.llama_mapper.security.rotation import RotationResult
        
        # Add some test results
        result1 = RotationResult(
            secret_name="test/secret",
            status=RotationStatus.COMPLETED,
            old_version="v1",
            new_version="v2"
        )
        
        result2 = RotationResult(
            secret_name="test/secret",
            status=RotationStatus.FAILED,
            old_version="v2",
            new_version=None,
            error_message="Test error"
        )
        
        rotation_manager._add_to_history("test/secret", result1)
        rotation_manager._add_to_history("test/secret", result2)
        
        history = rotation_manager.get_rotation_history("test/secret")
        assert "test/secret" in history
        assert len(history["test/secret"]) == 2
        assert history["test/secret"][0].status == RotationStatus.COMPLETED
        assert history["test/secret"][1].status == RotationStatus.FAILED
    
    def test_secure_password_generation(self, rotation_manager):
        """Test secure password generation."""
        password = rotation_manager._generate_secure_password(32)
        
        assert len(password) == 32
        assert any(c.isupper() for c in password)  # Has uppercase
        assert any(c.islower() for c in password)  # Has lowercase
        assert any(c.isdigit() for c in password)  # Has digits
        
        # Generate multiple passwords to ensure randomness
        passwords = [rotation_manager._generate_secure_password(16) for _ in range(10)]
        assert len(set(passwords)) == 10  # All should be unique
    
    def test_secure_api_key_generation(self, rotation_manager):
        """Test secure API key generation."""
        api_key = rotation_manager._generate_secure_api_key()
        
        assert len(api_key) == 64  # 32 bytes as hex = 64 characters
        assert all(c in '0123456789abcdef' for c in api_key)  # Valid hex
        
        # Generate multiple keys to ensure randomness
        keys = [rotation_manager._generate_secure_api_key() for _ in range(10)]
        assert len(set(keys)) == 10  # All should be unique


@pytest.mark.asyncio
async def test_correlation_middleware_integration():
    """Test correlation middleware integration."""
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from src.llama_mapper.api.middleware.correlation import CorrelationMiddleware
    
    app = FastAPI()
    app.add_middleware(CorrelationMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        from src.llama_mapper.utils.correlation import get_correlation_id
        return {"correlation_id": get_correlation_id()}
    
    client = TestClient(app)
    
    # Test without correlation ID header
    response = client.get("/test")
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    
    correlation_id = response.headers["X-Correlation-ID"]
    assert len(correlation_id) == 36  # UUID format
    
    # Test with provided correlation ID header
    custom_id = "custom-correlation-id"
    response = client.get("/test", headers={"X-Correlation-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == custom_id


if __name__ == "__main__":
    pytest.main([__file__])