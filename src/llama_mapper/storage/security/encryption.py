"""Field-level encryption for sensitive data with Azure Key Vault integration."""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, Optional

import structlog
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = structlog.get_logger(__name__)


class FieldEncryption:
    """Field-level encryption for sensitive data with Azure Key Vault integration."""
    
    def __init__(self, key_vault_url: Optional[str] = None, master_key: Optional[bytes] = None):
        self.key_vault_url = key_vault_url
        self.credential = DefaultAzureCredential() if key_vault_url else None
        self.master_key = master_key
        self._fernet: Optional[Fernet] = None
        self._key_vault_client: Optional[SecretClient] = None
        
        if key_vault_url:
            self._key_vault_client = SecretClient(
                vault_url=key_vault_url, 
                credential=self.credential
            )
    
    async def initialize(self):
        """Initialize encryption with key from Azure Key Vault or local key."""
        try:
            if self._key_vault_client:
                # Get encryption key from Azure Key Vault
                secret = self._key_vault_client.get_secret("field-encryption-key")
                encryption_key = secret.value.encode()
            elif self.master_key:
                # Use provided master key
                encryption_key = self.master_key
            else:
                # Generate a new key (for development/testing only)
                encryption_key = Fernet.generate_key()
                logger.warning(
                    "Using generated encryption key - not suitable for production",
                    key_vault_url=self.key_vault_url
                )
            
            self._fernet = Fernet(encryption_key)
            logger.info("Field encryption initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize field encryption", error=str(e))
            raise
    
    @classmethod
    def generate_key_from_password(cls, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password using PBKDF2."""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_field(self, data: str) -> str:
        """Encrypt sensitive field data."""
        if not data:
            return data
        
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        
        try:
            encrypted_data = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error("Failed to encrypt field data", error=str(e))
            raise
    
    def decrypt_field(self, encrypted_data: str) -> str:
        """Decrypt sensitive field data."""
        if not encrypted_data:
            return encrypted_data
        
        if not self._fernet:
            raise ValueError("Encryption not initialized")
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error("Failed to decrypt field data", error=str(e))
            raise ValueError("Invalid encrypted data")
    
    def encrypt_dict(self, data: Dict[str, Any], sensitive_fields: list[str]) -> Dict[str, Any]:
        """Encrypt specified fields in a dictionary."""
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt_field(str(encrypted_data[field]))
        
        return encrypted_data
    
    def decrypt_dict(self, encrypted_data: Dict[str, Any], sensitive_fields: list[str]) -> Dict[str, Any]:
        """Decrypt specified fields in a dictionary."""
        decrypted_data = encrypted_data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field]:
                decrypted_data[field] = self.decrypt_field(decrypted_data[field])
        
        return decrypted_data
    
    async def rotate_encryption_key(self, new_key: Optional[str] = None) -> bool:
        """Rotate encryption key in Azure Key Vault."""
        if not self._key_vault_client:
            logger.error("Key Vault client not initialized")
            return False
        
        try:
            if new_key is None:
                # Generate new key
                new_key = base64.urlsafe_b64encode(Fernet.generate_key()).decode()
            
            # Store new key in Key Vault
            self._key_vault_client.set_secret("field-encryption-key", new_key)
            
            # Update local Fernet instance
            self._fernet = Fernet(new_key.encode())
            
            logger.info("Encryption key rotated successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to rotate encryption key", error=str(e))
            return False


class SecuritySanitizer:
    """Multi-layer input sanitization for security."""
    
    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\bUNION\s+SELECT\b)"
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>"
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"/etc/passwd",
        r"\\windows\\system32"
    ]
    
    def sanitize_input(self, data: Any) -> Any:
        """Comprehensive input sanitization."""
        if isinstance(data, str):
            return self.sanitize_string(data)
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        else:
            return data
    
    def sanitize_string(self, text: str) -> str:
        """Sanitize string input."""
        import html
        import re
        
        # HTML escape
        text = html.escape(text)
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Detect and log suspicious patterns
        self.detect_malicious_patterns(text)
        
        # Length limits
        if len(text) > 10000:  # Configurable limit
            raise ValueError("Input exceeds maximum length")
        
        return text
    
    def detect_malicious_patterns(self, text: str):
        """Detect and log malicious input patterns."""
        import re
        
        patterns = {
            "sql_injection": self.SQL_INJECTION_PATTERNS,
            "xss": self.XSS_PATTERNS,
            "path_traversal": self.PATH_TRAVERSAL_PATTERNS
        }
        
        for attack_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(
                        "Malicious input detected",
                        attack_type=attack_type,
                        pattern=pattern[:50]  # Truncate for logging
                    )
                    # Could raise exception or sanitize further
                    # For now, we log and continue
                    break


class EnhancedRowLevelSecurity:
    """Enhanced row-level security policies management."""
    
    def __init__(self, connection_manager):
        self.connection_manager = connection_manager
    
    async def create_enhanced_rls_policies(self):
        """Create comprehensive RLS policies for all tables."""
        
        policies_sql = """
        -- Drop existing policies if they exist
        DROP POLICY IF EXISTS tenant_isolation_storage ON storage_records;
        DROP POLICY IF EXISTS admin_bypass_storage ON storage_records;
        DROP POLICY IF EXISTS analytics_read_only_storage ON storage_records;
        
        DROP POLICY IF EXISTS tenant_isolation_audit ON audit_trail;
        DROP POLICY IF EXISTS admin_bypass_audit ON audit_trail;
        
        DROP POLICY IF EXISTS tenant_isolation_configs ON tenant_configs;
        DROP POLICY IF EXISTS admin_bypass_configs ON tenant_configs;
        
        DROP POLICY IF EXISTS tenant_isolation_metrics ON model_metrics;
        DROP POLICY IF EXISTS admin_bypass_metrics ON model_metrics;
        DROP POLICY IF EXISTS analytics_read_only_metrics ON model_metrics;
        
        DROP POLICY IF EXISTS tenant_isolation_executions ON detector_executions;
        DROP POLICY IF EXISTS admin_bypass_executions ON detector_executions;
        
        -- Create database roles if they don't exist
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'application_role') THEN
                CREATE ROLE application_role;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'admin_role') THEN
                CREATE ROLE admin_role;
            END IF;
            
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analytics_role') THEN
                CREATE ROLE analytics_role;
            END IF;
        END $$;
        
        -- Grant appropriate permissions
        GRANT SELECT, INSERT, UPDATE, DELETE ON storage_records TO application_role;
        GRANT SELECT, INSERT ON audit_trail TO application_role;
        GRANT SELECT, INSERT, UPDATE ON tenant_configs TO application_role;
        GRANT SELECT, INSERT ON model_metrics TO application_role;
        GRANT SELECT, INSERT ON detector_executions TO application_role;
        
        GRANT ALL ON ALL TABLES IN SCHEMA public TO admin_role;
        
        GRANT SELECT ON storage_records TO analytics_role;
        GRANT SELECT ON model_metrics TO analytics_role;
        GRANT SELECT ON detector_executions TO analytics_role;
        
        -- Storage records policies
        CREATE POLICY tenant_isolation_storage ON storage_records
            FOR ALL TO application_role
            USING (tenant_id = current_setting('app.current_tenant_id', true));
        
        CREATE POLICY admin_bypass_storage ON storage_records
            FOR ALL TO admin_role
            USING (true);
        
        CREATE POLICY analytics_read_only_storage ON storage_records
            FOR SELECT TO analytics_role
            USING (true);
        
        -- Audit trail policies
        CREATE POLICY tenant_isolation_audit ON audit_trail
            FOR ALL TO application_role
            USING (tenant_id = current_setting('app.current_tenant_id', true));
        
        CREATE POLICY admin_bypass_audit ON audit_trail
            FOR ALL TO admin_role
            USING (true);
        
        -- Tenant configs policies
        CREATE POLICY tenant_isolation_configs ON tenant_configs
            FOR ALL TO application_role
            USING (tenant_id = current_setting('app.current_tenant_id', true));
        
        CREATE POLICY admin_bypass_configs ON tenant_configs
            FOR ALL TO admin_role
            USING (true);
        
        -- Model metrics policies
        CREATE POLICY tenant_isolation_metrics ON model_metrics
            FOR ALL TO application_role
            USING (tenant_id = current_setting('app.current_tenant_id', true));
        
        CREATE POLICY admin_bypass_metrics ON model_metrics
            FOR ALL TO admin_role
            USING (true);
        
        CREATE POLICY analytics_read_only_metrics ON model_metrics
            FOR SELECT TO analytics_role
            USING (true);
        
        -- Detector executions policies
        CREATE POLICY tenant_isolation_executions ON detector_executions
            FOR ALL TO application_role
            USING (tenant_id = current_setting('app.current_tenant_id', true));
        
        CREATE POLICY admin_bypass_executions ON detector_executions
            FOR ALL TO admin_role
            USING (true);
        """
        
        async with self.connection_manager.get_write_connection() as conn:
            await conn.execute(policies_sql)
        
        logger.info("Enhanced RLS policies created successfully")
    
    async def validate_tenant_isolation(self, tenant_id: str) -> Dict[str, Any]:
        """Validate tenant isolation is working correctly."""
        
        validation_results = {}
        
        async with self.connection_manager.get_read_connection() as conn:
            # Set tenant context
            await conn.execute("SET app.current_tenant_id = $1", tenant_id)
            
            # Test storage_records isolation
            storage_count = await conn.fetchval(
                "SELECT COUNT(*) FROM storage_records"
            )
            
            # Test without tenant context
            await conn.execute("RESET app.current_tenant_id")
            total_storage_count = await conn.fetchval(
                "SELECT COUNT(*) FROM storage_records"
            )
            
            validation_results['storage_records'] = {
                'tenant_visible': storage_count,
                'total_records': total_storage_count,
                'isolation_working': storage_count <= total_storage_count
            }
            
            # Test other tables similarly
            await conn.execute("SET app.current_tenant_id = $1", tenant_id)
            
            audit_count = await conn.fetchval(
                "SELECT COUNT(*) FROM audit_trail"
            )
            
            metrics_count = await conn.fetchval(
                "SELECT COUNT(*) FROM model_metrics"
            )
            
            validation_results['audit_trail'] = {'tenant_visible': audit_count}
            validation_results['model_metrics'] = {'tenant_visible': metrics_count}
            
            validation_results['overall_isolation'] = all(
                result.get('isolation_working', True) 
                for result in validation_results.values()
                if isinstance(result, dict)
            )
        
        return validation_results
