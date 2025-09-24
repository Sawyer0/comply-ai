"""Field-level encryption for sensitive data using Azure Key Vault."""

from __future__ import annotations

import base64
import hashlib
import os
from typing import Any, Dict, Optional

import structlog

try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
    from cryptography.fernet import Fernet
    AZURE_AVAILABLE = True
    CRYPTO_AVAILABLE = True
except ImportError:
    DefaultAzureCredential = None
    SecretClient = None
    Fernet = None
    AZURE_AVAILABLE = False
    CRYPTO_AVAILABLE = False

logger = structlog.get_logger(__name__)


class FieldEncryption:
    """Field-level encryption for sensitive data."""
    
    def __init__(self, key_vault_url: Optional[str] = None, master_key: Optional[bytes] = None):
        self.key_vault_url = key_vault_url
        self.master_key = master_key
        self._fernet: Optional[Any] = None
        self._credential = DefaultAzureCredential() if AZURE_AVAILABLE else None
        
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available, using dummy encryption")
            self._fernet = DummyFernet()
        else:
            self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption with Azure Key Vault or local key."""
        try:
            if self.key_vault_url and self._credential:
                # Get encryption key from Azure Key Vault
                encryption_key = self._get_key_from_vault("field-encryption-key")
            elif self.master_key:
                encryption_key = self.master_key
            else:
                # Generate a local key (for development only)
                encryption_key = self._generate_local_key()
                logger.warning("Using local encryption key - not suitable for production")
            
            # Ensure key is 32 bytes for Fernet
            if len(encryption_key) < 32:
                encryption_key = encryption_key.ljust(32, b'0')
            else:
                encryption_key = encryption_key[:32]
            
            fernet_key = base64.urlsafe_b64encode(encryption_key)
            self._fernet = Fernet(fernet_key)
            
            logger.info("Field encryption initialized", 
                       key_vault_enabled=bool(self.key_vault_url))
            
        except Exception as e:
            logger.error("Failed to initialize field encryption", error=str(e))
            self._fernet = DummyFernet()
    
    def _get_key_from_vault(self, secret_name: str) -> bytes:
        """Get encryption key from Azure Key Vault."""
        try:
            client = SecretClient(vault_url=self.key_vault_url, credential=self._credential)
            secret = client.get_secret(secret_name)
            return secret.value.encode()
        except Exception as e:
            logger.error("Failed to get encryption key from Key Vault", error=str(e))
            raise ValueError(f"Failed to get encryption key: {e}") from e
    
    def _generate_local_key(self) -> bytes:
        """Generate a local encryption key."""
        return os.urandom(32)
    
    def encrypt_field(self, data: str) -> str:
        """Encrypt sensitive field data."""
        if not data:
            return data
        
        try:
            encrypted_data = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error("Failed to encrypt field data", error=str(e))
            raise ValueError("Encryption failed") from e
    
    def decrypt_field(self, encrypted_data: str) -> str:
        """Decrypt sensitive field data."""
        if not encrypted_data:
            return encrypted_data
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error("Failed to decrypt field data", error=str(e))
            raise ValueError("Decryption failed") from e
    
    def hash_field(self, data: str) -> str:
        """Create SHA-256 hash of field data for privacy compliance."""
        if not data:
            return ""
        
        return hashlib.sha256(data.encode()).hexdigest()


class DummyFernet:
    """Dummy encryption for testing/fallback."""
    
    def encrypt(self, data: bytes) -> bytes:
        """Return data unchanged (for testing)."""
        return data
    
    def decrypt(self, data: bytes) -> bytes:
        """Return data unchanged (for testing)."""
        return data


class EncryptedField:
    """Pydantic field type for automatic encryption."""
    
    def __init__(self, encryption_manager: FieldEncryption):
        self.encryption = encryption_manager
    
    def __get_validators__(self):
        yield self.validate
    
    def validate(self, value):
        if isinstance(value, str) and value:
            return self.encryption.encrypt_field(value)
        return value
    
    def decrypt(self, encrypted_value: str) -> str:
        return self.encryption.decrypt_field(encrypted_value)


# Global encryption instance
_global_encryption: Optional[FieldEncryption] = None


def get_field_encryption() -> FieldEncryption:
    """Get global field encryption instance."""
    global _global_encryption
    if _global_encryption is None:
        _global_encryption = FieldEncryption()
    return _global_encryption


def initialize_field_encryption(key_vault_url: Optional[str] = None, 
                               master_key: Optional[bytes] = None):
    """Initialize global field encryption."""
    global _global_encryption
    _global_encryption = FieldEncryption(key_vault_url, master_key)


def encrypt_sensitive_data(data: Dict[str, Any], sensitive_fields: list) -> Dict[str, Any]:
    """Encrypt sensitive fields in a dictionary."""
    encryption = get_field_encryption()
    encrypted_data = data.copy()
    
    for field in sensitive_fields:
        if field in encrypted_data and encrypted_data[field]:
            encrypted_data[field] = encryption.encrypt_field(str(encrypted_data[field]))
    
    return encrypted_data


def decrypt_sensitive_data(data: Dict[str, Any], sensitive_fields: list) -> Dict[str, Any]:
    """Decrypt sensitive fields in a dictionary."""
    encryption = get_field_encryption()
    decrypted_data = data.copy()
    
    for field in sensitive_fields:
        if field in decrypted_data and decrypted_data[field]:
            decrypted_data[field] = encryption.decrypt_field(decrypted_data[field])
    
    return decrypted_data