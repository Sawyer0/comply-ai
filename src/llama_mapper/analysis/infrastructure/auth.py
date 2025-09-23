"""
Authentication infrastructure for the Analysis Module.

This module provides API key authentication, validation, and management
capabilities including key rotation and access control.
"""

import hashlib
import secrets
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import redis
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class APIKeyStatus(Enum):
    """API key status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"


class APIKeyScope(Enum):
    """API key scope enumeration."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    ANALYZE = "analyze"
    BATCH_ANALYZE = "batch_analyze"
    HEALTH = "health"


@dataclass
class APIKey:
    """API key data structure."""
    key_id: str
    key_hash: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    scopes: List[APIKeyScope] = None
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    created_at: datetime = None
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: Optional[int] = None  # requests per minute
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = [APIKeyScope.ANALYZE]
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "description": self.description,
            "scopes": [scope.value for scope in self.scopes],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "rate_limit": self.rate_limit
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'APIKey':
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            tenant_id=data["tenant_id"],
            name=data["name"],
            description=data.get("description"),
            scopes=[APIKeyScope(scope) for scope in data.get("scopes", ["analyze"])],
            status=APIKeyStatus(data.get("status", "active")),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data.get("last_used_at") else None,
            usage_count=data.get("usage_count", 0),
            rate_limit=data.get("rate_limit")
        )


class APIKeyRequest(BaseModel):
    """API key creation request model."""
    tenant_id: str = Field(..., description="Tenant ID for the API key")
    name: str = Field(..., description="Human-readable name for the API key")
    description: Optional[str] = Field(None, description="Description of the API key")
    scopes: List[str] = Field(default=["analyze"], description="List of allowed scopes")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days (None for no expiration)")
    rate_limit: Optional[int] = Field(None, description="Rate limit in requests per minute")


class APIKeyResponse(BaseModel):
    """API key response model."""
    key_id: str
    api_key: str  # Only returned on creation
    tenant_id: str
    name: str
    description: Optional[str]
    scopes: List[str]
    status: str
    created_at: str
    expires_at: Optional[str]
    rate_limit: Optional[int]


class APIKeyManager:
    """
    API key manager for authentication and authorization.
    
    Handles API key generation, validation, storage, and rotation.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize the API key manager.
        
        Args:
            redis_client: Redis client for key storage (optional)
        """
        self.redis_client = redis_client
        self.key_prefix = "api_key:"
        self.tenant_prefix = "tenant_keys:"
        
        # In-memory fallback storage
        self._memory_storage: Dict[str, APIKey] = {}
    
    def _get_storage_key(self, key_id: str) -> str:
        """Get Redis storage key for API key."""
        return f"{self.key_prefix}{key_id}"
    
    def _get_tenant_key(self, tenant_id: str) -> str:
        """Get Redis storage key for tenant's API keys."""
        return f"{self.tenant_prefix}{tenant_id}"
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _generate_api_key(self) -> Tuple[str, str]:
        """
        Generate a new API key and its hash.
        
        Returns:
            Tuple of (api_key, key_hash)
        """
        # Generate a secure random API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_api_key(api_key)
        return api_key, key_hash
    
    def _store_api_key(self, api_key: APIKey) -> bool:
        """
        Store API key in storage.
        
        Args:
            api_key: API key to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client:
                # Store in Redis
                storage_key = self._get_storage_key(api_key.key_id)
                tenant_key = self._get_tenant_key(api_key.tenant_id)
                
                # Store the API key data
                self.redis_client.hset(storage_key, mapping=api_key.to_dict())
                
                # Add to tenant's key set
                self.redis_client.sadd(tenant_key, api_key.key_id)
                
                # Set expiration if specified
                if api_key.expires_at:
                    ttl = int((api_key.expires_at - datetime.now(timezone.utc)).total_seconds())
                    if ttl > 0:
                        self.redis_client.expire(storage_key, ttl)
                
                return True
            else:
                # Store in memory
                self._memory_storage[api_key.key_id] = api_key
                return True
                
        except Exception as e:
            logger.error(f"Failed to store API key {api_key.key_id}: {e}")
            return False
    
    def _load_api_key(self, key_id: str) -> Optional[APIKey]:
        """
        Load API key from storage.
        
        Args:
            key_id: API key ID
            
        Returns:
            API key if found, None otherwise
        """
        try:
            if self.redis_client:
                # Load from Redis
                storage_key = self._get_storage_key(key_id)
                data = self.redis_client.hgetall(storage_key)
                
                if not data:
                    return None
                
                # Convert bytes to strings
                data = {k.decode(): v.decode() for k, v in data.items()}
                return APIKey.from_dict(data)
            else:
                # Load from memory
                return self._memory_storage.get(key_id)
                
        except Exception as e:
            logger.error(f"Failed to load API key {key_id}: {e}")
            return None
    
    def create_api_key(self, request: APIKeyRequest) -> Optional[APIKeyResponse]:
        """
        Create a new API key.
        
        Args:
            request: API key creation request
            
        Returns:
            API key response with the new key, or None if failed
        """
        try:
            # Generate API key
            api_key_str, key_hash = self._generate_api_key()
            key_id = secrets.token_urlsafe(16)
            
            # Calculate expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.now(timezone.utc) + timedelta(days=request.expires_in_days)
            
            # Create API key object
            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                tenant_id=request.tenant_id,
                name=request.name,
                description=request.description,
                scopes=[APIKeyScope(scope) for scope in request.scopes],
                expires_at=expires_at,
                rate_limit=request.rate_limit
            )
            
            # Store the API key
            if not self._store_api_key(api_key):
                return None
            
            logger.info(f"Created API key {key_id} for tenant {request.tenant_id}")
            
            # Return response with the actual API key (only time it's returned)
            return APIKeyResponse(
                key_id=key_id,
                api_key=api_key_str,
                tenant_id=request.tenant_id,
                name=request.name,
                description=request.description,
                scopes=request.scopes,
                status=api_key.status.value,
                created_at=api_key.created_at.isoformat(),
                expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
                rate_limit=api_key.rate_limit
            )
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            return None
    
    def validate_api_key(self, api_key: str, required_scopes: List[APIKeyScope]) -> Optional[APIKey]:
        """
        Validate an API key and check required scopes.
        
        Args:
            api_key: API key to validate
            required_scopes: List of required scopes
            
        Returns:
            API key object if valid, None otherwise
        """
        try:
            # Hash the provided API key
            key_hash = self._hash_api_key(api_key)
            
            # Find the API key by hash
            stored_key = None
            if self.redis_client:
                # Search in Redis (this is inefficient for large datasets)
                # In production, you'd want to maintain a hash-to-key-id mapping
                for key_id in self.redis_client.scan_iter(match=f"{self.key_prefix}*"):
                    data = self.redis_client.hgetall(key_id)
                    if data and data.get(b"key_hash", b"").decode() == key_hash:
                        data = {k.decode(): v.decode() for k, v in data.items()}
                        stored_key = APIKey.from_dict(data)
                        break
            else:
                # Search in memory
                for key in self._memory_storage.values():
                    if key.key_hash == key_hash:
                        stored_key = key
                        break
            
            if not stored_key:
                logger.warning("API key not found")
                return None
            
            # Check if key is active
            if stored_key.status != APIKeyStatus.ACTIVE:
                logger.warning(f"API key {stored_key.key_id} is not active: {stored_key.status}")
                return None
            
            # Check if key is expired
            if stored_key.expires_at and stored_key.expires_at < datetime.now(timezone.utc):
                logger.warning(f"API key {stored_key.key_id} has expired")
                # Mark as expired
                stored_key.status = APIKeyStatus.EXPIRED
                self._store_api_key(stored_key)
                return None
            
            # Check required scopes
            if not all(scope in stored_key.scopes for scope in required_scopes):
                logger.warning(f"API key {stored_key.key_id} lacks required scopes")
                return None
            
            # Update usage statistics
            stored_key.last_used_at = datetime.now(timezone.utc)
            stored_key.usage_count += 1
            self._store_api_key(stored_key)
            
            return stored_key
            
        except Exception as e:
            logger.error(f"Failed to validate API key: {e}")
            return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: API key ID to revoke
            
        Returns:
            True if successful, False otherwise
        """
        try:
            api_key = self._load_api_key(key_id)
            if not api_key:
                return False
            
            api_key.status = APIKeyStatus.REVOKED
            return self._store_api_key(api_key)
            
        except Exception as e:
            logger.error(f"Failed to revoke API key {key_id}: {e}")
            return False
    
    def rotate_api_key(self, key_id: str) -> Optional[APIKeyResponse]:
        """
        Rotate an API key (create new key, revoke old one).
        
        Args:
            key_id: API key ID to rotate
            
        Returns:
            New API key response, or None if failed
        """
        try:
            # Load existing key
            old_key = self._load_api_key(key_id)
            if not old_key:
                return None
            
            # Create new key with same properties
            new_key_str, new_key_hash = self._generate_api_key()
            new_key_id = secrets.token_urlsafe(16)
            
            new_key = APIKey(
                key_id=new_key_id,
                key_hash=new_key_hash,
                tenant_id=old_key.tenant_id,
                name=f"{old_key.name} (rotated)",
                description=old_key.description,
                scopes=old_key.scopes,
                expires_at=old_key.expires_at,
                rate_limit=old_key.rate_limit
            )
            
            # Store new key
            if not self._store_api_key(new_key):
                return None
            
            # Revoke old key
            old_key.status = APIKeyStatus.REVOKED
            self._store_api_key(old_key)
            
            logger.info(f"Rotated API key {key_id} -> {new_key_id}")
            
            return APIKeyResponse(
                key_id=new_key_id,
                api_key=new_key_str,
                tenant_id=new_key.tenant_id,
                name=new_key.name,
                description=new_key.description,
                scopes=[scope.value for scope in new_key.scopes],
                status=new_key.status.value,
                created_at=new_key.created_at.isoformat(),
                expires_at=new_key.expires_at.isoformat() if new_key.expires_at else None,
                rate_limit=new_key.rate_limit
            )
            
        except Exception as e:
            logger.error(f"Failed to rotate API key {key_id}: {e}")
            return None
    
    def list_tenant_keys(self, tenant_id: str) -> List[APIKey]:
        """
        List all API keys for a tenant.
        
        Args:
            tenant_id: Tenant ID
            
        Returns:
            List of API keys
        """
        try:
            keys = []
            if self.redis_client:
                # Get from Redis
                tenant_key = self._get_tenant_key(tenant_id)
                key_ids = self.redis_client.smembers(tenant_key)
                
                for key_id in key_ids:
                    api_key = self._load_api_key(key_id.decode())
                    if api_key:
                        keys.append(api_key)
            else:
                # Get from memory
                for key in self._memory_storage.values():
                    if key.tenant_id == tenant_id:
                        keys.append(key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to list keys for tenant {tenant_id}: {e}")
            return []
    
    def cleanup_expired_keys(self) -> int:
        """
        Clean up expired API keys.
        
        Returns:
            Number of keys cleaned up
        """
        try:
            cleaned_count = 0
            current_time = datetime.now(timezone.utc)
            
            if self.redis_client:
                # Clean up in Redis
                for key_id in self.redis_client.scan_iter(match=f"{self.key_prefix}*"):
                    api_key = self._load_api_key(key_id.decode().replace(self.key_prefix, ""))
                    if api_key and api_key.expires_at and api_key.expires_at < current_time:
                        api_key.status = APIKeyStatus.EXPIRED
                        self._store_api_key(api_key)
                        cleaned_count += 1
            else:
                # Clean up in memory
                for key in self._memory_storage.values():
                    if key.expires_at and key.expires_at < current_time:
                        key.status = APIKeyStatus.EXPIRED
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired API keys")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired keys: {e}")
            return 0
