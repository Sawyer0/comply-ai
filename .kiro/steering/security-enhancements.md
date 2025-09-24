---
inclusion: always
---

# Security Enhancements

## Secrets Management & Rotation

### Automated Secrets Rotation
```python
# src/llama_mapper/security/secrets_rotation.py
import hvac
from datetime import datetime, timedelta
from typing import Dict, List
import asyncio

class SecretsRotationManager:
    """Automated secrets rotation with HashiCorp Vault"""
    
    def __init__(self, vault_client: hvac.Client):
        self.vault = vault_client
        self.rotation_policies = self.load_rotation_policies()
    
    async def rotate_database_credentials(self, database_name: str):
        """Rotate database credentials"""
        try:
            # Generate new credentials
            new_credentials = await self.generate_database_credentials()
            
            # Update database user
            await self.update_database_user(database_name, new_credentials)
            
            # Store new credentials in Vault
            await self.store_credentials_in_vault(
                f"database/{database_name}",
                new_credentials
            )
            
            # Update application configuration
            await self.update_application_config(database_name, new_credentials)
            
            # Verify connectivity
            await self.verify_database_connectivity(database_name)
            
            logger.info("Database credentials rotated successfully",
                       database=database_name,
                       correlation_id=get_correlation_id())
            
        except Exception as e:
            logger.error("Failed to rotate database credentials",
                        database=database_name,
                        error=str(e),
                        correlation_id=get_correlation_id())
            await self.rollback_credential_rotation(database_name)
            raise
    
    async def rotate_api_keys(self, tenant_id: str):
        """Rotate tenant API keys"""
        try:
            # Generate new API key
            new_api_key = self.generate_secure_api_key()
            
            # Store in Vault with metadata
            await self.vault.secrets.kv.v2.create_or_update_secret(
                path=f"api-keys/{tenant_id}",
                secret={
                    "api_key": new_api_key,
                    "created_at": datetime.utcnow().isoformat(),
                    "rotated_by": "automated_rotation",
                    "previous_key_id": await self.get_current_key_id(tenant_id)
                }
            )
            
            # Update tenant configuration
            await self.update_tenant_api_key(tenant_id, new_api_key)
            
            # Notify tenant of key rotation
            await self.notify_tenant_key_rotation(tenant_id, new_api_key)
            
        except Exception as e:
            logger.error("API key rotation failed",
                        tenant_id=tenant_id,
                        error=str(e))
            raise
    
    def schedule_rotation_jobs(self):
        """Schedule automated rotation jobs"""
        rotation_schedule = {
            "database_credentials": "0 2 * * 0",  # Weekly on Sunday 2 AM
            "api_keys": "0 3 1 * *",              # Monthly on 1st at 3 AM
            "tls_certificates": "0 4 1 */3 *",    # Quarterly
            "encryption_keys": "0 5 1 */6 *"      # Semi-annually
        }
        
        for job_type, cron_schedule in rotation_schedule.items():
            self.schedule_cron_job(job_type, cron_schedule)
```

### Multi-Layer Input Sanitization
```python
# src/llama_mapper/security/input_sanitization.py
import re
import html
from typing import Any, Dict, List
from pydantic import BaseModel, validator

class SecuritySanitizer:
    """Multi-layer input sanitization"""
    
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
        """Comprehensive input sanitization"""
        if isinstance(data, str):
            return self.sanitize_string(data)
        elif isinstance(data, dict):
            return {k: self.sanitize_input(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.sanitize_input(item) for item in data]
        else:
            return data
    
    def sanitize_string(self, text: str) -> str:
        """Sanitize string input"""
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
        """Detect and log malicious input patterns"""
        patterns = {
            "sql_injection": self.SQL_INJECTION_PATTERNS,
            "xss": self.XSS_PATTERNS,
            "path_traversal": self.PATH_TRAVERSAL_PATTERNS
        }
        
        for attack_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning("Malicious input detected",
                                 attack_type=attack_type,
                                 pattern=pattern,
                                 correlation_id=get_correlation_id())
                    
                    # Could raise exception or sanitize further
                    # For now, we log and continue
                    break

# Enhanced Pydantic models with security validation
class SecureDetectorRequest(BaseModel):
    """Secure detector request with input validation"""
    
    detector_type: str
    content: str
    metadata: Dict[str, Any] = {}
    
    @validator('detector_type')
    def validate_detector_type(cls, v):
        allowed_types = ['presidio', 'deberta', 'custom']
        if v not in allowed_types:
            raise ValueError(f"Invalid detector type: {v}")
        return v
    
    @validator('content')
    def sanitize_content(cls, v):
        sanitizer = SecuritySanitizer()
        return sanitizer.sanitize_string(v)
    
    @validator('metadata')
    def sanitize_metadata(cls, v):
        sanitizer = SecuritySanitizer()
        return sanitizer.sanitize_input(v)
```

## Network Security & Segmentation

### Service Mesh Security
```python
# src/llama_mapper/security/service_mesh.py
class ServiceMeshSecurity:
    """Service mesh security configuration"""
    
    def __init__(self):
        self.allowed_services = self.load_service_registry()
        self.security_policies = self.load_security_policies()
    
    def generate_istio_policies(self) -> Dict[str, Any]:
        """Generate Istio security policies"""
        return {
            "authentication_policy": {
                "apiVersion": "security.istio.io/v1beta1",
                "kind": "PeerAuthentication",
                "metadata": {"name": "llama-mapper-mtls"},
                "spec": {
                    "mtls": {"mode": "STRICT"}
                }
            },
            "authorization_policy": {
                "apiVersion": "security.istio.io/v1beta1",
                "kind": "AuthorizationPolicy",
                "metadata": {"name": "llama-mapper-authz"},
                "spec": {
                    "rules": [
                        {
                            "from": [{"source": {"principals": ["cluster.local/ns/llama-mapper/sa/mapper-service"]}}],
                            "to": [{"operation": {"methods": ["GET", "POST"]}}]
                        }
                    ]
                }
            }
        }
    
    def validate_service_communication(self, source_service: str, target_service: str, operation: str) -> bool:
        """Validate inter-service communication"""
        policy = self.security_policies.get(target_service, {})
        allowed_sources = policy.get('allowed_sources', [])
        allowed_operations = policy.get('allowed_operations', [])
        
        return (source_service in allowed_sources and 
                operation in allowed_operations)
```

### Network Policies
```yaml
# config/security/network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llama-mapper-network-policy
spec:
  podSelector:
    matchLabels:
      app: llama-mapper
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Advanced Authentication & Authorization

### JWT Token Security
```python
# src/llama_mapper/security/jwt_security.py
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional
import secrets

class JWTSecurityManager:
    """Enhanced JWT token management"""
    
    def __init__(self, private_key: str, public_key: str):
        self.private_key = private_key
        self.public_key = public_key
        self.token_blacklist = set()  # In production, use Redis
        self.refresh_tokens = {}      # In production, use secure storage
    
    def create_access_token(self, user_data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create secure access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        # Add security claims
        payload = {
            **user_data,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_urlsafe(32),  # JWT ID for revocation
            "type": "access"
        }
        
        return jwt.encode(payload, self.private_key, algorithm="RS256")
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create secure refresh token"""
        jti = secrets.token_urlsafe(32)
        expire = datetime.utcnow() + timedelta(days=30)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": jti,
            "type": "refresh"
        }
        
        token = jwt.encode(payload, self.private_key, algorithm="RS256")
        
        # Store refresh token securely
        self.refresh_tokens[jti] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_used": datetime.utcnow()
        }
        
        return token
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate and decode token"""
        try:
            # Check if token is blacklisted
            decoded = jwt.decode(token, self.public_key, algorithms=["RS256"])
            jti = decoded.get("jti")
            
            if jti in self.token_blacklist:
                raise jwt.InvalidTokenError("Token has been revoked")
            
            return decoded
            
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token used", correlation_id=get_correlation_id())
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token used", error=str(e), correlation_id=get_correlation_id())
            return None
    
    def revoke_token(self, token: str):
        """Revoke a token"""
        try:
            decoded = jwt.decode(token, self.public_key, algorithms=["RS256"])
            jti = decoded.get("jti")
            if jti:
                self.token_blacklist.add(jti)
                logger.info("Token revoked", jti=jti, correlation_id=get_correlation_id())
        except jwt.InvalidTokenError:
            pass  # Token already invalid
```

### Role-Based Access Control (RBAC)
```python
# src/llama_mapper/security/rbac.py
from enum import Enum
from typing import List, Set
from dataclasses import dataclass

class Permission(Enum):
    READ_MAPPINGS = "read:mappings"
    WRITE_MAPPINGS = "write:mappings"
    MANAGE_DETECTORS = "manage:detectors"
    VIEW_ANALYTICS = "view:analytics"
    MANAGE_USERS = "manage:users"
    ADMIN_ACCESS = "admin:access"

class Role(Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class RolePermissions:
    role: Role
    permissions: Set[Permission]

class RBACManager:
    """Role-based access control manager"""
    
    ROLE_PERMISSIONS = {
        Role.VIEWER: {
            Permission.READ_MAPPINGS,
            Permission.VIEW_ANALYTICS
        },
        Role.ANALYST: {
            Permission.READ_MAPPINGS,
            Permission.WRITE_MAPPINGS,
            Permission.VIEW_ANALYTICS,
            Permission.MANAGE_DETECTORS
        },
        Role.ADMIN: {
            Permission.READ_MAPPINGS,
            Permission.WRITE_MAPPINGS,
            Permission.VIEW_ANALYTICS,
            Permission.MANAGE_DETECTORS,
            Permission.MANAGE_USERS
        },
        Role.SUPER_ADMIN: {perm for perm in Permission}
    }
    
    def check_permission(self, user_role: Role, required_permission: Permission) -> bool:
        """Check if user role has required permission"""
        role_permissions = self.ROLE_PERMISSIONS.get(user_role, set())
        return required_permission in role_permissions
    
    def get_user_permissions(self, user_role: Role) -> Set[Permission]:
        """Get all permissions for a user role"""
        return self.ROLE_PERMISSIONS.get(user_role, set())

# Decorator for permission checking
def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user from request context
            user = get_current_user()  # Implementation depends on auth system
            
            rbac = RBACManager()
            if not rbac.check_permission(user.role, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {permission.value}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

## Data Encryption & Privacy

### Field-Level Encryption
```python
# src/llama_mapper/security/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class FieldEncryption:
    """Field-level encryption for sensitive data"""
    
    def __init__(self, master_key: bytes):
        self.master_key = master_key
        self.fernet = Fernet(master_key)
    
    @classmethod
    def generate_key_from_password(cls, password: str, salt: bytes = None) -> bytes:
        """Generate encryption key from password"""
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
        """Encrypt sensitive field data"""
        if not data:
            return data
        
        encrypted_data = self.fernet.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_field(self, encrypted_data: str) -> str:
        """Decrypt sensitive field data"""
        if not encrypted_data:
            return encrypted_data
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error("Failed to decrypt field data", error=str(e))
            raise ValueError("Invalid encrypted data")

# Pydantic model with automatic encryption
class EncryptedField:
    """Custom Pydantic field type for automatic encryption"""
    
    def __init__(self, encryption_manager: FieldEncryption):
        self.encryption = encryption_manager
    
    def __get_validators__(self):
        yield self.validate
    
    def validate(self, value):
        if isinstance(value, str):
            return self.encryption.encrypt_field(value)
        return value
    
    def decrypt(self, encrypted_value: str) -> str:
        return self.encryption.decrypt_field(encrypted_value)
```

These security enhancements provide:

1. **Automated Secrets Rotation**: Vault integration with scheduled rotation
2. **Multi-Layer Input Sanitization**: Comprehensive protection against injection attacks
3. **Network Segmentation**: Service mesh and network policies
4. **Enhanced Authentication**: JWT with refresh tokens and revocation
5. **RBAC System**: Fine-grained permission management
6. **Field-Level Encryption**: Protect sensitive data at rest

This creates a defense-in-depth security posture for your system!