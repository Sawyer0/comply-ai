"""Security functionality for orchestration service following SRP.

This module provides security capabilities with clear separation of concerns:
- Authentication: API key management and JWT security
- Authorization: RBAC and permission management
- WAF: Web Application Firewall and attack detection
- Validation: Input sanitization and security validation
"""

# Authentication components
from .auth.api_key_manager import (
    ApiKeyManager,
    ApiKey,
    ApiKeyStatus,
)

# RBAC components
from .rbac.rbac_manager import (
    RBACManager,
    Permission,
    Role,
    RoleDefinition,
)

# WAF components
from .waf.attack_detector import (
    AttackDetector,
    AttackPattern,
    AttackDetection,
    AttackType,
    AttackSeverity,
)

# Validation components
from .validation.input_sanitizer import (
    InputSanitizer,
)

__all__ = [
    # Authentication
    "ApiKeyManager",
    "ApiKey",
    "ApiKeyStatus",
    # RBAC
    "RBACManager",
    "Permission",
    "Role",
    "RoleDefinition",
    # WAF
    "AttackDetector",
    "AttackPattern",
    "AttackDetection",
    "AttackType",
    "AttackSeverity",
    # Validation
    "InputSanitizer",
]
