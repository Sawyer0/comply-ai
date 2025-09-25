"""
Security configuration for the Analysis Service.
"""

import secrets
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    # Authentication settings
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60

    # Rate limiting
    rate_limit_requests_per_minute: int = 100
    rate_limit_burst_size: int = 20

    # Input validation
    max_input_size_bytes: int = 1024 * 1024  # 1MB
    allowed_content_types: Set[str] = field(
        default_factory=lambda: {"application/json", "text/plain"}
    )

    # Audit settings
    enable_audit_logging: bool = True
    audit_log_retention_days: int = 90

    # API security
    require_api_key: bool = True
    api_key_header: str = "X-API-Key"

    # Content security
    enable_content_scanning: bool = True
    blocked_patterns: List[str] = field(default_factory=list)
