"""
API Key Manager for Mapper Service

Handles API key creation, validation, and lifecycle management
following Single Responsibility Principle.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class APIKeyInfo(BaseModel):
    """API key information model."""

    key_id: str
    tenant_id: str
    name: str
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    is_active: bool = True
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_per_minute: int = 100
    created_at: datetime
    created_by: Optional[str] = None


class APIKeyCreateRequest(BaseModel):
    """API key creation request model."""

    tenant_id: str
    name: str
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)
    scopes: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = None
    rate_limit_per_minute: int = 100


class APIKeyUsage(BaseModel):
    """API key usage tracking model."""

    api_key_id: str
    tenant_id: str
    endpoint: str
    method: str
    status_code: int
    response_time_ms: int
    tokens_processed: int = 0
    cost_cents: float = 0.0
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    request_size_bytes: Optional[int] = None
    response_size_bytes: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class APIKeyManager:
    """
    API Key Manager responsible for API key lifecycle management.

    Follows SRP by focusing solely on API key operations:
    - Creating and revoking API keys
    - Validating API keys
    - Tracking API key usage
    - Managing API key permissions
    """

    def __init__(self, database_manager):
        self.db = database_manager
        self.logger = logger.bind(component="api_key_manager")

    async def create_api_key(
        self, request: APIKeyCreateRequest, created_by: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create a new API key.

        Args:
            request: API key creation request
            created_by: User who created the key

        Returns:
            Dictionary containing the API key and metadata
        """
        try:
            # Generate secure API key
            api_key = f"mk_{secrets.token_urlsafe(32)}"  # mk = mapper key
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            key_id = f"key_{secrets.token_urlsafe(16)}"

            # Calculate expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

            # Insert into database
            query = """
                INSERT INTO api_keys (
                    key_id, key_hash, tenant_id, name, description, 
                    permissions, scopes, expires_at, rate_limit_per_minute, created_by
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                RETURNING id
            """

            await self.db.execute_query(
                query,
                key_id,
                key_hash,
                request.tenant_id,
                request.name,
                request.description,
                request.permissions,
                request.scopes,
                expires_at,
                request.rate_limit_per_minute,
                created_by,
            )

            self.logger.info(
                "API key created successfully",
                key_id=key_id,
                tenant_id=request.tenant_id,
                name=request.name,
                expires_at=expires_at.isoformat() if expires_at else None,
            )

            return {
                "api_key": api_key,
                "key_id": key_id,
                "tenant_id": request.tenant_id,
                "name": request.name,
                "expires_at": expires_at.isoformat() if expires_at else None,
                "permissions": request.permissions,
                "scopes": request.scopes,
            }

        except Exception as e:
            self.logger.error(
                "Failed to create API key", error=str(e), tenant_id=request.tenant_id
            )
            raise

    async def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """
        Validate an API key and return its information.

        Args:
            api_key: The API key to validate

        Returns:
            API key information if valid, None otherwise
        """
        try:
            # Hash the API key for lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Query database
            query = """
                SELECT 
                    key_id, tenant_id, name, description, permissions, scopes,
                    is_active, expires_at, last_used_at, usage_count,
                    rate_limit_per_minute, created_at, created_by
                FROM api_keys 
                WHERE key_hash = $1
            """

            record = await self.db.fetch_one(query, key_hash)

            if not record:
                self.logger.warning("API key not found", key_prefix=api_key[:8])
                return None

            # Check if key is active
            if not record["is_active"]:
                self.logger.warning("Inactive API key used", key_id=record["key_id"])
                return None

            # Check if key is expired
            if record["expires_at"] and datetime.utcnow() > record["expires_at"]:
                self.logger.warning("Expired API key used", key_id=record["key_id"])
                return None

            # Update last used timestamp
            await self._update_last_used(record["key_id"])

            return APIKeyInfo(
                key_id=record["key_id"],
                tenant_id=record["tenant_id"],
                name=record["name"],
                description=record["description"],
                permissions=record["permissions"] or [],
                scopes=record["scopes"] or [],
                is_active=record["is_active"],
                expires_at=record["expires_at"],
                last_used_at=record["last_used_at"],
                usage_count=record["usage_count"],
                rate_limit_per_minute=record["rate_limit_per_minute"],
                created_at=record["created_at"],
                created_by=record["created_by"],
            )

        except Exception as e:
            self.logger.error("Failed to validate API key", error=str(e))
            return None

    async def revoke_api_key(self, key_id: str, tenant_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            key_id: The key ID to revoke
            tenant_id: Tenant ID for authorization

        Returns:
            True if revoked successfully, False otherwise
        """
        try:
            query = """
                UPDATE api_keys 
                SET is_active = false, updated_at = NOW()
                WHERE key_id = $1 AND tenant_id = $2
            """

            result = await self.db.execute_query(query, key_id, tenant_id)

            if "UPDATE 1" in result:
                self.logger.info(
                    "API key revoked successfully", key_id=key_id, tenant_id=tenant_id
                )
                return True
            else:
                self.logger.warning(
                    "API key not found for revocation",
                    key_id=key_id,
                    tenant_id=tenant_id,
                )
                return False

        except Exception as e:
            self.logger.error("Failed to revoke API key", error=str(e), key_id=key_id)
            return False

    async def list_api_keys(
        self, tenant_id: str, include_inactive: bool = False
    ) -> List[APIKeyInfo]:
        """
        List API keys for a tenant.

        Args:
            tenant_id: Tenant ID
            include_inactive: Whether to include inactive keys

        Returns:
            List of API key information
        """
        try:
            query = """
                SELECT 
                    key_id, tenant_id, name, description, permissions, scopes,
                    is_active, expires_at, last_used_at, usage_count,
                    rate_limit_per_minute, created_at, created_by
                FROM api_keys 
                WHERE tenant_id = $1
            """

            if not include_inactive:
                query += " AND is_active = true"

            query += " ORDER BY created_at DESC"

            records = await self.db.fetch_many(query, tenant_id)

            return [
                APIKeyInfo(
                    key_id=record["key_id"],
                    tenant_id=record["tenant_id"],
                    name=record["name"],
                    description=record["description"],
                    permissions=record["permissions"] or [],
                    scopes=record["scopes"] or [],
                    is_active=record["is_active"],
                    expires_at=record["expires_at"],
                    last_used_at=record["last_used_at"],
                    usage_count=record["usage_count"],
                    rate_limit_per_minute=record["rate_limit_per_minute"],
                    created_at=record["created_at"],
                    created_by=record["created_by"],
                )
                for record in records
            ]

        except Exception as e:
            self.logger.error(
                "Failed to list API keys", error=str(e), tenant_id=tenant_id
            )
            return []

    async def track_usage(self, usage: APIKeyUsage) -> None:
        """
        Track API key usage.

        Args:
            usage: Usage information to track
        """
        try:
            # Get API key ID from database
            key_query = "SELECT id FROM api_keys WHERE key_id = $1"
            key_record = await self.db.fetch_one(key_query, usage.api_key_id)

            if not key_record:
                self.logger.warning(
                    "API key not found for usage tracking", key_id=usage.api_key_id
                )
                return

            # Insert usage record
            usage_query = """
                INSERT INTO api_key_usage (
                    api_key_id, tenant_id, endpoint, method, status_code,
                    response_time_ms, tokens_processed, cost_cents,
                    user_agent, ip_address, request_size_bytes, response_size_bytes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """

            await self.db.execute_query(
                usage_query,
                key_record["id"],
                usage.tenant_id,
                usage.endpoint,
                usage.method,
                usage.status_code,
                usage.response_time_ms,
                usage.tokens_processed,
                usage.cost_cents,
                usage.user_agent,
                usage.ip_address,
                usage.request_size_bytes,
                usage.response_size_bytes,
            )

            # Update usage count
            count_query = """
                UPDATE api_keys 
                SET usage_count = usage_count + 1, updated_at = NOW()
                WHERE id = $1
            """
            await self.db.execute_query(count_query, key_record["id"])

        except Exception as e:
            self.logger.error(
                "Failed to track API key usage", error=str(e), key_id=usage.api_key_id
            )

    async def get_usage_statistics(
        self, tenant_id: str, key_id: Optional[str] = None, days: int = 30
    ) -> Dict[str, any]:
        """
        Get usage statistics for API keys.

        Args:
            tenant_id: Tenant ID
            key_id: Optional specific key ID
            days: Number of days to look back

        Returns:
            Usage statistics
        """
        try:
            # Base query
            base_query = (
                """
                SELECT 
                    COUNT(*) as total_requests,
                    AVG(response_time_ms) as avg_response_time,
                    SUM(tokens_processed) as total_tokens,
                    SUM(cost_cents) as total_cost_cents,
                    COUNT(DISTINCT DATE(created_at)) as active_days
                FROM api_key_usage aku
                JOIN api_keys ak ON aku.api_key_id = ak.id
                WHERE ak.tenant_id = $1 
                AND aku.created_at >= NOW() - INTERVAL '%s days'
            """
                % days
            )

            params = [tenant_id]

            if key_id:
                base_query += " AND ak.key_id = $2"
                params.append(key_id)

            stats = await self.db.fetch_one(base_query, *params)

            # Get endpoint breakdown
            endpoint_query = (
                """
                SELECT 
                    endpoint,
                    COUNT(*) as request_count,
                    AVG(response_time_ms) as avg_response_time
                FROM api_key_usage aku
                JOIN api_keys ak ON aku.api_key_id = ak.id
                WHERE ak.tenant_id = $1 
                AND aku.created_at >= NOW() - INTERVAL '%s days'
                GROUP BY endpoint
                ORDER BY request_count DESC
                LIMIT 10
            """
                % days
            )

            endpoint_params = [tenant_id]
            if key_id:
                endpoint_query = endpoint_query.replace(
                    "WHERE ak.tenant_id = $1",
                    "WHERE ak.tenant_id = $1 AND ak.key_id = $2",
                )
                endpoint_params.append(key_id)

            endpoints = await self.db.fetch_many(endpoint_query, *endpoint_params)

            return {
                "period_days": days,
                "total_requests": stats["total_requests"] or 0,
                "avg_response_time_ms": float(stats["avg_response_time"] or 0),
                "total_tokens": stats["total_tokens"] or 0,
                "total_cost_cents": float(stats["total_cost_cents"] or 0),
                "active_days": stats["active_days"] or 0,
                "top_endpoints": [
                    {
                        "endpoint": ep["endpoint"],
                        "request_count": ep["request_count"],
                        "avg_response_time_ms": float(ep["avg_response_time"]),
                    }
                    for ep in endpoints
                ],
            }

        except Exception as e:
            self.logger.error(
                "Failed to get usage statistics", error=str(e), tenant_id=tenant_id
            )
            return {}

    async def cleanup_expired_keys(self) -> int:
        """
        Clean up expired API keys.

        Returns:
            Number of keys cleaned up
        """
        try:
            query = """
                UPDATE api_keys 
                SET is_active = false, updated_at = NOW()
                WHERE expires_at < NOW() AND is_active = true
            """

            result = await self.db.execute_query(query)

            # Extract count from result string like "UPDATE 5"
            count = int(result.split()[-1]) if result.startswith("UPDATE") else 0

            if count > 0:
                self.logger.info("Cleaned up expired API keys", count=count)

            return count

        except Exception as e:
            self.logger.error("Failed to cleanup expired keys", error=str(e))
            return 0

    async def _update_last_used(self, key_id: str) -> None:
        """Update the last used timestamp for an API key."""
        try:
            query = """
                UPDATE api_keys 
                SET last_used_at = NOW(), updated_at = NOW()
                WHERE key_id = $1
            """
            await self.db.execute_query(query, key_id)

        except Exception as e:
            self.logger.error(
                "Failed to update last used timestamp", error=str(e), key_id=key_id
            )
