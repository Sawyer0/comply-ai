"""Tenant management functionality following SRP.

This module provides ONLY tenant management - creating, updating, and managing tenants.
Single Responsibility: Manage tenant lifecycle and basic tenant operations.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import ValidationError, AuthenticationError

logger = logging.getLogger(__name__)


class TenantStatus(str, Enum):
    """Tenant status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"


class TenantTier(str, Enum):
    """Tenant tier enumeration for different service levels."""

    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuotas:
    """Tenant resource quotas."""

    max_requests_per_hour: int = 1000
    max_detectors: int = 10
    max_policies: int = 5
    max_api_keys: int = 10
    max_storage_mb: int = 100
    max_concurrent_requests: int = 10


@dataclass
class Tenant:
    """Tenant data structure."""

    tenant_id: str
    name: str
    status: TenantStatus = TenantStatus.PENDING
    tier: TenantTier = TenantTier.FREE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    quotas: TenantQuotas = field(default_factory=TenantQuotas)
    metadata: Dict[str, Any] = field(default_factory=dict)
    contact_email: Optional[str] = None
    organization: Optional[str] = None
    allowed_domains: List[str] = field(default_factory=list)
    features: Set[str] = field(default_factory=set)


class TenantManager:
    """Manages tenant lifecycle and operations.

    Single Responsibility: Create, update, and manage tenant information.
    Does NOT handle: tenant isolation, routing, configuration, billing.
    """

    def __init__(self):
        """Initialize tenant manager."""
        self._tenants: Dict[str, Tenant] = {}
        self._tenant_name_index: Dict[str, str] = {}  # name -> tenant_id
        self._default_quotas = {
            TenantTier.FREE: TenantQuotas(
                max_requests_per_hour=100,
                max_detectors=3,
                max_policies=2,
                max_api_keys=2,
                max_storage_mb=10,
                max_concurrent_requests=2,
            ),
            TenantTier.BASIC: TenantQuotas(
                max_requests_per_hour=1000,
                max_detectors=10,
                max_policies=5,
                max_api_keys=5,
                max_storage_mb=100,
                max_concurrent_requests=5,
            ),
            TenantTier.PREMIUM: TenantQuotas(
                max_requests_per_hour=10000,
                max_detectors=25,
                max_policies=15,
                max_api_keys=15,
                max_storage_mb=1000,
                max_concurrent_requests=15,
            ),
            TenantTier.ENTERPRISE: TenantQuotas(
                max_requests_per_hour=100000,
                max_detectors=100,
                max_policies=50,
                max_api_keys=50,
                max_storage_mb=10000,
                max_concurrent_requests=50,
            ),
        }

    def create_tenant(
        self,
        tenant_id: str,
        name: str,
        tier: TenantTier = TenantTier.FREE,
        contact_email: Optional[str] = None,
        organization: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            tenant_id: Unique tenant identifier
            name: Tenant display name
            tier: Tenant service tier
            contact_email: Contact email address
            organization: Organization name
            metadata: Additional tenant metadata

        Returns:
            Created tenant object
        """
        correlation_id = get_correlation_id()

        try:
            # Validate tenant doesn't already exist
            if tenant_id in self._tenants:
                raise ValidationError(
                    f"Tenant {tenant_id} already exists", correlation_id=correlation_id
                )

            if name in self._tenant_name_index:
                raise ValidationError(
                    f"Tenant name '{name}' already exists",
                    correlation_id=correlation_id,
                )

            # Validate inputs
            self._validate_tenant_id(tenant_id)
            self._validate_tenant_name(name)

            if contact_email:
                self._validate_email(contact_email)

            # Create tenant with appropriate quotas
            quotas = self._default_quotas.get(tier, TenantQuotas())

            tenant = Tenant(
                tenant_id=tenant_id,
                name=name,
                tier=tier,
                quotas=quotas,
                contact_email=contact_email,
                organization=organization,
                metadata=metadata or {},
                status=TenantStatus.ACTIVE,  # Start as active for simplicity
            )

            # Store tenant
            self._tenants[tenant_id] = tenant
            self._tenant_name_index[name] = tenant_id

            logger.info(
                "Created tenant %s (%s) with tier %s",
                tenant_id,
                name,
                tier.value,
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "tenant_name": name,
                    "tier": tier.value,
                    "organization": organization,
                },
            )

            return tenant

        except Exception as e:
            logger.error(
                "Failed to create tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            raise

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant object if found, None otherwise
        """
        return self._tenants.get(tenant_id)

    def get_tenant_by_name(self, name: str) -> Optional[Tenant]:
        """Get a tenant by name.

        Args:
            name: Tenant name

        Returns:
            Tenant object if found, None otherwise
        """
        tenant_id = self._tenant_name_index.get(name)
        if tenant_id:
            return self._tenants.get(tenant_id)
        return None

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        tier: Optional[TenantTier] = None,
        status: Optional[TenantStatus] = None,
        contact_email: Optional[str] = None,
        organization: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update tenant information.

        Args:
            tenant_id: Tenant identifier
            name: New tenant name (optional)
            tier: New tenant tier (optional)
            status: New tenant status (optional)
            contact_email: New contact email (optional)
            organization: New organization (optional)
            metadata: New metadata (optional)

        Returns:
            True if update successful, False otherwise
        """
        correlation_id = get_correlation_id()

        try:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                logger.warning(
                    "Cannot update tenant: tenant %s not found",
                    tenant_id,
                    extra={"correlation_id": correlation_id, "tenant_id": tenant_id},
                )
                return False

            # Update name if provided
            if name and name != tenant.name:
                if name in self._tenant_name_index:
                    raise ValidationError(
                        f"Tenant name '{name}' already exists",
                        correlation_id=correlation_id,
                    )

                self._validate_tenant_name(name)

                # Update name index
                del self._tenant_name_index[tenant.name]
                self._tenant_name_index[name] = tenant_id
                tenant.name = name

            # Update tier and quotas if provided
            if tier and tier != tenant.tier:
                tenant.tier = tier
                tenant.quotas = self._default_quotas.get(tier, TenantQuotas())

            # Update other fields
            if status:
                tenant.status = status

            if contact_email is not None:
                if contact_email:
                    self._validate_email(contact_email)
                tenant.contact_email = contact_email

            if organization is not None:
                tenant.organization = organization

            if metadata is not None:
                tenant.metadata.update(metadata)

            tenant.updated_at = datetime.utcnow()

            logger.info(
                "Updated tenant %s",
                tenant_id,
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "updated_fields": {
                        "name": name,
                        "tier": tier.value if tier else None,
                        "status": status.value if status else None,
                    },
                },
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to update tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            return False

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant (soft delete by marking as deleted).

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if deletion successful, False otherwise
        """
        correlation_id = get_correlation_id()

        try:
            tenant = self._tenants.get(tenant_id)
            if not tenant:
                logger.warning(
                    "Cannot delete tenant: tenant %s not found",
                    tenant_id,
                    extra={"correlation_id": correlation_id, "tenant_id": tenant_id},
                )
                return False

            # Soft delete - mark as deleted but keep record
            tenant.status = TenantStatus.DELETED
            tenant.updated_at = datetime.utcnow()

            # Remove from name index
            self._tenant_name_index.pop(tenant.name, None)

            logger.info(
                "Deleted tenant %s (%s)",
                tenant_id,
                tenant.name,
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "tenant_name": tenant.name,
                },
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to delete tenant %s: %s",
                tenant_id,
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )
            return False

    def list_tenants(
        self, status: Optional[TenantStatus] = None, tier: Optional[TenantTier] = None
    ) -> List[Tenant]:
        """List tenants with optional filtering.

        Args:
            status: Filter by tenant status (optional)
            tier: Filter by tenant tier (optional)

        Returns:
            List of tenants matching criteria
        """
        tenants = list(self._tenants.values())

        if status:
            tenants = [t for t in tenants if t.status == status]

        if tier:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants

    def is_tenant_active(self, tenant_id: str) -> bool:
        """Check if a tenant is active.

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if tenant is active, False otherwise
        """
        tenant = self._tenants.get(tenant_id)
        return tenant is not None and tenant.status == TenantStatus.ACTIVE

    def get_tenant_quotas(self, tenant_id: str) -> Optional[TenantQuotas]:
        """Get quotas for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant quotas if found, None otherwise
        """
        tenant = self._tenants.get(tenant_id)
        return tenant.quotas if tenant else None

    def add_tenant_feature(self, tenant_id: str, feature: str) -> bool:
        """Add a feature to a tenant.

        Args:
            tenant_id: Tenant identifier
            feature: Feature name to add

        Returns:
            True if feature added successfully
        """
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.features.add(feature)
            tenant.updated_at = datetime.utcnow()
            return True
        return False

    def remove_tenant_feature(self, tenant_id: str, feature: str) -> bool:
        """Remove a feature from a tenant.

        Args:
            tenant_id: Tenant identifier
            feature: Feature name to remove

        Returns:
            True if feature removed successfully
        """
        tenant = self._tenants.get(tenant_id)
        if tenant and feature in tenant.features:
            tenant.features.remove(feature)
            tenant.updated_at = datetime.utcnow()
            return True
        return False

    def has_tenant_feature(self, tenant_id: str, feature: str) -> bool:
        """Check if a tenant has a specific feature.

        Args:
            tenant_id: Tenant identifier
            feature: Feature name to check

        Returns:
            True if tenant has the feature
        """
        tenant = self._tenants.get(tenant_id)
        return tenant is not None and feature in tenant.features

    def _validate_tenant_id(self, tenant_id: str):
        """Validate tenant ID format."""
        if not tenant_id or not isinstance(tenant_id, str):
            raise ValidationError("Tenant ID must be a non-empty string")

        if len(tenant_id) < 3 or len(tenant_id) > 50:
            raise ValidationError("Tenant ID must be between 3 and 50 characters")

        if not re.match(r"^[a-zA-Z0-9_-]+$", tenant_id):
            raise ValidationError(
                "Tenant ID can only contain letters, numbers, hyphens, and underscores"
            )

    def _validate_tenant_name(self, name: str):
        """Validate tenant name format."""
        if not name or not isinstance(name, str):
            raise ValidationError("Tenant name must be a non-empty string")

        if len(name) < 2 or len(name) > 100:
            raise ValidationError("Tenant name must be between 2 and 100 characters")

    def _validate_email(self, email: str):
        """Validate email format."""
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            raise ValidationError("Invalid email format")

    def get_tenant_stats(self) -> Dict[str, Any]:
        """Get statistics about tenants.

        Returns:
            Dictionary with tenant statistics
        """
        total_tenants = len(self._tenants)
        if total_tenants == 0:
            return {
                "total_tenants": 0,
                "by_status": {},
                "by_tier": {},
                "active_tenants": 0,
            }

        by_status = {}
        by_tier = {}
        active_count = 0

        for tenant in self._tenants.values():
            # Count by status
            status_key = tenant.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            # Count by tier
            tier_key = tenant.tier.value
            by_tier[tier_key] = by_tier.get(tier_key, 0) + 1

            # Count active tenants
            if tenant.status == TenantStatus.ACTIVE:
                active_count += 1

        return {
            "total_tenants": total_tenants,
            "by_status": by_status,
            "by_tier": by_tier,
            "active_tenants": active_count,
        }


# Export only the tenant management functionality
__all__ = [
    "TenantManager",
    "Tenant",
    "TenantStatus",
    "TenantTier",
    "TenantQuotas",
]
