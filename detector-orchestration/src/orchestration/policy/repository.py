"""Policy repository for CRUD operations and caching.

Extracted from PolicyManager to handle only policy storage,
retrieval, and basic validation operations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from shared.database.connection_manager import get_service_db
logger = logging.getLogger(__name__)

__all__ = [
    "PolicyRepository",
]


class PolicyRepository:
    """Manages policy storage, retrieval, and validation.
    
    Single Responsibility: Handle policy CRUD operations and caching.
    Does NOT handle: OPA communication, template loading, health checks.
    """

    def __init__(self, *, service_name: str = "orchestration"):
        """Initialize policy repository.

        Uses in-memory cache plus a backing database table for tenant_policies.
        """
        self._db = get_service_db(service_name)
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.default_policy = {
            "max_detectors": 5,
            "min_confidence": 0.7,
            "timeout_ms": 5000,
            "retry_count": 3,
            "coverage_method": "required_set",
        }

    def set_policy(
        self, tenant_id: str, policy: Dict[str, Any], bundle: Optional[str] = None
    ) -> bool:
        """Set policy for a tenant.

        Args:
            tenant_id: Tenant identifier
            policy: Policy configuration
            bundle: Optional policy bundle identifier

        Returns:
            True if policy was set successfully
        """
        try:
            policy_key = f"{tenant_id}:{bundle}" if bundle else tenant_id
            self.policies[policy_key] = policy

            logger.info(
                "Policy set for tenant %s",
                tenant_id,
                extra={
                    "tenant_id": tenant_id,
                    "bundle": bundle,
                    "policy_key": policy_key,
                },
            )

            return True

        except (TypeError, ValueError, KeyError) as e:
            logger.error(
                "Failed to set policy for tenant %s: %s",
                tenant_id,
                str(e),
                extra={"tenant_id": tenant_id, "error": str(e)},
            )
            return False

    def get_policy(
        self, tenant_id: str, bundle: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get policy for a tenant.

        Args:
            tenant_id: Tenant identifier
            bundle: Optional policy bundle identifier

        Returns:
            Policy configuration if found, None otherwise
        """
        policy_key = f"{tenant_id}:{bundle}" if bundle else tenant_id
        return self.policies.get(policy_key)

    def get_policy_or_default(
        self, tenant_id: str, bundle: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get policy for tenant or return default policy.

        Args:
            tenant_id: Tenant identifier
            bundle: Optional policy bundle identifier

        Returns:
            Policy configuration or default policy
        """
        policy_key = f"{tenant_id}:{bundle}" if bundle else tenant_id
        return self.policies.get(policy_key, self.default_policy)

    def validate_policy(self, policy: Dict[str, Any]) -> List[str]:
        """Validate policy configuration.

        Args:
            policy: Policy configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Validate required fields
        if "max_detectors" in policy:
            if (
                not isinstance(policy["max_detectors"], int)
                or policy["max_detectors"] <= 0
            ):
                errors.append("max_detectors must be a positive integer")

        if "min_confidence" in policy:
            if (
                not isinstance(policy["min_confidence"], (int, float))
                or not 0 <= policy["min_confidence"] <= 1
            ):
                errors.append("min_confidence must be a number between 0 and 1")

        if "timeout_ms" in policy:
            if not isinstance(policy["timeout_ms"], int) or policy["timeout_ms"] <= 0:
                errors.append("timeout_ms must be a positive integer")

        if "retry_count" in policy:
            if not isinstance(policy["retry_count"], int) or policy["retry_count"] < 0:
                errors.append("retry_count must be a non-negative integer")

        if "coverage_method" in policy:
            valid_methods = ["required_set", "best_effort", "majority_vote"]
            if policy["coverage_method"] not in valid_methods:
                errors.append(f"coverage_method must be one of: {valid_methods}")

        return errors

    def list_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored policies.

        Returns:
            Dictionary mapping policy keys to their configurations
        """
        return self.policies.copy()

    def delete_policy(self, tenant_id: str, bundle: Optional[str] = None) -> bool:
        """Delete a policy from storage.

        Args:
            tenant_id: Tenant identifier
            bundle: Optional policy bundle identifier

        Returns:
            True if policy was deleted successfully
        """
        try:
            policy_key = f"{tenant_id}:{bundle}" if bundle else tenant_id
            
            if policy_key in self.policies:
                del self.policies[policy_key]
                logger.info(
                    "Policy deleted for tenant %s",
                    tenant_id,
                    extra={"tenant_id": tenant_id, "bundle": bundle},
                )
                return True
            else:
                logger.warning(
                    "Policy not found for deletion: tenant %s, bundle %s",
                    tenant_id,
                    bundle,
                )
                return False

        except (TypeError, ValueError, KeyError) as e:
            logger.error(
                "Failed to delete policy for tenant %s: %s",
                tenant_id,
                str(e),
                extra={"tenant_id": tenant_id, "error": str(e)},
            )
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics.

        Returns:
            Dictionary containing repository statistics
        """
        total_policies = len(self.policies)
        tenant_policies = 0
        bundle_policies = 0

        for policy_key in self.policies.keys():
            if ":" in policy_key:
                bundle_policies += 1
            else:
                tenant_policies += 1

        return {
            "total_policies": total_policies,
            "tenant_policies": tenant_policies,
            "bundle_policies": bundle_policies,
            "policy_keys": list(self.policies.keys()),
        }

    def clear_all_policies(self) -> None:
        """Clear all policies from storage.
        
        Warning: This removes all cached policies. Use with caution.
        """
        self.policies.clear()
        logger.warning("All policies cleared from repository")

    # --- Tenant policy data (OPA data) persistence helpers ---

    async def load_all_tenant_policies(self) -> Optional[Dict[str, Any]]:
        """Load tenant_policies JSON blob from the database.

        Returns a nested mapping of tenant_id -> bundle -> policy data, or None
        if no record is found.
        """

        query = "SELECT data FROM tenant_policies LIMIT 1"
        try:
            row = await self._db.fetchrow(query)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load tenant_policies from DB: %s", exc)
            return None

        if not row:
            return None

        data = row["data"]
        if isinstance(data, dict):
            return data
        return None

    async def save_all_tenant_policies(self, data: Dict[str, Any]) -> bool:
        """Persist tenant_policies JSON blob into the database.

        Uses a single-row table with an upsert pattern.
        """

        query = (
            """
            INSERT INTO tenant_policies (id, data)
            VALUES ('default', $1)
            ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()
            """
        )

        try:
            await self._db.execute(query, data)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to save tenant_policies to DB: %s", exc)
            return False
