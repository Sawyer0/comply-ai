"""Policy management and OPA integration following SRP.

This module provides ONLY policy management - enforcing policies and OPA integration.
Single Responsibility: Manage and enforce policies for detector orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

from shared.utils.correlation import get_correlation_id

logger = logging.getLogger(__name__)


class PolicyDecision:
    """Policy decision result - data structure only."""

    def __init__(
        self,
        selected_detectors: List[str],
        coverage_method: str,
        coverage_requirements: Dict[str, Any],
        routing_reason: str,
        policy_violations: Optional[List[str]] = None,
    ):
        self.selected_detectors = selected_detectors
        self.coverage_method = coverage_method
        self.coverage_requirements = coverage_requirements
        self.routing_reason = routing_reason
        self.policy_violations = policy_violations or []


class PolicyManager:
    """Manages policy enforcement and OPA integration.

    Single Responsibility: Enforce policies and integrate with OPA.
    Does NOT handle: routing, health monitoring, service discovery.
    """

    def __init__(self, opa_endpoint: Optional[str] = None):
        """Initialize policy manager.

        Args:
            opa_endpoint: Optional OPA endpoint URL
        """
        self.opa_endpoint = opa_endpoint
        self.policies = {}
        self.default_policy = {
            "max_detectors": 5,
            "min_confidence": 0.7,
            "timeout_ms": 5000,
            "retry_count": 3,
            "coverage_method": "required_set",
        }

    async def decide(
        self,
        tenant_id: str,
        bundle: Optional[str],
        content_type: str,
        candidate_detectors: List[str],
    ) -> PolicyDecision:
        """Make routing decision based on policy.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle identifier
            content_type: Type of content being analyzed
            candidate_detectors: List of candidate detectors

        Returns:
            Policy decision with selected detectors and requirements
        """
        correlation_id = get_correlation_id()

        logger.info(
            "Making policy decision for tenant %s with bundle %s",
            tenant_id,
            bundle,
            extra={
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "bundle": bundle,
                "content_type": content_type,
                "candidate_count": len(candidate_detectors),
            },
        )

        try:
            # Get tenant-specific policy or use default
            policy = await self._get_policy(tenant_id, bundle)

            # Apply policy rules
            max_detectors = policy.get("max_detectors", 5)
            selected = candidate_detectors[:max_detectors]

            # Create policy decision
            decision = PolicyDecision(
                selected_detectors=selected,
                coverage_method="required_set",
                coverage_requirements={"min_success_fraction": 0.8},
                routing_reason=f"policy_applied_for_{tenant_id}",
            )

            logger.info(
                "Policy decision completed: selected %d detectors",
                len(selected),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "selected_detectors": selected,
                    "routing_reason": decision.routing_reason,
                },
            )

            return decision

        except Exception as e:
            logger.error(
                "Policy decision failed: %s",
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "error": str(e),
                },
            )

            # Return default decision on error
            return PolicyDecision(
                selected_detectors=candidate_detectors[:3],  # Default limit
                coverage_method="required_set",
                coverage_requirements={"min_success_fraction": 0.8},
                routing_reason="default_policy_on_error",
            )

    async def _get_policy(
        self, tenant_id: str, bundle: Optional[str]
    ) -> Dict[str, Any]:
        """Get policy for tenant and bundle."""

        # Try OPA integration if endpoint is configured
        if self.opa_endpoint and bundle:
            try:
                return await self._query_opa(tenant_id, bundle)
            except Exception as e:
                logger.warning(
                    "OPA query failed, using cached policy: %s",
                    str(e),
                    extra={"tenant_id": tenant_id, "bundle": bundle},
                )

        # Fall back to cached policies
        policy_key = f"{tenant_id}:{bundle}" if bundle else tenant_id
        return self.policies.get(policy_key, self.default_policy)

    async def _query_opa(self, tenant_id: str, bundle: str) -> Dict[str, Any]:
        """Query OPA for policy decision."""
        # This would integrate with actual OPA endpoint
        # For now, return default policy
        await asyncio.sleep(0.01)  # Simulate network call
        return self.default_policy

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

        except Exception as e:
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

        return errors


# Export only the policy management functionality
__all__ = [
    "PolicyManager",
    "PolicyDecision",
]
