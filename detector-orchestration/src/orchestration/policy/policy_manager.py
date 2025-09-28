"""Policy management and OPA integration following SRP.

This module provides ONLY policy management - enforcing policies and OPA integration.
Single Responsibility: Manage and enforce policies for detector orchestration.
"""

import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import ValidationError, ServiceUnavailableError
from shared.clients.opa_client import (
    create_opa_client_with_config,
    OPAError,
)
from shared.interfaces.orchestration import DetectorResult, PolicyViolation

from .models import PolicyDecision
from .policy_loader import PolicyLoader, PolicyLoadError
from .violations import evaluate_policy_violations as _eval_violations
from .opa_adapter import (
    query_policy as _opa_query_policy,
    load_policy as _opa_load_policy,
    unload_policy as _opa_unload_policy,
    load_data as _opa_load_data,
)
from .evaluator import (
    evaluate_policy_decision as _evaluate_policy,
    evaluate_policy_fallback as _evaluate_fallback,
)
from .repository import PolicyRepository
from .initialization import PolicyInitializer

if TYPE_CHECKING:
    from orchestration.core import AggregatedOutput

logger = logging.getLogger(__name__)


class PolicyManager:
    """Manages policy enforcement and OPA integration.

    Single Responsibility: Enforce policies and integrate with OPA.
    Does NOT handle: routing, health monitoring, service discovery.
    """

    def __init__(
        self, opa_endpoint: Optional[str] = None, policies_directory: str = "policies"
    ):
        """Initialize policy manager with production OPA integration.

        Args:
            opa_endpoint: Optional OPA endpoint URL
            policies_directory: Directory containing policy files
        """
        self.opa_endpoint = opa_endpoint or os.getenv(
            "OPA_SERVICE_URL", "http://localhost:8181"
        )
        self.policy_loader = PolicyLoader(policies_directory)
        self.repository = PolicyRepository()

        # Initialize OPA client using global shared pattern
        self.opa_client = create_opa_client_with_config(
            base_url=self.opa_endpoint,
            timeout=30.0,
            max_retries=3,
        )

        # Initialize policy initialization handler
        self.initializer = PolicyInitializer(self.policy_loader, self.opa_client)

        # Load policy templates and tenant data on initialization
        self.initializer.load_policy_templates()
        asyncio.create_task(self.initializer.load_tenant_policies_to_opa())

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

            # Evaluate policy decision using extracted evaluator
            decision = await _evaluate_policy(
                opa_client=self.opa_client,
                tenant_id=tenant_id,
                bundle=bundle,
                content_type=content_type,
                candidate_detectors=candidate_detectors,
                policy=policy,
                correlation_id=correlation_id,
            )

            logger.info(
                "Policy decision completed: selected %d detectors",
                len(decision.selected_detectors),
                extra={
                    "correlation_id": correlation_id,
                    "tenant_id": tenant_id,
                    "selected_detectors": decision.selected_detectors,
                    "routing_reason": decision.routing_reason,
                    "violations": decision.get_violation_count(),
                },
            )

            return decision

        except (ValueError, TypeError, KeyError) as e:
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

    # _evaluate_policy_decision moved to evaluator.py

    # _process_opa_evaluation_results and _evaluate_policy_fallback moved to evaluator.py

    async def evaluate_policy_violations(
        self,
        *,
        tenant_id: str,
        policy_bundle: Optional[str],
        detector_results: List[DetectorResult],
        aggregated_output: Optional["AggregatedOutput"],
        coverage: float,
        correlation_id: str,
    ) -> List[PolicyViolation]:
        """Evaluate results against policy via helper (kept for compatibility)."""

        policy = await self._get_policy(tenant_id, policy_bundle)

        violations = _eval_violations(
            policy=policy,
            _tenant_id=tenant_id,
            policy_bundle=policy_bundle,
            detector_results=detector_results,
            aggregated_output=aggregated_output,
            coverage=coverage,
            _correlation_id=correlation_id,
        )

        logger.debug(
            "Policy evaluation completed",
            extra={
                "tenant_id": tenant_id,
                "correlation_id": correlation_id,
                "policy_bundle": policy_bundle,
                "violations": len(violations),
            },
        )

        return violations

    async def _get_policy(
        self, tenant_id: str, bundle: Optional[str]
    ) -> Dict[str, Any]:
        """Get policy for tenant and bundle."""

        if self.opa_endpoint and bundle:
            try:
                return await _opa_query_policy(
                    client=self.opa_client,
                    tenant_id=tenant_id,
                    bundle=bundle,
                    correlation_id=get_correlation_id(),
                )
            except (
                OPAError,
                ValidationError,
                ServiceUnavailableError,
                asyncio.TimeoutError,
            ):
                logger.warning(
                    "OPA query failed, using cached policy",
                    extra={"tenant_id": tenant_id, "bundle": bundle},
                )

        # Fall back to cached policies
        return self.repository.get_policy_or_default(tenant_id, bundle)

    # _query_opa deprecated – replaced by opa_adapter

    def set_policy(
        self, tenant_id: str, policy: Dict[str, Any], bundle: Optional[str] = None
    ) -> bool:
        """Set policy for a tenant."""
        return self.repository.set_policy(tenant_id, policy, bundle)

    def get_policy(
        self, tenant_id: str, bundle: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get policy for a tenant."""
        return self.repository.get_policy(tenant_id, bundle)

    def validate_policy(self, policy: Dict[str, Any]) -> List[str]:
        """Validate policy configuration."""
        return self.repository.validate_policy(policy)

    # Template and tenant data loading moved to PolicyInitializer

    async def load_policy(
        self, policy_content: str, policy_name: Optional[str] = None
    ) -> bool:
        """Load a policy into OPA.

        Args:
            policy_content: The Rego policy content
            policy_name: Optional name for the policy

        Returns:
            True if policy was loaded successfully
        """
        try:
            # Validate policy syntax using OPA client for production validation
            validation_errors = await self.opa_client.validate_policy_syntax(
                policy_content
            )

            if validation_errors:
                logger.error(
                    "Policy validation failed",
                    extra={
                        "policy_name": policy_name,
                        "errors": validation_errors,
                        "error_count": len(validation_errors),
                    },
                )
                return False

            # Load to OPA using production client
            name = policy_name or f"policy_{self.initializer.get_template_count()}_{int(time.time())}"
            success = await _opa_load_policy(
                client=self.opa_client,
                policy_content=policy_content,
                policy_name=name,
                correlation_id=get_correlation_id(),
            )

            if success:
                # Policy loaded successfully - initializer tracks template status

                logger.info(
                    "Policy loaded successfully",
                    extra={
                        "policy_name": name,
                        "opa_endpoint": self.opa_endpoint,
                    },
                )
                return True

            logger.error(
                "Failed to load policy to OPA",
                extra={
                    "policy_name": name,
                    "opa_endpoint": self.opa_endpoint,
                },
            )
            return False

        except (ValueError, TypeError) as e:
            logger.error(
                "Failed to load policy",
                extra={"policy_name": policy_name, "error": str(e)},
            )
            return False

    async def _load_policy_to_opa(
        self, policy_content: str, policy_name: Optional[str]
    ) -> bool:
        """Thin wrapper delegating to opa_adapter.load_policy."""

        if not policy_name:
            policy_name = f"policy_{get_correlation_id()[:8]}"

        return await _opa_load_policy(
            client=self.opa_client,
            policy_content=policy_content,
            policy_name=policy_name,
            correlation_id=get_correlation_id(),
        )

    async def unload_policy(self, policy_name: str) -> bool:
        """Unload a policy from OPA.

        Args:
            policy_name: Name of policy to unload

        Returns:
            True if policy was unloaded successfully
        """
        try:
            # Template status managed by initializer

            # Remove from OPA if endpoint is configured
            if self.opa_endpoint:
                success = await _opa_unload_policy(
                    client=self.opa_client,
                    policy_name=policy_name,
                    correlation_id=get_correlation_id(),
                )
                if not success:
                    logger.warning(
                        "Failed to unload policy from OPA",
                        extra={"policy_name": policy_name},
                    )

            logger.info("Policy unloaded", extra={"policy_name": policy_name})
            return True

        except (KeyError, ValueError) as e:
            logger.error(
                "Failed to unload policy",
                extra={"policy_name": policy_name, "error": str(e)},
            )
            return False

    async def _unload_policy_from_opa(self, policy_name: str) -> bool:
        """Thin wrapper delegating to opa_adapter.unload_policy."""

        return await _opa_unload_policy(
            client=self.opa_client,
            policy_name=policy_name,
            correlation_id=get_correlation_id(),
        )

    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy management statistics.

        Returns:
            Dictionary containing policy statistics
        """
        return {
            "opa_endpoint": self.opa_endpoint,
            "loaded_templates": self.initializer.get_template_count(),
            "policy_loader_stats": self.policy_loader.get_policy_statistics(),
            "policy_templates": list(self.policy_loader.get_loaded_policies().keys()),
            "repository_stats": self.repository.get_statistics(),
            "template_status": self.initializer.get_template_status(),
        }

    async def health_check(self) -> bool:
        """Check health of policy manager and OPA integration.

        Single Responsibility: Verify all policy management components are healthy.
        """
        correlation_id = get_correlation_id()

        try:
            # Check OPA client health
            opa_healthy = await self.opa_client.health_check()

            # Check policy loader health
            loader_healthy = bool(
                self.policy_loader and self.policy_loader.policies_directory.exists()
            )

            # Check if any templates are loaded
            templates_loaded = self.initializer.get_template_count() > 0

            overall_health = opa_healthy and loader_healthy

            logger.info(
                "Policy manager health check completed",
                extra={
                    "correlation_id": correlation_id,
                    "opa_healthy": opa_healthy,
                    "loader_healthy": loader_healthy,
                    "templates_loaded": templates_loaded,
                    "overall_healthy": overall_health,
                },
            )

            return overall_health

        except (OPAError, ValidationError, ServiceUnavailableError) as e:
            logger.error(
                "Policy manager health check failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

    async def reload_policies(self) -> bool:
        """Reload all policies from templates and tenant data."""
        return await self.initializer.reload_all_policies()

    async def close(self):
        """Close the policy manager and clean up resources.

        Single Responsibility: Clean shutdown of all policy manager resources.
        """
        try:
            if hasattr(self, "opa_client") and self.opa_client:
                await self.opa_client.close()

            logger.info(
                "Policy manager closed successfully",
                extra={
                    "opa_endpoint": self.opa_endpoint,
                },
            )

        except (OPAError, ValidationError, ServiceUnavailableError) as e:
            logger.error(
                "Error closing policy manager",
                extra={"error": str(e)},
            )


# Export only the policy management functionality
__all__ = [
    "PolicyManager",
    "PolicyDecision",
    "PolicyLoader",
    "PolicyLoadError",
]
