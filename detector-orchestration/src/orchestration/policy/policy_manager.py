"""Policy management and OPA integration following SRP.

This module provides ONLY policy management - enforcing policies and OPA integration.
Single Responsibility: Manage and enforce policies for detector orchestration.
"""

import asyncio
import logging
import time
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import (
    ValidationError,
    ServiceUnavailableError,
)
from shared.clients.opa_client import (
    create_opa_client_with_config,
    OPAError,
    OPAPolicyError,
)
from shared.interfaces.common import Severity
from shared.interfaces.orchestration import DetectorResult, PolicyViolation

if TYPE_CHECKING:
    from orchestration.core.aggregator import AggregatedOutput

from .policy_loader import PolicyLoader, PolicyLoadError

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """Policy decision result with comprehensive decision data."""

    selected_detectors: List[str]
    coverage_method: str
    coverage_requirements: Dict[str, Any]
    routing_reason: str
    policy_violations: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize after dataclass creation."""
        if self.policy_violations is None:
            self.policy_violations = []

    def has_violations(self) -> bool:
        """Check if there are any policy violations."""
        return bool(self.policy_violations)

    def add_violation(self, violation: str) -> None:
        """Add a policy violation to the decision."""
        if self.policy_violations is None:
            self.policy_violations = []
        self.policy_violations.append(violation)

    def get_violation_count(self) -> int:
        """Get the total number of policy violations."""
        return len(self.policy_violations) if self.policy_violations else 0


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
        self.policies = {}
        self.policy_loader = PolicyLoader(policies_directory)
        self.default_policy = {
            "max_detectors": 5,
            "min_confidence": 0.7,
            "timeout_ms": 5000,
            "retry_count": 3,
            "coverage_method": "required_set",
        }

        # Initialize OPA client using global shared pattern
        self.opa_client = create_opa_client_with_config(
            base_url=self.opa_endpoint,
            timeout=30.0,
            max_retries=3,
        )

        # Load policy templates and tenant data on initialization
        self._load_policy_templates()
        asyncio.create_task(self._load_tenant_policies_to_opa())

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

            # Evaluate policy decision using OPA with real policy evaluation
            decision = await self._evaluate_policy_decision(
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

    async def _evaluate_policy_decision(
        self,
        tenant_id: str,
        bundle: Optional[str],
        content_type: str,
        candidate_detectors: List[str],
        policy: Dict[str, Any],
        correlation_id: str,
    ) -> PolicyDecision:
        """Evaluate policy decision using OPA with comprehensive policy evaluation.

        Single Responsibility: Execute policy evaluation logic using OPA for detector selection.
        """
        try:
            # Prepare input data for OPA policy evaluation
            input_data = {
                "tenant_id": tenant_id,
                "bundle": bundle,
                "content_type": content_type,
                "candidate_detectors": candidate_detectors,
                "policy_config": policy,
                "correlation_id": correlation_id,
                "timestamp": correlation_id,
            }

            # Evaluate detector selection policy
            detector_result = await self.opa_client.evaluate_policy(
                policy_path="detector_orchestration/detector_selection",
                input_data=input_data,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            # Evaluate PII detection policy if applicable
            pii_result = await self.opa_client.evaluate_policy(
                policy_path="detector_orchestration/pii",
                input_data=input_data,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            # Evaluate compliance policy if applicable
            compliance_result = await self.opa_client.evaluate_policy(
                policy_path="detector_orchestration/compliance",
                input_data=input_data,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            # Evaluate security policy if applicable
            security_result = await self.opa_client.evaluate_policy(
                policy_path="detector_orchestration/security",
                input_data=input_data,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            # Process OPA results and create decision
            decision = self._process_opa_evaluation_results(
                detector_result=detector_result,
                pii_result=pii_result,
                compliance_result=compliance_result,
                security_result=security_result,
                candidate_detectors=candidate_detectors,
                policy=policy,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            return decision

        except (OPAError, ValidationError, ServiceUnavailableError) as e:
            logger.warning(
                "OPA policy evaluation failed, using fallback logic",
                extra={
                    "tenant_id": tenant_id,
                    "bundle": bundle,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

            # Fallback to local policy evaluation
            return self._evaluate_policy_fallback(
                policy, candidate_detectors, tenant_id, correlation_id
            )

    def _process_opa_evaluation_results(
        self,
        detector_result: Dict[str, Any],
        pii_result: Dict[str, Any],
        compliance_result: Dict[str, Any],
        security_result: Dict[str, Any],
        candidate_detectors: List[str],
        policy: Dict[str, Any],
        tenant_id: str,
        correlation_id: str,
    ) -> PolicyDecision:
        """Process OPA evaluation results into a policy decision.

        Single Responsibility: Transform OPA results into PolicyDecision object.
        """
        violations = []

        # Extract detector selection from OPA results
        selected_detectors = []
        coverage_method = "required_set"
        routing_reason = f"opa_policy_evaluation_{tenant_id}"

        # Process detector selection results
        if "result" in detector_result and detector_result["result"]:
            det_result = detector_result["result"]
            if "selected_detectors" in det_result:
                selected_detectors = det_result["selected_detectors"]
            if "coverage_method" in det_result:
                coverage_method = det_result["coverage_method"]

        # Process PII detection results
        if "result" in pii_result and pii_result["result"]:
            pii_data = pii_result["result"]
            if not pii_data.get("allow", False):
                violations.append("PII detection policy violation")
            if pii_data.get("pii_detected", False):
                routing_reason += "_pii_detected"

        # Process compliance results
        if "result" in compliance_result and compliance_result["result"]:
            comp_data = compliance_result["result"]
            if not comp_data.get("allow", False):
                violations.append("Compliance policy violation")
            if not comp_data.get("compliant", False):
                violations.append("Compliance framework requirements not met")

        # Process security results
        if "result" in security_result and security_result["result"]:
            sec_data = security_result["result"]
            if not sec_data.get("allow", False):
                violations.append("Security policy violation")
            if not sec_data.get("secure", False):
                violations.append("Security validation failed")

        # Fallback to policy-based selection if OPA didn't provide detectors
        if not selected_detectors:
            max_detectors = policy.get("max_detectors", 5)
            selected_detectors = candidate_detectors[:max_detectors]
            routing_reason += "_fallback_selection"

        # Create comprehensive policy decision
        decision = PolicyDecision(
            selected_detectors=selected_detectors,
            coverage_method=coverage_method,
            coverage_requirements={
                "min_success_fraction": policy.get("min_confidence", 0.8),
                "timeout_ms": policy.get("timeout_ms", 5000),
                "retry_count": policy.get("retry_count", 3),
            },
            routing_reason=routing_reason,
            policy_violations=violations,
        )

        logger.info(
            "OPA evaluation results processed",
            extra={
                "tenant_id": tenant_id,
                "correlation_id": correlation_id,
                "selected_count": len(selected_detectors),
                "violations_count": len(violations),
                "coverage_method": coverage_method,
            },
        )

        return decision

    def _evaluate_policy_fallback(
        self,
        policy: Dict[str, Any],
        candidate_detectors: List[str],
        tenant_id: str,
        correlation_id: str,
    ) -> PolicyDecision:
        """Fallback policy evaluation when OPA is unavailable.

        Single Responsibility: Provide reliable fallback when OPA is unavailable.
        """
        max_detectors = policy.get("max_detectors", 5)
        selected = candidate_detectors[:max_detectors]

        decision = PolicyDecision(
            selected_detectors=selected,
            coverage_method="required_set",
            coverage_requirements={
                "min_success_fraction": policy.get("min_confidence", 0.8),
                "timeout_ms": policy.get("timeout_ms", 5000),
                "retry_count": policy.get("retry_count", 3),
            },
            routing_reason=f"fallback_policy_{tenant_id}",
        )

        logger.info(
            "Used fallback policy evaluation",
            extra={
                "tenant_id": tenant_id,
                "correlation_id": correlation_id,
                "selected_count": len(selected),
            },
        )

        return decision

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
        """Evaluate detector and aggregate results against configured policies."""
        policy = await self._get_policy(tenant_id, policy_bundle)
        min_confidence = float(policy.get("min_confidence", 0.7))
        coverage_target = float(
            policy.get("coverage_requirements", {}).get("min_success_fraction", 0.5)
        )
        policy_id_prefix = policy_bundle or "default"
        violations: List[PolicyViolation] = []

        for result in detector_results:
            if result.confidence < min_confidence:
                severity = (
                    Severity.CRITICAL
                    if result.severity in {Severity.CRITICAL, Severity.HIGH}
                    else Severity.MEDIUM
                )
                violations.append(
                    PolicyViolation(
                        policy_id=f"policy:{policy_id_prefix}:confidence",
                        violation_type="low_confidence_detector",
                        message=(
                            f"Detector {result.detector_id} confidence {result.confidence:.2f} "
                            f"below threshold {min_confidence:.2f}"
                        ),
                        severity=severity,
                    )
                )

        if aggregated_output and aggregated_output.confidence_score < min_confidence:
            violations.append(
                PolicyViolation(
                    policy_id=f"policy:{policy_id_prefix}:aggregate",
                    violation_type="low_confidence_aggregate",
                    message=(
                        "Aggregated confidence "
                        f"{aggregated_output.confidence_score:.2f} below threshold "
                        f"{min_confidence:.2f}"
                    ),
                    severity=Severity.HIGH,
                )
            )

        if coverage_target > 0 and coverage < coverage_target:
            severity = (
                Severity.HIGH if coverage < coverage_target / 2 else Severity.MEDIUM
            )
            violations.append(
                PolicyViolation(
                    policy_id=f"policy:{policy_id_prefix}:coverage",
                    violation_type="insufficient_coverage",
                    message=(
                        f"Achieved coverage {coverage:.2f} below required "
                        f"{coverage_target:.2f}"
                    ),
                    severity=severity,
                )
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

        # Try OPA integration if endpoint is configured
        if self.opa_endpoint and bundle:
            try:
                return await self._query_opa(tenant_id, bundle)
            except (asyncio.TimeoutError, ConnectionError, ValueError) as e:
                logger.warning(
                    "OPA query failed, using cached policy: %s",
                    str(e),
                    extra={"tenant_id": tenant_id, "bundle": bundle},
                )

        # Fall back to cached policies
        policy_key = f"{tenant_id}:{bundle}" if bundle else tenant_id
        return self.policies.get(policy_key, self.default_policy)

    async def _query_opa(self, tenant_id: str, bundle: str) -> Dict[str, Any]:
        """Query OPA for policy decision using production client."""
        correlation_id = get_correlation_id()

        try:
            # Query OPA for tenant-specific policy using production client
            policy_path = f"tenant_policies/{tenant_id}/{bundle}"

            input_data = {
                "tenant_id": tenant_id,
                "bundle": bundle,
                "query_type": "policy_retrieval",
                "timestamp": correlation_id,
            }

            result = await self.opa_client.evaluate_policy(
                policy_path=policy_path,
                input_data=input_data,
                tenant_id=tenant_id,
                correlation_id=correlation_id,
            )

            # Extract policy from OPA result
            if "result" in result and result["result"]:
                policy_data = result["result"]
                logger.info(
                    "Retrieved policy from OPA",
                    extra={
                        "tenant_id": tenant_id,
                        "bundle": bundle,
                        "correlation_id": correlation_id,
                        "policy_keys": (
                            list(policy_data.keys())
                            if isinstance(policy_data, dict)
                            else []
                        ),
                    },
                )
                return policy_data

            # If no result, fall back to default policy
            logger.info(
                "No policy found in OPA, using default",
                extra={
                    "tenant_id": tenant_id,
                    "bundle": bundle,
                    "correlation_id": correlation_id,
                },
            )
            return self.default_policy

        except (OPAError, ValidationError, ServiceUnavailableError) as e:
            logger.warning(
                "OPA query failed, falling back to default policy",
                extra={
                    "tenant_id": tenant_id,
                    "bundle": bundle,
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_code": getattr(e, "error_code", "UNKNOWN"),
                },
            )
            return self.default_policy
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error querying OPA",
                extra={
                    "tenant_id": tenant_id,
                    "bundle": bundle,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
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

    def _load_policy_templates(self) -> None:
        """Load policy templates from directory and queue them for OPA loading."""
        try:
            loaded_policies = self.policy_loader.load_all_policies()

            if loaded_policies:
                # Queue policy templates for loading to OPA
                for policy_name, policy_content in loaded_policies.items():
                    self.policies[policy_name] = {
                        "content": policy_content,
                        "status": "template_loaded",
                        "loaded_at": time.time(),
                    }

                logger.info(
                    "Policy templates loaded successfully",
                    extra={
                        "count": len(loaded_policies),
                        "templates": list(loaded_policies.keys()),
                    },
                )

                # Schedule loading templates to OPA in background
                asyncio.create_task(self._load_templates_to_opa(loaded_policies))
            else:
                logger.warning("No policy templates found")

        except PolicyLoadError as e:
            logger.error("Failed to load policy templates", extra={"error": str(e)})

    async def _load_templates_to_opa(self, templates: Dict[str, str]):
        """Load policy templates to OPA in background.

        Single Responsibility: Load templates to OPA without blocking initialization.
        """
        correlation_id = get_correlation_id()

        for policy_name, policy_content in templates.items():
            try:
                success = await self._load_policy_to_opa(policy_content, policy_name)

                if success and policy_name in self.policies:
                    self.policies[policy_name]["status"] = "loaded_to_opa"

                logger.debug(
                    "Template loaded to OPA",
                    extra={
                        "policy_name": policy_name,
                        "success": success,
                        "correlation_id": correlation_id,
                    },
                )

            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Failed to load template to OPA",
                    extra={
                        "policy_name": policy_name,
                        "correlation_id": correlation_id,
                        "error": str(e),
                    },
                )

                if policy_name in self.policies:
                    self.policies[policy_name]["status"] = "load_failed"

    async def _load_tenant_policies_to_opa(self):
        """Load tenant policy data into OPA for policy evaluation.

        Single Responsibility: Initialize OPA with tenant configuration data.
        """
        correlation_id = get_correlation_id()

        try:
            # Load tenant policy data from policy loader
            tenant_data = self.policy_loader.load_tenant_policy_data()

            if tenant_data:
                # Load tenant policies into OPA data store
                success = await self.opa_client.load_data(
                    data_path="tenant_policies",
                    data=tenant_data,
                    correlation_id=correlation_id,
                )

                if success:
                    logger.info(
                        "Tenant policy data loaded to OPA successfully",
                        extra={
                            "correlation_id": correlation_id,
                            "tenant_count": len(tenant_data),
                            "tenants": list(tenant_data.keys()),
                        },
                    )
                else:
                    logger.error(
                        "Failed to load tenant policy data to OPA",
                        extra={"correlation_id": correlation_id},
                    )
            else:
                logger.warning(
                    "No tenant policy data found to load",
                    extra={"correlation_id": correlation_id},
                )

        except (OPAError, PolicyLoadError, ValidationError) as e:
            logger.error(
                "Failed to load tenant policies to OPA",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_code": getattr(e, "error_code", "UNKNOWN"),
                },
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error loading tenant policies to OPA",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

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
            name = policy_name or f"policy_{len(self.policies)}_{int(time.time())}"
            success = await self._load_policy_to_opa(policy_content, name)

            if success:
                # Store policy metadata locally for tracking
                self.policies[name] = {
                    "content": policy_content,
                    "loaded_at": time.time(),
                    "status": "loaded",
                }

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
        """Load policy content to OPA endpoint using production OPA client.

        Args:
            policy_content: Rego policy content
            policy_name: Optional policy name

        Returns:
            True if successfully loaded to OPA
        """
        correlation_id = get_correlation_id()

        if not policy_name:
            policy_name = f"policy_{correlation_id[:8]}"

        try:
            # Use production OPA client for actual HTTP request
            success = await self.opa_client.load_policy(
                policy_content=policy_content,
                policy_name=policy_name,
                correlation_id=correlation_id,
            )

            if success:
                logger.info(
                    "Policy loaded to OPA successfully",
                    extra={
                        "policy_name": policy_name,
                        "correlation_id": correlation_id,
                        "endpoint": self.opa_endpoint,
                    },
                )

            return success

        except (
            OPAError,
            OPAPolicyError,
            ValidationError,
            ServiceUnavailableError,
        ) as e:
            logger.error(
                "OPA policy loading failed",
                extra={
                    "policy_name": policy_name,
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_code": getattr(e, "error_code", "UNKNOWN"),
                },
            )
            return False
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error loading policy to OPA",
                extra={
                    "policy_name": policy_name,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

    async def unload_policy(self, policy_name: str) -> bool:
        """Unload a policy from OPA.

        Args:
            policy_name: Name of policy to unload

        Returns:
            True if policy was unloaded successfully
        """
        try:
            # Remove from local storage
            if policy_name in self.policies:
                del self.policies[policy_name]

            # Remove from OPA if endpoint is configured
            if self.opa_endpoint:
                success = await self._unload_policy_from_opa(policy_name)
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
        """Unload policy from OPA endpoint using production OPA client.

        Args:
            policy_name: Name of policy to unload

        Returns:
            True if successfully unloaded from OPA
        """
        correlation_id = get_correlation_id()

        try:
            # Use production OPA client for actual HTTP DELETE request
            success = await self.opa_client.unload_policy(
                policy_name=policy_name,
                correlation_id=correlation_id,
            )

            if success:
                logger.info(
                    "Policy unloaded from OPA successfully",
                    extra={
                        "policy_name": policy_name,
                        "correlation_id": correlation_id,
                        "endpoint": self.opa_endpoint,
                    },
                )

            return success

        except (
            OPAError,
            OPAPolicyError,
            ValidationError,
            ServiceUnavailableError,
        ) as e:
            logger.error(
                "OPA policy unloading failed",
                extra={
                    "policy_name": policy_name,
                    "correlation_id": correlation_id,
                    "error": str(e),
                    "error_code": getattr(e, "error_code", "UNKNOWN"),
                },
            )
            return False
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error unloading policy from OPA",
                extra={
                    "policy_name": policy_name,
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy management statistics.

        Returns:
            Dictionary containing policy statistics
        """
        return {
            "opa_endpoint": self.opa_endpoint,
            "loaded_policies": len(self.policies),
            "policy_loader_stats": self.policy_loader.get_policy_statistics(),
            "policy_templates": list(self.policy_loader.get_loaded_policies().keys()),
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

            # Check if any policies are loaded
            policies_loaded = len(self.policies) > 0

            overall_health = opa_healthy and loader_healthy

            logger.info(
                "Policy manager health check completed",
                extra={
                    "correlation_id": correlation_id,
                    "opa_healthy": opa_healthy,
                    "loader_healthy": loader_healthy,
                    "policies_loaded": policies_loaded,
                    "overall_healthy": overall_health,
                },
            )

            return overall_health

        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Policy manager health check failed",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

    async def reload_policies(self) -> bool:
        """Reload all policies from templates and tenant data.

        Single Responsibility: Refresh all policy data in OPA.
        """
        correlation_id = get_correlation_id()

        try:
            # Reload policy templates
            self._load_policy_templates()

            # Reload tenant data to OPA
            await self._load_tenant_policies_to_opa()

            logger.info(
                "Policies reloaded successfully",
                extra={
                    "correlation_id": correlation_id,
                    "policy_count": len(self.policies),
                },
            )

            return True

        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Failed to reload policies",
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            return False

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

        except Exception as e:  # pylint: disable=broad-except
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
