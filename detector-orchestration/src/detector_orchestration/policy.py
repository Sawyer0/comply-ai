"""Policy management and decision engine for detector orchestration.

This module provides policy-based detector selection and management including:
- Policy storage and versioning with approval workflows
- OPA (Open Policy Agent) integration for dynamic policy evaluation
- Coverage-based detector selection algorithms
- Policy validation and conflict resolution

Key components:
- PolicyStore: Persistent storage for tenant policies with versioning
- OPAPolicyEngine: Integration with OPA for policy evaluation
- PolicyManager: Main decision engine for detector selection
- PolicyValidationCLI: Command-line validation utilities
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from .config import Settings
from .models import ContentType


@dataclass
class ReviewRecord:
    """Data structure for policy review submissions."""

    reviewer: str
    decision: ApprovalDecision
    note: Optional[str] = None

    def is_approved(self) -> bool:
        """Check if the review decision is approved."""
        return self.decision == ApprovalDecision.APPROVED

    def is_rejected(self) -> bool:
        """Check if the review decision is rejected."""
        return self.decision == ApprovalDecision.REJECTED

    def requires_changes(self) -> bool:
        """Check if the review requests changes."""
        return self.decision == ApprovalDecision.CHANGES_REQUESTED


@dataclass
class RollbackRequest:
    """Data structure for policy rollback requests."""

    actor: str
    reason: Optional[str] = None

    def has_reason(self) -> bool:
        """Check if a reason is provided for the rollback."""
        return self.reason is not None

    def get_description(self, version_id: str) -> str:
        """Get a formatted description for the rollback."""
        if self.reason:
            return self.reason
        return f"Rollback to {version_id}"


class CoverageMethod(str, Enum):
    """Enumeration of coverage calculation methods for detector selection."""

    REQUIRED_SET = "required_set"
    WEIGHTED = "weighted"
    TAXONOMY = "taxonomy"


class TenantPolicy(BaseModel):
    """Configuration for detector selection policies per tenant and bundle.

    Defines which detectors should be run, how coverage is calculated,
    and what content types are allowed for processing.
    """

    tenant_id: str
    bundle: str
    version: str = "v1"
    required_detectors: List[str] = Field(default_factory=list)
    optional_detectors: List[str] = Field(default_factory=list)
    coverage_method: CoverageMethod = CoverageMethod.REQUIRED_SET
    required_coverage: float = 1.0
    detector_weights: Dict[str, float] = Field(default_factory=dict)
    required_taxonomy_categories: List[str] = Field(default_factory=list)
    allowed_content_types: List[ContentType] = Field(
        default_factory=lambda: [ContentType.TEXT]
    )


class PolicyDecision(BaseModel):
    """Result of policy evaluation containing detector selection and coverage rules."""

    selected_detectors: List[str]
    coverage_method: CoverageMethod
    coverage_requirements: Dict[str, Any]
    routing_reason: str


class ApprovalDecision(str, Enum):
    """Enumeration of possible decisions for policy approval workflow."""

    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class PolicyVersionStatus(str, Enum):
    """Enumeration of possible states for policy versions in approval workflow."""

    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    ROLLED_BACK = "rolled_back"


class ApprovalRecord(BaseModel):
    """Record of an individual approval decision in the policy approval workflow."""

    reviewer: str
    decision: ApprovalDecision
    note: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PolicyVersion(BaseModel):
    """Versioned snapshot of a policy with approval workflow tracking."""

    version_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str
    description: Optional[str] = None
    status: PolicyVersionStatus = PolicyVersionStatus.PENDING_APPROVAL
    policy: TenantPolicy
    approvals: List[ApprovalRecord] = Field(default_factory=list)
    approved_at: Optional[datetime] = None
    rolled_back_from: Optional[str] = None
    conflict_warnings: List[str] = Field(default_factory=list)


class PolicySubmission(BaseModel):
    """Request to submit a new policy for approval and deployment."""

    policy: TenantPolicy
    submitted_by: str
    description: Optional[str] = None
    requires_approval: bool = True


class PolicyStore:
    """Persistent storage for policy management with versioning and approval workflows.

    Handles storage of active policies and version history in JSON format.
    Supports tenant isolation and bundle-based organization.
    """

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def _path(self, tenant_id: str, bundle: str) -> Path:
        """Get the file path for active policy storage.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name

        Returns:
            Path to the active policy file
        """
        return self.base_dir / tenant_id / f"{bundle}.json"

    def _history_path(self, tenant_id: str, bundle: str) -> Path:
        """Get the file path for policy version history storage.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name

        Returns:
            Path to the policy version history file
        """
        return self.base_dir / tenant_id / f"{bundle}.history.json"

    # ------------------------------------------------------------------
    # Active policy helpers
    # ------------------------------------------------------------------

    def get_policy(self, tenant_id: str, bundle: str) -> Optional[TenantPolicy]:
        """Retrieve the active policy for a tenant and bundle."""
        p = self._path(tenant_id, bundle)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return TenantPolicy(**data)

    def save_policy(self, policy: TenantPolicy) -> None:
        """Save a policy as the active policy for a tenant and bundle."""
        p = self._path(policy.tenant_id, policy.bundle)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(policy.model_dump_json(indent=2), encoding="utf-8")

    def list_policies(self, tenant_id: str) -> List[str]:
        """List all policy bundles for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of policy bundle names
        """
        d = self.base_dir / tenant_id
        if not d.exists():
            return []
        return [fp.stem for fp in d.glob("*.json")]

    def delete_policy(self, tenant_id: str, bundle: str) -> bool:
        """Delete a policy and its version history.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name

        Returns:
            True if policy was deleted, False if it didn't exist
        """
        p = self._path(tenant_id, bundle)
        if p.exists():
            p.unlink()
            history = self._history_path(tenant_id, bundle)
            if history.exists():
                history.unlink()
            return True
        return False

    # ------------------------------------------------------------------
    # Version management helpers
    # ------------------------------------------------------------------

    def _load_versions(self, tenant_id: str, bundle: str) -> List[PolicyVersion]:
        """Load policy version history from disk."""
        hp = self._history_path(tenant_id, bundle)
        if not hp.exists():
            return []
        raw = json.loads(hp.read_text(encoding="utf-8"))
        versions: List[PolicyVersion] = []
        if isinstance(raw, list):
            for item in raw:
                try:
                    versions.append(PolicyVersion.model_validate(item))
                except (ValueError, TypeError, KeyError):
                    # Skip invalid version data
                    continue
        return versions

    def _save_versions(
        self, tenant_id: str, bundle: str, versions: List[PolicyVersion]
    ) -> None:
        """Save policy version history to disk."""
        hp = self._history_path(tenant_id, bundle)
        hp.parent.mkdir(parents=True, exist_ok=True)
        payload = [v.model_dump(mode="json") for v in versions]
        hp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def list_versions(self, tenant_id: str, bundle: str) -> List[PolicyVersion]:
        """List all versions for a policy bundle, sorted by creation time."""
        versions = self._load_versions(tenant_id, bundle)
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_version(
        self, tenant_id: str, bundle: str, version_id: str
    ) -> Optional[PolicyVersion]:
        """Retrieve a specific version of a policy."""
        for v in self._load_versions(tenant_id, bundle):
            if v.version_id == version_id:
                return v
        return None

    def _generate_version_id(self) -> str:
        """Generate a unique timestamp-based version identifier."""
        return datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S%f")

    # ------------------------------------------------------------------
    # Submission / approval lifecycle
    # ------------------------------------------------------------------

    def submit_policy(
        self,
        submission: PolicySubmission,
        *,
        detector_catalog: Optional[Dict[str, Any]] = None,
    ) -> PolicyVersion:
        """Submit a new policy for approval and potential deployment.

        Args:
            submission: Policy submission request with policy and metadata
            detector_catalog: Optional detector catalog for validation

        Returns:
            Created policy version

        Raises:
            ValueError: If policy validation fails
        """
        policy = submission.policy
        policy.tenant_id = submission.policy.tenant_id
        policy.bundle = submission.policy.bundle

        warnings, errors = self._validate_policy(policy, detector_catalog)
        if errors:
            raise ValueError("; ".join(errors))

        versions = self._load_versions(policy.tenant_id, policy.bundle)
        version_id = self._generate_version_id()
        policy.version = version_id

        status = (
            PolicyVersionStatus.PENDING_APPROVAL
            if submission.requires_approval
            else PolicyVersionStatus.APPROVED
        )
        approved_at: Optional[datetime] = None
        approvals: List[ApprovalRecord] = []
        if not submission.requires_approval:
            approved_at = datetime.now(timezone.utc)
            approvals.append(
                ApprovalRecord(
                    reviewer=submission.submitted_by,
                    decision=ApprovalDecision.APPROVED,
                    note="auto-approved",
                )
            )

        version = PolicyVersion(
            version_id=version_id,
            created_by=submission.submitted_by,
            description=submission.description,
            status=status,
            policy=policy,
            approvals=approvals,
            approved_at=approved_at,
            conflict_warnings=warnings,
        )

        versions.append(version)
        self._save_versions(policy.tenant_id, policy.bundle, versions)

        if status == PolicyVersionStatus.APPROVED:
            self.save_policy(policy)

        return version

    def record_review(
        self,
        tenant_id: str,
        bundle: str,
        version_id: str,
        review_record: ReviewRecord,
    ) -> PolicyVersion:
        """Record a review decision for a policy version.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name
            version_id: Version identifier to review
            review_record: Review record containing reviewer, decision, and note

        Returns:
            Updated PolicyVersion with recorded review

        Raises:
            ValueError: If version not found or not reviewable
        """
        version = self.get_version(tenant_id, bundle, version_id)
        if version is None:
            raise ValueError("version_not_found")
        if version.status not in {
            PolicyVersionStatus.PENDING_APPROVAL,
            PolicyVersionStatus.CHANGES_REQUESTED,
        }:
            raise ValueError("version_not_reviewable")

        version.approvals.append(
            ApprovalRecord(
                reviewer=review_record.reviewer,
                decision=review_record.decision,
                note=review_record.note,
            )
        )

        if review_record.decision == ApprovalDecision.APPROVED:
            version.status = PolicyVersionStatus.APPROVED
            version.approved_at = datetime.now(timezone.utc)
            self.save_policy(version.policy)
        elif review_record.decision == ApprovalDecision.REJECTED:
            version.status = PolicyVersionStatus.REJECTED
        else:
            version.status = PolicyVersionStatus.CHANGES_REQUESTED

        versions = self._load_versions(tenant_id, bundle)
        updated = [v if v.version_id != version.version_id else version for v in versions]
        self._save_versions(tenant_id, bundle, updated)
        return version

    def rollback_policy(
        self,
        tenant_id: str,
        bundle: str,
        version_id: str,
        rollback_request: RollbackRequest,
    ) -> PolicyVersion:
        """Rollback to a previous approved policy version.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name
            version_id: Version to rollback to
            rollback_request: Rollback request containing actor and reason

        Returns:
            New PolicyVersion created for the rollback

        Raises:
            ValueError: If version not found or not rollbackable
        """
        target = self.get_version(tenant_id, bundle, version_id)
        if target is None or target.status != PolicyVersionStatus.APPROVED:
            raise ValueError("version_not_rollbackable")

        rollback_policy = target.policy.model_copy(deep=True)
        rollback_policy.version = self._generate_version_id()

        new_version = PolicyVersion(
            version_id=rollback_policy.version,
            created_by=rollback_request.actor,
            description=rollback_request.get_description(version_id),
            status=PolicyVersionStatus.APPROVED,
            policy=rollback_policy,
            approvals=[
                ApprovalRecord(
                    reviewer=rollback_request.actor,
                    decision=ApprovalDecision.APPROVED,
                    note="rollback",
                )
            ],
            approved_at=datetime.now(timezone.utc),
            rolled_back_from=version_id,
        )

        versions = self._load_versions(tenant_id, bundle)
        versions.append(new_version)
        self._save_versions(tenant_id, bundle, versions)
        self.save_policy(rollback_policy)
        return new_version

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_policy(
        self,
        policy: TenantPolicy,
        detector_catalog: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Validate a policy configuration and return warnings and errors.

        Args:
            policy: Policy to validate
            detector_catalog: Optional detector catalog for cross-reference validation

        Returns:
            Tuple of (warnings: List[str], errors: List[str])
        """
        warnings: List[str] = []
        errors: List[str] = []

        if not policy.required_detectors:
            errors.append("policy must include at least one required_detector")

        if not 0.0 < policy.required_coverage <= 1.0:
            errors.append("required_coverage must be within (0.0, 1.0]")

        catalog = detector_catalog or {}
        for detector in policy.required_detectors + policy.optional_detectors:
            if catalog and detector not in catalog:
                warnings.append(f"detector '{detector}' is not registered in catalogue")

        if policy.detector_weights:
            unknown_weights = [
                d for d in policy.detector_weights.keys()
                if d not in policy.required_detectors + policy.optional_detectors
            ]
            if unknown_weights:
                warnings.append(
                    "weights specified for unknown detectors: "
                    + ", ".join(sorted(unknown_weights))
                )
        elif policy.coverage_method == CoverageMethod.WEIGHTED:
            warnings.append("weighted coverage specified without detector_weights")

        if not policy.allowed_content_types:
            warnings.append("allowed_content_types empty; defaulting to TEXT")
            policy.allowed_content_types = [ContentType.TEXT]

        return warnings, errors


class OPAPolicyEngine:
    """Open Policy Agent integration for dynamic policy evaluation.

    Provides integration with OPA server for policy-based detector selection
    and conflict resolution decisions.
    """

    def __init__(self, settings: Settings):
        """Initialize OPA policy engine.

        Args:
            settings: Configuration settings containing OPA server details
        """
        self.settings = settings

    async def evaluate(
        self, tenant_id: str, bundle: str, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate detector selection policy using OPA.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name
            input_data: Input data for policy evaluation

        Returns:
            OPA evaluation result or None if OPA unavailable
        """
        if not self.settings.config.opa_enabled or not self.settings.config.opa_url:
            return None
        url = f"{self.settings.config.opa_url.rstrip('/')}/v1/data/{tenant_id}/{bundle}/select"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(url, json={"input": input_data})
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        result = payload.get("result")
                        if isinstance(result, dict):
                            return result
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError):
            # Network errors, HTTP errors, or JSON parsing errors
            return None
        return None

    async def evaluate_conflict(
        self, tenant_id: str, bundle: str, input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate conflict resolution strategy and decision via OPA.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name
            input_data: Input data for conflict resolution evaluation

        Returns:
            OPA conflict resolution result or None if OPA unavailable

        Note:
            Expects a Rego package with a 'conflict' decision that returns a result like:
            {
              "strategy": (
                  "highest_confidence|weighted_average|majority_vote|"
                  "most_restrictive|tenant_preference"
              ),
              "preferred_detector": "optional-detector-name",
              "tie_breaker": "optional-hint"
            }
        """
        if not self.settings.config.opa_enabled or not self.settings.config.opa_url:
            return None
        url = f"{self.settings.config.opa_url.rstrip('/')}/v1/data/{tenant_id}/{bundle}/conflict"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(url, json={"input": input_data})
                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, dict):
                        result = payload.get("result")
                        if isinstance(result, dict):
                            return result
        except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError):
            # Network errors, HTTP errors, or JSON parsing errors
            return None
        return None


class PolicyManager:
    """Main policy decision engine for detector selection and evaluation.

    Coordinates policy evaluation using OPA integration and tenant policies
    to determine which detectors should be executed for given content.
    """

    def __init__(self, store: PolicyStore, engine: OPAPolicyEngine):
        """Initialize policy manager.

        Args:
            store: Policy store for tenant policies
            engine: OPA engine for dynamic policy evaluation
        """
        self.store = store
        self.engine = engine

    async def decide(
        self,
        tenant_id: str,
        bundle: str,
        content_type: ContentType,
        candidate_detectors: List[str],
    ) -> PolicyDecision:
        """Make detector selection decision based on policies and OPA evaluation.

        Args:
            tenant_id: Tenant identifier
            bundle: Policy bundle name
            content_type: Content type being processed
            candidate_detectors: List of available detectors

        Returns:
            Policy decision with selected detectors and coverage requirements
        """
        # Query OPA first if configured
        opa_result = await self.engine.evaluate(
            tenant_id,
            bundle,
            {
                "content_type": content_type.value,
                "candidates": candidate_detectors,
            },
        )
        if opa_result and isinstance(opa_result, dict) and opa_result.get("selected"):
            selected = [
                d for d in opa_result.get("selected", []) if d in candidate_detectors
            ]
            cov_method = CoverageMethod(
                opa_result.get("coverage_method", CoverageMethod.REQUIRED_SET)
            )
            cov_req = opa_result.get(
                "coverage_requirements", {"min_success_fraction": 1.0}
            )
            return PolicyDecision(
                selected_detectors=selected,
                coverage_method=cov_method,
                coverage_requirements=cov_req,
                routing_reason="opa",
            )

        # Fallback: Tenant policy
        pol = self.store.get_policy(tenant_id, bundle)
        if pol:
            selected = [d for d in pol.required_detectors if d in candidate_detectors]
            return PolicyDecision(
                selected_detectors=selected,
                coverage_method=pol.coverage_method,
                coverage_requirements={
                    "min_success_fraction": pol.required_coverage,
                    "weights": pol.detector_weights,
                    "required_taxonomy_categories": pol.required_taxonomy_categories,
                },
                routing_reason="tenant_policy",
            )

        # Default: pass through all candidates, required set
        return PolicyDecision(
            selected_detectors=candidate_detectors,
            coverage_method=CoverageMethod.REQUIRED_SET,
            coverage_requirements={"min_success_fraction": 1.0},
            routing_reason="default",
        )

    def get_policy_store(self) -> PolicyStore:
        """Get the policy store instance."""
        return self.store

    def get_opa_engine(self) -> OPAPolicyEngine:
        """Get the OPA policy engine instance."""
        return self.engine

    def is_opa_enabled(self) -> bool:
        """Check if OPA integration is enabled and configured."""
        return (
            self.engine.settings.config.opa_enabled and
            bool(self.engine.settings.config.opa_url)
        )


class PolicyValidationCLI:
    """Command-line utilities for policy validation and testing."""

    @staticmethod
    def validate_policy_file(path: str) -> Tuple[bool, str]:
        """Validate a policy file against the TenantPolicy schema.

        Args:
            path: Path to the JSON policy file to validate

        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            TenantPolicy(**data)
            return True, "ok"
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # JSON parsing errors, validation errors, or type errors
            return False, str(e)

    @staticmethod
    def validate_policy_data(data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate policy data directly without reading from file.

        Args:
            data: Policy data dictionary to validate

        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        try:
            TenantPolicy(**data)
            return True, "ok"
        except (ValueError, TypeError) as e:
            return False, str(e)

    @staticmethod
    def create_example_policy() -> Dict[str, Any]:
        """Create an example policy for testing purposes.

        Returns:
            Example policy data dictionary
        """
        return {
            "tenant_id": "example-tenant",
            "bundle": "example-bundle",
            "required_detectors": ["detector1", "detector2"],
            "optional_detectors": ["detector3"],
            "coverage_method": "required_set",
            "required_coverage": 1.0,
            "detector_weights": {"detector1": 0.6, "detector2": 0.4},
            "required_taxonomy_categories": ["category1"],
            "allowed_content_types": ["TEXT"]
        }
