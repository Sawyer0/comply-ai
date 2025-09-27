"""Policy plugin system following SRP."""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import PluginBase, PluginInterface

logger = logging.getLogger(__name__)


class PolicyEvaluationDecision(Enum):
    """Policy evaluation decisions."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    REQUIRE_REVIEW = "require_review"


@dataclass
class PolicyResult:
    """Policy evaluation result."""

    decision: PolicyEvaluationDecision
    confidence: float
    violations: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float


@dataclass
class PolicyCapabilities:
    """Capabilities supported by a policy plugin."""

    supported_data_types: List[str]
    evaluation_modes: List[str]
    requires_context: bool
    supports_batch: bool


class PolicyPluginInterface(PluginInterface):
    """Contract for policy plugins."""

    @property
    @abstractmethod
    def policy_type(self) -> str:
        """Return the policy type identifier."""

    @abstractmethod
    async def evaluate(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> PolicyResult:
        """Evaluate the policy using the supplied data."""

    @abstractmethod
    async def validate_policy(self, policy_definition: Dict[str, Any]) -> bool:
        """Validate a policy definition."""

    @abstractmethod
    def get_capabilities(self) -> PolicyCapabilities:
        """Return the plugin capabilities."""


class PolicyPlugin(PluginBase, PolicyPluginInterface):
    """Base policy plugin with shared lifecycle behaviour."""

    def __init__(self, name: str, version: str, policy_type: str) -> None:
        super().__init__(name, version)
        self._policy_type = policy_type
        self._policy_rules: Dict[str, Any] = {}

    @property
    def policy_type(self) -> str:
        return self._policy_type

    async def initialize(self, config: Dict[str, Any]) -> bool:
        self._policy_rules = config.get("policy_rules", {})
        return await super().initialize(config)

    async def evaluate(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> PolicyResult:
        self._ensure_initialized()

        start = time.perf_counter()
        try:
            self._validate_input(data, context)
            result = await self._perform_evaluation(data, context)
            result.processing_time_ms = (time.perf_counter() - start) * 1000
            self._validate_result(result)
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Policy evaluation failed in plugin %s: %s", self.plugin_name, exc
            )
            return PolicyResult(
                decision=PolicyEvaluationDecision.DENY,
                confidence=0.0,
                violations=[f"Policy evaluation error: {exc}"],
                metadata={"error": True, "plugin": self.plugin_name},
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

    async def validate_policy(self, policy_definition: Dict[str, Any]) -> bool:
        try:
            for field in ("name", "type", "rules"):
                if field not in policy_definition:
                    logger.error("Missing required field in policy: %s", field)
                    return False

            return await self._validate_policy_definition(policy_definition)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Policy validation failed in plugin %s: %s", self.plugin_name, exc
            )
            return False

    def get_capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(
            supported_data_types=["dict"],
            evaluation_modes=["standard"],
            requires_context=False,
            supports_batch=False,
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        self._policy_rules = {}

    async def _initialize_plugin(self) -> None:
        """Plugin specific initialization (optional)."""

    async def _cleanup_plugin(self) -> None:
        """Plugin specific cleanup (optional)."""

    @abstractmethod
    async def _perform_evaluation(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> PolicyResult:
        """Execute the policy specific evaluation."""

    async def _validate_policy_definition(
        self, policy_definition: Dict[str, Any]
    ) -> bool:
        """Hook for plugin specific policy validation."""

        return True

    def _validate_input(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> None:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        if not data:
            raise ValueError("Data cannot be empty")

        capabilities = self.get_capabilities()
        if capabilities.requires_context and not context:
            raise ValueError("Context is required for this policy")

    def _validate_result(self, result: PolicyResult) -> None:
        if not isinstance(result, PolicyResult):
            raise ValueError("Result must be PolicyResult instance")
        if not 0.0 <= result.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not isinstance(result.decision, PolicyEvaluationDecision):
            raise ValueError("Decision must be PolicyEvaluationDecision enum")


class ExamplePolicyPlugin(PolicyPlugin):
    """Example policy plugin implementation."""

    def __init__(self) -> None:
        super().__init__(
            name="example-policy",
            version="1.0.0",
            policy_type="content_filter",
        )

    def get_capabilities(self) -> PolicyCapabilities:
        return PolicyCapabilities(
            supported_data_types=["dict"],
            evaluation_modes=["standard"],
            requires_context=False,
            supports_batch=False,
        )

    async def _perform_evaluation(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> PolicyResult:
        violations: List[str] = []
        prohibited_words = self._policy_rules.get(
            "prohibited_words", ["spam", "malicious"]
        )
        content = data.get("content", "")

        for word in prohibited_words:
            if word.lower() in content.lower():
                violations.append(f"Prohibited word found: {word}")

        if violations:
            decision = PolicyEvaluationDecision.DENY
            confidence = 0.9
        else:
            decision = PolicyEvaluationDecision.ALLOW
            confidence = 0.8

        return PolicyResult(
            decision=decision,
            confidence=confidence,
            violations=violations,
            metadata={
                "plugin": self.plugin_name,
                "checked_words": prohibited_words,
                "content_length": len(content),
            },
            processing_time_ms=0.0,
        )


__all__ = [
    "PolicyPluginInterface",
    "PolicyPlugin",
    "PolicyResult",
    "PolicyEvaluationDecision",
    "PolicyCapabilities",
    "ExamplePolicyPlugin",
]
