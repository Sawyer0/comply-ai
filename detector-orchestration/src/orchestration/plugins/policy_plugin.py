"""Policy plugin system following SRP.

This module provides ONLY policy plugin interfaces and base implementations.
Single Responsibility: Define policy plugin contracts and base functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PolicyEvaluationDecision(Enum):
    """Policy evaluation decision enumeration."""

    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"
    REQUIRE_REVIEW = "require_review"


@dataclass
class PolicyResult:
    """Policy evaluation result.

    Single Responsibility: Represent policy evaluation outcome.
    """

    decision: PolicyEvaluationDecision
    confidence: float
    violations: List[str]
    metadata: Dict[str, Any]
    processing_time_ms: float


@dataclass
class PolicyCapabilities:
    """Policy capabilities definition.

    Single Responsibility: Define what a policy can evaluate.
    """

    supported_data_types: List[str]
    evaluation_modes: List[str]
    requires_context: bool
    supports_batch: bool


class PolicyPluginInterface(ABC):
    """Interface for policy plugins.

    Single Responsibility: Define policy plugin contract.
    Does NOT handle: implementation details, orchestration logic.
    """

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Get plugin name."""
        pass

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Get plugin version."""
        pass

    @property
    @abstractmethod
    def policy_type(self) -> str:
        """Get policy type."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the policy plugin.

        Args:
            config: Plugin configuration

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def evaluate(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> PolicyResult:
        """Evaluate policy against data.

        Args:
            data: Data to evaluate
            context: Optional evaluation context

        Returns:
            Policy evaluation result
        """
        pass

    @abstractmethod
    async def validate_policy(self, policy_definition: Dict[str, Any]) -> bool:
        """Validate policy definition.

        Args:
            policy_definition: Policy definition to validate

        Returns:
            True if valid
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> PolicyCapabilities:
        """Get policy capabilities.

        Returns:
            Policy capabilities
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup policy resources."""
        pass


class PolicyPlugin(PolicyPluginInterface):
    """Base policy plugin implementation.

    Single Responsibility: Provide base policy plugin functionality.
    Does NOT handle: specific policy logic, orchestration.
    """

    def __init__(self, name: str, version: str, policy_type: str):
        """Initialize policy plugin.

        Args:
            name: Plugin name
            version: Plugin version
            policy_type: Type of policy
        """
        self._name = name
        self._version = version
        self._policy_type = policy_type
        self._config: Dict[str, Any] = {}
        self._initialized = False
        self._policy_rules: Dict[str, Any] = {}

    @property
    def plugin_name(self) -> str:
        """Get plugin name."""
        return self._name

    @property
    def plugin_version(self) -> str:
        """Get plugin version."""
        return self._version

    @property
    def policy_type(self) -> str:
        """Get policy type."""
        return self._policy_type

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the policy plugin.

        Args:
            config: Plugin configuration

        Returns:
            True if initialization successful
        """
        try:
            self._config = config.copy()

            # Load policy rules if provided
            if "policy_rules" in config:
                self._policy_rules = config["policy_rules"]

            # Perform plugin-specific initialization
            await self._initialize_plugin()

            self._initialized = True
            logger.info("Policy plugin initialized: %s", self._name)
            return True

        except Exception as e:
            logger.error(
                "Failed to initialize policy plugin %s: %s", self._name, str(e)
            )
            return False

    async def evaluate(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> PolicyResult:
        """Evaluate policy against data.

        Args:
            data: Data to evaluate
            context: Optional evaluation context

        Returns:
            Policy evaluation result
        """
        if not self._initialized:
            raise RuntimeError(f"Policy plugin {self._name} not initialized")

        try:
            import time

            start_time = time.perf_counter()

            # Validate input
            self._validate_input(data, context)

            # Perform evaluation
            result = await self._perform_evaluation(data, context)

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000
            result.processing_time_ms = processing_time

            # Validate result
            self._validate_result(result)

            return result

        except Exception as e:
            logger.error(
                "Policy evaluation failed in plugin %s: %s", self._name, str(e)
            )
            # Return error result
            return PolicyResult(
                decision=PolicyEvaluationDecision.DENY,
                confidence=0.0,
                violations=[f"Policy evaluation error: {str(e)}"],
                metadata={"error": True, "plugin": self._name},
                processing_time_ms=0.0,
            )

    async def validate_policy(self, policy_definition: Dict[str, Any]) -> bool:
        """Validate policy definition.

        Args:
            policy_definition: Policy definition to validate

        Returns:
            True if valid
        """
        try:
            # Basic validation
            required_fields = ["name", "type", "rules"]
            for field in required_fields:
                if field not in policy_definition:
                    logger.error("Missing required field in policy: %s", field)
                    return False

            # Plugin-specific validation
            return await self._validate_policy_definition(policy_definition)

        except Exception as e:
            logger.error(
                "Policy validation failed in plugin %s: %s", self._name, str(e)
            )
            return False

    def get_capabilities(self) -> PolicyCapabilities:
        """Get policy capabilities.

        Returns:
            Policy capabilities
        """
        # Default capabilities - override in subclasses
        return PolicyCapabilities(
            supported_data_types=["dict"],
            evaluation_modes=["standard"],
            requires_context=False,
            supports_batch=False,
        )

    async def cleanup(self) -> None:
        """Cleanup policy resources."""
        try:
            if self._initialized:
                await self._cleanup_plugin()
                self._initialized = False
                logger.info("Policy plugin cleaned up: %s", self._name)
        except Exception as e:
            logger.error("Failed to cleanup policy plugin %s: %s", self._name, str(e))

    # Protected methods for subclasses to override

    async def _initialize_plugin(self) -> None:
        """Plugin-specific initialization logic.

        Override in subclasses for custom initialization.
        """
        pass

    @abstractmethod
    async def _perform_evaluation(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> PolicyResult:
        """Perform the actual policy evaluation logic.

        Args:
            data: Data to evaluate
            context: Optional evaluation context

        Returns:
            Policy evaluation result
        """
        pass

    async def _validate_policy_definition(
        self, policy_definition: Dict[str, Any]
    ) -> bool:
        """Plugin-specific policy validation logic.

        Override in subclasses for custom validation.

        Args:
            policy_definition: Policy definition to validate

        Returns:
            True if valid
        """
        return True

    async def _cleanup_plugin(self) -> None:
        """Plugin-specific cleanup logic.

        Override in subclasses for custom cleanup.
        """
        pass

    def _validate_input(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> None:
        """Validate input parameters.

        Args:
            data: Data to validate
            context: Context to validate

        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")

        if not data:
            raise ValueError("Data cannot be empty")

        capabilities = self.get_capabilities()
        if capabilities.requires_context and not context:
            raise ValueError("Context is required for this policy")

    def _validate_result(self, result: PolicyResult) -> None:
        """Validate policy result.

        Args:
            result: Result to validate

        Raises:
            ValueError: If result is invalid
        """
        if not isinstance(result, PolicyResult):
            raise ValueError("Result must be PolicyResult instance")

        if result.confidence < 0.0 or result.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        if not isinstance(result.decision, PolicyEvaluationDecision):
            raise ValueError("Decision must be PolicyEvaluationDecision enum")


# Example concrete policy plugin
class ExamplePolicyPlugin(PolicyPlugin):
    """Example policy plugin implementation.

    Single Responsibility: Demonstrate policy plugin implementation.
    """

    def __init__(self):
        """Initialize example policy plugin."""
        super().__init__(
            name="example-policy", version="1.0.0", policy_type="content_filter"
        )

    async def _perform_evaluation(
        self, data: Dict[str, Any], context: Optional[Dict[str, Any]]
    ) -> PolicyResult:
        """Perform example policy evaluation.

        Args:
            data: Data to evaluate
            context: Optional evaluation context

        Returns:
            Policy evaluation result
        """
        violations = []

        # Simple example: check for prohibited words
        prohibited_words = self._policy_rules.get(
            "prohibited_words", ["spam", "malicious"]
        )
        content = data.get("content", "")

        for word in prohibited_words:
            if word.lower() in content.lower():
                violations.append(f"Prohibited word found: {word}")

        # Determine decision
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
                "plugin": self._name,
                "checked_words": prohibited_words,
                "content_length": len(content),
            },
            processing_time_ms=0.0,  # Will be set by base class
        )


# Export only the policy plugin functionality
__all__ = [
    "PolicyPluginInterface",
    "PolicyPlugin",
    "PolicyResult",
    "PolicyEvaluationDecision",
    "PolicyCapabilities",
    "ExamplePolicyPlugin",
]
