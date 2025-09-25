"""
Feature flag system for the mapper service.

This module provides feature flag capabilities with gradual rollout,
runtime evaluation, and A/B testing support.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel
import hashlib

logger = logging.getLogger(__name__)


class FlagType(Enum):
    """Types of feature flags."""

    BOOLEAN = "boolean"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    JSON = "json"


class RolloutStrategy(Enum):
    """Rollout strategies for feature flags."""

    IMMEDIATE = "immediate"
    GRADUAL = "gradual"
    CANARY = "canary"
    A_B_TEST = "ab_test"
    SCHEDULED = "scheduled"


class FlagStatus(Enum):
    """Feature flag status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    PAUSED = "paused"


@dataclass
class FlagRule:
    """Rule for feature flag evaluation."""

    attribute: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, not_in, contains
    value: Any
    weight: float = 1.0


@dataclass
class RolloutConfig:
    """Configuration for gradual rollout."""

    strategy: RolloutStrategy
    start_percentage: float = 0.0
    end_percentage: float = 100.0
    increment_percentage: float = 10.0
    increment_interval_minutes: int = 60

    # A/B testing configuration
    variants: Dict[str, Any] = field(default_factory=dict)
    traffic_allocation: Dict[str, float] = field(default_factory=dict)

    # Scheduling configuration
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Targeting rules
    rules: List[FlagRule] = field(default_factory=list)
    user_segments: List[str] = field(default_factory=list)


@dataclass
class FeatureFlag:
    """Feature flag definition."""

    flag_id: str
    name: str
    description: str
    flag_type: FlagType
    default_value: Any

    # Current configuration
    enabled: bool = False
    status: FlagStatus = FlagStatus.INACTIVE
    current_value: Any = None

    # Rollout configuration
    rollout_config: Optional[RolloutConfig] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"

    # Tracking
    evaluation_count: int = 0
    last_evaluated: Optional[datetime] = None

    # Callbacks
    on_change_callback: Optional[Callable] = None


class EvaluationContext(BaseModel):
    """Context for feature flag evaluation."""

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None

    # User attributes
    user_attributes: Dict[str, Any] = {}

    # Request context
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    # Custom attributes
    custom_attributes: Dict[str, Any] = {}


class FlagEvaluationResult(BaseModel):
    """Result of feature flag evaluation."""

    flag_id: str
    value: Any
    variant: Optional[str] = None
    reason: str
    evaluation_time_ms: float
    context_hash: str


class FlagStore:
    """Storage interface for feature flags."""

    def __init__(self):
        self._flags: Dict[str, FeatureFlag] = {}
        self._flag_history: Dict[str, List[Dict[str, Any]]] = {}

    def save_flag(self, flag: FeatureFlag) -> bool:
        """Save a feature flag."""
        try:
            flag.updated_at = datetime.utcnow()
            self._flags[flag.flag_id] = flag

            # Record history
            if flag.flag_id not in self._flag_history:
                self._flag_history[flag.flag_id] = []

            self._flag_history[flag.flag_id].append(
                {
                    "timestamp": flag.updated_at,
                    "enabled": flag.enabled,
                    "value": flag.current_value,
                    "status": flag.status.value,
                }
            )

            # Keep only last 100 history entries
            if len(self._flag_history[flag.flag_id]) > 100:
                self._flag_history[flag.flag_id] = self._flag_history[flag.flag_id][
                    -100:
                ]

            logger.debug(f"Saved feature flag: {flag.flag_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save flag {flag.flag_id}: {e}")
            return False

    def get_flag(self, flag_id: str) -> Optional[FeatureFlag]:
        """Get a feature flag by ID."""
        return self._flags.get(flag_id)

    def list_flags(
        self, status_filter: Optional[FlagStatus] = None
    ) -> List[FeatureFlag]:
        """List all feature flags, optionally filtered by status."""
        flags = list(self._flags.values())

        if status_filter:
            flags = [flag for flag in flags if flag.status == status_filter]

        return flags

    def delete_flag(self, flag_id: str) -> bool:
        """Delete a feature flag."""
        try:
            self._flags.pop(flag_id, None)
            self._flag_history.pop(flag_id, None)
            logger.info(f"Deleted feature flag: {flag_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete flag {flag_id}: {e}")
            return False

    def get_flag_history(self, flag_id: str) -> List[Dict[str, Any]]:
        """Get history for a feature flag."""
        return self._flag_history.get(flag_id, [])


class RuleEvaluator:
    """Evaluates feature flag rules."""

    def evaluate_rules(self, rules: List[FlagRule], context: EvaluationContext) -> bool:
        """Evaluate all rules against context."""
        if not rules:
            return True  # No rules means always match

        total_weight = 0.0
        matched_weight = 0.0

        for rule in rules:
            total_weight += rule.weight

            if self._evaluate_single_rule(rule, context):
                matched_weight += rule.weight

        # Return true if majority of weighted rules match
        return matched_weight / total_weight >= 0.5 if total_weight > 0 else True

    def _evaluate_single_rule(self, rule: FlagRule, context: EvaluationContext) -> bool:
        """Evaluate a single rule."""
        try:
            # Get attribute value from context
            attr_value = self._get_attribute_value(rule.attribute, context)

            if attr_value is None:
                return False

            # Evaluate based on operator
            return self._apply_operator(rule.operator, attr_value, rule.value)

        except Exception as e:
            logger.warning(
                f"Error evaluating rule {rule.attribute} {rule.operator} {rule.value}: {e}"
            )
            return False

    def _get_attribute_value(self, attribute: str, context: EvaluationContext) -> Any:
        """Get attribute value from evaluation context."""
        # Handle nested attributes with dot notation
        parts = attribute.split(".")

        # Start with context object
        current = context.dict()

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _apply_operator(self, operator: str, attr_value: Any, rule_value: Any) -> bool:
        """Apply comparison operator."""
        try:
            if operator == "eq":
                return attr_value == rule_value
            elif operator == "ne":
                return attr_value != rule_value
            elif operator == "gt":
                return attr_value > rule_value
            elif operator == "lt":
                return attr_value < rule_value
            elif operator == "gte":
                return attr_value >= rule_value
            elif operator == "lte":
                return attr_value <= rule_value
            elif operator == "in":
                return (
                    attr_value in rule_value
                    if isinstance(rule_value, (list, tuple, set))
                    else False
                )
            elif operator == "not_in":
                return (
                    attr_value not in rule_value
                    if isinstance(rule_value, (list, tuple, set))
                    else True
                )
            elif operator == "contains":
                return rule_value in str(attr_value)
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.warning(f"Error applying operator {operator}: {e}")
            return False


class PercentageCalculator:
    """Calculates percentage-based rollouts."""

    def is_user_in_percentage(
        self, user_id: str, flag_id: str, percentage: float
    ) -> bool:
        """Determine if user is in the rollout percentage."""
        if percentage <= 0:
            return False
        if percentage >= 100:
            return True

        # Create consistent hash for user + flag combination
        hash_input = f"{user_id}:{flag_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)

        # Convert to percentage (0-100)
        user_percentage = (hash_value % 10000) / 100.0

        return user_percentage < percentage

    def get_variant_for_user(
        self, user_id: str, flag_id: str, variants: Dict[str, float]
    ) -> Optional[str]:
        """Get A/B test variant for user based on traffic allocation."""
        if not variants:
            return None

        # Create hash for consistent variant assignment
        hash_input = f"{user_id}:{flag_id}:variant"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        user_percentage = (hash_value % 10000) / 100.0

        # Find variant based on cumulative percentages
        cumulative = 0.0
        for variant, allocation in variants.items():
            cumulative += allocation
            if user_percentage < cumulative:
                return variant

        # Default to first variant if no match
        return list(variants.keys())[0] if variants else None


class FeatureFlagManager:
    """Main feature flag manager."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flag_store = FlagStore()
        self.rule_evaluator = RuleEvaluator()
        self.percentage_calculator = PercentageCalculator()

        # Rollout management
        self._rollout_tasks: Dict[str, asyncio.Task] = {}
        self._evaluation_cache: Dict[str, FlagEvaluationResult] = {}
        self._cache_ttl_seconds = config.get("cache_ttl_seconds", 300)  # 5 minutes

    async def initialize(self):
        """Initialize the feature flag manager."""
        try:
            # Load flags from configuration
            await self._load_flags_from_config()

            # Start rollout management
            await self._start_rollout_management()

            logger.info("Feature flag manager initialized")

        except Exception as e:
            logger.error(f"Failed to initialize feature flag manager: {e}")
            raise

    async def shutdown(self):
        """Shutdown the feature flag manager."""
        try:
            # Cancel rollout tasks
            for task in self._rollout_tasks.values():
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete
            if self._rollout_tasks:
                await asyncio.gather(
                    *self._rollout_tasks.values(), return_exceptions=True
                )

            logger.info("Feature flag manager shutdown complete")

        except Exception as e:
            logger.error(f"Error during feature flag manager shutdown: {e}")

    def create_flag(
        self,
        flag_id: str,
        name: str,
        description: str,
        flag_type: FlagType,
        default_value: Any,
        **kwargs,
    ) -> FeatureFlag:
        """Create a new feature flag."""
        flag = FeatureFlag(
            flag_id=flag_id,
            name=name,
            description=description,
            flag_type=flag_type,
            default_value=default_value,
            current_value=default_value,
            **kwargs,
        )

        self.flag_store.save_flag(flag)
        logger.info(f"Created feature flag: {flag_id}")

        return flag

    def update_flag(self, flag_id: str, **updates) -> Optional[FeatureFlag]:
        """Update a feature flag."""
        flag = self.flag_store.get_flag(flag_id)
        if not flag:
            return None

        # Update attributes
        for key, value in updates.items():
            if hasattr(flag, key):
                setattr(flag, key, value)

        # Execute change callback
        if flag.on_change_callback:
            try:
                flag.on_change_callback(flag)
            except Exception as e:
                logger.error(f"Error executing change callback for flag {flag_id}: {e}")

        self.flag_store.save_flag(flag)

        # Clear cache for this flag
        self._clear_flag_cache(flag_id)

        logger.info(f"Updated feature flag: {flag_id}")
        return flag

    async def evaluate_flag(
        self, flag_id: str, context: EvaluationContext, use_cache: bool = True
    ) -> FlagEvaluationResult:
        """Evaluate a feature flag for given context."""
        start_time = time.time()

        try:
            # Check cache first
            if use_cache:
                cached_result = self._get_cached_result(flag_id, context)
                if cached_result:
                    return cached_result

            flag = self.flag_store.get_flag(flag_id)
            if not flag:
                return FlagEvaluationResult(
                    flag_id=flag_id,
                    value=None,
                    reason="flag_not_found",
                    evaluation_time_ms=(time.time() - start_time) * 1000,
                    context_hash=self._hash_context(context),
                )

            # Update evaluation tracking
            flag.evaluation_count += 1
            flag.last_evaluated = datetime.utcnow()

            # Evaluate flag
            result = await self._evaluate_flag_internal(flag, context, start_time)

            # Cache result
            if use_cache:
                self._cache_result(result)

            return result

        except Exception as e:
            logger.error(f"Error evaluating flag {flag_id}: {e}")
            return FlagEvaluationResult(
                flag_id=flag_id,
                value=None,
                reason=f"evaluation_error: {e}",
                evaluation_time_ms=(time.time() - start_time) * 1000,
                context_hash=self._hash_context(context),
            )

    async def start_gradual_rollout(
        self, flag_id: str, rollout_config: RolloutConfig
    ) -> bool:
        """Start gradual rollout for a feature flag."""
        flag = self.flag_store.get_flag(flag_id)
        if not flag:
            return False

        try:
            flag.rollout_config = rollout_config
            flag.status = FlagStatus.ACTIVE
            self.flag_store.save_flag(flag)

            # Start rollout task
            if rollout_config.strategy == RolloutStrategy.GRADUAL:
                task = asyncio.create_task(self._manage_gradual_rollout(flag))
                self._rollout_tasks[flag_id] = task

            logger.info(f"Started gradual rollout for flag: {flag_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to start rollout for flag {flag_id}: {e}")
            return False

    def get_flag_status(self, flag_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive status of a feature flag."""
        flag = self.flag_store.get_flag(flag_id)
        if not flag:
            return None

        return {
            "flag_id": flag.flag_id,
            "name": flag.name,
            "enabled": flag.enabled,
            "status": flag.status.value,
            "current_value": flag.current_value,
            "evaluation_count": flag.evaluation_count,
            "last_evaluated": flag.last_evaluated,
            "rollout_config": (
                flag.rollout_config.__dict__ if flag.rollout_config else None
            ),
        }

    def list_flags(
        self, status_filter: Optional[FlagStatus] = None
    ) -> List[Dict[str, Any]]:
        """List all feature flags with their status."""
        flags = self.flag_store.list_flags(status_filter)
        return [self.get_flag_status(flag.flag_id) for flag in flags]

    async def _evaluate_flag_internal(
        self, flag: FeatureFlag, context: EvaluationContext, start_time: float
    ) -> FlagEvaluationResult:
        """Internal flag evaluation logic."""
        # Check if flag is enabled
        if not flag.enabled or flag.status != FlagStatus.ACTIVE:
            return FlagEvaluationResult(
                flag_id=flag.flag_id,
                value=flag.default_value,
                reason="flag_disabled",
                evaluation_time_ms=(time.time() - start_time) * 1000,
                context_hash=self._hash_context(context),
            )

        # Check rollout configuration
        if flag.rollout_config:
            rollout_result = await self._evaluate_rollout(flag, context)
            if rollout_result:
                return FlagEvaluationResult(
                    flag_id=flag.flag_id,
                    value=rollout_result["value"],
                    variant=rollout_result.get("variant"),
                    reason=rollout_result["reason"],
                    evaluation_time_ms=(time.time() - start_time) * 1000,
                    context_hash=self._hash_context(context),
                )

        # Default evaluation
        return FlagEvaluationResult(
            flag_id=flag.flag_id,
            value=(
                flag.current_value
                if flag.current_value is not None
                else flag.default_value
            ),
            reason="default_value",
            evaluation_time_ms=(time.time() - start_time) * 1000,
            context_hash=self._hash_context(context),
        )

    async def _evaluate_rollout(
        self, flag: FeatureFlag, context: EvaluationContext
    ) -> Optional[Dict[str, Any]]:
        """Evaluate rollout configuration."""
        rollout = flag.rollout_config
        if not rollout:
            return None

        # Check targeting rules
        if rollout.rules and not self.rule_evaluator.evaluate_rules(
            rollout.rules, context
        ):
            return {"value": flag.default_value, "reason": "targeting_rules_not_met"}

        # Handle different rollout strategies
        if rollout.strategy == RolloutStrategy.GRADUAL:
            return await self._evaluate_gradual_rollout(flag, context, rollout)
        elif rollout.strategy == RolloutStrategy.A_B_TEST:
            return await self._evaluate_ab_test(flag, context, rollout)
        elif rollout.strategy == RolloutStrategy.SCHEDULED:
            return await self._evaluate_scheduled_rollout(flag, context, rollout)

        return None

    async def _evaluate_gradual_rollout(
        self, flag: FeatureFlag, context: EvaluationContext, rollout: RolloutConfig
    ) -> Dict[str, Any]:
        """Evaluate gradual rollout."""
        user_id = context.user_id or context.session_id or "anonymous"

        # Calculate current rollout percentage (this would be managed by rollout task)
        current_percentage = rollout.start_percentage  # Simplified for now

        if self.percentage_calculator.is_user_in_percentage(
            user_id, flag.flag_id, current_percentage
        ):
            return {
                "value": flag.current_value,
                "reason": f"gradual_rollout_{current_percentage}%",
            }
        else:
            return {"value": flag.default_value, "reason": "not_in_rollout_percentage"}

    async def _evaluate_ab_test(
        self, flag: FeatureFlag, context: EvaluationContext, rollout: RolloutConfig
    ) -> Dict[str, Any]:
        """Evaluate A/B test."""
        user_id = context.user_id or context.session_id or "anonymous"

        variant = self.percentage_calculator.get_variant_for_user(
            user_id, flag.flag_id, rollout.traffic_allocation
        )

        if variant and variant in rollout.variants:
            return {
                "value": rollout.variants[variant],
                "variant": variant,
                "reason": f"ab_test_variant_{variant}",
            }
        else:
            return {"value": flag.default_value, "reason": "ab_test_control"}

    async def _evaluate_scheduled_rollout(
        self, flag: FeatureFlag, context: EvaluationContext, rollout: RolloutConfig
    ) -> Dict[str, Any]:
        """Evaluate scheduled rollout."""
        now = datetime.utcnow()

        if rollout.start_time and now < rollout.start_time:
            return {"value": flag.default_value, "reason": "scheduled_not_started"}

        if rollout.end_time and now > rollout.end_time:
            return {"value": flag.default_value, "reason": "scheduled_ended"}

        return {"value": flag.current_value, "reason": "scheduled_active"}

    async def _manage_gradual_rollout(self, flag: FeatureFlag):
        """Manage gradual rollout progression."""
        try:
            rollout = flag.rollout_config
            if not rollout or rollout.strategy != RolloutStrategy.GRADUAL:
                return

            current_percentage = rollout.start_percentage

            while current_percentage < rollout.end_percentage:
                # Wait for increment interval
                await asyncio.sleep(rollout.increment_interval_minutes * 60)

                # Increase percentage
                current_percentage = min(
                    current_percentage + rollout.increment_percentage,
                    rollout.end_percentage,
                )

                # Update rollout config
                rollout.start_percentage = current_percentage
                self.flag_store.save_flag(flag)

                logger.info(
                    f"Gradual rollout progress for {flag.flag_id}: {current_percentage}%"
                )

            # Mark rollout as completed
            flag.status = FlagStatus.COMPLETED
            self.flag_store.save_flag(flag)

            logger.info(f"Gradual rollout completed for flag: {flag.flag_id}")

        except asyncio.CancelledError:
            logger.info(f"Gradual rollout cancelled for flag: {flag.flag_id}")
        except Exception as e:
            logger.error(f"Error in gradual rollout for flag {flag.flag_id}: {e}")

    def _hash_context(self, context: EvaluationContext) -> str:
        """Create hash of evaluation context for caching."""
        context_str = json.dumps(context.dict(), sort_keys=True)
        return hashlib.md5(context_str.encode()).hexdigest()

    def _get_cached_result(
        self, flag_id: str, context: EvaluationContext
    ) -> Optional[FlagEvaluationResult]:
        """Get cached evaluation result."""
        cache_key = f"{flag_id}:{self._hash_context(context)}"
        return self._evaluation_cache.get(cache_key)

    def _cache_result(self, result: FlagEvaluationResult):
        """Cache evaluation result."""
        cache_key = f"{result.flag_id}:{result.context_hash}"
        self._evaluation_cache[cache_key] = result

        # Simple TTL cleanup (in production, use proper cache with TTL)
        asyncio.create_task(self._cleanup_cache_entry(cache_key))

    async def _cleanup_cache_entry(self, cache_key: str):
        """Cleanup cache entry after TTL."""
        await asyncio.sleep(self._cache_ttl_seconds)
        self._evaluation_cache.pop(cache_key, None)

    def _clear_flag_cache(self, flag_id: str):
        """Clear all cached results for a flag."""
        keys_to_remove = [
            key
            for key in self._evaluation_cache.keys()
            if key.startswith(f"{flag_id}:")
        ]
        for key in keys_to_remove:
            self._evaluation_cache.pop(key, None)

    async def _load_flags_from_config(self):
        """Load feature flags from configuration."""
        flags_config = self.config.get("flags", {})

        for flag_id, flag_config in flags_config.items():
            try:
                flag = FeatureFlag(
                    flag_id=flag_id,
                    name=flag_config.get("name", flag_id),
                    description=flag_config.get("description", ""),
                    flag_type=FlagType(flag_config.get("type", "boolean")),
                    default_value=flag_config.get("default_value"),
                    enabled=flag_config.get("enabled", False),
                    current_value=flag_config.get("current_value"),
                )

                self.flag_store.save_flag(flag)
                logger.debug(f"Loaded flag from config: {flag_id}")

            except Exception as e:
                logger.error(f"Failed to load flag {flag_id} from config: {e}")

    async def _start_rollout_management(self):
        """Start rollout management for active flags."""
        flags = self.flag_store.list_flags(FlagStatus.ACTIVE)

        for flag in flags:
            if (
                flag.rollout_config
                and flag.rollout_config.strategy == RolloutStrategy.GRADUAL
                and flag.flag_id not in self._rollout_tasks
            ):

                task = asyncio.create_task(self._manage_gradual_rollout(flag))
                self._rollout_tasks[flag.flag_id] = task


# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> Optional[FeatureFlagManager]:
    """Get the global feature flag manager instance."""
    return _feature_flag_manager


def initialize_feature_flag_manager(config: Dict[str, Any]) -> FeatureFlagManager:
    """Initialize the global feature flag manager."""
    global _feature_flag_manager
    _feature_flag_manager = FeatureFlagManager(config)
    return _feature_flag_manager
