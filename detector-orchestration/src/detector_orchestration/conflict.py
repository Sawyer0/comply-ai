"""Conflict resolution strategies for orchestrated detector responses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from .models import ContentType, DetectorResult
from .policy import OPAPolicyEngine


class ConflictResolutionStrategy(str, Enum):
    """Available resolution strategies for detector disagreements."""

    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    MOST_RESTRICTIVE = "most_restrictive"
    TENANT_PREFERENCE = "tenant_preference"


class ConflictResolutionOutcome(BaseModel):
    """Detailed decision returned after resolving detector conflicts."""

    strategy_used: ConflictResolutionStrategy
    winning_output: Optional[str] = None
    winning_detector: Optional[str] = None
    conflicting_detectors: List[str] = Field(default_factory=list)
    tie_breaker_applied: Optional[str] = None
    confidence_delta: float = 0.0
    normalized_scores: Dict[str, float] = Field(default_factory=dict)
    audit_decision: Dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class ConflictResolutionRequest:
    """Input payload provided to the conflict resolver."""

    tenant_id: str
    policy_bundle: str
    content_type: ContentType
    detector_results: List[DetectorResult]
    weights: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class ResolutionContext:
    """Intermediate aggregates used when resolving detector conflicts."""

    successes: Sequence[DetectorResult]
    groups: Dict[str, List[DetectorResult]]
    normalized_scores: Dict[str, float]
    weights: Dict[str, float]

    @property
    def unique_outputs(self) -> List[str]:
        """Unique output labels observed across successful detectors."""

        return list(self.groups.keys())

    def has_conflict(self) -> bool:
        """Return True when multiple distinct outputs are present."""

        return len(self.unique_outputs) > 1


@dataclass(frozen=True)
class TrivialResolution:  # pylint: disable=too-few-public-methods
    """Resolution details for scenarios with no actual conflict."""

    winning_output: Optional[str]
    winning_detector: Optional[str]
    confidence_delta: float
    conflicting_detectors: List[str]

    def to_outcome(
        self, normalized_scores: Dict[str, float]
    ) -> ConflictResolutionOutcome:
        """Convert the trivial decision into a full outcome payload."""

        return ConflictResolutionOutcome(
            strategy_used=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            winning_output=self.winning_output,
            winning_detector=self.winning_detector,
            conflicting_detectors=self.conflicting_detectors,
            tie_breaker_applied=None,
            confidence_delta=self.confidence_delta,
            normalized_scores=normalized_scores,
            audit_decision={},
        )


@dataclass(frozen=True)
class StrategySelection:
    """Selected resolution strategy and related policy hints."""

    strategy: ConflictResolutionStrategy
    preferred_detector: Optional[str]
    tie_breaker_hint: Optional[str]
    audit_decision: Dict[str, Any]


@dataclass(frozen=True)
class StrategyDecision:
    """Result returned by the strategy executor."""

    winning_output: Optional[str]
    winning_detector: Optional[str]
    tie_breaker_applied: Optional[str]
    confidence_delta: float
    conflicting_detectors: List[str]


@dataclass(frozen=True)
class GroupMetrics:
    """Aggregate statistics computed for each output group."""

    counts: Dict[str, int]
    max_confidence: Dict[str, float]
    weighted_scores: Dict[str, float]


@dataclass(frozen=True)
class StrategyHints:
    """Hints provided by policy or defaults when applying a strategy."""

    preferred_detector: Optional[str]
    primary_hint: str
    tie_breaker_hint: Optional[str] = None

    def has_preference(self) -> bool:
        """Return True when a detector has been explicitly preferred."""

        return self.preferred_detector is not None


class ConflictResolver:  # pylint: disable=too-few-public-methods
    """Resolves conflicting detector outputs using policies or sensible defaults."""

    def __init__(self, opa_engine: Optional[OPAPolicyEngine] = None):
        self.opa_engine = opa_engine

    async def resolve(
        self, request: ConflictResolutionRequest
    ) -> ConflictResolutionOutcome:
        """Resolve detector disagreements according to policy guidance."""

        context = self._build_context(
            results=request.detector_results, weights=request.weights
        )

        trivial = self._resolve_trivial_case(context)
        if trivial:
            return trivial.to_outcome(context.normalized_scores)

        selection = await self._select_strategy(request, context)
        decision = self._execute_strategy(selection, context)

        return ConflictResolutionOutcome(
            strategy_used=selection.strategy,
            winning_output=decision.winning_output,
            winning_detector=decision.winning_detector,
            conflicting_detectors=decision.conflicting_detectors,
            tie_breaker_applied=decision.tie_breaker_applied,
            confidence_delta=decision.confidence_delta,
            normalized_scores=context.normalized_scores,
            audit_decision=selection.audit_decision,
        )

    def _build_context(
        self, results: Sequence[DetectorResult], weights: Optional[Dict[str, float]]
    ) -> ResolutionContext:
        successes = [
            r
            for r in results
            if r.status.value == "success" and r.output is not None
        ]
        groups: Dict[str, List[DetectorResult]] = {}
        for result in successes:
            key = (result.output or "none").strip()
            groups.setdefault(key, []).append(result)

        normalized_scores: Dict[str, float] = {}
        normalized_weights = {**(weights or {})}
        for output, grouped_results in groups.items():
            normalized_scores[output] = self._normalized_score(
                grouped_results, normalized_weights
            )

        return ResolutionContext(
            successes=successes,
            groups=groups,
            normalized_scores=normalized_scores,
            weights=normalized_weights,
        )

    def _normalized_score(
        self, results: Sequence[DetectorResult], weights: Dict[str, float]
    ) -> float:
        if not results:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for result in results:
            weight = weights.get(result.detector, 1.0)
            weighted_sum += weight * (result.confidence or 0.0)
            total_weight += weight
        if total_weight == 0:
            return 0.0
        score = weighted_sum / total_weight
        return float(max(0.0, min(score, 1.0)))

    def _resolve_trivial_case(
        self, context: ResolutionContext
    ) -> Optional[TrivialResolution]:
        outputs = context.unique_outputs
        if len(outputs) <= 1:
            winning_output = outputs[0] if outputs else None
            ordered = sorted(
                context.successes,
                key=lambda res: (res.confidence or 0.0),
                reverse=True,
            )
            winning_detector = ordered[0].detector if ordered else None
            delta = 0.0
            if len(ordered) > 1:
                delta = (ordered[0].confidence or 0.0) - (
                    ordered[1].confidence or 0.0
                )
            conflicting = [
                res.detector
                for res in context.successes
                if (res.output or "none").strip()
                != (winning_output or "none").strip()
            ]
            return TrivialResolution(
                winning_output=winning_output,
                winning_detector=winning_detector,
                confidence_delta=float(delta),
                conflicting_detectors=conflicting,
            )
        return None

    async def _select_strategy(
        self, request: ConflictResolutionRequest, context: ResolutionContext
    ) -> StrategySelection:
        strategy: Optional[ConflictResolutionStrategy] = None
        preferred_detector: Optional[str] = None
        tie_breaker_hint: Optional[str] = None
        audit_decision: Dict[str, Any] = {}

        if self.opa_engine is not None:
            decision = await self.opa_engine.evaluate_conflict(
                request.tenant_id,
                request.policy_bundle,
                self._build_opa_input(context, request),
            )
            if isinstance(decision, dict):
                audit_decision = decision
                raw_strategy = decision.get("strategy")
                if isinstance(raw_strategy, str):
                    try:
                        strategy = ConflictResolutionStrategy(raw_strategy)
                    except ValueError:
                        strategy = None
                preferred_detector_val = decision.get("preferred_detector")
                if isinstance(preferred_detector_val, str):
                    preferred_detector = preferred_detector_val
                tie_breaker_val = decision.get("tie_breaker")
                if isinstance(tie_breaker_val, str):
                    tie_breaker_hint = tie_breaker_val

        if strategy is None:
            strategy = self._default_strategy(request.content_type)

        return StrategySelection(
            strategy=strategy,
            preferred_detector=preferred_detector,
            tie_breaker_hint=tie_breaker_hint,
            audit_decision=audit_decision,
        )

    def _build_opa_input(
        self, context: ResolutionContext, request: ConflictResolutionRequest
    ) -> Dict[str, Any]:
        return {
            "content_type": request.content_type.value,
            "candidates": [
                {
                    "detector": result.detector,
                    "output": result.output,
                    "confidence": result.confidence,
                }
                for result in context.successes
            ],
            "weights": context.weights,
            "unique_outputs": context.unique_outputs,
        }

    def _default_strategy(
        self, content_type: ContentType
    ) -> ConflictResolutionStrategy:
        if content_type == ContentType.TEXT:
            return ConflictResolutionStrategy.WEIGHTED_AVERAGE
        if content_type == ContentType.IMAGE:
            return ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        if content_type == ContentType.DOCUMENT:
            return ConflictResolutionStrategy.MOST_RESTRICTIVE
        if content_type == ContentType.CODE:
            return ConflictResolutionStrategy.MAJORITY_VOTE
        return ConflictResolutionStrategy.HIGHEST_CONFIDENCE

    def _execute_strategy(
        self, selection: StrategySelection, context: ResolutionContext
    ) -> StrategyDecision:
        metrics = self._compute_metrics(context)

        if selection.strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            hints = StrategyHints(
                preferred_detector=selection.preferred_detector,
                primary_hint="highest_confidence",
                tie_breaker_hint=selection.tie_breaker_hint,
            )
            return self._select_by_metric(
                metric=metrics.max_confidence,
                context=context,
                hints=hints,
            )
        if selection.strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            hints = StrategyHints(
                preferred_detector=selection.preferred_detector,
                primary_hint="weighted_average",
                tie_breaker_hint=selection.tie_breaker_hint,
            )
            return self._select_by_metric(
                metric=metrics.weighted_scores,
                context=context,
                hints=hints,
            )
        if selection.strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            counts_metric = {key: float(value) for key, value in metrics.counts.items()}
            hints = StrategyHints(
                preferred_detector=selection.preferred_detector,
                primary_hint="majority_vote",
                tie_breaker_hint=selection.tie_breaker_hint,
            )
            choice = self._select_by_metric(
                metric=counts_metric,
                context=context,
                hints=hints,
            )
            return StrategyDecision(
                winning_output=choice.winning_output,
                winning_detector=choice.winning_detector,
                tie_breaker_applied=choice.tie_breaker_applied,
                confidence_delta=0.0,
                conflicting_detectors=choice.conflicting_detectors,
            )
        if selection.strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE:
            hints = StrategyHints(
                preferred_detector=selection.preferred_detector,
                primary_hint="most_restrictive",
                tie_breaker_hint=selection.tie_breaker_hint,
            )
            return self._select_by_metric(
                metric=metrics.weighted_scores,
                context=context,
                hints=hints,
            )
        if selection.strategy == ConflictResolutionStrategy.TENANT_PREFERENCE:
            return self._tenant_preference_choice(selection, context, metrics)
        hints = StrategyHints(
            preferred_detector=selection.preferred_detector,
            primary_hint="fallback_highest_confidence",
            tie_breaker_hint=selection.tie_breaker_hint,
        )
        return self._select_by_metric(
            metric=metrics.max_confidence,
            context=context,
            hints=hints,
        )

    def _tenant_preference_choice(
        self,
        selection: StrategySelection,
        context: ResolutionContext,
        metrics: GroupMetrics,
    ) -> StrategyDecision:
        preferred = selection.preferred_detector
        if preferred:
            for output, results in context.groups.items():
                if any(res.detector == preferred for res in results):
                    return StrategyDecision(
                        winning_output=output,
                        winning_detector=preferred,
                        tie_breaker_applied="preferred_detector",
                        confidence_delta=0.0,
                        conflicting_detectors=self._conflicting_detectors(
                            context, output
                        ),
                    )
        hints = StrategyHints(
            preferred_detector=preferred,
            primary_hint="tenant_preference",
            tie_breaker_hint=selection.tie_breaker_hint,
        )
        chosen = self._select_by_metric(
            metric=metrics.weighted_scores,
            context=context,
            hints=hints,
        )
        return StrategyDecision(
            winning_output=chosen.winning_output,
            winning_detector=chosen.winning_detector,
            tie_breaker_applied=chosen.tie_breaker_applied,
            confidence_delta=chosen.confidence_delta,
            conflicting_detectors=self._conflicting_detectors(
                context, chosen.winning_output
            ),
        )

    def _compute_metrics(self, context: ResolutionContext) -> GroupMetrics:
        counts: Dict[str, int] = {}
        max_confidence: Dict[str, float] = {}
        weighted_scores: Dict[str, float] = {}
        for output, results in context.groups.items():
            counts[output] = len(results)
            max_confidence[output] = (
                max((res.confidence or 0.0) for res in results)
                if results
                else 0.0
            )
            total_weight = 0.0
            weighted_sum = 0.0
            for res in results:
                weight = context.weights.get(res.detector, 1.0)
                total_weight += weight
                weighted_sum += weight * (res.confidence or 0.0)
            score = weighted_sum / total_weight if total_weight else 0.0
            weighted_scores[output] = float(max(0.0, min(score, 1.0)))
        return GroupMetrics(
            counts=counts,
            max_confidence=max_confidence,
            weighted_scores=weighted_scores,
        )

    def _select_by_metric(
        self,
        *,
        metric: Dict[str, float],
        context: ResolutionContext,
        hints: StrategyHints,
    ) -> StrategyDecision:
        ordered = sorted(metric.items(), key=lambda item: item[1], reverse=True)
        if not ordered:
            return StrategyDecision(
                winning_output=None,
                winning_detector=None,
                tie_breaker_applied=hints.tie_breaker_hint,
                confidence_delta=0.0,
                conflicting_detectors=self._conflicting_detectors(context, None),
            )

        best_output, best_value = ordered[0]
        second_value = ordered[1][1] if len(ordered) > 1 else None
        if second_value is None or best_value > second_value:
            detector = self._highest_confidence_detector(
                context.groups.get(best_output, [])
            )
            delta = float(best_value - (second_value or 0.0))
            return StrategyDecision(
                winning_output=best_output,
                winning_detector=detector,
                tie_breaker_applied=hints.tie_breaker_hint,
                confidence_delta=delta,
                conflicting_detectors=self._conflicting_detectors(
                    context, best_output
                ),
            )

        return self._resolve_metric_tie(
            ordered=ordered,
            context=context,
            hints=hints,
        )

    def _resolve_metric_tie(
        self,
        *,
        ordered: List[Tuple[str, float]],
        context: ResolutionContext,
        hints: StrategyHints,
    ) -> StrategyDecision:
        top_value = ordered[0][1]
        tied_outputs = [
            output
            for output, value in ordered
            if value == top_value
        ]

        preferred = self._preferred_detector_tiebreak(
            tied_outputs=tied_outputs,
            context=context,
            hints=hints,
        )
        if preferred:
            return preferred

        highest_confidence = self._highest_confidence_tiebreak(
            tied_outputs=tied_outputs,
            context=context,
            hints=hints,
        )
        if highest_confidence:
            return highest_confidence

        return self._alphabetical_tiebreak(
            tied_outputs=tied_outputs,
            context=context,
            primary_hint=hints.primary_hint,
        )

    def _preferred_detector_tiebreak(
        self,
        *,
        tied_outputs: List[str],
        context: ResolutionContext,
        hints: StrategyHints,
    ) -> Optional[StrategyDecision]:
        preferred = hints.preferred_detector
        if preferred is None:
            return None
        for output in tied_outputs:
            results = context.groups.get(output, [])
            if any(res.detector == preferred for res in results):
                return StrategyDecision(
                    winning_output=output,
                    winning_detector=preferred,
                    tie_breaker_applied=f"{hints.primary_hint}->preferred_detector",
                    confidence_delta=0.0,
                    conflicting_detectors=self._conflicting_detectors(
                        context, output
                    ),
                )
        return None

    def _highest_confidence_tiebreak(
        self,
        *,
        tied_outputs: List[str],
        context: ResolutionContext,
        hints: StrategyHints,
    ) -> Optional[StrategyDecision]:
        best_output: Optional[str] = None
        best_detector: Optional[str] = None
        best_confidence = -1.0
        duplicate_best = False

        for output in tied_outputs:
            results = context.groups.get(output, [])
            confidence = (
                max((res.confidence or 0.0) for res in results)
                if results
                else 0.0
            )
            if confidence > best_confidence:
                best_confidence = confidence
                best_output = output
                best_detector = self._highest_confidence_detector(results)
                duplicate_best = False
            elif confidence == best_confidence:
                duplicate_best = True

        if best_output is None or duplicate_best:
            return None

        return StrategyDecision(
            winning_output=best_output,
            winning_detector=best_detector,
            tie_breaker_applied=f"{hints.primary_hint}->highest_confidence",
            confidence_delta=0.0,
            conflicting_detectors=self._conflicting_detectors(
                context, best_output
            ),
        )

    def _alphabetical_tiebreak(
        self,
        *,
        tied_outputs: List[str],
        context: ResolutionContext,
        primary_hint: str,
    ) -> StrategyDecision:
        winning_output = sorted(tied_outputs)[0]
        detectors = sorted(
            res.detector for res in context.groups.get(winning_output, [])
        )
        detector = detectors[0] if detectors else None
        return StrategyDecision(
            winning_output=winning_output,
            winning_detector=detector,
            tie_breaker_applied=f"{primary_hint}->alphabetical",
            confidence_delta=0.0,
            conflicting_detectors=self._conflicting_detectors(
                context, winning_output
            ),
        )

    def _highest_confidence_detector(
        self, results: Sequence[DetectorResult]
    ) -> Optional[str]:
        if not results:
            return None
        best_confidence = max(result.confidence or 0.0 for result in results)
        best_detector_names = sorted(
            result.detector
            for result in results
            if (result.confidence or 0.0) == best_confidence
        )
        return best_detector_names[0] if best_detector_names else None

    def _conflicting_detectors(
        self, context: ResolutionContext, winning_output: Optional[str]
    ) -> List[str]:
        target = (winning_output or "none").strip()
        return [
            result.detector
            for result in context.successes
            if (result.output or "none").strip() != target
        ]


__all__ = [
    "ConflictResolutionStrategy",
    "ConflictResolutionOutcome",
    "ConflictResolutionRequest",
    "ConflictResolver",
]
