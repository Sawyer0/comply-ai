from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from .models import ContentType, DetectorResult
from .policy import OPAPolicyEngine


class ConflictResolutionStrategy(str, Enum):
    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    MOST_RESTRICTIVE = "most_restrictive"
    TENANT_PREFERENCE = "tenant_preference"


class ConflictResolutionOutcome(BaseModel):
    strategy_used: ConflictResolutionStrategy
    winning_output: Optional[str] = None
    winning_detector: Optional[str] = None
    conflicting_detectors: List[str] = []
    tie_breaker_applied: Optional[str] = None
    confidence_delta: float = 0.0
    normalized_scores: Dict[str, float] = {}
    audit_decision: Dict[str, Any] = {}


class ConflictResolver:
    """Resolves conflicting detector outputs using policy (OPA) or sensible defaults.

    Notes:
    - Never assigns canonical taxonomy. Operates only on raw detector outputs.
    - Produces normalized score hints per raw output label for downstream mapper context.
    - When OPA is enabled, will call tenant/bundle conflict decision endpoint to choose strategy.
    """

    def __init__(self, opa_engine: Optional[OPAPolicyEngine] = None):
        self.opa_engine = opa_engine

    async def resolve(
        self,
        *,
        tenant_id: str,
        policy_bundle: str,
        content_type: ContentType,
        detector_results: List[DetectorResult],
        weights: Optional[Dict[str, float]] = None,
    ) -> ConflictResolutionOutcome:
        successes = [r for r in detector_results if r.status.value == "success" and (r.output is not None)]
        # Deduplicate identical detector-output pairs if needed
        # Group by output value
        groups: Dict[str, List[DetectorResult]] = {}
        for r in successes:
            key = (r.output or "none").strip()
            groups.setdefault(key, []).append(r)

        # Build normalized score hints per raw output label (0..1)
        norm_scores: Dict[str, float] = {}
        for out, rs in groups.items():
            if not rs:
                norm_scores[out] = 0.0
                continue
            # Weighted sum of confidences as hint
            if weights:
                total_w = sum(weights.get(r.detector, 1.0) for r in rs) or 1.0
                score = sum((weights.get(r.detector, 1.0) * (r.confidence or 0.0)) for r in rs) / total_w
                norm_scores[out] = float(max(0.0, min(score, 1.0)))
            else:
                # Average confidence
                avg = sum((r.confidence or 0.0) for r in rs) / float(len(rs))
                norm_scores[out] = float(max(0.0, min(avg, 1.0)))

        unique_outputs = list(groups.keys())
        # Trivial case: 0 or 1 unique outputs
        if len(unique_outputs) <= 1:
            out = unique_outputs[0] if unique_outputs else None
            # Find winning detector (highest confidence) if any
            winning_detector = None
            delta = 0.0
            if successes:
                ordered = sorted(successes, key=lambda r: (r.confidence or 0.0), reverse=True)
                winning_detector = ordered[0].detector
                if len(ordered) > 1:
                    delta = (ordered[0].confidence or 0.0) - (ordered[1].confidence or 0.0)
            return ConflictResolutionOutcome(
                strategy_used=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
                winning_output=out,
                winning_detector=winning_detector,
                conflicting_detectors=[r.detector for r in successes if (r.output or "none").strip() != (out or "none")],
                tie_breaker_applied=None,
                confidence_delta=float(delta),
                normalized_scores=norm_scores,
                audit_decision={},
            )

        # Strategy selection via OPA (if available)
        selected_strategy: Optional[ConflictResolutionStrategy] = None
        preferred_detector: Optional[str] = None
        tie_breaker_hint: Optional[str] = None
        audit_decision: Dict[str, Any] = {}

        if self.opa_engine is not None:
            try:
                opa_input = {
                    "content_type": content_type.value,
                    "candidates": [
                        {"detector": r.detector, "output": r.output, "confidence": r.confidence}
                        for r in successes
                    ],
                    "weights": weights or {},
                    "unique_outputs": unique_outputs,
                }
                decision = await self.opa_engine.evaluate_conflict(tenant_id, policy_bundle, opa_input)  # type: ignore[attr-defined]
                if isinstance(decision, dict):
                    audit_decision = decision
                    s = decision.get("strategy")
                    if isinstance(s, str):
                        try:
                            selected_strategy = ConflictResolutionStrategy(s)
                        except Exception:  # noqa: BLE001
                            selected_strategy = None
                    preferred_detector = decision.get("preferred_detector")
                    tie_breaker_hint = decision.get("tie_breaker")
            except Exception:
                # Fail open to defaults
                selected_strategy = None

        # Default strategy by content type when OPA not decisive
        if selected_strategy is None:
            if content_type == ContentType.TEXT:
                selected_strategy = ConflictResolutionStrategy.WEIGHTED_AVERAGE
            elif content_type == ContentType.IMAGE:
                selected_strategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
            elif content_type == ContentType.DOCUMENT:
                selected_strategy = ConflictResolutionStrategy.MOST_RESTRICTIVE
            elif content_type == ContentType.CODE:
                selected_strategy = ConflictResolutionStrategy.MAJORITY_VOTE
            else:
                selected_strategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE

        # Compute winner based on selected strategy
        winning_output, winning_detector, tie_breaker_used, delta = self._apply_strategy(
            selected_strategy, groups, weights or {}, preferred_detector, tie_breaker_hint
        )

        conflicting = [r.detector for r in successes if (r.output or "none").strip() != (winning_output or "none")]

        return ConflictResolutionOutcome(
            strategy_used=selected_strategy,
            winning_output=winning_output,
            winning_detector=winning_detector,
            conflicting_detectors=conflicting,
            tie_breaker_applied=tie_breaker_used,
            confidence_delta=float(delta),
            normalized_scores=norm_scores,
            audit_decision=audit_decision,
        )

    def _apply_strategy(
        self,
        strategy: ConflictResolutionStrategy,
        groups: Dict[str, List[DetectorResult]],
        weights: Dict[str, float],
        preferred_detector: Optional[str],
        tie_breaker_hint: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """Returns: (winning_output, winning_detector, tie_breaker_used, delta)"""
        # Prepare aggregates
        out_counts: Dict[str, int] = {out: len(rs) for out, rs in groups.items()}
        out_conf_max: Dict[str, float] = {
            out: max((r.confidence or 0.0) for r in rs) if rs else 0.0 for out, rs in groups.items()
        }
        out_weighted: Dict[str, float] = {}
        for out, rs in groups.items():
            if rs:
                out_weighted[out] = sum((weights.get(r.detector, 1.0) * (r.confidence or 0.0)) for r in rs)
            else:
                out_weighted[out] = 0.0

        # Helper to pick by metric with deterministic tie-breakers
        def pick_by_metric(metric: Dict[str, float], primary_tie_breaker: str) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
            # Highest metric wins; delta is diff between top two
            ordered = sorted(metric.items(), key=lambda kv: kv[1], reverse=True)
            if not ordered:
                return None, None, None, 0.0
            winner_out, winner_score = ordered[0]
            if len(ordered) > 1 and ordered[1][1] == winner_score:
                # Tie — apply tie-breakers
                tied = [o for o, v in ordered if v == winner_score]
                # Tie-breaker 1: preferred detector if provided and it produced any tied output
                if preferred_detector:
                    for out in tied:
                        if any(r.detector == preferred_detector for r in groups.get(out, [])):
                            win_det = preferred_detector
                            delta = 0.0
                            return out, win_det, f"preferred_detector:{preferred_detector}", delta
                # Tie-breaker 2: use highest individual confidence among tied outputs
                # If all tied outputs also have the exact same highest individual confidence,
                # fall back to alphabetical determinism instead of arbitrary order.
                tied_conf_values = {out: out_conf_max.get(out, 0.0) for out in tied}
                if len(set(tied_conf_values.values())) == 1:
                    best_out = sorted(tied)[0]
                    dets = sorted([r.detector for r in groups.get(best_out, [])])
                    det = dets[0] if dets else None
                    return best_out, det, f"{primary_tie_breaker}->alphabetical", 0.0
                best_out = None
                best_conf = -1.0
                best_det = None
                for out in tied:
                    for r in groups.get(out, []):
                        c = r.confidence or 0.0
                        if c > best_conf:
                            best_conf = c
                            best_out = out
                            best_det = r.detector
                if best_out is not None:
                    return best_out, best_det, f"{primary_tie_breaker}->highest_confidence", 0.0
                # Tie-breaker 3: alphabetical by output for determinism
                best_out = sorted(tied)[0]
                # Pick first detector alphabetically that produced it
                dets = sorted([r.detector for r in groups.get(best_out, [])])
                det = dets[0] if dets else None
                return best_out, det, f"{primary_tie_breaker}->alphabetical", 0.0
            # No tie
            # Choose detector with highest confidence for the winning output
            rs = groups.get(winner_out, [])
            det = None
            best = -1.0
            for r in rs:
                c = r.confidence or 0.0
                if c > best:
                    best = c
                    det = r.detector
            # Delta between top two metric scores
            second = ordered[1][1] if len(ordered) > 1 else 0.0
            return winner_out, det, None, float(winner_score - second)

        if strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            # Build per-output highest individual confidence
            return pick_by_metric(out_conf_max, "highest_confidence")
        elif strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            return pick_by_metric(out_weighted, "weighted_average")
        elif strategy == ConflictResolutionStrategy.MAJORITY_VOTE:
            # Use counts; tie-break with weighted then highest confidence
            out, det, tb, _ = pick_by_metric({k: float(v) for k, v in out_counts.items()}, "majority_vote")
            return out, det, tb, 0.0
        elif strategy == ConflictResolutionStrategy.MOST_RESTRICTIVE:
            # Without taxonomy semantics, approximate with weighted then highest confidence
            return pick_by_metric(out_weighted, "most_restrictive")
        elif strategy == ConflictResolutionStrategy.TENANT_PREFERENCE:
            # Prefer provided detector if it exists
            if preferred_detector:
                for out, rs in groups.items():
                    for r in rs:
                        if r.detector == preferred_detector:
                            return out, preferred_detector, "preferred_detector", 0.0
            # Fallback to weighted
            return pick_by_metric(out_weighted, "tenant_preference")
        # Fallback
        return pick_by_metric(out_conf_max, "highest_confidence")
