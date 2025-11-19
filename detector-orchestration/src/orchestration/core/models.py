"""Dataclasses shared across core orchestration components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from shared.interfaces.orchestration import DetectorResult


@dataclass(slots=True)
class DetectorConfig:
    """Configuration for a detector endpoint."""

    name: str
    endpoint: str
    timeout_ms: int = 5000
    max_retries: int = 3
    supported_content_types: List[str] = field(default_factory=lambda: ["text"])
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.supported_content_types:
            self.supported_content_types = ["text"]
        else:
            self.supported_content_types = list(self.supported_content_types)


@dataclass(slots=True)
class DetectorClientConfig:
    """Runtime configuration for a detector HTTP client."""

    name: str
    endpoint: str
    timeout: float = 5.0
    max_retries: int = 3
    default_headers: Dict[str, str] = field(default_factory=dict)
    response_parser: Callable[[Dict[str, Any]], DetectorResult] | None = None

    def __post_init__(self) -> None:
        self.default_headers = dict(self.default_headers)


@dataclass(slots=True)
class RoutingDecision:
    """Decision produced by the router describing detector selection."""

    selected_detectors: List[str]
    routing_reason: str
    policy_applied: str | None = None
    coverage_requirements: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.selected_detectors = list(self.selected_detectors)
        self.coverage_requirements = dict(self.coverage_requirements)


@dataclass(slots=True)
class RoutingPlan:
    """Plan describing how detectors should be executed."""

    primary_detectors: List[str]
    secondary_detectors: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    timeout_config: Dict[str, int] = field(default_factory=dict)
    retry_config: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.primary_detectors = list(self.primary_detectors)
        self.secondary_detectors = list(self.secondary_detectors)
        if self.parallel_groups:
            self.parallel_groups = [list(group) for group in self.parallel_groups]
        elif self.primary_detectors:
            self.parallel_groups = [list(self.primary_detectors)]
        self.timeout_config = dict(self.timeout_config)
        self.retry_config = dict(self.retry_config)


@dataclass(slots=True)
class AggregatedOutput:
    """Unified detector output returned by the aggregation stage."""

    combined_output: str
    confidence_score: float
    contributing_detectors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.contributing_detectors = list(self.contributing_detectors)
        self.metadata = dict(self.metadata)

    @classmethod
    def empty(cls) -> "AggregatedOutput":
        """Return a sentinel aggregated output representing no results."""

        return cls(
            combined_output="none:info",
            confidence_score=0.0,
            metadata={
                "strategy": "empty",
                "reason": "no_successful_results",
                "total_detectors": 0,
            },
        )


__all__ = [
    "AggregatedOutput",
    "DetectorClientConfig",
    "DetectorConfig",
    "RoutingDecision",
    "RoutingPlan",
]
