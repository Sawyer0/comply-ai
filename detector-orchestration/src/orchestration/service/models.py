"""Data models used by the orchestration service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from shared.interfaces.detector_output import CanonicalDetectorOutputs
from shared.interfaces.orchestration import DetectorResult, PolicyViolation

from ..core import (
    AggregatedOutput,
    ContentRouter,
    CustomerDetectorClient,
    DetectorCoordinator,
    ResponseAggregator,
    RoutingDecision,
    RoutingPlan as CoordinatorRoutingPlan,
)
from ..discovery import ServiceDiscoveryManager
from ..ml import (
    AdaptiveLoadBalancer,
    ContentAnalyzer,
    PerformancePredictor,
    RiskScorer,
    RoutingOptimizer,
)
from ..ml.feedback_service import MLFeedbackService
from ..monitoring import PrometheusMetricsCollector
from ..pipelines import AsyncJobProcessor
from ..policy import PolicyManager
from ..repository import (
    DetectorMappingConfigRepository,
    DetectorRepository,
    RiskAnalysisRepository,
)
from ..resilience import RateLimiter
from ..security import ApiKeyManager, AttackDetector, InputSanitizer, RBACManager
from ..tenancy.tenant_isolation import TenantContext, TenantIsolationManager
from ..tenancy.tenant_manager import TenantManager


@dataclass
class OrchestrationArtifacts:
    """Artifacts produced by the orchestration pipeline."""

    detector_results: List[DetectorResult]
    aggregated_output: Optional[AggregatedOutput]
    coverage: float
    policy_violations: List[PolicyViolation]
    recommendations: List[str]
    canonical_outputs: Optional[CanonicalDetectorOutputs] = None


@dataclass
class AggregationContext:
    """Intermediate aggregation data shared across pipeline steps."""

    detector_results: List[DetectorResult]
    aggregated_output: Optional[AggregatedOutput]
    coverage: float


@dataclass(frozen=True)
class OrchestrationConfig:
    """Configuration container for orchestration service following SRP."""

    # Core feature toggles used by the new orchestration stack
    enable_health_monitoring: bool = True
    enable_service_discovery: bool = True
    enable_policy_management: bool = True
    health_check_interval: int = 30
    service_ttl_minutes: int = 30

    # Rate limiting configuration used by the new orchestration stack.
    rate_limit_enabled: bool = True
    rate_limit_tenant_limit: int = 120
    rate_limit_window_seconds: int = 60
    rate_limit_tenant_overrides: Dict[str, int] = field(default_factory=dict)


@dataclass  # pylint: disable=too-many-instance-attributes
class OrchestrationComponents:
    """Container for orchestration service components following SRP."""

    detector_clients: Dict[str, CustomerDetectorClient] = field(default_factory=dict)

    # Core components
    detector_coordinator: Optional[DetectorCoordinator] = None
    content_router: Optional[ContentRouter] = None
    response_aggregator: Optional[ResponseAggregator] = None

    # Optional components
    health_monitor: Optional[Any] = None
    service_discovery: Optional[ServiceDiscoveryManager] = None
    policy_manager: Optional[PolicyManager] = None
    rate_limiter: Optional[RateLimiter] = None
    job_processor: Optional[AsyncJobProcessor] = None

    # Security components
    api_key_manager: Optional[ApiKeyManager] = None
    rbac_manager: Optional[RBACManager] = None
    attack_detector: Optional[AttackDetector] = None
    input_sanitizer: Optional[InputSanitizer] = None

    # Tenancy components
    tenant_manager: Optional[TenantManager] = None
    tenant_isolation: Optional[TenantIsolationManager] = None

    # ML components
    performance_predictor: Optional[PerformancePredictor] = None
    content_analyzer: Optional[ContentAnalyzer] = None
    load_balancer: Optional[AdaptiveLoadBalancer] = None
    routing_optimizer: Optional[RoutingOptimizer] = None
    ml_feedback: Optional[MLFeedbackService] = None
    risk_scorer: Optional[RiskScorer] = None

    # Persistence components
    detector_repository: Optional[DetectorRepository] = None
    detector_mapping_repository: Optional[DetectorMappingConfigRepository] = None
    risk_repository: Optional[RiskAnalysisRepository] = None

    # Monitoring
    metrics_collector: Optional[PrometheusMetricsCollector] = None

    # Cache components using shared library
    idempotency_cache: Optional[Any] = None
    response_cache: Optional[Any] = None


@dataclass  # pylint: disable=too-many-instance-attributes
class PipelineContext:
    """Shared context for orchestrating a single request."""

    tenant_id: str
    correlation_id: str
    tenant_context: TenantContext
    processing_mode: str
    routing_plan: Optional[CoordinatorRoutingPlan] = None
    routing_decision: Optional[RoutingDecision] = None
    content_features: Optional[Dict[str, Any]] = None

    def build_metadata(self) -> Dict[str, Any]:
        """Create detector metadata payload."""
        metadata: Dict[str, Any] = {
            "tenant_id": self.tenant_id,
            "correlation_id": self.correlation_id,
            "tenant_context": {
                "tenant_id": self.tenant_context.tenant_id,
                "user_id": self.tenant_context.user_id,
                "request_id": self.tenant_context.request_id,
            },
        }

        if self.routing_decision:
            metadata.update(
                {
                    "routing_reason": getattr(
                        self.routing_decision, "routing_reason", ""
                    ),
                    "policy_applied": getattr(
                        self.routing_decision, "policy_applied", None
                    ),
                    "coverage_requirements": getattr(
                        self.routing_decision, "coverage_requirements", None
                    ),
                }
            )

        if self.content_features is not None:
            metadata["content_features"] = self.content_features

        return metadata


@dataclass
class OrchestrationRequestContext:
    """Optional metadata provided when orchestrating a request."""

    correlation_id: Optional[str] = None
    api_key: Optional[str] = None
    user_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    processing_mode: str = "standard"


@dataclass  # pylint: disable=too-many-instance-attributes
class DetectorRegistrationConfig:
    """Descriptor for registering a detector with the service."""

    detector_id: str
    endpoint: str
    detector_type: str
    tenant_id: str = "default"
    timeout_ms: int = 5000
    max_retries: int = 3
    supported_content_types: Optional[List[str]] = None
    auth_headers: Optional[Dict[str, str]] = None
    analyze_path: Optional[str] = "/analyze"
    response_parser: Optional[str] = None


__all__ = [
    "AggregationContext",
    "DetectorRegistrationConfig",
    "OrchestrationArtifacts",
    "OrchestrationComponents",
    "OrchestrationConfig",
    "OrchestrationRequestContext",
    "PipelineContext",
]
