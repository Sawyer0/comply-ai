"""Content routing functionality following SRP.

This module provides ONLY content routing - determining which detectors to use for content.
Other responsibilities are handled by separate modules:
- Health monitoring: ../monitoring/health_monitor.py
- Policy management: ../policy/policy_manager.py
- Load balancing: ../discovery/load_balancer.py
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

from shared.interfaces.orchestration import OrchestrationRequest, ProcessingMode
from shared.utils.correlation import get_correlation_id
from shared.exceptions.base import ValidationError

logger = logging.getLogger(__name__)


class RoutingDecision:
    """Routing decision with selected detectors and reasoning - data structure only."""

    def __init__(
        self,
        selected_detectors: List[str],
        routing_reason: str,
        policy_applied: Optional[str] = None,
        coverage_requirements: Optional[Dict[str, Any]] = None,
    ):
        self.selected_detectors = selected_detectors
        self.routing_reason = routing_reason
        self.policy_applied = policy_applied
        self.coverage_requirements = coverage_requirements or {}


class RoutingPlan:
    """Enhanced routing plan - data structure only."""

    def __init__(
        self,
        primary_detectors: List[str],
        secondary_detectors: Optional[List[str]] = None,
        parallel_groups: Optional[List[List[str]]] = None,
        timeout_config: Optional[Dict[str, int]] = None,
        retry_config: Optional[Dict[str, int]] = None,
    ):
        self.primary_detectors = primary_detectors
        self.secondary_detectors = secondary_detectors or []
        self.parallel_groups = (
            parallel_groups or [primary_detectors] if primary_detectors else []
        )
        self.timeout_config = timeout_config or {}
        self.retry_config = retry_config or {}


class DetectorConfig:
    """Configuration for a detector - data structure only."""

    def __init__(
        self,
        name: str,
        endpoint: str,
        timeout_ms: int = 5000,
        max_retries: int = 3,
        supported_content_types: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        self.name = name
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.supported_content_types = supported_content_types or ["text"]
        self.enabled = enabled


class ContentRouter:
    """Routes content to appropriate detectors based on content type and requirements.

    Single Responsibility: Determine which detectors should process given content.
    Does NOT handle: health monitoring, policy enforcement, load balancing, circuit breakers.
    """

    def __init__(
        self,
        detector_configs: Optional[Dict[str, DetectorConfig]] = None,
        default_timeout_ms: int = 5000,
        default_max_retries: int = 3,
    ):
        """Initialize router with detector configurations.

        Args:
            detector_configs: Dictionary of detector name -> configuration
            default_timeout_ms: Default timeout for detectors
            default_max_retries: Default retry count for detectors
        """
        self.detector_configs = detector_configs or {}
        self.default_timeout_ms = default_timeout_ms
        self.default_max_retries = default_max_retries

    async def route_request(
        self, request: OrchestrationRequest
    ) -> Tuple[RoutingPlan, RoutingDecision]:
        """Route orchestration request to appropriate detectors.

        Single responsibility: determine detector selection based on content and requirements.

        Args:
            request: Orchestration request with content and requirements

        Returns:
            Tuple of (routing_plan, routing_decision)
        """
        correlation_id = get_correlation_id()

        logger.info(
            "Routing request for content analysis",
            extra={
                "correlation_id": correlation_id,
                "detector_types": getattr(request, "detector_types", []),
                "processing_mode": (
                    request.processing_mode.value
                    if hasattr(request, "processing_mode")
                    else "standard"
                ),
            },
        )

        try:
            # Step 1: Get candidate detectors
            candidates = self._get_candidate_detectors(request)

            # Step 2: Apply content type filtering
            content_filtered = self._filter_by_content_type(
                candidates, "text"
            )  # Default to text

            # Step 3: Apply processing mode optimization
            selected_detectors = self._optimize_for_processing_mode(
                content_filtered,
                getattr(request, "processing_mode", ProcessingMode.STANDARD),
            )

            # Step 4: Create routing plan
            routing_plan = self._create_routing_plan(selected_detectors, request)

            # Step 5: Create routing decision
            routing_decision = RoutingDecision(
                selected_detectors=selected_detectors,
                routing_reason=f"content_type_filtered_{len(selected_detectors)}_detectors",
                policy_applied=getattr(request, "policy_bundle", None),
            )

            logger.info(
                "Routing completed with %d selected detectors",
                len(selected_detectors),
                extra={
                    "correlation_id": correlation_id,
                    "selected_detectors": selected_detectors,
                    "routing_reason": routing_decision.routing_reason,
                },
            )

            return routing_plan, routing_decision

        except Exception as e:
            logger.error(
                "Routing failed: %s",
                str(e),
                extra={
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )
            raise ValidationError(
                f"Routing failed: {str(e)}", correlation_id=correlation_id
            ) from e

    def _get_candidate_detectors(self, request: OrchestrationRequest) -> List[str]:
        """Get candidate detectors based on request requirements."""

        # If specific detectors requested, use those
        if hasattr(request, "detector_types") and request.detector_types:
            candidates = request.detector_types
        else:
            # Use all available detectors
            candidates = list(self.detector_configs.keys())

        # Remove excluded detectors if specified
        if hasattr(request, "excluded_detectors") and request.excluded_detectors:
            candidates = [d for d in candidates if d not in request.excluded_detectors]

        return candidates

    def _filter_by_content_type(
        self, candidates: List[str], content_type: str
    ) -> List[str]:
        """Filter detectors by supported content type."""

        filtered = []
        for detector_name in candidates:
            config = self.detector_configs.get(detector_name)
            if (
                config
                and config.enabled
                and content_type in config.supported_content_types
            ):
                filtered.append(detector_name)

        return filtered

    def _optimize_for_processing_mode(
        self, candidates: List[str], processing_mode: ProcessingMode
    ) -> List[str]:
        """Optimize detector selection based on processing mode."""

        if processing_mode == ProcessingMode.FAST:
            # For fast mode, limit to 2 detectors
            return candidates[:2]
        elif processing_mode == ProcessingMode.THOROUGH:
            # For thorough mode, use all available detectors
            return candidates
        else:
            # Standard mode: balanced approach with up to 3 detectors
            return candidates[:3]

    def _create_routing_plan(
        self, selected_detectors: List[str], request: OrchestrationRequest
    ) -> RoutingPlan:
        """Create routing plan from selected detectors."""

        # Build timeout and retry configuration
        timeout_config = {}
        retry_config = {}

        for detector_name in selected_detectors:
            config = self.detector_configs.get(detector_name)
            if config:
                timeout_config[detector_name] = config.timeout_ms
                retry_config[detector_name] = config.max_retries
            else:
                timeout_config[detector_name] = self.default_timeout_ms
                retry_config[detector_name] = self.default_max_retries

        # Determine parallel groups based on processing mode
        processing_mode = getattr(request, "processing_mode", ProcessingMode.STANDARD)
        if processing_mode == ProcessingMode.FAST:
            # Execute fewer detectors in parallel for speed
            parallel_groups = [selected_detectors[:2]] if selected_detectors else []
        elif processing_mode == ProcessingMode.THOROUGH:
            # Execute all detectors in parallel for thoroughness
            parallel_groups = [selected_detectors] if selected_detectors else []
        else:
            # Standard mode: balanced approach
            parallel_groups = [selected_detectors[:3]] if selected_detectors else []

        return RoutingPlan(
            primary_detectors=selected_detectors,
            secondary_detectors=[],  # Could be populated based on requirements
            parallel_groups=parallel_groups,
            timeout_config=timeout_config,
            retry_config=retry_config,
        )

    def register_detector(
        self,
        name: str,
        endpoint: str,
        timeout_ms: int = 5000,
        max_retries: int = 3,
        supported_content_types: Optional[List[str]] = None,
    ) -> bool:
        """Register a new detector configuration."""

        try:
            config = DetectorConfig(
                name=name,
                endpoint=endpoint,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                supported_content_types=supported_content_types or ["text"],
                enabled=True,
            )

            self.detector_configs[name] = config

            logger.info("Registered detector configuration: %s", name)
            return True

        except Exception as e:
            logger.error("Failed to register detector %s: %s", name, str(e))
            return False

    def unregister_detector(self, name: str) -> bool:
        """Unregister a detector configuration."""

        try:
            if name in self.detector_configs:
                del self.detector_configs[name]

            logger.info("Unregistered detector configuration: %s", name)
            return True

        except Exception as e:
            logger.error("Failed to unregister detector %s: %s", name, str(e))
            return False

    def get_available_detectors(self) -> List[str]:
        """Get list of available detector configurations."""
        return [
            name for name, config in self.detector_configs.items() if config.enabled
        ]

    def get_detector_config(self, detector_name: str) -> Optional[DetectorConfig]:
        """Get configuration for a specific detector."""
        return self.detector_configs.get(detector_name)

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and configuration summary.

        Returns:
            Dictionary with routing statistics
        """
        enabled_detectors = [
            name for name, config in self.detector_configs.items() if config.enabled
        ]

        disabled_detectors = [
            name for name, config in self.detector_configs.items() if not config.enabled
        ]

        content_type_support = {}
        for name, config in self.detector_configs.items():
            for content_type in config.supported_content_types:
                if content_type not in content_type_support:
                    content_type_support[content_type] = []
                content_type_support[content_type].append(name)

        return {
            "total_detectors": len(self.detector_configs),
            "enabled_detectors": len(enabled_detectors),
            "disabled_detectors": len(disabled_detectors),
            "enabled_detector_list": enabled_detectors,
            "disabled_detector_list": disabled_detectors,
            "content_type_support": content_type_support,
            "default_timeout_ms": self.default_timeout_ms,
            "default_max_retries": self.default_max_retries,
        }


# Export only the core routing functionality
__all__ = [
    "ContentRouter",
    "RoutingDecision",
    "RoutingPlan",
    "DetectorConfig",
]
