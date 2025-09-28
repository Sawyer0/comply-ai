"""Content routing functionality following SRP.

This module provides ONLY content routing - determining which detectors to use for content.
Other responsibilities are handled by separate modules:
- Health monitoring: ../monitoring/health_monitor.py
- Policy management: ../policy/policy_manager.py
- Load balancing: ../discovery/load_balancer.py
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from shared.exceptions.base import ValidationError
from shared.interfaces.orchestration import OrchestrationRequest, ProcessingMode
from shared.utils.correlation import get_correlation_id

from .models import DetectorConfig, RoutingDecision, RoutingPlan

logger = logging.getLogger(__name__)


class ContentRouter:
    """Routes content to appropriate detectors based on content type and requirements."""

    def __init__(
        self,
        detector_configs: Optional[Dict[str, DetectorConfig]] = None,
        default_timeout_ms: int = 5000,
        default_max_retries: int = 3,
    ) -> None:
        self.detector_configs: Dict[str, DetectorConfig] = dict(detector_configs or {})
        self.default_timeout_ms = default_timeout_ms
        self.default_max_retries = default_max_retries

    async def route_request(
        self, request: OrchestrationRequest
    ) -> tuple[RoutingPlan, RoutingDecision]:
        """Route orchestration request to appropriate detectors."""

        correlation_id = get_correlation_id()
        logger.info(
            "Routing request for content analysis",
            extra={
                "correlation_id": correlation_id,
                "detector_types": getattr(request, "detector_types", []),
                "processing_mode": getattr(
                    request, "processing_mode", ProcessingMode.STANDARD
                ),
            },
        )

        try:
            candidates = self._get_candidate_detectors(request)
            filtered = self._filter_by_content_type(candidates, "text")
            if not filtered:
                raise ValidationError(
                    "No detectors matched the requested content type",
                    correlation_id=correlation_id,
                )

            selected = self._optimize_for_processing_mode(
                filtered,
                getattr(request, "processing_mode", ProcessingMode.STANDARD),
            )
            routing_plan = self._create_routing_plan(selected, request)
            routing_decision = RoutingDecision(
                selected_detectors=selected,
                routing_reason=f"content_type_filtered_{len(selected)}_detectors",
                policy_applied=getattr(request, "policy_bundle", None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(
                "Routing failed: %s",
                exc,
                extra={"correlation_id": correlation_id},
            )
            raise ValidationError(
                f"Routing failed: {exc}",
                correlation_id=correlation_id,
            ) from exc

        logger.info(
            "Routing completed with %d selected detectors",
            len(selected),
            extra={
                "correlation_id": correlation_id,
                "selected_detectors": selected,
                "routing_reason": routing_decision.routing_reason,
            },
        )

        return routing_plan, routing_decision

    def register_detector(self, config: DetectorConfig) -> bool:
        """Register a new detector configuration."""

        self.detector_configs[config.name] = config
        logger.info("Registered detector configuration: %s", config.name)
        return True

    def unregister_detector(self, name: str) -> bool:
        """Unregister a detector configuration."""

        removed = self.detector_configs.pop(name, None) is not None
        if removed:
            logger.info("Unregistered detector configuration: %s", name)
        return removed

    def get_available_detectors(self) -> List[str]:
        """Get list of available detector configurations."""

        return [
            name for name, config in self.detector_configs.items() if config.enabled
        ]

    def get_detector_config(self, detector_name: str) -> Optional[DetectorConfig]:
        """Get configuration for a specific detector."""

        return self.detector_configs.get(detector_name)

    def get_routing_statistics(self) -> Dict[str, int | List[str] | Dict[str, List[str]]]:
        """Get routing statistics and configuration summary."""

        enabled_detectors = [
            name for name, config in self.detector_configs.items() if config.enabled
        ]
        disabled_detectors = [
            name for name, config in self.detector_configs.items() if not config.enabled
        ]

        content_type_support: Dict[str, List[str]] = {}
        for name, config in self.detector_configs.items():
            for content_type in config.supported_content_types:
                content_type_support.setdefault(content_type, []).append(name)

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

    def _get_candidate_detectors(self, request: OrchestrationRequest) -> List[str]:
        if getattr(request, "detector_types", None):
            candidates = list(request.detector_types)
        else:
            candidates = list(self.detector_configs.keys())

        excluded = set(getattr(request, "excluded_detectors", []) or [])
        if excluded:
            candidates = [detector for detector in candidates if detector not in excluded]
        return candidates

    def _filter_by_content_type(
        self, candidates: List[str], content_type: str
    ) -> List[str]:
        filtered = []
        for detector_name in candidates:
            config = self.detector_configs.get(detector_name)
            if config and content_type in config.supported_content_types:
                filtered.append(detector_name)
        return filtered

    def _optimize_for_processing_mode(
        self,
        candidates: List[str],
        processing_mode: ProcessingMode,
    ) -> List[str]:
        if not candidates:
            return []

        if processing_mode == ProcessingMode.FAST:
            return candidates[: min(2, len(candidates))]
        if processing_mode == ProcessingMode.THOROUGH:
            return candidates
        return candidates[: min(3, len(candidates))]

    def _create_routing_plan(
        self,
        selected_detectors: List[str],
        request: OrchestrationRequest,
    ) -> RoutingPlan:
        timeout_config: Dict[str, int] = {}
        retry_config: Dict[str, int] = {}

        for detector_name in selected_detectors:
            config = self.detector_configs.get(detector_name)
            if config:
                timeout_config[detector_name] = config.timeout_ms
                retry_config[detector_name] = config.max_retries
            else:
                timeout_config[detector_name] = self.default_timeout_ms
                retry_config[detector_name] = self.default_max_retries

        processing_mode = getattr(request, "processing_mode", ProcessingMode.STANDARD)
        if processing_mode == ProcessingMode.FAST:
            parallel_groups = [selected_detectors[: min(2, len(selected_detectors))]]
        elif processing_mode == ProcessingMode.THOROUGH:
            parallel_groups = [selected_detectors]
        else:
            parallel_groups = [selected_detectors[: min(3, len(selected_detectors))]]

        return RoutingPlan(
            primary_detectors=selected_detectors,
            secondary_detectors=[],
            parallel_groups=parallel_groups,
            timeout_config=timeout_config,
            retry_config=retry_config,
        )


__all__ = ["ContentRouter"]
