"""Primary orchestration service factory orchestrating dependency wiring."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Tuple

from detector_orchestration.metrics import OrchestrationMetricsCollector

from .components import ServiceFactoryComponentsMixin
from .container import ServiceContainer
from .events import ServiceFactoryEventsMixin
from .pipeline import ServiceFactoryPipelineMixin

logger = logging.getLogger(__name__)


class OrchestrationServiceFactory(
    ServiceFactoryComponentsMixin,
    ServiceFactoryEventsMixin,
    ServiceFactoryPipelineMixin,
):
    """Factory for creating and configuring orchestration service components."""

    # pylint: disable=too-many-instance-attributes

    def __init__(self, settings) -> None:
        self.settings = settings
        self.container = ServiceContainer(settings)
        self.metrics = OrchestrationMetricsCollector()

        self._event_subscribers: List[asyncio.Queue] = []
        self._incident_history: deque[Dict[str, Any]] = deque(
            maxlen=self.settings.config.event_history_limit
        )

        self.detector_clients: Dict[str, Any] = {}
        self.health_monitor = None
        self.circuit_breaker = None
        self.registry = None
        self.policy_store = None
        self.policy_manager = None
        self.policy_engine = None
        self.mapper_client = None
        self.job_manager = None
        self.idempotency_cache = None
        self.response_cache = None
        self.conflict_resolver = None
        self.aggregator = None
        self.router = None
        self.coordinator = None
        self.pending_idempotent_jobs: Dict[str, Tuple[str, str]] = {}


__all__ = ["OrchestrationServiceFactory"]
