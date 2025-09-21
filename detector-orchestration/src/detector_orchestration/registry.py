from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

from .clients import DetectorClient
from .config import DetectorEndpoint, Settings
from .health_monitor import HealthMonitor
from .models import DetectorCapabilities


class DetectorRegistration(BaseModel):
    name: str
    endpoint: str
    timeout_ms: int = 3000
    max_retries: int = 1
    supported_content_types: List[str] = ["text"]
    auth: dict = {}


class DetectorRegistry:
    def __init__(self, settings: Settings, clients: Dict[str, DetectorClient], monitor: Optional[HealthMonitor] = None):
        self.settings = settings
        self.clients = clients
        self.monitor = monitor

    def register(self, reg: DetectorRegistration) -> None:
        # Update settings registry
        self.settings.detectors[reg.name] = DetectorEndpoint(
            name=reg.name,
            endpoint=reg.endpoint,
            timeout_ms=reg.timeout_ms,
            max_retries=reg.max_retries,
            auth=reg.auth,
            supported_content_types=reg.supported_content_types,
        )
        # Create/replace client
        self.clients[reg.name] = DetectorClient(
            name=reg.name,
            endpoint=reg.endpoint,
            timeout_ms=reg.timeout_ms,
            max_retries=reg.max_retries,
            auth=reg.auth,
        )
        if self.monitor:
            # Rebuild monitor map (simple approach)
            self.monitor.clients[reg.name] = self.clients[reg.name]

    def update(self, name: str, reg: DetectorRegistration) -> None:
        if name not in self.settings.detectors:
            raise KeyError("detector_not_found")
        self.register(reg)

    def remove(self, name: str) -> None:
        self.settings.detectors.pop(name, None)
        self.clients.pop(name, None)
        if self.monitor:
            self.monitor.clients.pop(name, None)

    def list(self) -> List[str]:
        return list(self.settings.detectors.keys())

    def get(self, name: str) -> Optional[DetectorClient]:
        return self.clients.get(name)

