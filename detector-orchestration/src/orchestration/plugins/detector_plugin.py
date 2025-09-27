"""Detector plugin system following SRP."""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from shared.interfaces.orchestration import DetectorResult

from .base import PluginBase, PluginInterface

logger = logging.getLogger(__name__)


@dataclass
class DetectorCapabilities:
    """Capabilities supported by a detector plugin."""

    supported_content_types: List[str]
    supported_languages: List[str]
    max_content_size: int
    processing_modes: List[str]
    confidence_threshold: float


class DetectorPluginInterface(PluginInterface):
    """Contract for detector plugins."""

    @property
    @abstractmethod
    def detector_type(self) -> str:
        """Return the detector type identifier."""

    @abstractmethod
    async def detect(self, content: str, metadata: Dict[str, Any]) -> DetectorResult:
        """Run detection for the supplied content."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Check whether the detector is healthy."""

    @abstractmethod
    def get_capabilities(self) -> DetectorCapabilities:
        """Return the supported capabilities for this detector."""


class DetectorPlugin(PluginBase, DetectorPluginInterface):
    """Base detector plugin with shared lifecycle behaviour."""

    def __init__(self, name: str, version: str, detector_type: str) -> None:
        super().__init__(name, version)
        self._detector_type = detector_type

    @property
    def detector_type(self) -> str:
        return self._detector_type

    async def detect(self, content: str, metadata: Dict[str, Any]) -> DetectorResult:
        self._ensure_initialized()
        self._validate_input(content, metadata)
        result = await self._perform_detection(content, metadata)
        self._validate_result(result)
        return result

    async def health_check(self) -> bool:
        self._ensure_initialized()
        return await self._check_health()

    async def _initialize_plugin(self) -> None:
        """Plugin specific initialization (optional)."""

    async def _cleanup_plugin(self) -> None:
        """Plugin specific cleanup (optional)."""

    def _validate_input(self, content: str, metadata: Dict[str, Any]) -> None:
        if not content:
            raise ValueError("Content cannot be empty")

        capabilities = self.get_capabilities()
        if len(content) > capabilities.max_content_size:
            raise ValueError(
                f"Content size exceeds maximum: {capabilities.max_content_size}"
            )

    def _validate_result(self, result: DetectorResult) -> None:
        if not isinstance(result, DetectorResult):
            raise ValueError("Result must be DetectorResult instance")

        if not 0.0 <= result.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    async def _check_health(self) -> bool:
        """Default health check implementation."""

        return True

    @abstractmethod
    async def _perform_detection(
        self, content: str, metadata: Dict[str, Any]
    ) -> DetectorResult:
        """Execute the detector specific logic."""


class ExampleDetectorPlugin(DetectorPlugin):
    """Example detector plugin implementation."""

    def __init__(self) -> None:
        super().__init__(
            name="example-detector",
            version="1.0.0",
            detector_type="example",
        )

    def get_capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            supported_content_types=["text"],
            supported_languages=["en"],
            max_content_size=10_000,
            processing_modes=["sync"],
            confidence_threshold=0.5,
        )

    async def _perform_detection(
        self, content: str, metadata: Dict[str, Any]
    ) -> DetectorResult:
        findings: List[Dict[str, Any]] = []
        lowered = content.lower()
        index = lowered.find("example")
        if index != -1:
            findings.append(
                {
                    "type": "EXAMPLE_KEYWORD",
                    "value": "example",
                    "confidence": 0.9,
                    "start": index,
                    "end": index + len("example"),
                }
            )

        return DetectorResult(
            detector_id=self.plugin_name,
            detector_type=self.detector_type,
            success=True,
            confidence=0.9 if findings else 0.1,
            findings=findings,
        )


__all__ = [
    "DetectorPluginInterface",
    "DetectorPlugin",
    "DetectorCapabilities",
    "ExampleDetectorPlugin",
]
