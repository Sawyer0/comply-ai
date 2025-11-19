"""Detector plugin system following SRP."""

from __future__ import annotations

import logging
import importlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from shared.interfaces.common import Severity
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
            confidence=0.9 if findings else 0.1,
            category="example",
            subcategory=None,
            severity=Severity.MEDIUM if findings else Severity.LOW,
            findings=findings,
            metadata=None,
            processing_time_ms=None,
        )


class PiiranhaPIIDetectorPlugin(DetectorPlugin):
    def __init__(self) -> None:
        super().__init__(
            name="piiranha-pii-detector",
            version="1.0.0",
            detector_type="piiranha-pii",
        )
        self._pipeline = None

    def get_capabilities(self) -> DetectorCapabilities:
        return DetectorCapabilities(
            supported_content_types=["text"],
            supported_languages=["en", "es", "fr", "de", "it", "nl"],
            max_content_size=10_000,
            processing_modes=["sync"],
            confidence_threshold=0.5,
        )

    async def _initialize_plugin(self) -> None:
        try:
            transformers_module = importlib.import_module("transformers")
            pipeline = getattr(transformers_module, "pipeline")
        except ImportError as exc:  # pragma: no cover - optional dependency
            logger.error("transformers is required for PiiranhaPIIDetectorPlugin: %s", exc)
            self._pipeline = None
            return
        self._pipeline = pipeline(
            "token-classification",
            model="iiiorg/piiranha-v1-detect-personal-information",
            aggregation_strategy="simple",
        )

    async def _perform_detection(
        self, content: str, metadata: Dict[str, Any]
    ) -> DetectorResult:
        if self._pipeline is None:
            raise RuntimeError(
                "PiiranhaPIIDetectorPlugin is not initialized or transformers is missing"
            )

        raw_entities = self._pipeline(content)
        findings: List[Dict[str, Any]] = []

        for entity in raw_entities:
            entity_type = entity.get("entity_group") or entity.get("entity")
            value = entity.get("word") or entity.get("text")
            if not entity_type or not value:
                continue
            start = entity.get("start")
            end = entity.get("end")
            score = float(entity.get("score", 0.0))
            findings.append(
                {
                    "type": str(entity_type),
                    "value": str(value),
                    "confidence": score,
                    "start": start,
                    "end": end,
                }
            )

        has_pii = bool(findings)
        severity = Severity.MEDIUM if has_pii else Severity.LOW
        confidence = (
            sum(f["confidence"] for f in findings) / len(findings)
            if findings
            else 0.0
        )

        return DetectorResult(
            detector_id=self.plugin_name,
            detector_type=self.detector_type,
            confidence=confidence,
            category="pii",
            subcategory=None,
            severity=severity,
            findings=findings,
            metadata={"source_model": "iiiorg/piiranha-v1-detect-personal-information"},
            processing_time_ms=None,
        )


__all__ = [
    "DetectorPluginInterface",
    "DetectorPlugin",
    "DetectorCapabilities",
    "ExampleDetectorPlugin",
    "PiiranhaPIIDetectorPlugin",
]
