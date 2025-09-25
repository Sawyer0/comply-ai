"""Detector plugin system following SRP.

This module provides ONLY detector plugin interfaces and base implementations.
Single Responsibility: Define detector plugin contracts and base functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from shared.interfaces.orchestration import DetectorResult

logger = logging.getLogger(__name__)


@dataclass
class DetectorCapabilities:
    """Detector capabilities definition.

    Single Responsibility: Define what a detector can do.
    """

    supported_content_types: List[str]
    supported_languages: List[str]
    max_content_size: int
    processing_modes: List[str]
    confidence_threshold: float


class DetectorPluginInterface(ABC):
    """Interface for detector plugins.

    Single Responsibility: Define detector plugin contract.
    Does NOT handle: implementation details, orchestration logic.
    """

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Get plugin name."""
        pass

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Get plugin version."""
        pass

    @property
    @abstractmethod
    def detector_type(self) -> str:
        """Get detector type."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the detector plugin.

        Args:
            config: Plugin configuration

        Returns:
            True if initialization successful
        """
        pass

    @abstractmethod
    async def detect(self, content: str, metadata: Dict[str, Any]) -> DetectorResult:
        """Perform detection on content.

        Args:
            content: Content to analyze
            metadata: Additional metadata

        Returns:
            Detection result
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if detector is healthy.

        Returns:
            True if healthy
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> DetectorCapabilities:
        """Get detector capabilities.

        Returns:
            Detector capabilities
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup detector resources."""
        pass


class DetectorPlugin(DetectorPluginInterface):
    """Base detector plugin implementation.

    Single Responsibility: Provide base detector plugin functionality.
    Does NOT handle: specific detection logic, orchestration.
    """

    def __init__(self, name: str, version: str, detector_type: str):
        """Initialize detector plugin.

        Args:
            name: Plugin name
            version: Plugin version
            detector_type: Type of detector
        """
        self._name = name
        self._version = version
        self._detector_type = detector_type
        self._config: Dict[str, Any] = {}
        self._initialized = False

    @property
    def plugin_name(self) -> str:
        """Get plugin name."""
        return self._name

    @property
    def plugin_version(self) -> str:
        """Get plugin version."""
        return self._version

    @property
    def detector_type(self) -> str:
        """Get detector type."""
        return self._detector_type

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the detector plugin.

        Args:
            config: Plugin configuration

        Returns:
            True if initialization successful
        """
        try:
            self._config = config.copy()

            # Perform plugin-specific initialization
            await self._initialize_plugin()

            self._initialized = True
            logger.info("Detector plugin initialized: %s", self._name)
            return True

        except Exception as e:
            logger.error(
                "Failed to initialize detector plugin %s: %s", self._name, str(e)
            )
            return False

    async def detect(self, content: str, metadata: Dict[str, Any]) -> DetectorResult:
        """Perform detection on content.

        Args:
            content: Content to analyze
            metadata: Additional metadata

        Returns:
            Detection result
        """
        if not self._initialized:
            raise RuntimeError(f"Detector plugin {self._name} not initialized")

        try:
            # Validate input
            self._validate_input(content, metadata)

            # Perform detection
            result = await self._perform_detection(content, metadata)

            # Validate result
            self._validate_result(result)

            return result

        except Exception as e:
            logger.error("Detection failed in plugin %s: %s", self._name, str(e))
            # Return error result
            return DetectorResult(
                detector_id=self._name,
                detector_type=self._detector_type,
                success=False,
                error_message=str(e),
                confidence=0.0,
                findings=[],
            )

    async def health_check(self) -> bool:
        """Check if detector is healthy.

        Returns:
            True if healthy
        """
        try:
            if not self._initialized:
                return False

            # Perform plugin-specific health check
            return await self._check_health()

        except Exception as e:
            logger.error("Health check failed for plugin %s: %s", self._name, str(e))
            return False

    def get_capabilities(self) -> DetectorCapabilities:
        """Get detector capabilities.

        Returns:
            Detector capabilities
        """
        # Default capabilities - override in subclasses
        return DetectorCapabilities(
            supported_content_types=["text/plain"],
            supported_languages=["en"],
            max_content_size=1024 * 1024,  # 1MB
            processing_modes=["standard"],
            confidence_threshold=0.5,
        )

    async def cleanup(self) -> None:
        """Cleanup detector resources."""
        try:
            if self._initialized:
                await self._cleanup_plugin()
                self._initialized = False
                logger.info("Detector plugin cleaned up: %s", self._name)
        except Exception as e:
            logger.error("Failed to cleanup detector plugin %s: %s", self._name, str(e))

    # Protected methods for subclasses to override

    async def _initialize_plugin(self) -> None:
        """Plugin-specific initialization logic.

        Override in subclasses for custom initialization.
        """
        pass

    @abstractmethod
    async def _perform_detection(
        self, content: str, metadata: Dict[str, Any]
    ) -> DetectorResult:
        """Perform the actual detection logic.

        Args:
            content: Content to analyze
            metadata: Additional metadata

        Returns:
            Detection result
        """
        pass

    async def _check_health(self) -> bool:
        """Plugin-specific health check logic.

        Override in subclasses for custom health checks.

        Returns:
            True if healthy
        """
        return True

    async def _cleanup_plugin(self) -> None:
        """Plugin-specific cleanup logic.

        Override in subclasses for custom cleanup.
        """
        pass

    def _validate_input(self, content: str, metadata: Dict[str, Any]) -> None:
        """Validate input parameters.

        Args:
            content: Content to validate
            metadata: Metadata to validate

        Raises:
            ValueError: If input is invalid
        """
        if not content:
            raise ValueError("Content cannot be empty")

        capabilities = self.get_capabilities()
        if len(content) > capabilities.max_content_size:
            raise ValueError(
                f"Content size exceeds maximum: {capabilities.max_content_size}"
            )

    def _validate_result(self, result: DetectorResult) -> None:
        """Validate detection result.

        Args:
            result: Result to validate

        Raises:
            ValueError: If result is invalid
        """
        if not isinstance(result, DetectorResult):
            raise ValueError("Result must be DetectorResult instance")

        if result.confidence < 0.0 or result.confidence > 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


# Example concrete detector plugin
class ExampleDetectorPlugin(DetectorPlugin):
    """Example detector plugin implementation.

    Single Responsibility: Demonstrate detector plugin implementation.
    """

    def __init__(self):
        """Initialize example detector plugin."""
        super().__init__(
            name="example-detector", version="1.0.0", detector_type="example"
        )

    async def _perform_detection(
        self, content: str, metadata: Dict[str, Any]
    ) -> DetectorResult:
        """Perform example detection.

        Args:
            content: Content to analyze
            metadata: Additional metadata

        Returns:
            Detection result
        """
        # Simple example: detect if content contains "example"
        findings = []
        if "example" in content.lower():
            findings.append(
                {
                    "type": "EXAMPLE_KEYWORD",
                    "value": "example",
                    "confidence": 0.9,
                    "start": content.lower().find("example"),
                    "end": content.lower().find("example") + len("example"),
                }
            )

        return DetectorResult(
            detector_id=self._name,
            detector_type=self._detector_type,
            success=True,
            confidence=0.9 if findings else 0.1,
            findings=findings,
        )


# Export only the detector plugin functionality
__all__ = [
    "DetectorPluginInterface",
    "DetectorPlugin",
    "DetectorCapabilities",
    "ExampleDetectorPlugin",
]
