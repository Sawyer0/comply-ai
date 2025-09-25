"""
Plugin interfaces for the mapper service.

This module defines the core interfaces that plugins must implement
to extend the mapper service functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum

# Import shared components
from shared.interfaces.base import BaseRequest, BaseResponse
from shared.interfaces.common import ProcessingMode, RiskLevel
from shared.exceptions.base import BaseServiceException


class PluginType(Enum):
    """Types of plugins supported by the mapper service."""

    MAPPING_ENGINE = "mapping_engine"
    MODEL_SERVING = "model_serving"
    VALIDATION = "validation"
    PREPROCESSING = "preprocessing"
    POSTPROCESSING = "postprocessing"


class PluginCapability(BaseResponse):
    """Describes a plugin's capabilities."""

    name: str
    version: str
    description: str
    supported_formats: List[str]
    required_config: Dict[str, Any]
    optional_config: Dict[str, Any] = {}


class PluginMetadata(BaseResponse):
    """Plugin metadata and registration information."""

    plugin_id: str
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    capabilities: List[PluginCapability]
    dependencies: List[str] = []
    config_schema: Dict[str, Any] = {}


class PluginResult(BaseResponse):
    """Result returned by plugin execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    processing_time_ms: float = 0.0


class IPlugin(ABC):
    """Base interface for all plugins."""

    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass


class IMappingEnginePlugin(IPlugin):
    """Interface for mapping engine plugins."""

    @abstractmethod
    async def map_to_canonical(
        self, detector_output: Dict[str, Any], context: Dict[str, Any]
    ) -> PluginResult:
        """Map detector output to canonical taxonomy."""
        pass

    @abstractmethod
    async def map_to_framework(
        self, canonical_result: Dict[str, Any], framework: str, context: Dict[str, Any]
    ) -> PluginResult:
        """Map canonical result to specific compliance framework."""
        pass

    @abstractmethod
    def get_supported_frameworks(self) -> List[str]:
        """Return list of supported compliance frameworks."""
        pass


class IModelServingPlugin(IPlugin):
    """Interface for model serving plugins."""

    @abstractmethod
    async def load_model(self, model_path: str, config: Dict[str, Any]) -> PluginResult:
        """Load a model for serving."""
        pass

    @abstractmethod
    async def predict(self, input_data: Any, model_id: str) -> PluginResult:
        """Make prediction using loaded model."""
        pass

    @abstractmethod
    async def unload_model(self, model_id: str) -> PluginResult:
        """Unload a model from memory."""
        pass

    @abstractmethod
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a loaded model."""
        pass

    @abstractmethod
    def list_loaded_models(self) -> List[str]:
        """List all currently loaded models."""
        pass


class IValidationPlugin(IPlugin):
    """Interface for validation plugins."""

    @abstractmethod
    async def validate_input(self, data: Any, schema: Dict[str, Any]) -> PluginResult:
        """Validate input data against schema."""
        pass

    @abstractmethod
    async def validate_output(self, data: Any, schema: Dict[str, Any]) -> PluginResult:
        """Validate output data against schema."""
        pass

    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules supported by this plugin."""
        pass


class IPreprocessingPlugin(IPlugin):
    """Interface for preprocessing plugins."""

    @abstractmethod
    async def preprocess(self, data: Any, config: Dict[str, Any]) -> PluginResult:
        """Preprocess input data."""
        pass

    @abstractmethod
    def get_preprocessing_steps(self) -> List[str]:
        """Get list of preprocessing steps this plugin performs."""
        pass


class IPostprocessingPlugin(IPlugin):
    """Interface for postprocessing plugins."""

    @abstractmethod
    async def postprocess(self, data: Any, config: Dict[str, Any]) -> PluginResult:
        """Postprocess output data."""
        pass

    @abstractmethod
    def get_postprocessing_steps(self) -> List[str]:
        """Get list of postprocessing steps this plugin performs."""
        pass


# Plugin registry interface
class IPluginRegistry(ABC):
    """Interface for plugin registry."""

    @abstractmethod
    def register_plugin(self, plugin: IPlugin) -> bool:
        """Register a plugin."""
        pass

    @abstractmethod
    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin."""
        pass

    @abstractmethod
    def get_plugin(self, plugin_id: str) -> Optional[IPlugin]:
        """Get a plugin by ID."""
        pass

    @abstractmethod
    def list_plugins(
        self, plugin_type: Optional[PluginType] = None
    ) -> List[PluginMetadata]:
        """List all registered plugins, optionally filtered by type."""
        pass

    @abstractmethod
    def get_plugins_by_capability(self, capability: str) -> List[IPlugin]:
        """Get plugins that support a specific capability."""
        pass
