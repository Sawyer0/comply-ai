"""
Model serving infrastructure for Analysis Service.

Handles Phi-3-Mini model serving for context-aware risk assessments.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json

from ..shared_integration import get_shared_logger
from ..ml import ModelServer as MLModelServer

logger = get_shared_logger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""

    model_name: str
    model_path: str
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    batch_size: int = 1
    device: str = "cpu"


@dataclass
class InferenceRequest:
    """Model inference request."""

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class InferenceResponse:
    """Model inference response."""

    text: str
    tokens_used: int
    processing_time_ms: float
    model_name: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelServer:
    """Model serving infrastructure for Phi-3-Mini."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger: Any = logger.bind(
            component="model_server", model=config.model_name
        )

        # Underlying unified ML model server
        self._ml_server: Optional[MLModelServer] = None

        # Model state
        self.is_loaded = False
        self.load_time = None

        # Performance tracking
        self.request_count = 0
        self.total_tokens_processed = 0
        self.total_processing_time = 0.0

    async def load_model(self) -> bool:
        """Load the model into memory."""
        try:
            start_time = datetime.utcnow()

            self.logger.info(
                "Initializing ML model server", model_path=self.config.model_path
            )

            # Build ML server configuration from the serving model config
            ml_config = self._build_ml_config()

            ml_server = MLModelServer(ml_config)
            await ml_server.initialize()

            self._ml_server = ml_server
            self.is_loaded = True
            self.load_time = datetime.utcnow()

            load_duration = (self.load_time - start_time).total_seconds()

            self.logger.info(
                "Model loaded successfully",
                load_time_seconds=load_duration,
                model_name=self.config.model_name,
            )

            return True

        except Exception as e:
            self.logger.error("Failed to load model", error=str(e))
            return False

    async def unload_model(self) -> None:
        """Unload the model from memory."""
        try:
            self.logger.info("Unloading model")

            # Clear model references
            self._ml_server = None
            self.is_loaded = False
            self.load_time = None

            # Force garbage collection
            import gc

            gc.collect()

            self.logger.info("Model unloaded successfully")

        except Exception as e:
            self.logger.error("Error unloading model", error=str(e))

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text using the loaded model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = datetime.utcnow()

        try:
            self.logger.debug(
                "Processing inference request", prompt_length=len(request.prompt)
            )

            # Use request parameters or fall back to config defaults
            max_tokens = request.max_tokens or self.config.max_tokens
            temperature = request.temperature or self.config.temperature
            top_p = request.top_p or self.config.top_p

            if not self._ml_server:
                raise RuntimeError("ML model server not initialized")

            # Delegate generation to the unified ML model server
            ml_result = await self._ml_server.generate_analysis(
                request.prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=request.stop_sequences,
            )

            generated_text = ml_result.get("generated_text", "")
            tokens_used = int(
                ml_result.get("tokens_generated")
                or len(generated_text.split())
            )

            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds() * 1000

            # Update statistics
            self.request_count += 1
            self.total_tokens_processed += tokens_used
            self.total_processing_time += processing_time

            response = InferenceResponse(
                text=generated_text,
                tokens_used=tokens_used,
                processing_time_ms=processing_time,
                model_name=self.config.model_name,
                confidence=None,
                metadata={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "backend_metadata": ml_result.get("metadata", {}),
                },
            )

            self.logger.debug(
                "Inference completed",
                tokens_used=tokens_used,
                processing_time_ms=processing_time,
            )

            return response

        except Exception as e:
            self.logger.error("Inference failed", error=str(e))
            raise

    async def batch_generate(
        self, requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Generate text for multiple requests in batch."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.logger.info("Processing batch inference", batch_size=len(requests))

        # For simplicity, process sequentially (in production, would batch)
        responses = []
        for request in requests:
            try:
                response = await self.generate(request)
                responses.append(response)
            except Exception as e:
                # Create error response
                error_response = InferenceResponse(
                    text="",
                    tokens_used=0,
                    processing_time_ms=0.0,
                    model_name=self.config.model_name,
                    metadata={"error": str(e)},
                )
                responses.append(error_response)

        return responses

    def get_stats(self) -> Dict[str, Any]:
        """Get model server statistics."""
        avg_processing_time = (
            self.total_processing_time / self.request_count
            if self.request_count > 0
            else 0.0
        )

        avg_tokens_per_request = (
            self.total_tokens_processed / self.request_count
            if self.request_count > 0
            else 0.0
        )

        return {
            "model_name": self.config.model_name,
            "is_loaded": self.is_loaded,
            "load_time": self.load_time.isoformat() if self.load_time else None,
            "request_count": self.request_count,
            "total_tokens_processed": self.total_tokens_processed,
            "avg_processing_time_ms": avg_processing_time,
            "avg_tokens_per_request": avg_tokens_per_request,
            "config": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "device": self.config.device,
            },
        }

    def _build_ml_config(self) -> Dict[str, Any]:
        """Build configuration dictionary for the unified ML ModelServer.

        This maps the serving-level ModelConfig into the generic ML
        configuration structure used by analysis.ml.model_server.ModelServer
        without changing the external serving API.
        """

        ml_section: Dict[str, Any] = {
            "model_name": self.config.model_name,
            # Use CPU backend by default; more advanced backends (vLLM/TGI)
            # can be enabled via the ML configuration in the future.
            "preferred_backend": "cpu",
            "fallback_backends": ["cpu"],
            "max_length": self.config.max_tokens,
            "temperature": self.config.temperature,
            "do_sample": True,
            # Preserve path so future backends can use it for loading.
            "model_path": self.config.model_path,
        }

        return {"ml": ml_section}


class ModelManager:
    """Manages multiple model servers."""

    def __init__(self):
        self.logger: Any = logger.bind(component="model_manager")
        self.servers: Dict[str, ModelServer] = {}

    async def add_model(self, name: str, config: ModelConfig) -> bool:
        """Add a new model server."""
        try:
            if name in self.servers:
                self.logger.warning("Model already exists", model_name=name)
                return False

            server = ModelServer(config)
            success = await server.load_model()

            if success:
                self.servers[name] = server
                self.logger.info("Model added successfully", model_name=name)
                return True
            else:
                self.logger.error("Failed to add model", model_name=name)
                return False

        except Exception as e:
            self.logger.error("Error adding model", model_name=name, error=str(e))
            return False

    async def remove_model(self, name: str) -> bool:
        """Remove a model server."""
        try:
            if name not in self.servers:
                self.logger.warning("Model not found", model_name=name)
                return False

            server = self.servers[name]
            await server.unload_model()
            del self.servers[name]

            self.logger.info("Model removed successfully", model_name=name)
            return True

        except Exception as e:
            self.logger.error("Error removing model", model_name=name, error=str(e))
            return False

    async def generate(
        self, model_name: str, request: InferenceRequest
    ) -> InferenceResponse:
        """Generate using specified model."""
        if model_name not in self.servers:
            raise ValueError(f"Model '{model_name}' not found")

        return await self.servers[model_name].generate(request)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        return [
            {"name": name, "stats": server.get_stats()}
            for name, server in self.servers.items()
        ]

    def get_model_stats(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific model."""
        if model_name in self.servers:
            return self.servers[model_name].get_stats()
        return None
