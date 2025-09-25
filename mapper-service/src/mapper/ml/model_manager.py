"""
Model Manager for Mapper Service

Single Responsibility: Manage ML model lifecycle, loading, and inference.
Handles model versioning, caching, and performance optimization.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TRANSFORMERS_AVAILABLE = False

import structlog

# Import resilience patterns
try:
    from ..resilience import (
        ComprehensiveResilienceManager,
        CircuitBreakerConfig,
        RetryConfig,
        BulkheadConfig,
    )

    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

logger = structlog.get_logger(__name__)


class ModelBackend(str, Enum):
    """Supported model backends."""

    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TGI = "tgi"
    ONNX = "onnx"


class ModelStatus(str, Enum):
    """Model loading status."""

    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


@dataclass
class ModelConfig:
    """Model configuration."""

    model_id: str
    model_path: str
    backend: ModelBackend
    device: str = "auto"
    max_length: int = 2048
    torch_dtype: str = "auto"
    trust_remote_code: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "backend": self.backend.value,
            "device": self.device,
            "max_length": self.max_length,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    model_id: str
    inference_count: int = 0
    total_inference_time: float = 0.0
    avg_inference_time: float = 0.0
    error_count: int = 0
    last_inference: Optional[datetime] = None
    memory_usage_mb: float = 0.0

    def update_inference(self, inference_time: float, success: bool = True):
        """Update inference metrics."""
        self.inference_count += 1
        self.total_inference_time += inference_time
        self.avg_inference_time = self.total_inference_time / self.inference_count
        self.last_inference = datetime.utcnow()

        if not success:
            self.error_count += 1


class ModelManager:
    """
    Model Manager for ML model lifecycle management.

    Single Responsibility: Manage model loading, caching, and inference coordination.

    This class handles:
    - Model loading and unloading
    - Model version management
    - Performance metrics collection
    - Memory management
    - Model health monitoring
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "mapper-models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_status: Dict[str, ModelStatus] = {}

        self.logger = logger.bind(component="model_manager")

        # Initialize resilience patterns
        self._setup_resilience_patterns()

    async def load_model(self, config: ModelConfig) -> bool:
        """
        Load a model with the given configuration.

        Single Responsibility: Load and cache a single model.
        """
        model_id = config.model_id

        if model_id in self.loaded_models:
            self.logger.info("Model already loaded", model_id=model_id)
            return True

        self.model_status[model_id] = ModelStatus.LOADING
        self.logger.info("Loading model", model_id=model_id, backend=config.backend)

        try:
            if config.backend == ModelBackend.TRANSFORMERS:
                model, tokenizer = await self._load_transformers_model(config)
                self.loaded_models[model_id] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "backend": config.backend,
                }
            elif config.backend == ModelBackend.VLLM:
                model = await self._load_vllm_model(config)
                self.loaded_models[model_id] = {
                    "model": model,
                    "backend": config.backend,
                }
            else:
                raise ValueError(f"Unsupported backend: {config.backend}")

            self.model_configs[model_id] = config
            self.model_metrics[model_id] = ModelMetrics(model_id=model_id)
            self.model_status[model_id] = ModelStatus.LOADED

            self.logger.info("Model loaded successfully", model_id=model_id)
            return True

        except (ImportError, RuntimeError, ValueError, OSError) as e:
            self.model_status[model_id] = ModelStatus.ERROR
            self.logger.error("Failed to load model", model_id=model_id, error=str(e))
            return False

    async def _load_transformers_model(self, config: ModelConfig) -> tuple:
        """Load model using Transformers library."""

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers not installed. Install with: pip install transformers"
            )

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Install with: pip install torch")

        # Determine torch dtype
        torch_dtype = (
            torch.float16
            if config.torch_dtype == "auto"
            else getattr(torch, config.torch_dtype)
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=config.trust_remote_code,
            cache_dir=str(self.cache_dir),
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
            cache_dir=str(self.cache_dir),
            device_map=config.device if config.device != "auto" else "auto",
        )

        return model, tokenizer

    async def _load_vllm_model(self, config: ModelConfig):
        """Load model using vLLM."""
        try:
            from vllm import LLM

            model = LLM(
                model=config.model_path,
                tensor_parallel_size=1,
                dtype=config.torch_dtype,
                trust_remote_code=config.trust_remote_code,
                download_dir=str(self.cache_dir),
            )

            return model

        except ImportError as exc:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm"
            ) from exc

    async def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Single Responsibility: Clean up a single model's resources.
        """
        if model_id not in self.loaded_models:
            self.logger.warning("Model not loaded", model_id=model_id)
            return False

        try:
            model_data = self.loaded_models[model_id]

            # Clean up model resources
            if "model" in model_data:
                if hasattr(model_data["model"], "cpu"):
                    model_data["model"].cpu()
                del model_data["model"]

            if "tokenizer" in model_data:
                del model_data["tokenizer"]

            # Clear from memory
            del self.loaded_models[model_id]
            self.model_status[model_id] = ModelStatus.UNLOADED

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("Model unloaded successfully", model_id=model_id)
            return True

        except (RuntimeError, AttributeError) as e:
            self.logger.error("Failed to unload model", model_id=model_id, error=str(e))
            return False

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a loaded model.

        Single Responsibility: Retrieve a loaded model for inference.
        """
        return self.loaded_models.get(model_id)

    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get model loading status."""
        return self.model_status.get(model_id, ModelStatus.UNLOADED)

    def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Get model performance metrics."""
        return self.model_metrics.get(model_id)

    def list_loaded_models(self) -> List[str]:
        """List all loaded model IDs."""
        return list(self.loaded_models.keys())

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all loaded models.

        Single Responsibility: Monitor model health and availability.
        """
        health_status = {
            "healthy": True,
            "loaded_models": len(self.loaded_models),
            "models": {},
        }

        for model_id, model_data in self.loaded_models.items():
            try:
                metrics = self.model_metrics.get(model_id)

                model_health = {
                    "status": self.model_status[model_id].value,
                    "backend": model_data["backend"].value,
                    "inference_count": metrics.inference_count if metrics else 0,
                    "error_count": metrics.error_count if metrics else 0,
                    "avg_inference_time": (
                        metrics.avg_inference_time if metrics else 0.0
                    ),
                    "memory_usage_mb": metrics.memory_usage_mb if metrics else 0.0,
                }

                health_status["models"][model_id] = model_health

            except (KeyError, AttributeError, ValueError) as e:
                health_status["models"][model_id] = {"status": "error", "error": str(e)}
                health_status["healthy"] = False

        return health_status

    async def cleanup(self):
        """Clean up all models and resources."""
        self.logger.info("Cleaning up model manager")

        for model_id in list(self.loaded_models.keys()):
            await self.unload_model(model_id)

        self.loaded_models.clear()
        self.model_configs.clear()
        self.model_metrics.clear()
        self.model_status.clear()

        self.logger.info("Model manager cleanup complete")

    def _setup_resilience_patterns(self) -> None:
        """Setup resilience patterns for model operations."""

        if not RESILIENCE_AVAILABLE:
            self.logger.warning("Resilience patterns not available")
            self.resilience_manager = None
            return

        self.resilience_manager = ComprehensiveResilienceManager()

        # Circuit breaker for model loading
        load_cb_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            expected_exception_types=[ImportError, RuntimeError, OSError],
        )

        # Retry configuration for model operations
        model_retry_config = RetryConfig(
            max_attempts=2,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=True,
        )

        # Bulkhead for model loading (prevent resource exhaustion)
        load_bulkhead_config = BulkheadConfig(
            max_concurrent_calls=2,  # Limit concurrent model loads
            queue_size=10,
            timeout=300.0,  # 5 minutes for model loading
        )

        # Create resilience stack for model loading
        self.load_resilience = self.resilience_manager.create_resilience_stack(
            name="model_loading",
            circuit_breaker_config=load_cb_config,
            retry_config=model_retry_config,
            bulkhead_config=load_bulkhead_config,
        )

        self.logger.info("Resilience patterns configured for model manager")
