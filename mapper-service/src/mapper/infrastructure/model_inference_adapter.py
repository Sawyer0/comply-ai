"""Infrastructure adapter for model-based canonical mapping.

Implements the ModelInferencePort using the service's ModelServer and
resilience stack so that the core mapper depends only on a port.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..config.settings import MapperSettings
from ..core.ports import ModelInferencePort
from ..serving.model_server import ModelServer, create_model_server, GenerationConfig
from ..resilience import (
    ComprehensiveResilienceManager,
    CircuitBreakerConfig,
    RetryConfig,
    BulkheadConfig,
    FallbackConfig,
)
from ..shared_integration import get_shared_logger


logger = get_shared_logger(__name__)


class SharedModelInferenceAdapter(ModelInferencePort):
    """Adapter that provides model-based canonical mapping.

    This adapter owns the concrete ModelServer instance and resilience
    stack. CoreMapper interacts with it only through the
    ModelInferencePort abstraction.
    """

    def __init__(self, settings: MapperSettings) -> None:
        self._settings = settings
        self._model_server: Optional[ModelServer] = None
        self._resilience_manager = ComprehensiveResilienceManager()
        self._model_resilience: Dict[str, Any] = self._create_resilience_stack()

    def _create_resilience_stack(self) -> Dict[str, Any]:
        """Create a resilience stack for model inference calls."""
        model_server_cb_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=30.0,
            expected_exception=(ConnectionError, TimeoutError, RuntimeError),
        )

        model_retry_config = RetryConfig(
            max_attempts=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_enabled=True,
        )

        model_bulkhead_config = BulkheadConfig(
            max_concurrent_calls=10,
            queue_size=50,
            timeout=30.0,
        )

        model_fallback_config = FallbackConfig(
            fallback_timeout=5.0,
            enable_fallback=True,
        )

        return self._resilience_manager.create_resilience_stack(
            name="model_server",
            circuit_breaker_config=model_server_cb_config,
            retry_config=model_retry_config,
            bulkhead_config=model_bulkhead_config,
            fallback_config=model_fallback_config,
        )

    async def initialize(self) -> None:
        """Initialize the underlying model server if needed."""
        if self._model_server is not None:
            return

        try:
            self._model_server = create_model_server(
                backend=self._settings.model_backend,
                model_path=self._settings.model_path,
                generation_config=GenerationConfig(
                    temperature=self._settings.temperature,
                    top_p=self._settings.top_p,
                    max_new_tokens=self._settings.max_new_tokens,
                ),
                **self._settings.backend_kwargs,
            )

            await self._model_server.load_model()

            logger.info(
                "Model inference adapter initialized",
                backend=self._settings.model_backend,
                model_path=self._settings.model_path,
            )
        except (ImportError, RuntimeError, ValueError) as e:
            logger.error("Failed to initialize model inference adapter", error=str(e))
            # Leave model server as None - caller will fall back as needed
            self._model_server = None

    async def is_available(self) -> bool:  # type: ignore[override]
        """Return True if the model server is initialized."""
        return self._model_server is not None

    async def generate_mapping(  # type: ignore[override]
        self,
        detector: str,
        output: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a canonical mapping JSON string via the model server."""
        if self._model_server is None:
            raise RuntimeError("Model server not available")

        async def _model_call() -> str:
            return await self._model_server.generate_mapping(
                detector=detector,
                output=output,
                metadata=metadata,
            )

        circuit_breaker = self._model_resilience.get("circuit_breaker")
        retry_manager = self._model_resilience.get("retry_manager")

        try:
            if circuit_breaker and retry_manager:
                return await retry_manager.execute_with_retry(
                    lambda: circuit_breaker.call(_model_call)
                )
            return await _model_call()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(
                "Model inference call failed",
                error=str(e),
                detector=detector,
            )
            # Propagate so core can decide on fallback behaviour
            raise

    async def health_check(self) -> bool:  # type: ignore[override]
        """Check health of the underlying model server."""
        if self._model_server is None:
            return False

        try:
            return bool(await self._model_server.health_check())
        except (RuntimeError, ConnectionError, TimeoutError) as e:  # pragma: no cover
            logger.warning("Model server health check failed", error=str(e))
            return False

    async def shutdown(self) -> None:  # type: ignore[override]
        """Cleanly shut down model resources."""
        if self._model_server is None:
            return

        try:
            if hasattr(self._model_server, "close"):
                await self._model_server.close()
        except (RuntimeError, ConnectionError) as e:  # pragma: no cover
            logger.error("Error during model server shutdown", error=str(e))
        finally:
            self._model_server = None
