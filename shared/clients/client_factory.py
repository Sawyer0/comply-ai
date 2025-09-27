"""Client factory for creating service clients with enhanced configuration."""

import os
from typing import Dict, Any, Optional, Type, TypeVar
from dataclasses import dataclass
import logging

from .orchestration_client import OrchestrationServiceClient
from .analysis_client import AnalysisServiceClient
from .mapper_client import MapperServiceClient
from .opa_client import OPAClient
from ..utils.retry import RetryConfig
from ..utils.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ClientConfig:
    """Configuration for service clients."""

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    retry_config: Optional[RetryConfig] = None

    def __post_init__(self):
        if self.retry_config is None:
            self.retry_config = RetryConfig(
                max_attempts=self.max_retries, base_delay=1.0, max_delay=60.0
            )


class ClientFactory:
    """Factory for creating enhanced service clients."""

    def __init__(self, default_config: Optional[ClientConfig] = None):
        self.default_config = default_config or ClientConfig()
        self._clients: Dict[str, Any] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

    def create_orchestration_client(
        self, config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
    ) -> OrchestrationServiceClient:
        """Create an orchestration service client."""
        effective_config = self._merge_config(config)

        base_url = effective_config.base_url or os.getenv(
            "ORCHESTRATION_SERVICE_URL", "http://localhost:8000"
        )
        api_key = effective_config.api_key or os.getenv("ORCHESTRATION_API_KEY")

        client = OrchestrationServiceClient(
            base_url=base_url,
            api_key=api_key,
            timeout=effective_config.timeout,
            max_retries=effective_config.max_retries,
        )

        if effective_config.enable_circuit_breaker:
            self._setup_circuit_breaker(
                "orchestration",
                effective_config.circuit_breaker_threshold,
                effective_config.circuit_breaker_timeout,
            )

        return self._wrap_client(client, "orchestration", effective_config, tenant_id)

    def create_analysis_client(
        self, config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
    ) -> AnalysisServiceClient:
        """Create an analysis service client."""
        effective_config = self._merge_config(config)

        base_url = effective_config.base_url or os.getenv(
            "ANALYSIS_SERVICE_URL", "http://localhost:8001"
        )
        api_key = effective_config.api_key or os.getenv("ANALYSIS_API_KEY")

        client = AnalysisServiceClient(
            base_url=base_url,
            api_key=api_key,
            timeout=effective_config.timeout,
            max_retries=effective_config.max_retries,
        )

        if effective_config.enable_circuit_breaker:
            self._setup_circuit_breaker(
                "analysis",
                effective_config.circuit_breaker_threshold,
                effective_config.circuit_breaker_timeout,
            )

        return self._wrap_client(client, "analysis", effective_config, tenant_id)

    def create_mapper_client(
        self, config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
    ) -> MapperServiceClient:
        """Create a mapper service client."""
        effective_config = self._merge_config(config)

        base_url = effective_config.base_url or os.getenv(
            "MAPPER_SERVICE_URL", "http://localhost:8002"
        )
        api_key = effective_config.api_key or os.getenv("MAPPER_API_KEY")

        client = MapperServiceClient(
            base_url=base_url,
            api_key=api_key,
            timeout=effective_config.timeout,
            max_retries=effective_config.max_retries,
        )

        if effective_config.enable_circuit_breaker:
            self._setup_circuit_breaker(
                "mapper",
                effective_config.circuit_breaker_threshold,
                effective_config.circuit_breaker_timeout,
            )

        return self._wrap_client(client, "mapper", effective_config, tenant_id)

    def create_opa_client(
        self, config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
    ) -> OPAClient:
        """Create an OPA client."""
        effective_config = self._merge_config(config)

        base_url = effective_config.base_url or os.getenv(
            "OPA_ENDPOINT", "http://localhost:8181"
        )

        client = OPAClient(
            base_url=base_url,
            timeout=effective_config.timeout,
            max_retries=effective_config.max_retries,
        )

        if effective_config.enable_circuit_breaker:
            self._setup_circuit_breaker(
                "opa",
                effective_config.circuit_breaker_threshold,
                effective_config.circuit_breaker_timeout,
            )

        return self._wrap_client(client, "opa", effective_config, tenant_id)

    def get_client(self, service_name: str, client_type: Type[T]) -> Optional[T]:
        """Get a cached client instance."""
        return self._clients.get(f"{service_name}_{client_type.__name__}")

    def clear_clients(self):
        """Clear all cached clients."""
        self._clients.clear()
        self._circuit_breakers.clear()

    def _merge_config(self, config: Optional[ClientConfig]) -> ClientConfig:
        """Merge provided config with default config."""
        if config is None:
            return self.default_config

        # Create a new config with default values overridden by provided values
        merged = ClientConfig(
            base_url=config.base_url or self.default_config.base_url,
            api_key=config.api_key or self.default_config.api_key,
            timeout=(
                config.timeout
                if config.timeout != 30.0
                else self.default_config.timeout
            ),
            max_retries=(
                config.max_retries
                if config.max_retries != 3
                else self.default_config.max_retries
            ),
            enable_circuit_breaker=config.enable_circuit_breaker,
            circuit_breaker_threshold=(
                config.circuit_breaker_threshold
                if config.circuit_breaker_threshold != 5
                else self.default_config.circuit_breaker_threshold
            ),
            circuit_breaker_timeout=(
                config.circuit_breaker_timeout
                if config.circuit_breaker_timeout != 60.0
                else self.default_config.circuit_breaker_timeout
            ),
            retry_config=config.retry_config or self.default_config.retry_config,
        )

        return merged

    def _setup_circuit_breaker(self, service_name: str, threshold: int, timeout: float):
        """Setup circuit breaker for a service."""
        if service_name not in self._circuit_breakers:
            self._circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=threshold,
                recovery_timeout=timeout,
                name=f"{service_name}_circuit_breaker",
            )

    def _wrap_client(
        self,
        client: T,
        service_name: str,
        config: ClientConfig,
        tenant_id: Optional[str] = None,
    ) -> T:
        """Wrap client with additional functionality."""
        # Cache the client with tenant-specific key
        tenant_suffix = f"_{tenant_id}" if tenant_id else ""
        client_key = f"{service_name}_{client.__class__.__name__}{tenant_suffix}"
        self._clients[client_key] = client

        # Apply circuit breaker configuration if enabled
        if config.enable_circuit_breaker and service_name in self._circuit_breakers:
            circuit_breaker = self._circuit_breakers[service_name]
            logger.debug(
                "Client wrapped with circuit breaker",
                extra={
                    "service_name": service_name,
                    "tenant_id": tenant_id,
                    "circuit_breaker_state": circuit_breaker.state.name,
                    "failure_count": circuit_breaker.failure_count,
                    "timeout": config.timeout,
                    "max_retries": config.max_retries,
                },
            )
        else:
            logger.debug(
                "Client wrapped without circuit breaker",
                extra={"service_name": service_name, "tenant_id": tenant_id},
            )
        return client


# Global factory instance
default_factory = ClientFactory()


def create_orchestration_client_with_config(
    config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
) -> OrchestrationServiceClient:
    """Create orchestration client with enhanced configuration."""
    return default_factory.create_orchestration_client(config, tenant_id)


def create_analysis_client_with_config(
    config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
) -> AnalysisServiceClient:
    """Create analysis client with enhanced configuration."""
    return default_factory.create_analysis_client(config, tenant_id)


def create_mapper_client_with_config(
    config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
) -> MapperServiceClient:
    """Create mapper client with enhanced configuration."""
    return default_factory.create_mapper_client(config, tenant_id)


def create_opa_client_with_config(
    config: Optional[ClientConfig] = None, tenant_id: Optional[str] = None
) -> OPAClient:
    """Create OPA client with enhanced configuration."""
    return default_factory.create_opa_client(config, tenant_id)
