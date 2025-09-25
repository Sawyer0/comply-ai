"""
Resilience factory for creating and configuring analysis service resilience components.

This module provides factory methods for creating resilience components
with proper configuration and dependency injection specific to analysis operations.
"""

import logging
from typing import Any, Dict, Optional

from .circuit_breaker import AnalysisCircuitBreaker
from .config import CircuitBreakerConfig, RetryConfig
from .interfaces import ICircuitBreaker, IResilienceManager, IRetryManager
from .retry_manager import AnalysisRetryManager

logger = logging.getLogger(__name__)


class AnalysisResilienceFactory:
    """Factory for creating analysis service resilience components."""

    @staticmethod
    def create_circuit_breaker(
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        metrics_collector: Optional[Any] = None,
    ) -> ICircuitBreaker:
        """
        Create a circuit breaker with configuration.

        Args:
            name: Circuit breaker name
            config: Circuit breaker configuration
            metrics_collector: Metrics collector for monitoring

        Returns:
            Configured circuit breaker
        """
        if config is None:
            config = CircuitBreakerConfig()

        return AnalysisCircuitBreaker(
            name=name, config=config, metrics_collector=metrics_collector
        )

    @staticmethod
    def create_retry_manager(
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
        metrics_collector: Optional[Any] = None,
    ) -> IRetryManager:
        """
        Create a retry manager with configuration.

        Args:
            config: Retry configuration
            circuit_breaker: Circuit breaker instance
            metrics_collector: Metrics collector for monitoring

        Returns:
            Configured retry manager
        """
        if config is None:
            config = RetryConfig()

        return AnalysisRetryManager(
            config=config,
            circuit_breaker=circuit_breaker,
            metrics_collector=metrics_collector,
        )

    @staticmethod
    def create_resilience_stack(
        service_name: str,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        metrics_collector: Optional[Any] = None,
    ) -> tuple[ICircuitBreaker, IRetryManager]:
        """
        Create a complete resilience stack with configuration.

        Args:
            service_name: Name of the service
            retry_config: Retry configuration
            circuit_breaker_config: Circuit breaker configuration
            metrics_collector: Metrics collector for monitoring

        Returns:
            Tuple of (circuit_breaker, retry_manager)
        """
        if circuit_breaker_config is None:
            circuit_breaker_config = CircuitBreakerConfig()

        if retry_config is None:
            retry_config = RetryConfig()

        circuit_breaker = AnalysisResilienceFactory.create_circuit_breaker(
            name=service_name,
            config=circuit_breaker_config,
            metrics_collector=metrics_collector,
        )

        retry_manager = AnalysisResilienceFactory.create_retry_manager(
            config=retry_config,
            circuit_breaker=circuit_breaker,
            metrics_collector=metrics_collector,
        )

        logger.info("Analysis resilience stack created for service '%s'", service_name)
        return circuit_breaker, retry_manager


class AnalysisResilienceManager(IResilienceManager):
    """
    Manager for multiple analysis service resilience components.

    Provides centralized management and monitoring of resilience components
    across different analysis operations and components.
    """

    def __init__(self, metrics_collector: Optional[Any] = None):
        """
        Initialize resilience manager.

        Args:
            metrics_collector: Metrics collector for monitoring
        """
        self._circuit_breakers: Dict[str, ICircuitBreaker] = {}
        self._retry_managers: Dict[str, IRetryManager] = {}
        self._metrics_collector = metrics_collector

        logger.info("Analysis resilience manager initialized")

    def create_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> ICircuitBreaker:
        """Create a new circuit breaker."""
        if name in self._circuit_breakers:
            logger.warning(
                "Circuit breaker '%s' already exists, returning existing instance", name
            )
            return self._circuit_breakers[name]

        circuit_breaker = AnalysisResilienceFactory.create_circuit_breaker(
            name=name, config=config, metrics_collector=self._metrics_collector
        )

        self._circuit_breakers[name] = circuit_breaker
        logger.info("Analysis circuit breaker '%s' created", name)

        return circuit_breaker

    def get_circuit_breaker(self, name: str) -> Optional[ICircuitBreaker]:
        """Get circuit breaker by name."""
        return self._circuit_breakers.get(name)

    def create_retry_manager(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
    ) -> IRetryManager:
        """Create a new retry manager."""
        if name in self._retry_managers:
            logger.warning(
                "Retry manager '%s' already exists, returning existing instance", name
            )
            return self._retry_managers[name]

        retry_manager = AnalysisResilienceFactory.create_retry_manager(
            config=config,
            circuit_breaker=circuit_breaker,
            metrics_collector=self._metrics_collector,
        )

        self._retry_managers[name] = retry_manager
        logger.info("Analysis retry manager '%s' created", name)

        return retry_manager

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all resilience components."""
        stats = {"circuit_breakers": {}, "retry_managers": {}}

        for name, cb in self._circuit_breakers.items():
            stats["circuit_breakers"][name] = cb.get_statistics()

        for name, rm in self._retry_managers.items():
            stats["retry_managers"][name] = rm.get_statistics()

        return stats

    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for cb in self._circuit_breakers.values():
            cb.reset()
        logger.info("All analysis circuit breakers reset")
