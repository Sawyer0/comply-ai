"""
Comprehensive resilience manager for mapper service.

This module provides a unified manager that coordinates all resilience patterns
including circuit breakers, retry logic, bulkhead isolation, and fallback mechanisms.
"""

import logging
from typing import Any, Dict, Optional

from .bulkhead import BulkheadConfig, BulkheadManager
from .circuit_breaker import MapperCircuitBreaker
from .config import CircuitBreakerConfig, RetryConfig
from .factory import MapperResilienceFactory
from .fallback import MapperFallbackManager, FallbackConfig
from .interfaces import ICircuitBreaker, IRetryManager

logger = logging.getLogger(__name__)


class ComprehensiveResilienceManager:
    """
    Comprehensive resilience manager that coordinates all resilience patterns.

    This manager provides a unified interface for creating and managing
    circuit breakers, retry managers, bulkhead isolation, and fallback mechanisms
    for mapper service operations.
    """

    def __init__(self, metrics_collector: Optional[Any] = None):
        """
        Initialize comprehensive resilience manager.

        Args:
            metrics_collector: Metrics collector for monitoring
        """
        self._metrics_collector = metrics_collector

        # Individual managers
        self._circuit_breakers: Dict[str, ICircuitBreaker] = {}
        self._retry_managers: Dict[str, IRetryManager] = {}
        self._bulkhead_manager = BulkheadManager(metrics_collector)
        self._fallback_managers: Dict[str, MapperFallbackManager] = {}

        logger.info("Comprehensive resilience manager initialized")

    def create_resilience_stack(
        self,
        name: str,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        bulkhead_config: Optional[BulkheadConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
    ) -> Dict[str, Any]:
        """
        Create a complete resilience stack for a service.

        Args:
            name: Service name
            circuit_breaker_config: Circuit breaker configuration
            retry_config: Retry configuration
            bulkhead_config: Bulkhead configuration
            fallback_config: Fallback configuration

        Returns:
            Dictionary containing all resilience components
        """
        components = {}

        # Create circuit breaker
        circuit_breaker = self.create_circuit_breaker(name, circuit_breaker_config)
        components["circuit_breaker"] = circuit_breaker

        # Create retry manager
        retry_manager = self.create_retry_manager(name, retry_config, circuit_breaker)
        components["retry_manager"] = retry_manager

        # Create bulkhead isolation
        if bulkhead_config:
            bulkhead = self._bulkhead_manager.create_bulkhead(name, bulkhead_config)
            components["bulkhead"] = bulkhead

        # Create fallback manager
        if fallback_config:
            fallback_manager = self.create_fallback_manager(name, fallback_config)
            components["fallback_manager"] = fallback_manager

        logger.info("Complete resilience stack created for service '%s'", name)
        return components

    def create_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> ICircuitBreaker:
        """Create a circuit breaker."""
        if name in self._circuit_breakers:
            logger.warning(
                "Circuit breaker '%s' already exists, returning existing instance", name
            )
            return self._circuit_breakers[name]

        circuit_breaker = MapperResilienceFactory.create_circuit_breaker(
            name=name,
            config=config,
            metrics_collector=self._metrics_collector,
        )

        self._circuit_breakers[name] = circuit_breaker
        logger.info("Circuit breaker '%s' created", name)

        return circuit_breaker

    def create_retry_manager(
        self,
        name: str,
        config: Optional[RetryConfig] = None,
        circuit_breaker: Optional[ICircuitBreaker] = None,
    ) -> IRetryManager:
        """Create a retry manager."""
        if name in self._retry_managers:
            logger.warning(
                "Retry manager '%s' already exists, returning existing instance", name
            )
            return self._retry_managers[name]

        retry_manager = MapperResilienceFactory.create_retry_manager(
            config=config,
            circuit_breaker=circuit_breaker,
            metrics_collector=self._metrics_collector,
        )

        self._retry_managers[name] = retry_manager
        logger.info("Retry manager '%s' created", name)

        return retry_manager

    def create_fallback_manager(
        self,
        name: str,
        config: Optional[FallbackConfig] = None,
    ) -> MapperFallbackManager:
        """Create a fallback manager."""
        if name in self._fallback_managers:
            logger.warning(
                "Fallback manager '%s' already exists, returning existing instance",
                name,
            )
            return self._fallback_managers[name]

        fallback_manager = MapperFallbackManager(
            name=name,
            config=config,
            metrics_collector=self._metrics_collector,
        )

        self._fallback_managers[name] = fallback_manager
        logger.info("Fallback manager '%s' created", name)

        return fallback_manager

    def get_circuit_breaker(self, name: str) -> Optional[ICircuitBreaker]:
        """Get circuit breaker by name."""
        return self._circuit_breakers.get(name)

    def get_retry_manager(self, name: str) -> Optional[IRetryManager]:
        """Get retry manager by name."""
        return self._retry_managers.get(name)

    def get_fallback_manager(self, name: str) -> Optional[MapperFallbackManager]:
        """Get fallback manager by name."""
        return self._fallback_managers.get(name)

    def get_bulkhead_manager(self) -> BulkheadManager:
        """Get bulkhead manager."""
        return self._bulkhead_manager

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all resilience components."""
        stats = {
            "circuit_breakers": {},
            "retry_managers": {},
            "bulkheads": {},
            "fallback_managers": {},
        }

        # Circuit breaker statistics
        for name, cb in self._circuit_breakers.items():
            stats["circuit_breakers"][name] = cb.get_statistics()

        # Retry manager statistics
        for name, rm in self._retry_managers.items():
            stats["retry_managers"][name] = rm.get_statistics()

        # Bulkhead statistics
        stats["bulkheads"] = self._bulkhead_manager.get_all_statistics()

        # Fallback manager statistics
        for name, fm in self._fallback_managers.items():
            stats["fallback_managers"][name] = fm.get_statistics()

        return stats

    def reset_all(self) -> None:
        """Reset all resilience components."""
        # Reset circuit breakers
        for cb in self._circuit_breakers.values():
            cb.reset()

        logger.info("All resilience components reset")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all resilience components."""
        health = {
            "status": "healthy",
            "components": {
                "circuit_breakers": {},
                "bulkheads": {},
                "fallback_managers": {},
            },
        }

        # Check circuit breakers
        for name, cb in self._circuit_breakers.items():
            stats = cb.get_statistics()
            health["components"]["circuit_breakers"][name] = {
                "state": stats["state"],
                "failure_rate": stats["failure_rate"],
                "healthy": stats["state"] != "open" and stats["failure_rate"] < 0.5,
            }

        # Check bulkheads
        bulkhead_stats = self._bulkhead_manager.get_all_statistics()
        for name, stats in bulkhead_stats.items():
            health["components"]["bulkheads"][name] = {
                "active_calls": stats["active_calls"],
                "rejection_rate": stats["rejection_rate"],
                "healthy": stats["rejection_rate"] < 0.1,
            }

        # Check fallback managers
        for name, fm in self._fallback_managers.items():
            stats = fm.get_statistics()
            health["components"]["fallback_managers"][name] = {
                "fallback_rate": stats["fallback_rate"],
                "fallback_success_rate": stats["fallback_success_rate"],
                "healthy": (
                    stats["fallback_success_rate"] > 0.8
                    if stats["fallback_calls"] > 0
                    else True
                ),
            }

        # Determine overall health
        all_healthy = all(
            component.get("healthy", True)
            for component_type in health["components"].values()
            for component in component_type.values()
        )

        health["status"] = "healthy" if all_healthy else "degraded"

        return health
