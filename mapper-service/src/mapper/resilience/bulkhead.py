"""
Bulkhead isolation pattern for mapper service.

This module provides bulkhead isolation to prevent resource exhaustion
and ensure that failures in one part of the mapper system don't
affect other parts.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, TypeVar
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""

    max_concurrent_calls: int = 10
    queue_size: int = 100
    timeout: float = 30.0

    def __post_init__(self):
        """Validate configuration."""
        if self.max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be at least 1")
        if self.queue_size < 0:
            raise ValueError("queue_size must be non-negative")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


class BulkheadIsolation:
    """
    Bulkhead isolation implementation for mapper operations.

    Provides resource isolation to prevent one failing component
    from exhausting resources needed by other components.
    """

    def __init__(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
        metrics_collector: Optional[Any] = None,
    ):
        """
        Initialize bulkhead isolation.

        Args:
            name: Name of the bulkhead
            config: Bulkhead configuration
            metrics_collector: Metrics collector for monitoring
        """
        self._name = name
        self._config = config or BulkheadConfig()
        self._metrics_collector = metrics_collector

        # Create semaphore for limiting concurrent calls
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_calls)

        # Statistics
        self._total_calls = 0
        self._active_calls = 0
        self._rejected_calls = 0
        self._completed_calls = 0
        self._failed_calls = 0

        logger.info(
            "Bulkhead isolation '%s' initialized with max_concurrent_calls=%d",
            name,
            self._config.max_concurrent_calls,
        )

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with bulkhead isolation.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            asyncio.TimeoutError: If operation times out
            RuntimeError: If bulkhead is at capacity
        """
        self._total_calls += 1

        # Try to acquire semaphore with timeout
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._config.timeout
            )
        except asyncio.TimeoutError:
            self._rejected_calls += 1
            logger.warning(
                "Bulkhead '%s' rejected call - at capacity (%d concurrent calls)",
                self._name,
                self._config.max_concurrent_calls,
            )
            raise RuntimeError(f"Bulkhead '{self._name}' at capacity")

        self._active_calls += 1

        try:
            # Execute the function
            result = await func(*args, **kwargs)
            self._completed_calls += 1

            logger.debug(
                "Bulkhead '%s' completed call successfully (active: %d)",
                self._name,
                self._active_calls,
            )

            return result

        except Exception as e:
            self._failed_calls += 1
            logger.warning(
                "Bulkhead '%s' call failed: %s (active: %d)",
                self._name,
                e,
                self._active_calls,
            )
            raise

        finally:
            self._active_calls -= 1
            self._semaphore.release()

    def get_statistics(self) -> Dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "name": self._name,
            "total_calls": self._total_calls,
            "active_calls": self._active_calls,
            "completed_calls": self._completed_calls,
            "failed_calls": self._failed_calls,
            "rejected_calls": self._rejected_calls,
            "success_rate": (
                self._completed_calls / self._total_calls
                if self._total_calls > 0
                else 0
            ),
            "rejection_rate": (
                self._rejected_calls / self._total_calls if self._total_calls > 0 else 0
            ),
            "config": {
                "max_concurrent_calls": self._config.max_concurrent_calls,
                "queue_size": self._config.queue_size,
                "timeout": self._config.timeout,
            },
        }

    @property
    def name(self) -> str:
        """Bulkhead name."""
        return self._name

    @property
    def active_calls(self) -> int:
        """Number of active calls."""
        return self._active_calls

    @property
    def available_capacity(self) -> int:
        """Available capacity."""
        return self._config.max_concurrent_calls - self._active_calls


class BulkheadManager:
    """Manager for multiple bulkhead isolations."""

    def __init__(self, metrics_collector: Optional[Any] = None):
        """
        Initialize bulkhead manager.

        Args:
            metrics_collector: Metrics collector for monitoring
        """
        self._bulkheads: Dict[str, BulkheadIsolation] = {}
        self._metrics_collector = metrics_collector

        logger.info("Bulkhead manager initialized")

    def create_bulkhead(
        self,
        name: str,
        config: Optional[BulkheadConfig] = None,
    ) -> BulkheadIsolation:
        """Create a new bulkhead isolation."""
        if name in self._bulkheads:
            logger.warning(
                "Bulkhead '%s' already exists, returning existing instance", name
            )
            return self._bulkheads[name]

        bulkhead = BulkheadIsolation(
            name=name,
            config=config,
            metrics_collector=self._metrics_collector,
        )

        self._bulkheads[name] = bulkhead
        logger.info("Bulkhead '%s' created", name)

        return bulkhead

    def get_bulkhead(self, name: str) -> Optional[BulkheadIsolation]:
        """Get bulkhead by name."""
        return self._bulkheads.get(name)

    def get_all_statistics(self) -> Dict[str, Any]:
        """Get statistics for all bulkheads."""
        return {
            name: bulkhead.get_statistics()
            for name, bulkhead in self._bulkheads.items()
        }
