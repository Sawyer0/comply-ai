"""
Fallback pattern implementation for analysis service.

This module provides fallback mechanisms to ensure graceful degradation
when primary analysis operations fail, maintaining service availability
with reduced functionality.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanisms."""

    enable_fallback: bool = True
    max_fallback_attempts: int = 3
    fallback_timeout: float = 10.0
    log_fallback_usage: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.max_fallback_attempts < 1:
            raise ValueError("max_fallback_attempts must be at least 1")
        if self.fallback_timeout <= 0:
            raise ValueError("fallback_timeout must be positive")


class IFallbackStrategy(ABC):
    """Interface for fallback strategies."""

    @abstractmethod
    async def execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback logic."""
        ...

    @abstractmethod
    def can_handle(self, exception: Exception) -> bool:
        """Check if this strategy can handle the given exception."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        ...


class SimpleFallbackStrategy(IFallbackStrategy):
    """Simple fallback strategy that returns a default value."""

    def __init__(self, name: str, default_value: Any):
        """
        Initialize simple fallback strategy.

        Args:
            name: Strategy name
            default_value: Default value to return
        """
        self._name = name
        self._default_value = default_value

    async def execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback logic."""
        logger.info("Using simple fallback strategy '%s'", self._name)
        return self._default_value

    def can_handle(self, exception: Exception) -> bool:
        """Check if this strategy can handle the given exception."""
        # Simple strategy can handle any exception
        return True

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name


class FunctionFallbackStrategy(IFallbackStrategy):
    """Fallback strategy that executes an alternative function."""

    def __init__(
        self,
        name: str,
        fallback_func: Callable,
        exception_types: Optional[tuple] = None,
    ):
        """
        Initialize function fallback strategy.

        Args:
            name: Strategy name
            fallback_func: Function to execute as fallback
            exception_types: Exception types this strategy can handle
        """
        self._name = name
        self._fallback_func = fallback_func
        self._exception_types = exception_types or (Exception,)

    async def execute_fallback(self, *args, **kwargs) -> Any:
        """Execute fallback logic."""
        logger.info("Using function fallback strategy '%s'", self._name)

        # Execute fallback function
        if asyncio.iscoroutinefunction(self._fallback_func):
            return await self._fallback_func(*args, **kwargs)
        else:
            return self._fallback_func(*args, **kwargs)

    def can_handle(self, exception: Exception) -> bool:
        """Check if this strategy can handle the given exception."""
        return isinstance(exception, self._exception_types)

    @property
    def name(self) -> str:
        """Strategy name."""
        return self._name


class AnalysisFallbackManager:
    """
    Fallback manager for analysis operations.

    Provides graceful degradation when primary analysis operations fail,
    ensuring service availability with reduced functionality.
    """

    def __init__(
        self,
        name: str,
        config: Optional[FallbackConfig] = None,
        metrics_collector: Optional[Any] = None,
    ):
        """
        Initialize fallback manager.

        Args:
            name: Manager name
            config: Fallback configuration
            metrics_collector: Metrics collector for monitoring
        """
        self._name = name
        self._config = config or FallbackConfig()
        self._metrics_collector = metrics_collector

        # Fallback strategies (ordered by priority)
        self._strategies: List[IFallbackStrategy] = []

        # Statistics
        self._total_calls = 0
        self._fallback_calls = 0
        self._fallback_successes = 0
        self._fallback_failures = 0

        logger.info("Analysis fallback manager '%s' initialized", name)

    def add_strategy(self, strategy: IFallbackStrategy) -> None:
        """Add a fallback strategy."""
        self._strategies.append(strategy)
        logger.info(
            "Added fallback strategy '%s' to manager '%s'", strategy.name, self._name
        )

    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a fallback strategy by name."""
        for i, strategy in enumerate(self._strategies):
            if strategy.name == strategy_name:
                del self._strategies[i]
                logger.info(
                    "Removed fallback strategy '%s' from manager '%s'",
                    strategy_name,
                    self._name,
                )
                return True
        return False

    async def execute_with_fallback(
        self, primary_func: Callable[..., T], *args, **kwargs
    ) -> T:
        """
        Execute function with fallback support.

        Args:
            primary_func: Primary function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result
        """
        self._total_calls += 1

        # Try primary function first
        try:
            result = await self._execute_primary(primary_func, *args, **kwargs)
            logger.debug(
                "Primary function succeeded in fallback manager '%s'", self._name
            )
            return result

        except Exception as e:
            logger.warning(
                "Primary function failed in fallback manager '%s': %s", self._name, e
            )

            if not self._config.enable_fallback:
                logger.info("Fallback disabled, re-raising exception")
                raise

            # Try fallback strategies
            return await self._execute_fallback(e, *args, **kwargs)

    async def _execute_primary(
        self, primary_func: Callable[..., T], *args, **kwargs
    ) -> T:
        """Execute primary function."""
        if asyncio.iscoroutinefunction(primary_func):
            return await primary_func(*args, **kwargs)
        else:
            return primary_func(*args, **kwargs)

    async def _execute_fallback(self, exception: Exception, *args, **kwargs) -> Any:
        """Execute fallback strategies."""
        self._fallback_calls += 1

        # Find suitable fallback strategy
        for strategy in self._strategies:
            if strategy.can_handle(exception):
                try:
                    result = await strategy.execute_fallback(*args, **kwargs)
                    self._fallback_successes += 1

                    if self._config.log_fallback_usage:
                        logger.info(
                            "Fallback strategy '%s' succeeded in manager '%s'",
                            strategy.name,
                            self._name,
                        )

                    return result

                except Exception as fallback_error:
                    logger.warning(
                        "Fallback strategy '%s' failed in manager '%s': %s",
                        strategy.name,
                        self._name,
                        fallback_error,
                    )
                    continue

        # No fallback strategy worked
        self._fallback_failures += 1
        logger.error("All fallback strategies failed in manager '%s'", self._name)
        raise exception

    def get_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        return {
            "name": self._name,
            "total_calls": self._total_calls,
            "fallback_calls": self._fallback_calls,
            "fallback_successes": self._fallback_successes,
            "fallback_failures": self._fallback_failures,
            "fallback_rate": (
                self._fallback_calls / self._total_calls if self._total_calls > 0 else 0
            ),
            "fallback_success_rate": (
                self._fallback_successes / self._fallback_calls
                if self._fallback_calls > 0
                else 0
            ),
            "strategies": [strategy.name for strategy in self._strategies],
            "config": {
                "enable_fallback": self._config.enable_fallback,
                "max_fallback_attempts": self._config.max_fallback_attempts,
                "fallback_timeout": self._config.fallback_timeout,
            },
        }

    @property
    def name(self) -> str:
        """Manager name."""
        return self._name


# Import asyncio at the top level to avoid issues
import asyncio


def with_fallback(
    strategies: List[IFallbackStrategy], config: Optional[FallbackConfig] = None
):
    """
    Decorator to add fallback support to a function.

    Args:
        strategies: List of fallback strategies
        config: Fallback configuration

    Returns:
        Decorated function with fallback support
    """

    def decorator(func):
        # Create fallback manager
        manager = AnalysisFallbackManager(name=func.__name__, config=config)

        # Add strategies
        for strategy in strategies:
            manager.add_strategy(strategy)

        async def wrapper(*args, **kwargs):
            return await manager.execute_with_fallback(func, *args, **kwargs)

        return wrapper

    return decorator
