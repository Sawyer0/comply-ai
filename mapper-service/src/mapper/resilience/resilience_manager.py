"""
Resilience Manager for comprehensive resilience orchestration.

This module provides a centralized resilience manager that coordinates
all resilience patterns including circuit breakers, retry logic, and
bulkhead isolation.
"""

from typing import Optional, Any, Dict, Callable
import structlog

from .circuit_breaker import MapperCircuitBreaker
from .retry_manager import MapperRetryManager
from .bulkhead import BulkheadManager
from .fallback import MapperFallbackManager

logger = structlog.get_logger(__name__)


class ResilienceManager:
    """
    Centralized resilience manager that orchestrates all resilience patterns.
    
    This class provides a unified interface for resilience operations including
    circuit breakers, retry logic, bulkhead isolation, and fallback strategies.
    """
    
    def __init__(
        self,
        database_manager: Optional[Any] = None,
        connection_pool: Optional[Any] = None,
    ):
        """
        Initialize the resilience manager.
        
        Args:
            database_manager: Database manager instance
            connection_pool: Connection pool manager instance
        """
        self.database_manager = database_manager
        self.connection_pool = connection_pool
        
        # Initialize circuit breaker
        self.circuit_breaker = MapperCircuitBreaker(
            name="api_request",
            failure_threshold=5,
            timeout_duration=60,
            expected_exception=Exception
        )
        
        # Initialize retry manager
        self.retry_manager = MapperRetryManager(
            max_retries=3,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )
        
        # Initialize bulkhead manager
        self.bulkhead_manager = BulkheadManager(
            max_concurrent_requests=100,
            max_wait_time=30.0
        )
        
        # Initialize fallback manager
        self.fallback_manager = MapperFallbackManager()
        
        logger.info("ResilienceManager initialized")
    
    async def execute_with_resilience(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute an operation with full resilience patterns.
        
        Args:
            operation: The operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
        """
        try:
            # Apply bulkhead isolation
            async with self.bulkhead_manager.acquire():
                # Apply circuit breaker
                async with self.circuit_breaker:
                    # Apply retry logic
                    return await self.retry_manager.execute_with_retry(
                        operation, *args, **kwargs
                    )
        except Exception as e:
            logger.error("Resilience execution failed", error=str(e))
            # Apply fallback strategy
            return await self.fallback_manager.execute_fallback(
                operation, *args, **kwargs
            )
    
    async def get_resilience_status(self) -> Dict[str, Any]:
        """
        Get the current resilience status.
        
        Returns:
            Dictionary containing resilience status information
        """
        status = {
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "last_failure_time": self.circuit_breaker.last_failure_time,
            },
            "retry_manager": {
                "max_retries": self.retry_manager.max_retries,
                "base_delay": self.retry_manager.base_delay,
                "max_delay": self.retry_manager.max_delay,
            },
            "bulkhead": {
                "max_concurrent": self.bulkhead_manager.max_concurrent_requests,
                "current_requests": self.bulkhead_manager.current_requests,
            },
            "fallback": {
                "available": self.fallback_manager is not None,
            },
        }
        
        return status
    
    async def update_resilience_config(self, config: Dict[str, Any]) -> bool:
        """
        Update resilience configuration.
        
        Args:
            config: Resilience configuration dictionary
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update circuit breaker config
            if "circuit_breaker" in config:
                cb_config = config["circuit_breaker"]
                self.circuit_breaker.failure_threshold = cb_config.get(
                    "failure_threshold", self.circuit_breaker.failure_threshold
                )
                self.circuit_breaker.timeout_duration = cb_config.get(
                    "timeout_duration", self.circuit_breaker.timeout_duration
                )
            
            # Update retry config
            if "retry" in config:
                retry_config = config["retry"]
                self.retry_manager.max_retries = retry_config.get(
                    "max_retries", self.retry_manager.max_retries
                )
                self.retry_manager.base_delay = retry_config.get(
                    "base_delay", self.retry_manager.base_delay
                )
            
            # Update bulkhead config
            if "bulkhead" in config:
                bulkhead_config = config["bulkhead"]
                self.bulkhead_manager.max_concurrent_requests = bulkhead_config.get(
                    "max_concurrent", self.bulkhead_manager.max_concurrent_requests
                )
            
            logger.info("Resilience configuration updated", config_keys=list(config.keys()))
            return True
            
        except Exception as e:
            logger.error("Failed to update resilience configuration", error=str(e))
            return False
