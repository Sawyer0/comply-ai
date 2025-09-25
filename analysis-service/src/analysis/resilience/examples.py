"""
Examples of using resilience patterns in analysis service.

This module provides practical examples of how to use the various
resilience patterns for analysis operations.
"""

import asyncio
import logging
import random
from typing import Any, Dict

from .bulkhead import BulkheadConfig
from .config import CircuitBreakerConfig, RetryConfig
from .interfaces import RetryStrategy
from .fallback import FallbackConfig, FunctionFallbackStrategy, SimpleFallbackStrategy
from .manager import ComprehensiveResilienceManager

logger = logging.getLogger(__name__)


class AnalysisServiceExample:
    """Example analysis service using resilience patterns."""

    def __init__(self):
        """Initialize example service with resilience manager."""
        self.resilience_manager = ComprehensiveResilienceManager()
        self._setup_resilience_patterns()

    def _setup_resilience_patterns(self):
        """Setup resilience patterns for different operations."""

        # Model inference resilience stack
        model_stack = self.resilience_manager.create_resilience_stack(
            name="model_inference",
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=3, recovery_timeout=30.0, timeout=10.0
            ),
            retry_config=RetryConfig(
                max_attempts=3,
                base_delay=1.0,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            ),
            bulkhead_config=BulkheadConfig(max_concurrent_calls=5, timeout=15.0),
            fallback_config=FallbackConfig(
                enable_fallback=True, max_fallback_attempts=2
            ),
        )

        # Add fallback strategies for model inference
        fallback_manager = model_stack["fallback_manager"]

        # Simple fallback with default result
        fallback_manager.add_strategy(
            SimpleFallbackStrategy(
                name="default_analysis",
                default_value={
                    "risk_score": 0.5,
                    "confidence": 0.3,
                    "source": "fallback",
                },
            )
        )

        # Function fallback with rule-based analysis
        fallback_manager.add_strategy(
            FunctionFallbackStrategy(
                name="rule_based_analysis",
                fallback_func=self._rule_based_analysis,
                exception_types=(TimeoutError, ConnectionError),
            )
        )

        # Pattern recognition resilience stack
        self.resilience_manager.create_resilience_stack(
            name="pattern_recognition",
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=60.0
            ),
            retry_config=RetryConfig(max_attempts=2, base_delay=0.5),
        )

        logger.info("Resilience patterns configured for analysis service")

    async def analyze_with_resilience(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform analysis with full resilience protection.

        Args:
            data: Input data for analysis

        Returns:
            Analysis result with resilience protection
        """
        # Get resilience components
        circuit_breaker = self.resilience_manager.get_circuit_breaker("model_inference")
        retry_manager = self.resilience_manager.get_retry_manager("model_inference")
        fallback_manager = self.resilience_manager.get_fallback_manager(
            "model_inference"
        )
        bulkhead = self.resilience_manager.get_bulkhead_manager().get_bulkhead(
            "model_inference"
        )

        # Execute with full resilience stack
        try:
            # Use bulkhead isolation
            if bulkhead:
                result = await bulkhead.execute(
                    self._execute_with_retry_and_fallback,
                    retry_manager,
                    fallback_manager,
                    data,
                )
            else:
                result = await self._execute_with_retry_and_fallback(
                    retry_manager, fallback_manager, data
                )

            return result

        except Exception as e:
            logger.error("Analysis failed even with resilience patterns: %s", e)
            raise

    async def _execute_with_retry_and_fallback(
        self, retry_manager, fallback_manager, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis with retry and fallback."""

        if fallback_manager:
            # Use fallback manager
            return await fallback_manager.execute_with_fallback(
                self._execute_with_retry, retry_manager, data
            )
        else:
            # Use retry only
            return await self._execute_with_retry(retry_manager, data)

    async def _execute_with_retry(
        self, retry_manager, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis with retry."""

        if retry_manager:
            return await retry_manager.execute_with_retry(self._perform_analysis, data)
        else:
            return await self._perform_analysis(data)

    async def _perform_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the actual analysis (this would be your real analysis logic).

        This is a mock implementation that sometimes fails to demonstrate
        the resilience patterns.
        """
        import random

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Simulate failures for demonstration
        if random.random() < 0.3:  # 30% failure rate
            if random.random() < 0.5:
                raise TimeoutError("Analysis timed out")
            else:
                raise ConnectionError("Model server unavailable")

        # Return successful analysis
        return {
            "risk_score": random.uniform(0.1, 0.9),
            "confidence": random.uniform(0.7, 0.95),
            "patterns": ["pattern_1", "pattern_2"],
            "source": "ml_model",
        }

    def _rule_based_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback analysis.

        This provides a simple rule-based analysis when ML models fail.
        """
        logger.info("Using rule-based fallback analysis")

        # Simple rule-based logic
        text_length = len(str(data.get("text", "")))

        if text_length > 1000:
            risk_score = 0.7
        elif text_length > 500:
            risk_score = 0.5
        else:
            risk_score = 0.3

        return {
            "risk_score": risk_score,
            "confidence": 0.6,
            "patterns": ["length_based"],
            "source": "rule_based_fallback",
        }

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all resilience components."""
        return self.resilience_manager.health_check()

    async def get_resilience_statistics(self) -> Dict[str, Any]:
        """Get statistics for all resilience components."""
        return self.resilience_manager.get_all_statistics()


# Example usage
async def main():
    """Example usage of resilience patterns."""

    # Create analysis service with resilience
    service = AnalysisServiceExample()

    # Perform multiple analyses to see resilience in action
    for i in range(10):
        try:
            result = await service.analyze_with_resilience(
                {"text": f"Sample analysis text {i}" * 50, "id": f"analysis_{i}"}
            )

            print(
                f"Analysis {i}: {result['source']} - Risk: {result['risk_score']:.2f}"
            )

        except Exception as e:
            print(f"Analysis {i} failed: {e}")

        # Small delay between requests
        await asyncio.sleep(0.1)

    # Print resilience statistics
    stats = await service.get_resilience_statistics()
    print("\nResilience Statistics:")
    print(f"Circuit Breakers: {len(stats['circuit_breakers'])}")
    print(f"Retry Managers: {len(stats['retry_managers'])}")
    print(f"Bulkheads: {len(stats['bulkheads'])}")
    print(f"Fallback Managers: {len(stats['fallback_managers'])}")

    # Print health status
    health = await service.get_health_status()
    print(f"\nOverall Health: {health['status']}")


if __name__ == "__main__":
    # Run example
    asyncio.run(main())
