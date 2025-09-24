"""
Unit tests for retry logic and circuit breaker implementation.

Tests the resilience components including retry managers, circuit breakers,
and their integration for model server reliability.
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llama_mapper.analysis.resilience.circuit_breaker.implementation import (
    CircuitBreaker,
    CircuitBreakerOpenException,
)
from src.llama_mapper.analysis.resilience.config.circuit_breaker_config import (
    CircuitBreakerConfig,
)
from src.llama_mapper.analysis.resilience.config.retry_config import RetryConfig
from src.llama_mapper.analysis.resilience.interfaces import CircuitState, RetryStrategy
from src.llama_mapper.analysis.resilience.retry.implementation import RetryManager


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_retry_config(self):
        """Test default retry configuration values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.1
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=0.2,
            strategy=RetryStrategy.LINEAR_BACKOFF,
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter == 0.2
        assert config.strategy == RetryStrategy.LINEAR_BACKOFF

    def test_retry_config_validation(self):
        """Test retry configuration validation."""
        # Test invalid max_attempts
        with pytest.raises(ValueError, match="max_attempts must be between 1 and 10"):
            RetryConfig(max_attempts=0)

        with pytest.raises(ValueError, match="max_attempts must be between 1 and 10"):
            RetryConfig(max_attempts=11)

        # Test invalid base_delay
        with pytest.raises(ValueError, match="base_delay must be between 0.1 and 10.0"):
            RetryConfig(base_delay=0.05)

        with pytest.raises(ValueError, match="base_delay must be between 0.1 and 10.0"):
            RetryConfig(base_delay=15.0)

        # Test invalid max_delay
        with pytest.raises(ValueError, match="max_delay must be between 1.0 and 300.0"):
            RetryConfig(max_delay=0.5)

        with pytest.raises(ValueError, match="max_delay must be between 1.0 and 300.0"):
            RetryConfig(max_delay=400.0)


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_circuit_breaker_config(self):
        """Test default circuit breaker configuration values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.expected_exception == (Exception,)

    def test_custom_circuit_breaker_config(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=15.0,
            expected_exception=(ConnectionError, TimeoutError),
        )

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.timeout == 15.0
        assert config.expected_exception == (ConnectionError, TimeoutError)

    def test_circuit_breaker_config_validation(self):
        """Test circuit breaker configuration validation."""
        # Test invalid failure_threshold
        with pytest.raises(
            ValueError, match="failure_threshold must be between 1 and 20"
        ):
            CircuitBreakerConfig(failure_threshold=0)

        with pytest.raises(
            ValueError, match="failure_threshold must be between 1 and 20"
        ):
            CircuitBreakerConfig(failure_threshold=25)

        # Test invalid recovery_timeout
        with pytest.raises(
            ValueError, match="recovery_timeout must be between 10.0 and 300.0"
        ):
            CircuitBreakerConfig(recovery_timeout=5.0)

        with pytest.raises(
            ValueError, match="recovery_timeout must be between 10.0 and 300.0"
        ):
            CircuitBreakerConfig(recovery_timeout=400.0)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=5.0,
        )
        return CircuitBreaker("test_circuit", config)

    def test_circuit_breaker_initialization(self, circuit_breaker):
        """Test circuit breaker initialization."""
        assert circuit_breaker.name == "test_circuit"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0
        assert circuit_breaker._success_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_successful_call(self, circuit_breaker):
        """Test successful circuit breaker call."""

        async def successful_func():
            return "success"

        result = await circuit_breaker.call(successful_func)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._success_count == 1
        assert circuit_breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_handling(self, circuit_breaker):
        """Test circuit breaker failure handling."""

        async def failing_func():
            raise ConnectionError("Connection failed")

        # First failure
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 1
        assert circuit_breaker._success_count == 0

        # Second failure
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 2

        # Third failure - should open circuit
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker._failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self, circuit_breaker):
        """Test circuit breaker in open state."""

        # Open the circuit
        async def failing_func():
            raise ConnectionError("Connection failed")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Try to call function when circuit is open
        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery from half-open state."""

        # Open the circuit
        async def failing_func():
            raise ConnectionError("Connection failed")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # First successful call should transition to half-open
        async def successful_func():
            return "success"

        result = await circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN
        assert circuit_breaker._success_count == 1

        # Second successful call should close circuit
        result = await circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._success_count == 0  # Reset on state change
        assert circuit_breaker._failure_count == 0  # Reset on state change

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure(self, circuit_breaker):
        """Test circuit breaker failure in half-open state."""

        # Open the circuit
        async def failing_func():
            raise ConnectionError("Connection failed")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Failure in half-open state should open circuit again
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self, circuit_breaker):
        """Test circuit breaker timeout handling."""

        async def slow_func():
            await asyncio.sleep(10)  # Longer than circuit breaker timeout
            return "success"

        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_func)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_unexpected_exception(self, circuit_breaker):
        """Test circuit breaker with unexpected exceptions."""

        async def unexpected_error_func():
            raise ValueError("Unexpected error")

        # Unexpected exceptions should not count as failures
        with pytest.raises(ValueError):
            await circuit_breaker.call(unexpected_error_func)

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0  # Should not increment

    @pytest.mark.asyncio
    async def test_circuit_breaker_metrics_collection(self, circuit_breaker):
        """Test circuit breaker metrics collection."""
        metrics_collector = Mock()
        circuit_breaker._metrics_collector = metrics_collector

        async def successful_func():
            return "success"

        await circuit_breaker.call(successful_func)

        # Verify metrics were recorded
        assert metrics_collector.record_circuit_breaker_call.called
        assert metrics_collector.record_circuit_breaker_success.called


class TestRetryManager:
    """Test retry manager functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create retry manager for testing."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            max_delay=1.0,
            exponential_base=2.0,
            jitter=0.0,  # No jitter for predictable testing
        )
        return RetryManager(config)

    def test_retry_manager_initialization(self, retry_manager):
        """Test retry manager initialization."""
        assert retry_manager._config.max_attempts == 3
        assert retry_manager._total_attempts == 0
        assert retry_manager._total_retries == 0
        assert retry_manager._total_failures == 0

    @pytest.mark.asyncio
    async def test_retry_manager_successful_call(self, retry_manager):
        """Test successful retry manager call."""

        async def successful_func():
            return "success"

        result = await retry_manager.execute_with_retry(successful_func)

        assert result == "success"
        assert retry_manager._total_attempts == 1
        assert retry_manager._total_retries == 0
        assert retry_manager._total_failures == 0

    @pytest.mark.asyncio
    async def test_retry_manager_retry_on_failure(self, retry_manager):
        """Test retry manager retry logic on failure."""
        call_count = 0

        async def failing_then_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"

        result = await retry_manager.execute_with_retry(failing_then_success_func)

        assert result == "success"
        assert call_count == 3
        assert retry_manager._total_attempts == 3
        assert retry_manager._total_retries == 2
        assert retry_manager._total_failures == 0

    @pytest.mark.asyncio
    async def test_retry_manager_max_attempts_exceeded(self, retry_manager):
        """Test retry manager when max attempts are exceeded."""

        async def always_failing_func():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            await retry_manager.execute_with_retry(always_failing_func)

        assert retry_manager._total_attempts == 3
        assert retry_manager._total_retries == 2
        assert retry_manager._total_failures == 1

    @pytest.mark.asyncio
    async def test_retry_manager_exponential_backoff(self, retry_manager):
        """Test retry manager exponential backoff timing."""
        call_times = []

        async def failing_func():
            call_times.append(time.time())
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            await retry_manager.execute_with_retry(failing_func)

        # Check that delays increase exponentially
        assert len(call_times) == 3

        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Second delay should be approximately double the first
        assert delay2 > delay1 * 1.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_retry_manager_with_circuit_breaker(self, retry_manager):
        """Test retry manager integration with circuit breaker."""
        circuit_breaker = CircuitBreaker(
            "test_circuit",
            CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=0.5,
                success_threshold=1,
                timeout=5.0,
            ),
        )
        retry_manager._circuit_breaker = circuit_breaker

        async def failing_func():
            raise ConnectionError("Connection failed")

        # Should fail after circuit breaker opens
        with pytest.raises(CircuitBreakerOpenException):
            await retry_manager.execute_with_retry(failing_func)

        # Circuit breaker should be open
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_retry_manager_metrics_collection(self, retry_manager):
        """Test retry manager metrics collection."""
        metrics_collector = Mock()
        retry_manager._metrics_collector = metrics_collector

        async def successful_func():
            return "success"

        await retry_manager.execute_with_retry(successful_func)

        # Verify metrics were recorded
        assert metrics_collector.record_retry_attempt.called

    @pytest.mark.asyncio
    async def test_retry_manager_linear_backoff(self):
        """Test retry manager with linear backoff strategy."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=0.0,
        )
        retry_manager = RetryManager(config)

        call_times = []

        async def failing_func():
            call_times.append(time.time())
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            await retry_manager.execute_with_retry(failing_func)

        # Check that delays are linear
        assert len(call_times) == 3

        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Delays should be approximately equal (linear)
        assert abs(delay1 - delay2) < 0.05  # Allow small tolerance

    @pytest.mark.asyncio
    async def test_retry_manager_fixed_delay(self):
        """Test retry manager with fixed delay strategy."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            strategy=RetryStrategy.FIXED_DELAY,
            jitter=0.0,
        )
        retry_manager = RetryManager(config)

        call_times = []

        async def failing_func():
            call_times.append(time.time())
            raise ConnectionError("Connection failed")

        with pytest.raises(ConnectionError):
            await retry_manager.execute_with_retry(failing_func)

        # Check that delays are fixed
        assert len(call_times) == 3

        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]

        # Delays should be approximately equal (fixed)
        assert abs(delay1 - delay2) < 0.05  # Allow small tolerance
        assert abs(delay1 - 0.1) < 0.05  # Should be close to base_delay


class TestResilienceIntegration:
    """Integration tests for resilience components."""

    @pytest.mark.asyncio
    async def test_full_resilience_pipeline(self):
        """Test complete resilience pipeline with retry and circuit breaker."""
        # Create circuit breaker
        circuit_breaker = CircuitBreaker(
            "model_server",
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=1.0,
                success_threshold=2,
                timeout=5.0,
            ),
        )

        # Create retry manager with circuit breaker
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            max_delay=1.0,
            exponential_base=2.0,
            jitter=0.0,
        )
        retry_manager = RetryManager(retry_config, circuit_breaker)

        # Test successful operation
        async def successful_model_call():
            return {"result": "analysis_complete", "confidence": 0.95}

        result = await retry_manager.execute_with_retry(successful_model_call)
        assert result["result"] == "analysis_complete"
        assert circuit_breaker.state == CircuitState.CLOSED

        # Test failure handling
        call_count = 0

        async def failing_then_success_model_call():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Model server unavailable")
            return {"result": "analysis_complete", "confidence": 0.95}

        result = await retry_manager.execute_with_retry(failing_then_success_model_call)
        assert result["result"] == "analysis_complete"
        assert call_count == 2

        # Test circuit breaker opening
        async def always_failing_model_call():
            raise ConnectionError("Model server down")

        for _ in range(3):
            with pytest.raises(ConnectionError):
                await retry_manager.execute_with_retry(always_failing_model_call)

        assert circuit_breaker.state == CircuitState.OPEN

        # Test circuit breaker blocking
        with pytest.raises(CircuitBreakerOpenException):
            await retry_manager.execute_with_retry(always_failing_model_call)

    @pytest.mark.asyncio
    async def test_resilience_with_metrics(self):
        """Test resilience components with metrics collection."""
        metrics_collector = Mock()

        # Create circuit breaker with metrics
        circuit_breaker = CircuitBreaker(
            "model_server", CircuitBreakerConfig(), metrics_collector
        )

        # Create retry manager with metrics
        retry_manager = RetryManager(RetryConfig(), circuit_breaker, metrics_collector)

        async def successful_func():
            return "success"

        await retry_manager.execute_with_retry(successful_func)

        # Verify metrics were collected
        assert metrics_collector.record_circuit_breaker_call.called
        assert metrics_collector.record_circuit_breaker_success.called
        assert metrics_collector.record_retry_attempt.called

    @pytest.mark.asyncio
    async def test_resilience_performance(self):
        """Test resilience components performance."""
        circuit_breaker = CircuitBreaker("test", CircuitBreakerConfig())
        retry_manager = RetryManager(RetryConfig(), circuit_breaker)

        async def fast_func():
            return "fast_result"

        # Measure execution time
        start_time = time.time()
        result = await retry_manager.execute_with_retry(fast_func)
        end_time = time.time()

        execution_time = end_time - start_time

        assert result == "fast_result"
        assert execution_time < 0.1  # Should be very fast for successful calls
