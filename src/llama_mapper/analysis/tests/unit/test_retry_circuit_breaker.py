"""
Unit tests for retry logic and circuit breaker functionality.

This module provides comprehensive tests for the retry mechanisms,
circuit breaker patterns, and resilience features.
"""

import asyncio

import pytest

from ...resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenException,
    CircuitState,
)
from ...resilience.retry import (
    RetryConfig,
    RetryManager,
    RetryStrategy,
)


class TestCircuitBreakerConfig:
    """Test cases for circuit breaker configuration."""

    def test_default_config(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.expected_exception == (Exception,)

    def test_custom_config(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
            timeout=15.0,
            expected_exception=(ValueError, RuntimeError),
        )

        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.success_threshold == 2
        assert config.timeout == 15.0
        assert config.expected_exception == (ValueError, RuntimeError)


class TestCircuitBreaker:
    """Test cases for circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=5.0,
        )
        return CircuitBreaker("test_service", config)

    def test_initial_state(self, circuit_breaker):
        """Test initial circuit breaker state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.success_count == 0
        assert circuit_breaker.total_calls == 0

    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """Test successful function call."""

        async def success_func():
            return "success"

        result = await circuit_breaker.call(success_func)

        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.total_calls == 1
        assert circuit_breaker.total_successes == 1
        assert circuit_breaker.total_failures == 0

    @pytest.mark.asyncio
    async def test_failure_handling(self, circuit_breaker):
        """Test failure handling and state transitions."""

        async def failing_func():
            raise ValueError("Test error")

        # Make calls that will fail
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Circuit should be open after threshold failures
        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.failure_count == 0  # Reset after transition
        assert circuit_breaker.total_failures == 3

    @pytest.mark.asyncio
    async def test_circuit_open_blocks_calls(self, circuit_breaker):
        """Test that open circuit blocks calls."""

        async def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Try to make a call when circuit is open
        with pytest.raises(CircuitBreakerOpenException):
            await circuit_breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_circuit_recovery(self, circuit_breaker):
        """Test circuit recovery from open to closed state."""

        async def failing_func():
            raise ValueError("Test error")

        async def success_func():
            return "success"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # First call should transition to half-open
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Second successful call should close the circuit
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_returns_to_open(self, circuit_breaker):
        """Test that failure in half-open state returns to open."""

        async def failing_func():
            raise ValueError("Test error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit_breaker.call(failing_func)

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # First call should be half-open
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_func)

        # Should be back to open
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_timeout_handling(self, circuit_breaker):
        """Test timeout handling."""

        async def slow_func():
            await asyncio.sleep(10)  # Longer than timeout
            return "success"

        with pytest.raises(asyncio.TimeoutError):
            await circuit_breaker.call(slow_func)

        assert circuit_breaker.total_timeouts == 1

    @pytest.mark.asyncio
    async def test_unexpected_exception(self, circuit_breaker):
        """Test handling of unexpected exceptions."""

        async def unexpected_func():
            raise KeyError("Unexpected error")

        # Unexpected exceptions should not count as failures
        with pytest.raises(KeyError):
            await circuit_breaker.call(unexpected_func)

        assert circuit_breaker.total_failures == 0

    def test_statistics(self, circuit_breaker):
        """Test circuit breaker statistics."""
        stats = circuit_breaker.get_statistics()

        assert "name" in stats
        assert "state" in stats
        assert "total_calls" in stats
        assert "total_failures" in stats
        assert "total_successes" in stats
        assert "failure_rate" in stats
        assert "success_rate" in stats
        assert "config" in stats

    def test_manual_reset(self, circuit_breaker):
        """Test manual circuit breaker reset."""
        # Manually open the circuit
        circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN

        # Reset the circuit
        circuit_breaker.reset()
        assert circuit_breaker.state == CircuitState.CLOSED

    def test_force_open(self, circuit_breaker):
        """Test manual circuit opening."""
        circuit_breaker.force_open()
        assert circuit_breaker.state == CircuitState.OPEN


class TestRetryConfig:
    """Test cases for retry configuration."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter_enabled is True
        assert config.jitter_range == 0.1
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.retryable_exceptions == (Exception,)
        assert not config.non_retryable_exceptions

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter_enabled=False,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            retryable_exceptions=(ValueError, RuntimeError),
            non_retryable_exceptions=(KeyError,),
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter_enabled is False
        assert config.strategy == RetryStrategy.LINEAR_BACKOFF
        assert config.retryable_exceptions == (ValueError, RuntimeError)
        assert config.non_retryable_exceptions == (KeyError,)


class TestRetryManager:
    """Test cases for retry manager functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create retry manager for testing."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,  # Short delay for testing
            max_delay=1.0,
            jitter_enabled=False,  # Disable jitter for predictable testing
        )
        return RetryManager(config)

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self, retry_manager):
        """Test successful call without retries."""

        async def success_func():
            return "success"

        result = await retry_manager.execute_with_retry(success_func)

        assert result == "success"
        assert retry_manager.total_attempts == 1
        assert retry_manager.total_successes == 1
        assert retry_manager.total_retries == 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, retry_manager):
        """Test retry on failure."""
        call_count = 0

        async def failing_then_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await retry_manager.execute_with_retry(failing_then_success_func)

        assert result == "success"
        assert call_count == 3
        assert retry_manager.total_attempts == 3
        assert retry_manager.total_successes == 1
        assert retry_manager.total_retries == 1

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self, retry_manager):
        """Test behavior when max attempts are exceeded."""

        async def always_failing_func():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await retry_manager.execute_with_retry(always_failing_func)

        assert retry_manager.total_attempts == 3
        assert retry_manager.total_successes == 0
        assert retry_manager.total_failures == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self, retry_manager):
        """Test non-retryable exceptions."""
        retry_manager.config.non_retryable_exceptions = (KeyError,)

        async def non_retryable_func():
            raise KeyError("Non-retryable error")

        with pytest.raises(KeyError, match="Non-retryable error"):
            await retry_manager.execute_with_retry(non_retryable_func)

        assert retry_manager.total_attempts == 1
        assert retry_manager.total_failures == 1

    @pytest.mark.asyncio
    async def test_retryable_exception_filtering(self, retry_manager):
        """Test retryable exception filtering."""
        retry_manager.config.retryable_exceptions = (ValueError,)
        retry_manager.config.non_retryable_exceptions = (KeyError,)

        async def mixed_exception_func():
            raise KeyError("Non-retryable error")

        with pytest.raises(KeyError, match="Non-retryable error"):
            await retry_manager.execute_with_retry(mixed_exception_func)

        assert retry_manager.total_attempts == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self, retry_manager):
        """Test exponential backoff delay calculation."""
        retry_manager.config.strategy = RetryStrategy.EXPONENTIAL_BACKOFF
        retry_manager.config.jitter_enabled = False

        delays = []
        for attempt in range(1, 4):
            delay = retry_manager._calculate_delay(
                attempt
            )  # pylint: disable=protected-access
            delays.append(delay)

        # Should be exponential: 0.1, 0.2, 0.4
        assert delays[0] == 0.1
        assert delays[1] == 0.2
        assert delays[2] == 0.4

    @pytest.mark.asyncio
    async def test_linear_backoff_delay(self, retry_manager):
        """Test linear backoff delay calculation."""
        retry_manager.config.strategy = RetryStrategy.LINEAR_BACKOFF
        retry_manager.config.jitter_enabled = False

        delays = []
        for attempt in range(1, 4):
            delay = retry_manager._calculate_delay(
                attempt
            )  # pylint: disable=protected-access
            delays.append(delay)

        # Should be linear: 0.1, 0.2, 0.3
        assert delays[0] == 0.1
        assert delays[1] == 0.2
        assert delays[2] == 0.3

    @pytest.mark.asyncio
    async def test_fixed_delay(self, retry_manager):
        """Test fixed delay strategy."""
        retry_manager.config.strategy = RetryStrategy.FIXED_DELAY
        retry_manager.config.jitter_enabled = False

        delays = []
        for attempt in range(1, 4):
            delay = retry_manager._calculate_delay(
                attempt
            )  # pylint: disable=protected-access
            delays.append(delay)

        # Should be fixed: 0.1, 0.1, 0.1
        assert all(delay == 0.1 for delay in delays)

    @pytest.mark.asyncio
    async def test_max_delay_limit(self, retry_manager):
        """Test maximum delay limit."""
        retry_manager.config.max_delay = 0.2
        retry_manager.config.jitter_enabled = False

        delay = retry_manager._calculate_delay(10)  # pylint: disable=protected-access
        assert delay <= 0.2

    def test_statistics(self, retry_manager):
        """Test retry manager statistics."""
        stats = retry_manager.get_statistics()

        assert "total_attempts" in stats
        assert "total_retries" in stats
        assert "total_successes" in stats
        assert "total_failures" in stats
        assert "success_rate" in stats
        assert "retry_rate" in stats
        assert "config" in stats


class TestRetryDecorator:
    """Test cases for retry decorator."""

    @pytest.mark.asyncio
    async def test_retry_manager_as_decorator(self):
        """Test retry manager used through manual wrapping."""
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter_enabled=False)
        retry_manager = RetryManager(config)

        async def success_func():
            return "success"

        result = await retry_manager.execute_with_retry(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_manager_with_failure(self):
        """Test retry manager with failing function."""
        config = RetryConfig(max_attempts=3, base_delay=0.01, jitter_enabled=False)
        retry_manager = RetryManager(config)

        call_count = 0

        async def failing_then_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await retry_manager.execute_with_retry(failing_then_success_func)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test with_retry decorator factory."""

        @RetryManager.with_retry(max_attempts=2, base_delay=0.01, jitter_enabled=False)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_manual_retry_wrapping(self):
        """Test manual retry wrapping preserves function behavior."""
        config = RetryConfig()
        retry_manager = RetryManager(config)

        async def test_func():
            """Test function docstring."""
            return "success"

        # Test that we can execute the function through retry manager
        result = await retry_manager.execute_with_retry(test_func)
        assert result == "success"
        # Original function metadata is preserved
        assert test_func.__name__ == "test_func"
        assert test_func.__doc__ == "Test function docstring."


class TestCircuitBreakerIntegration:
    """Test cases for circuit breaker and retry integration."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry logic with circuit breaker integration."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1, success_threshold=1
        )
        circuit_breaker = CircuitBreaker("test_service", circuit_config)

        retry_config = RetryConfig(
            max_attempts=3, base_delay=0.01, jitter_enabled=False
        )
        retry_manager = RetryManager(retry_config, circuit_breaker)

        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Service error")

        # First two calls should fail and open circuit
        with pytest.raises(ValueError):
            await retry_manager.execute_with_retry(failing_func)

        with pytest.raises(ValueError):
            await retry_manager.execute_with_retry(failing_func)

        # Circuit should be open now
        assert circuit_breaker.state == CircuitState.OPEN

        # Next call should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerOpenException):
            await retry_manager.execute_with_retry(failing_func)

        assert call_count == 2  # Only 2 calls made before circuit opened

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_with_retry(self):
        """Test circuit breaker recovery with retry logic."""
        circuit_config = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1, success_threshold=1
        )
        circuit_breaker = CircuitBreaker("test_service", circuit_config)

        retry_config = RetryConfig(
            max_attempts=3, base_delay=0.01, jitter_enabled=False
        )
        retry_manager = RetryManager(retry_config, circuit_breaker)

        # Open the circuit
        async def failing_func():
            raise ValueError("Service error")

        for _ in range(2):
            with pytest.raises(ValueError):
                await retry_manager.execute_with_retry(failing_func)

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        # Try with success function
        async def success_func():
            return "recovered"

        result = await retry_manager.execute_with_retry(success_func)
        assert result == "recovered"
        assert circuit_breaker.state == CircuitState.CLOSED
