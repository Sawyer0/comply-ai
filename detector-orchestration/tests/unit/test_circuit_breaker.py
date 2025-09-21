"""Tests for circuit breaker."""

import time
import pytest

from detector_orchestration.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerManager,
)


class TestCircuitBreaker:
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_seconds=60)

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout_seconds == 60
        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.last_failure == 0.0

    def test_circuit_breaker_initialization_defaults(self):
        """Test circuit breaker initialization with defaults."""
        breaker = CircuitBreaker()

        assert breaker.failure_threshold == 5
        assert breaker.recovery_timeout_seconds == 60
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_allow_request_closed_state(self):
        """Test allowing requests when circuit breaker is closed."""
        breaker = CircuitBreaker()
        assert breaker.allow_request() is True
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_allow_request_open_state(self):
        """Test allowing requests when circuit breaker is open."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30)

        # Record failures to open the circuit
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN
        assert breaker.allow_request() is False

    def test_allow_request_open_to_half_open(self):
        """Test transitioning from open to half-open state."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)

        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        time.sleep(1.1)

        # Should transition to half-open and allow request
        assert breaker.allow_request() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_record_success_from_closed(self):
        """Test recording success when circuit breaker is closed."""
        breaker = CircuitBreaker()

        breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0

    def test_record_success_from_half_open(self):
        """Test recording success when circuit breaker is half-open."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=1)

        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        # Transition to half-open
        time.sleep(1.1)
        breaker.allow_request()

        assert breaker.state == CircuitBreakerState.HALF_OPEN

        # Record success
        breaker.record_success()

        assert breaker.state == CircuitBreakerState.CLOSED
        assert breaker.failure_count == 0

    def test_record_failure_threshold_reached(self):
        """Test recording failures until threshold is reached."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Record failures up to threshold
        for i in range(3):
            breaker.record_failure()
            assert breaker.failure_count == i + 1
            assert breaker.state == CircuitBreakerState.CLOSED

        # One more failure should open the circuit
        breaker.record_failure()
        assert breaker.failure_count == 4
        assert breaker.state == CircuitBreakerState.OPEN

    def test_record_failure_no_threshold_change(self):
        """Test recording failure when already open."""
        breaker = CircuitBreaker(failure_threshold=3)

        # Open the circuit
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitBreakerState.OPEN

        # Record additional failure
        breaker.record_failure()
        assert breaker.failure_count == 4
        assert breaker.state == CircuitBreakerState.OPEN  # Should remain open

    def test_last_failure_timestamp(self):
        """Test that last failure timestamp is updated."""
        breaker = CircuitBreaker()

        initial_time = breaker.last_failure

        # Record a failure
        time.sleep(0.01)  # Small delay to ensure timestamp changes
        breaker.record_failure()

        assert breaker.last_failure > initial_time


class TestCircuitBreakerManager:
    def test_manager_initialization(self):
        """Test circuit breaker manager initialization."""
        manager = CircuitBreakerManager(failure_threshold=5, recovery_timeout_seconds=60)

        assert manager._ft == 5
        assert manager._rt == 60
        assert manager._breakers == {}

    def test_get_existing_breaker(self):
        """Test getting an existing circuit breaker."""
        manager = CircuitBreakerManager(failure_threshold=5, recovery_timeout_seconds=60)

        # Get a breaker (should create new one)
        breaker1 = manager.get("detector1")
        assert "detector1" in manager._breakers
        assert isinstance(breaker1, CircuitBreaker)
        assert breaker1.failure_threshold == 5
        assert breaker1.recovery_timeout_seconds == 60

        # Get the same breaker again (should return existing one)
        breaker2 = manager.get("detector1")
        assert breaker1 is breaker2

    def test_get_multiple_breakers(self):
        """Test getting multiple different circuit breakers."""
        manager = CircuitBreakerManager(failure_threshold=3, recovery_timeout_seconds=30)

        breaker1 = manager.get("detector1")
        breaker2 = manager.get("detector2")

        assert breaker1 is not breaker2
        assert "detector1" in manager._breakers
        assert "detector2" in manager._breakers
        assert breaker1.failure_threshold == 3
        assert breaker2.failure_threshold == 3

    def test_breaker_independence(self):
        """Test that different breakers maintain independent state."""
        manager = CircuitBreakerManager(failure_threshold=2, recovery_timeout_seconds=60)

        breaker1 = manager.get("detector1")
        breaker2 = manager.get("detector2")

        # Open breaker1
        breaker1.record_failure()
        breaker1.record_failure()

        assert breaker1.state == CircuitBreakerState.OPEN
        assert breaker2.state == CircuitBreakerState.CLOSED

        # Record success on breaker2
        breaker2.record_success()
        assert breaker2.state == CircuitBreakerState.CLOSED
        assert breaker2.failure_count == 0

        # breaker1 should still be open
        assert breaker1.state == CircuitBreakerState.OPEN
        assert breaker1.failure_count == 2
