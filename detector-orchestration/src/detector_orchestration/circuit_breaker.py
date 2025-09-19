from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Dict, Optional


class CircuitBreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure = 0.0

    def allow_request(self) -> bool:
        if self.state == CircuitBreakerState.OPEN:
            # Check if we can transition to HALF_OPEN
            if (time.time() - self.last_failure) >= self.recovery_timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        return True

    def record_success(self) -> None:
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class CircuitBreakerManager:
    def __init__(self, failure_threshold: int, recovery_timeout_seconds: int):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._ft = failure_threshold
        self._rt = recovery_timeout_seconds

    def get(self, detector: str) -> CircuitBreaker:
        br = self._breakers.get(detector)
        if not br:
            br = CircuitBreaker(self._ft, self._rt)
            self._breakers[detector] = br
        return br

