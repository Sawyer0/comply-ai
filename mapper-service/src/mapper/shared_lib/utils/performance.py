"""Performance utilities following DRY principle.

This module provides ONLY performance-related utility functions
that can be reused across all services to avoid code duplication.
"""

from typing import Optional, Dict, Any
from datetime import datetime


def calculate_performance_reward(
    confidence: float,
    response_time_ms: Optional[float] = None,
    success: bool = True,
    max_response_time: float = 1000.0,
) -> float:
    """Calculate performance reward for ML feedback following DRY principle.

    This function is reused across all services to ensure consistent
    reward calculation logic.

    Args:
        confidence: Confidence score (0.0-1.0)
        response_time_ms: Response time in milliseconds
        success: Whether the operation was successful
        max_response_time: Maximum acceptable response time

    Returns:
        Reward value between 0.0 and 1.0
    """
    # Base reward from confidence
    reward = confidence if success else 0.0

    # Time-based bonus (faster responses get bonus)
    if response_time_ms is not None and response_time_ms > 0:
        if response_time_ms < max_response_time:
            time_bonus = (
                0.1 * (max_response_time - response_time_ms) / max_response_time
            )
            reward += time_bonus

    # Success/failure adjustment
    if not success:
        reward = max(0.0, reward - 0.2)  # Penalty for failure

    # Clamp to valid range
    return max(0.0, min(1.0, reward))


def calculate_response_time_score(
    response_time_ms: float, target_time_ms: float = 100.0
) -> float:
    """Calculate response time score following DRY principle.

    Args:
        response_time_ms: Actual response time
        target_time_ms: Target response time

    Returns:
        Score between 0.0 and 1.0 (higher is better)
    """
    if response_time_ms <= target_time_ms:
        return 1.0

    # Exponential decay for longer response times
    ratio = response_time_ms / target_time_ms
    return max(0.0, 1.0 / ratio)


def calculate_success_rate_score(successes: int, total: int) -> float:
    """Calculate success rate score following DRY principle.

    Args:
        successes: Number of successful operations
        total: Total number of operations

    Returns:
        Success rate between 0.0 and 1.0
    """
    if total == 0:
        return 0.0
    return successes / total


def calculate_throughput_score(
    operations: int, time_window_seconds: float, target_ops_per_second: float = 10.0
) -> float:
    """Calculate throughput score following DRY principle.

    Args:
        operations: Number of operations completed
        time_window_seconds: Time window in seconds
        target_ops_per_second: Target operations per second

    Returns:
        Throughput score between 0.0 and 1.0
    """
    if time_window_seconds <= 0:
        return 0.0

    actual_ops_per_second = operations / time_window_seconds
    return min(1.0, actual_ops_per_second / target_ops_per_second)


class PerformanceTracker:
    """Performance tracking utility following SRP and DRY principles."""

    def __init__(self):
        """Initialize performance tracker."""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time_ms": 0.0,
            "start_time": datetime.utcnow(),
        }

    def record_request(self, success: bool, response_time_ms: float) -> None:
        """Record a request for performance tracking.

        Args:
            success: Whether the request was successful
            response_time_ms: Response time in milliseconds
        """
        self.metrics["total_requests"] += 1
        self.metrics["total_response_time_ms"] += response_time_ms

        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.

        Returns:
            Dictionary with performance metrics
        """
        total_requests = self.metrics["total_requests"]

        if total_requests == 0:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "average_response_time_ms": 0.0,
                "requests_per_second": 0.0,
            }

        # Calculate derived metrics
        success_rate = self.metrics["successful_requests"] / total_requests
        avg_response_time = self.metrics["total_response_time_ms"] / total_requests

        # Calculate requests per second
        elapsed_time = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
        requests_per_second = total_requests / max(1.0, elapsed_time)

        return {
            "total_requests": total_requests,
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": success_rate,
            "average_response_time_ms": avg_response_time,
            "requests_per_second": requests_per_second,
            "uptime_seconds": elapsed_time,
        }

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time_ms": 0.0,
            "start_time": datetime.utcnow(),
        }
