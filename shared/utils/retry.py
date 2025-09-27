"""Retry utilities with exponential backoff."""

import asyncio
import random
import time
from typing import Any, Callable, Optional, Type, Union, Tuple
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def exponential_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with optional jitter."""
    delay = min(base_delay * (2**attempt), max_delay)

    if jitter:
        # Add random jitter to avoid thundering herd
        delay = delay * (0.5 + random.random() * 0.5)

    return delay


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt == max_attempts - 1:
                            # Last attempt, don't retry
                            break

                        delay = exponential_backoff(
                            attempt, base_delay, max_delay, jitter
                        )

                        logger.warning(
                            "Function %s failed on attempt %d/%d, retrying in %.2fs",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            delay,
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "delay": delay,
                                "error": str(e),
                            },
                        )

                        if on_retry:
                            on_retry(attempt + 1, e)

                        await asyncio.sleep(delay)

                # All attempts failed
                if last_exception is None:
                    last_exception = Exception("Function failed without capturing an exception")
                logger.error(
                    "Function %s failed after %d attempts",
                    func.__name__,
                    max_attempts,
                    extra={
                        "function": func.__name__,
                        "max_attempts": max_attempts,
                        "final_error": str(last_exception),
                    },
                )
                raise last_exception

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                last_exception = None

                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e

                        if attempt == max_attempts - 1:
                            # Last attempt, don't retry
                            break

                        delay = exponential_backoff(
                            attempt, base_delay, max_delay, jitter
                        )

                        logger.warning(
                            "Function %s failed on attempt %d/%d, retrying in %.2fs",
                            func.__name__,
                            attempt + 1,
                            max_attempts,
                            delay,
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": max_attempts,
                                "delay": delay,
                                "error": str(e),
                            },
                        )

                        if on_retry:
                            on_retry(attempt + 1, e)

                        time.sleep(delay)

                # All attempts failed
                if last_exception is None:
                    last_exception = Exception("Function failed without capturing an exception")
                logger.error(
                    "Function %s failed after %d attempts",
                    func.__name__,
                    max_attempts,
                    extra={
                        "function": func.__name__,
                        "max_attempts": max_attempts,
                        "final_error": str(last_exception),
                    },
                )
                raise last_exception

            return sync_wrapper

    return decorator


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.exceptions = exceptions

    def create_decorator(
        self, on_retry: Optional[Callable[[int, Exception], None]] = None
    ):
        """Create a retry decorator with this configuration."""
        return retry_with_backoff(
            max_attempts=self.max_attempts,
            base_delay=self.base_delay,
            max_delay=self.max_delay,
            backoff_factor=self.backoff_factor,
            jitter=self.jitter,
            exceptions=self.exceptions,
            on_retry=on_retry,
        )
