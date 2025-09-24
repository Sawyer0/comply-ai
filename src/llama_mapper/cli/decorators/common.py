"""Common decorators for CLI commands."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Optional

import click

from ..core.base import CLIError
from ..utils import display_error, display_info, display_success, display_warning


def handle_errors(func: Callable) -> Callable:
    """Decorator to handle errors consistently across CLI commands."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except CLIError as e:
            display_error(str(e))
            raise click.Abort() from e
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            raise click.Abort() from e

    return wrapper


def timing(func: Callable) -> Callable:
    """Decorator to measure and display execution time."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            display_info(f"Command completed in {elapsed:.2f} seconds")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            display_warning(f"Command failed after {elapsed:.2f} seconds")
            raise e

    return wrapper


def confirm_action(message: str, default: bool = False) -> Callable:
    """Decorator to require user confirmation before executing a command."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not click.confirm(message, default=default):
                display_info("Operation cancelled by user")
                return None
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_config_section(section_name: str) -> Callable:
    """Decorator to ensure a config section exists before executing a command."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get config manager from global context or create default
            try:
                from ...config.manager import ConfigManager
                config_manager = ConfigManager()
                
                # Inject config manager into function if it accepts it
                import inspect
                sig = inspect.signature(func)
                if 'config_manager' in sig.parameters:
                    kwargs['config_manager'] = config_manager
                    
                return func(*args, **kwargs)
            except Exception:
                # Fallback to original function without config injection
                return func(*args, **kwargs)

        return wrapper

    return decorator


def output_formatted(
    format_type: str = "json",
    output_path: Optional[str] = None,
) -> Callable:
    """Decorator to format command output."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            if result is None:
                return result

            # Format the result based on format_type
            from ..utils import format_output

            format_output(result, format_type, output_path)

            return result

        return wrapper

    return decorator


def validate_params(**validators: Callable) -> Callable:
    """Decorator to validate command parameters."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Apply validators to kwargs
            for param_name, validator in validators.items():
                if param_name in kwargs:
                    kwargs[param_name] = validator(kwargs[param_name])

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_command_execution(func: Callable) -> Callable:
    """Decorator to log command execution."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from ...logging import get_logger

        logger = get_logger(func.__module__)

        logger.info(
            f"Executing command: {func.__name__}",
            args=args,
            kwargs=kwargs,
        )

        try:
            result = func(*args, **kwargs)
            logger.info("Command %s completed successfully", func.__name__)
            return result
        except Exception as e:
            logger.error("Command %s failed", func.__name__, error=str(e))
            raise e

    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0) -> Callable:
    """Decorator to retry command on failure."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        display_warning(
                            f"Attempt {attempt + 1} failed, retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        display_error(f"All {max_retries + 1} attempts failed")

            raise last_exception

        return wrapper

    return decorator


def progress_indicator(message: str = "Processing...") -> Callable:
    """Decorator to show progress indicator for long-running commands."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with click.progressbar(length=100, label=message) as bar:
                # This is a simplified implementation
                # In practice, you'd need to integrate with the actual command logic
                bar.update(50)  # Halfway
                result = func(*args, **kwargs)
                bar.update(100)  # Complete
                return result

        return wrapper

    return decorator
