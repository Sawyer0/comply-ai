"""Policy file loading operations.

Core file operations for reading and loading policy files from disk.
Extracted from PolicyLoader to focus on pure file I/O operations.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = [
    "PolicyLoadError",
    "handle_policy_errors",
    "load_policy_file",
    "load_all_policy_files",
]


class PolicyLoadError(Exception):
    """Exception raised when policy loading fails."""


def handle_policy_errors(default_return: Any = None, reraise: bool = False):
    """Decorator to handle common policy operation errors following DRY principle."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except PolicyLoadError as e:
                # Log policy load errors for tracing, then re-raise
                logger.error(
                    "Policy load error in %s: %s",
                    func.__name__,
                    str(e),
                    extra={"error": str(e), "function": func.__name__},
                )
                raise
            except (OSError, PermissionError, FileNotFoundError) as e:
                logger.error(
                    "File system error in %s",
                    func.__name__,
                    extra={"error": str(e), "function": func.__name__},
                )
                if reraise:
                    error_msg = f"File operation failed in {func.__name__}: {str(e)}"
                    raise PolicyLoadError(error_msg) from e
                return default_return
            except UnicodeDecodeError as e:
                logger.error(
                    "Data format error in %s",
                    func.__name__,
                    extra={"error": str(e), "function": func.__name__},
                )
                if reraise:
                    error_msg = f"Data format error in {func.__name__}: {str(e)}"
                    raise PolicyLoadError(error_msg) from e
                return default_return

        return wrapper

    return decorator


def load_policy_file(policy_file: Path) -> str:
    """Load content from a policy file.

    Args:
        policy_file: Path to the policy file

    Returns:
        Policy file content

    Raises:
        PolicyLoadError: If file cannot be read
    """
    try:
        with open(policy_file, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            raise PolicyLoadError(f"Policy file {policy_file} is empty")

        return content

    except FileNotFoundError as e:
        raise PolicyLoadError(f"Policy file not found: {policy_file}") from e
    except PermissionError as e:
        raise PolicyLoadError(f"Permission denied reading policy file: {policy_file}") from e
    except UnicodeDecodeError as e:
        raise PolicyLoadError(f"Invalid encoding in policy file: {policy_file}") from e


def load_all_policy_files(policies_directory: Path) -> Dict[str, str]:
    """Load all .rego policy files from the specified directory.

    Args:
        policies_directory: Directory containing policy files

    Returns:
        Dictionary mapping policy names to their content

    Raises:
        PolicyLoadError: If policy loading fails
    """
    try:
        policy_files = list(policies_directory.glob("*.rego"))

        if not policy_files:
            logger.warning(
                "No .rego policy files found",
                extra={"directory": str(policies_directory)},
            )
            return {}

        loaded_policies = {}

        for policy_file in policy_files:
            try:
                policy_name = policy_file.stem
                policy_content = load_policy_file(policy_file)
                loaded_policies[policy_name] = policy_content

            except (OSError, UnicodeDecodeError, PolicyLoadError) as e:
                logger.error(
                    "Failed to load policy file",
                    extra={"policy_file": str(policy_file), "error": str(e)},
                )
                # Continue loading other policies
                continue

        logger.info(
            "Successfully loaded policy files",
            extra={
                "count": len(loaded_policies),
                "policies": list(loaded_policies.keys()),
            },
        )

        return loaded_policies

    except (OSError, PermissionError) as e:
        error_msg = f"Failed to load policies from {policies_directory}: {str(e)}"
        logger.error("Policy loading failed", extra={"error": str(e)})
        raise PolicyLoadError(error_msg) from e


def load_single_policy_by_name(policies_directory: Path, policy_name: str) -> str | None:
    """Load a specific policy by name.

    Args:
        policies_directory: Directory containing policy files
        policy_name: Name of the policy to load (without .rego extension)

    Returns:
        Policy content if found, None otherwise
    """
    policy_file = policies_directory / f"{policy_name}.rego"

    if not policy_file.exists():
        logger.warning(
            "Policy file not found", 
            extra={"policy_name": policy_name, "directory": str(policies_directory)}
        )
        return None

    try:
        content = load_policy_file(policy_file)
        logger.info("Policy loaded successfully", extra={"policy_name": policy_name})
        return content
    except PolicyLoadError:
        logger.error("Failed to load policy", extra={"policy_name": policy_name})
        return None
