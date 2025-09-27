"""Shared helpers for registry-style operations."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional

from shared.utils.correlation import get_correlation_id


def run_registry_operation(
    operation: Callable[[], bool],
    *,
    logger: logging.Logger,
    success_message: str,
    error_message: str,
    success_args: Iterable[Any] = (),
    error_args: Iterable[Any] = (),
    log_context: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Execute a registry mutation with consistent logging/error handling."""

    correlation_id = get_correlation_id()
    extra: MutableMapping[str, Any] = {"correlation_id": correlation_id}
    if log_context:
        extra.update(log_context)

    try:
        result = operation()
        logger.info(success_message, *success_args, extra=dict(extra))
        return bool(result)
    except Exception as exc:  # pragma: no cover - logging wrapper
        error_extra = dict(extra)
        error_extra["error"] = str(exc)
        logger.error(error_message, *error_args, extra=error_extra)
        return False
