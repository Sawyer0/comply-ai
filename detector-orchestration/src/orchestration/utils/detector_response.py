"""Adapters for transforming customer detector responses into DetectorResult."""

from __future__ import annotations

from typing import Any, Callable, Dict

from shared.interfaces.orchestration import DetectorResult


def _default_parser(payload: Dict[str, Any]) -> DetectorResult:
    """Assume payload already matches DetectorResult schema."""
    return DetectorResult.model_validate(payload)


def _confidence_field_parser(payload: Dict[str, Any]) -> DetectorResult:
    """Handle detectors that return confidence in percentage form."""
    if "confidence_percent" in payload and "confidence" not in payload:
        payload = dict(payload)
        payload["confidence"] = float(payload.pop("confidence_percent")) / 100.0
    return DetectorResult.model_validate(payload)


_RESPONSE_PARSERS: Dict[str, Callable[[Dict[str, Any]], DetectorResult]] = {
    "default": _default_parser,
    "confidence_percent": _confidence_field_parser,
}


def get_response_parser(name: str | None) -> Callable[[Dict[str, Any]], DetectorResult]:
    """Return a response parser by name."""
    if not name:
        return _default_parser
    try:
        return _RESPONSE_PARSERS[name]
    except KeyError as exc:  # pragma: no cover - configuration error
        available = ", ".join(sorted(_RESPONSE_PARSERS))
        raise ValueError(
            f"Unknown detector response parser '{name}'. Available: {available}"
        ) from exc


class ResponseParserAlreadyRegisteredError(ValueError):
    """Raised when attempting to register a parser that already exists."""


class ResponseParserNotCallableError(TypeError):
    """Raised when the provided parser is not callable."""


def register_response_parser(
    name: str, parser: Callable[[Dict[str, Any]], DetectorResult]
) -> None:
    """Register a custom response parser."""
    if not callable(parser):
        raise ResponseParserNotCallableError(
            f"Response parser for '{name}' must be callable, got {type(parser)!r}"
        )
    if name in _RESPONSE_PARSERS:
        raise ResponseParserAlreadyRegisteredError(
            f"Response parser '{name}' is already registered"
        )
    _RESPONSE_PARSERS[name] = parser


__all__ = [
    "get_response_parser",
    "register_response_parser",
    "ResponseParserAlreadyRegisteredError",
    "ResponseParserNotCallableError",
]
