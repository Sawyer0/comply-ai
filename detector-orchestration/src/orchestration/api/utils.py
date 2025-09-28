"""Helper functions for building API responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, TypeVar

from shared.interfaces.base import ApiResponse

T = TypeVar("T")


def build_metadata(
    *,
    request_id: str,
    tenant_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return standard response metadata with optional extras."""

    metadata: Dict[str, Any] = {
        "requestId": request_id,
        "timestamp": datetime.utcnow().isoformat(),
        "tenantId": tenant_id,
    }
    if extra:
        metadata.update(extra)
    return metadata


def make_api_response(
    data: T,
    *,
    request_id: str,
    tenant_id: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> ApiResponse[T]:
    """Wrap data in a standard ApiResponse with consistent metadata."""

    metadata = build_metadata(
        request_id=request_id,
        tenant_id=tenant_id,
        extra=extra_metadata,
    )
    return ApiResponse(data=data, metadata=metadata)


__all__ = ["build_metadata", "make_api_response"]
