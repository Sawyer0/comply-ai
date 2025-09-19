from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from .models import MappingResponse, MapperPayload
from .config import Settings


class MapperClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def map(
        self,
        payload: MapperPayload,
        tenant_id: str,
        idempotency_key: Optional[str] = None,
    ) -> tuple[Optional[MappingResponse], Optional[str], int]:
        timeout = self.settings.config.sla.mapper_timeout_budget_ms / 1000
        headers: Dict[str, str] = {
            self.settings.config.tenant_header: tenant_id,
        }
        mapper_api_key = getattr(self.settings, "mapper_api_key", None)
        if mapper_api_key:
            headers[self.settings.config.api_key_header] = mapper_api_key
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.settings.config.mapper_endpoint, json=payload.model_dump(), headers=headers)
                if resp.status_code == 200:
                    return (MappingResponse(**resp.json()), None, 200)
                # Map errors to canonical codes per contract
                if resp.status_code == 429:
                    return (None, "DETECTOR_OVERLOADED", 429)
                if resp.status_code == 401:
                    return (None, "UNAUTHORIZED", 401)
                if resp.status_code == 403:
                    return (None, "INSUFFICIENT_RBAC", 403)
                if resp.status_code == 400:
                    return (None, "INVALID_REQUEST", 400)
                return (None, "DETECTOR_COMMUNICATION_FAILED", 502)
        except Exception:
            return (None, "DETECTOR_COMMUNICATION_FAILED", 502)

