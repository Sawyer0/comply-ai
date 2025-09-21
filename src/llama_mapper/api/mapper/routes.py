"""Route registration for the Mapper FastAPI application."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, TYPE_CHECKING, cast

from fastapi import Body, Depends, Header, HTTPException, Request, Response
from pydantic import ValidationError

from ..auth import AuthContext, build_idempotency_key
from ..errors import build_error_body, http_status_for
from ..models import BatchMappingResponse, DetectorRequest, MapperPayload, MappingResponse

if TYPE_CHECKING:  # pragma: no cover
    from .app import MapperAPI

logger = logging.getLogger(__name__)


def register_routes(mapper: "MapperAPI") -> None:
    """Attach all HTTP routes to the underlying FastAPI app."""

    app = mapper.app
    metrics = mapper.metrics_collector
    service = mapper.service

    try:
        tenant_header_name = getattr(
            getattr(mapper.config_manager, "security", None),
            "tenant_header",
            "X-Tenant-ID",
        )
    except (AttributeError, TypeError) as _:
        # Configuration retrieval failed, using default tenant header
        # This can happen if config_manager.security is None or doesn't have tenant_header
        tenant_header_name = "X-Tenant-ID"

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        return {"status": "healthy", "timestamp": time.time()}

    @app.get("/metrics")
    async def get_metrics() -> Response:
        metrics_data = metrics.get_prometheus_metrics()
        return Response(content=metrics_data, media_type="text/plain")

    @app.get("/metrics/summary")
    async def get_metrics_summary() -> Dict[str, Any]:
        return metrics.get_all_metrics()

    @app.get("/metrics/alerts")
    async def get_quality_alerts() -> Dict[str, Any]:
        alerts = metrics.check_quality_thresholds()
        return {"alerts": alerts, "count": len(alerts), "timestamp": time.time()}

    @app.post(
        "/map",
        response_model=MappingResponse,
        summary="Map detector output to canonical taxonomy",
        dependencies=[Depends(mapper.api_key_header_for_docs)],
        responses={
            200: {
                "description": "Mapping succeeded",
                "headers": {
                    "X-Request-ID": {
                        "description": "Request ID",
                        "schema": {"type": "string"},
                    },
                    "Idempotency-Key": {
                        "description": "Echoed if provided",
                        "schema": {"type": "string"},
                    },
                    "RateLimit-Limit": {
                        "description": "Rate limit and window",
                        "schema": {"type": "string"},
                    },
                    "RateLimit-Remaining": {
                        "description": "Remaining requests in window",
                        "schema": {"type": "integer"},
                    },
                    "RateLimit-Reset": {
                        "description": "Seconds until reset",
                        "schema": {"type": "integer"},
                    },
                    "X-RateLimit-Limit": {
                        "description": "Legacy limit header",
                        "schema": {"type": "integer"},
                    },
                    "X-RateLimit-Remaining": {
                        "description": "Legacy remaining header",
                        "schema": {"type": "integer"},
                    },
                },
            },
            400: {
                "description": "Bad request (payload rejected: oversize or raw content)"
            },
            401: {"description": "Unauthorized (missing/invalid API key)"},
            403: {"description": "Forbidden (insufficient scope or tenant mismatch)"},
            408: {"description": "Request timeout (mapper timeout budget exceeded)"},
            422: {"description": "Validation error"},
            429: {"description": "Too Many Requests (rate limited)"},
            500: {"description": "Internal server error"},
        },
    )
    async def map_detector_output(
        http_request: Request,
        response: Response,
        request_body: Dict[str, Any] = Body(...),
        auth: AuthContext = Depends(mapper.auth_dependency),
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        _tenant_header_for_docs: Optional[str] = Header(
            default=None, alias=cast(str, tenant_header_name)
        ),
    ) -> MappingResponse:
        request_id = getattr(http_request.state, "request_id", "unknown")
        start_time = time.time()

        try:
            payload_bytes = json.dumps(request_body).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as _:
            # JSON serialization failed, using empty dict as fallback
            payload_bytes = b"{}"
        try:
            max_kb = int(getattr(mapper.config_manager.serving, "max_payload_kb", 64))
        except (AttributeError, TypeError, ValueError) as _:
            # Configuration retrieval failed, using default payload limit
            max_kb = 64
        if len(payload_bytes) > max_kb * 1024:
            metrics.record_payload_rejection("oversize")
            code = "INVALID_REQUEST"
            raise HTTPException(
                status_code=http_status_for(code),
                detail=build_error_body(
                    request_id, code, f"Payload too large (> {max_kb} KB)"
                ).model_dump(),
            )
        try:
            reject_on_raw = bool(
                getattr(mapper.config_manager.serving, "reject_on_raw_content", True)
            )
        except (AttributeError, TypeError, ValueError) as _:
            # Configuration retrieval failed, using conservative default
            reject_on_raw = True
        if reject_on_raw and service.is_raw_content_like(request_body):
            metrics.record_payload_rejection("raw_content")
            code = "INVALID_REQUEST"
            raise HTTPException(
                status_code=http_status_for(code),
                detail=build_error_body(
                    request_id,
                    code,
                    "Raw content detected in payload; send detector outputs only",
                ).model_dump(),
            )

        request_type = "DetectorRequest"
        try:
            mapper_payload = MapperPayload.model_validate(request_body)
            request_type = "MapperPayload"
            normalized_request = DetectorRequest(
                detector=mapper_payload.detector,
                output=mapper_payload.output,
                metadata=mapper_payload.metadata.model_dump()
                if mapper_payload.metadata
                else None,
                tenant_id=mapper_payload.tenant_id,
            )
        except ValidationError:
            try:
                normalized_request = DetectorRequest.model_validate(request_body)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error_code": "INVALID_REQUEST",
                        "message": "Request body does not match MapperPayload or DetectorRequest",
                        "request_id": request_id,
                        "retryable": False,
                    },
                ) from exc

        auth_enabled = False
        try:
            enabled_val = getattr(
                getattr(mapper.config_manager, "auth", None), "enabled", False
            )
            auth_enabled = bool(enabled_val) if isinstance(enabled_val, bool) else False
        except (AttributeError, TypeError) as _:
            # Authentication configuration retrieval failed, disabling auth
            auth_enabled = False

        if auth and auth.tenant_id:
            if normalized_request.tenant_id and normalized_request.tenant_id != auth.tenant_id:
                raise HTTPException(
                    status_code=403,
                    detail="Tenant mismatch between header and request body",
                )
            if not normalized_request.tenant_id:
                normalized_request.tenant_id = auth.tenant_id
        elif auth_enabled and not normalized_request.tenant_id:
            code = "INVALID_REQUEST"
            raise HTTPException(
                status_code=http_status_for(code),
                detail=build_error_body(
                    request_id, code, "tenant_id is required"
                ).model_dump(),
            )

        cache_key = build_idempotency_key(
            auth.tenant_id if auth else None, "/map", idempotency_key
        )
        if cache_key:
            cached = mapper.idempotency_cache.get(cache_key)
            if cached is not None:
                response.headers["X-Request-ID"] = request_id
                response.headers["Idempotency-Key"] = idempotency_key or ""
                metrics.record_request("cache", 0.0, success=True)
                return cached

        try:
            result = await service.map_single_request(normalized_request, request_id)
            metrics.record_request("model", time.time() - start_time, success=True)
            response.headers["X-Request-ID"] = request_id
            if cache_key:
                mapper.idempotency_cache.set(cache_key, result)
                response.headers["Idempotency-Key"] = idempotency_key or ""
            if request_type == "DetectorRequest":
                response.headers["Deprecation"] = "true"
                response.headers["Sunset"] = "Fri, 31 Oct 2025 00:00:00 GMT"
                response.headers["Link"] = (
                    '<https://github.com/your-org/comply-ai/blob/main/docs/release/'
                    'mapper_migration.md>; rel="sunset"'
                )
                metrics.record_deprecated_request("DetectorRequest")
            return result
        except asyncio.TimeoutError as exc:
            metrics.record_error("REQUEST_TIMEOUT")
            raise HTTPException(status_code=408, detail="Mapper timeout exceeded") from exc
        except HTTPException as http_exc:
            raise http_exc
        except (RuntimeError, ConnectionError, OSError) as exc:
            metrics.record_error("INTERNAL_ERROR")
            logger.error("Mapping failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Mapping failed") from exc

    @app.post(
        "/map/batch",
        response_model=BatchMappingResponse,
        summary="Map a batch of detector outputs",
        dependencies=[Depends(mapper.api_key_header_for_docs)],
    )
    async def map_batch(
        http_request: Request,
        response: Response,
        request_body: Dict[str, Any] = Body(...),
        auth: AuthContext = Depends(mapper.auth_dependency),
        idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
        _tenant_header_for_docs: Optional[str] = Header(
            default=None, alias=cast(str, tenant_header_name)
        ),
    ) -> BatchMappingResponse:
        request_id = getattr(http_request.state, "request_id", "unknown")
        start_time = time.time()

        try:
            raw_items = list(request_body.get("requests", []))
        except (TypeError, AttributeError, KeyError) as exc:
            # Request body parsing failed - invalid structure
            raise HTTPException(status_code=422, detail="Invalid batch body") from exc
        if not raw_items:
            raise HTTPException(status_code=422, detail="Empty batch")

        normalized_items: list[DetectorRequest] = []
        used_legacy = False
        for idx, item in enumerate(raw_items):
            try:
                max_kb = int(getattr(mapper.config_manager.serving, "max_payload_kb", 64))
            except (AttributeError, TypeError, ValueError) as _:
                # Configuration retrieval failed, using default payload limit
                max_kb = 64
            if len(json.dumps(item).encode("utf-8")) > max_kb * 1024:
                metrics.record_payload_rejection("oversize")
                code = "INVALID_REQUEST"
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(
                        request_id, code, f"Item {idx} too large"
                    ).model_dump(),
                )
            try:
                reject_on_raw = bool(
                    getattr(mapper.config_manager.serving, "reject_on_raw_content", True)
                )
            except (AttributeError, TypeError, ValueError) as _:
                # Configuration retrieval failed, using conservative default
                reject_on_raw = True
            if reject_on_raw and service.is_raw_content_like(item):
                metrics.record_payload_rejection("raw_content")
                code = "INVALID_REQUEST"
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(
                        request_id, code, f"Item {idx} contains raw content"
                    ).model_dump(),
                )

            parsed_request: Optional[DetectorRequest] = None
            try:
                mapper_payload = MapperPayload.model_validate(item)
                parsed_request = DetectorRequest(
                    detector=mapper_payload.detector,
                    output=mapper_payload.output,
                    metadata=mapper_payload.metadata.model_dump()
                    if mapper_payload.metadata
                    else None,
                    tenant_id=mapper_payload.tenant_id,
                )
            except ValidationError:
                try:
                    parsed_request = DetectorRequest.model_validate(item)
                    used_legacy = True
                except ValidationError as exc:
                    raise HTTPException(
                        status_code=422, detail=f"Item {idx} invalid"
                    ) from exc

            auth_enabled = False
            try:
                enabled_val = getattr(
                    getattr(mapper.config_manager, "auth", None), "enabled", False
                )
                auth_enabled = bool(enabled_val) if isinstance(enabled_val, bool) else False
            except (AttributeError, TypeError) as _:
                # Authentication configuration retrieval failed, disabling auth
                auth_enabled = False

            if auth and auth.tenant_id:
                if parsed_request.tenant_id and parsed_request.tenant_id != auth.tenant_id:
                    raise HTTPException(
                        status_code=403, detail=f"Tenant mismatch at index {idx}"
                    )
                if not parsed_request.tenant_id:
                    parsed_request.tenant_id = auth.tenant_id
            elif auth_enabled and not parsed_request.tenant_id:
                code = "INVALID_REQUEST"
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(
                        request_id, code, f"Missing tenant_id at index {idx}"
                    ).model_dump(),
                )

            normalized_items.append(parsed_request)

        logger.info(
            "Processing batch mapping request %s with %s items",
            request_id,
            len(normalized_items),
        )

        results = []
        errors = []
        for idx, single_request in enumerate(normalized_items):
            try:
                result = await service.map_single_request(
                    single_request, f"{request_id}-{idx}"
                )
                results.append(result)
            except asyncio.TimeoutError:
                metrics.record_error("REQUEST_TIMEOUT")
                errors.append(
                    {
                        "index": idx,
                        "error": "REQUEST_TIMEOUT",
                        "detector": single_request.detector,
                    }
                )
                results.append(
                    service.create_error_response(
                        single_request.detector, "REQUEST_TIMEOUT"
                    )
                )
            except (RuntimeError, ConnectionError, OSError, ValueError) as exc:
                logger.error("Error processing batch item %s: %s", idx, exc)
                errors.append(
                    {
                        "index": idx,
                        "error": str(exc),
                        "detector": single_request.detector,
                    }
                )
                results.append(
                    service.create_error_response(single_request.detector, str(exc))
                )

        processing_time = time.time() - start_time
        metrics.record_histogram("batch_request_duration_seconds", processing_time)
        metrics.record_batch_request(len(normalized_items))
        logger.info(
            "Processed batch request %s in %.3fs", request_id, processing_time
        )

        if used_legacy:
            response.headers["Deprecation"] = "true"
            response.headers["Sunset"] = "Fri, 31 Oct 2025 00:00:00 GMT"
            response.headers["Link"] = (
                '<https://github.com/your-org/comply-ai/blob/main/docs/release/'
                'mapper_migration.md>; rel="sunset"'
            )
            metrics.record_deprecated_request("DetectorRequest")

        cache_key = build_idempotency_key(
            auth.tenant_id if auth else None, "/map/batch", idempotency_key
        )
        if cache_key:
            mapper.idempotency_cache.set(
                cache_key,
                BatchMappingResponse(results=results, errors=errors or None),
            )
            response.headers["Idempotency-Key"] = idempotency_key or ""

        return BatchMappingResponse(results=results, errors=errors or None)
