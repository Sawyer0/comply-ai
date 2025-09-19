"""
FastAPI application for the Llama Mapper service.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Optional, cast

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import ValidationError

from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector
from ..security.redaction import SENSITIVE_KEYS
from ..serving.fallback_mapper import FallbackMapper
from ..serving.json_validator import JSONValidator
from ..serving.model_server import ModelServer
from ..versioning import VersionManager
from .auth import (
    AuthContext,
    IdempotencyCache,
    build_api_key_auth,
    build_idempotency_key,
)
from .models import (
    BatchMappingResponse,
    DetectorRequest,
    MappingResponse,
    Provenance,
    MapperPayload,
    VersionInfo,
)
from .errors import build_error_body, http_status_for

logger = logging.getLogger(__name__)


class MapperAPI:
    """FastAPI application for the Llama Mapper service."""

    def __init__(
        self,
        model_server: ModelServer,
        json_validator: JSONValidator,
        fallback_mapper: FallbackMapper,
        config_manager: ConfigManager,
        metrics_collector: MetricsCollector,
    ):
        self.model_server = model_server
        self.json_validator = json_validator
        self.fallback_mapper = fallback_mapper
        self.config_manager = config_manager
        self.metrics_collector = metrics_collector
        self.version_manager = VersionManager()

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            # Startup: nothing required currently
            try:
                yield
            finally:
                # Shutdown: close backend resources (e.g., TGI HTTP session)
                try:
                    close_fn = getattr(self.model_server, "close", None)
                    if close_fn:
                        if asyncio.iscoroutinefunction(close_fn):
                            await close_fn()
                        else:
                            close_fn()
                    logger.info("Shutdown complete: model server resources released")
                except Exception as e:
                    logger.warning(f"Shutdown cleanup encountered an error: {e}")

        self.app = FastAPI(
            title="Llama Mapper API",
            description="AI safety detector output normalization service",
            version="1.0.0",
            lifespan=lifespan,
        )

        # For OpenAPI docs: API key security scheme (docs-only dependency)
        # Be resilient to mocked/partial ConfigManager in tests
        try:
            api_key_header_name = getattr(
                getattr(self.config_manager, "security", None),
                "api_key_header",
                "X-API-Key",
            )
        except Exception:
            api_key_header_name = "X-API-Key"
        self._api_key_header_for_docs = APIKeyHeader(
            name=cast(str, api_key_header_name), auto_error=False
        )

        # Idempotency cache (memory or Redis)
        # Resolve idempotency TTL safely (handle mocks)
        _ttl = 600
        try:
            _ttl_candidate = getattr(
                getattr(self.config_manager, "auth", None),
                "idempotency_cache_ttl_seconds",
                600,
            )
            if isinstance(_ttl_candidate, int):
                _ttl = _ttl_candidate
        except Exception:
            _ttl = 600
        cache_backend = "memory"
        redis_url = None
        redis_prefix = "idem:mapper:"
        try:
            auth_cfg = getattr(self.config_manager, "auth", None)
            cache_backend = getattr(auth_cfg, "idempotency_backend", "memory")
            redis_url = getattr(auth_cfg, "idempotency_redis_url", None)
            redis_prefix = getattr(auth_cfg, "idempotency_redis_prefix", "idem:mapper:")
        except Exception:
            cache_backend = "memory"

        # Annotate cache attribute to allow switching between backends
        self._idempotency_cache: Any
        if cache_backend == "redis" and redis_url:
            try:
                from .idempotency_redis import RedisIdempotencyCache

                redis_cache = RedisIdempotencyCache(
                    redis_url=redis_url, key_prefix=str(redis_prefix), ttl_seconds=_ttl
                )
                # Health check; if down, record metric and fall back
                if redis_cache.is_healthy():
                    self._idempotency_cache = redis_cache
                    try:
                        self.metrics_collector.redis_backend_up.labels(component="idempotency").set(1)  # type: ignore[attr-defined]
                    except Exception:
                        self.metrics_collector.set_gauge("redis_idempotency_up", 1)
                else:
                    self._idempotency_cache = IdempotencyCache(ttl_seconds=_ttl)
                    try:
                        self.metrics_collector.redis_backend_up.labels(component="idempotency").set(0)  # type: ignore[attr-defined]
                        self.metrics_collector.redis_backend_fallback_total.labels(component="idempotency").inc()  # type: ignore[attr-defined]
                    except Exception:
                        self.metrics_collector.set_gauge("redis_idempotency_up", 0)
                        self.metrics_collector.increment_counter(
                            "redis_idempotency_fallback_total"
                        )
            except Exception:
                # Fallback
                self._idempotency_cache = IdempotencyCache(ttl_seconds=_ttl)
                try:
                    self.metrics_collector.redis_backend_up.labels(component="idempotency").set(0)  # type: ignore[attr-defined]
                    self.metrics_collector.redis_backend_fallback_total.labels(component="idempotency").inc()  # type: ignore[attr-defined]
                except Exception:
                    self.metrics_collector.set_gauge("redis_idempotency_up", 0)
                    self.metrics_collector.increment_counter(
                        "redis_idempotency_fallback_total"
                    )
        else:
            self._idempotency_cache = IdempotencyCache(ttl_seconds=_ttl)

        # Auth dependency for mapping write operations
        self._auth_map_write = build_api_key_auth(
            self.config_manager, required_scopes=["map:write"]
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Add request ID middleware
        @self.app.middleware("http")
        async def add_request_id(
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        # Register routes
        self._register_routes()

        # Add RateLimit middleware (after route registration so it's global)
        from .rate_limit import RateLimitMiddleware

        self.app.add_middleware(
            RateLimitMiddleware,
            config_manager=self.config_manager,
            metrics_collector=self.metrics_collector,
        )

        # OpenAPI YAML endpoint (generated from FastAPI schema)
        @self.app.get("/openapi.yaml", include_in_schema=False)
        async def openapi_yaml() -> Response:
            import yaml  # type: ignore

            yaml_text = yaml.safe_dump(self.app.openapi(), sort_keys=False)
            return Response(content=yaml_text, media_type="application/yaml")

    def _is_raw_content_like(self, obj: Any, depth: int = 0) -> bool:
        """
        Heuristic to detect raw customer content in payloads.

        - Flags long free-text strings (length >= 2048 or many newlines)
        - Flags base64-like long blobs
        - Flags suspicious keys from SENSITIVE_KEYS or common content keys
        """
        if obj is None:
            return False
        if depth > 6:
            # Avoid deep recursion
            return False
        suspicious_keys = {
            "text",
            "content",
            "raw",
            "body",
            "document",
            "html",
            "markdown",
            "message",
            "attachment",
            "blob",
        }
        try:
            if isinstance(obj, str):
                if len(obj) >= 2048:
                    return True
                if obj.count("\n") >= 3:
                    return True
                # base64-like long continuous token
                if re.fullmatch(r"[A-Za-z0-9+/=]{256,}", obj or ""):
                    return True
                return False
            if isinstance(obj, dict):
                for k, v in obj.items():
                    k_lower = str(k).lower()
                    if k_lower in suspicious_keys or k_lower in {
                        s.lower() for s in SENSITIVE_KEYS
                    }:
                        if isinstance(v, str) and (len(v) >= 512 or v.count("\n") >= 2):
                            return True
                    if self._is_raw_content_like(v, depth + 1):
                        return True
                return False
            if isinstance(obj, (list, tuple)):
                for it in obj:
                    if self._is_raw_content_like(it, depth + 1):
                        return True
                return False
        except Exception:
            return False
        return False

    def _register_routes(self) -> None:
        """Register API routes."""

        # Resolve tenant header name safely for documentation alias
        try:
            tenant_header_name = getattr(
                getattr(self.config_manager, "security", None),
                "tenant_header",
                "X-Tenant-ID",
            )
        except Exception:
            tenant_header_name = "X-Tenant-ID"

        @self.app.get("/health")
        async def health_check() -> Dict[str, Any]:
            """Health check endpoint."""
            return {"status": "healthy", "timestamp": time.time()}

        @self.app.get("/metrics")
        async def get_metrics() -> Response:
            """Prometheus metrics endpoint."""
            from fastapi import Response

            metrics_data = self.metrics_collector.get_prometheus_metrics()
            return Response(content=metrics_data, media_type="text/plain")

        @self.app.get("/metrics/summary")
        async def get_metrics_summary() -> Dict[str, Any]:
            """Get metrics summary in JSON format."""
            return self.metrics_collector.get_all_metrics()

        @self.app.get("/metrics/alerts")
        async def get_quality_alerts() -> Dict[str, Any]:
            """Get current quality threshold violations."""
            alerts = self.metrics_collector.check_quality_thresholds()
            return {"alerts": alerts, "count": len(alerts), "timestamp": time.time()}

        @self.app.post(
            "/map",
            response_model=MappingResponse,
            summary="Map detector output to canonical taxonomy",
            dependencies=[Depends(self._api_key_header_for_docs)],
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
                403: {
                    "description": "Forbidden (insufficient scope or tenant mismatch)"
                },
                408: {
                    "description": "Request timeout (mapper timeout budget exceeded)"
                },
                422: {"description": "Validation error"},
                429: {"description": "Too Many Requests (rate limited)"},
                500: {"description": "Internal server error"},
            },
        )
        async def map_detector_output(
            http_request: Request,
            response: Response,
            request_body: Dict[str, Any] = Body(...),
            auth: AuthContext = Depends(self._auth_map_write),
            idempotency_key: Optional[str] = Header(
                default=None, alias="Idempotency-Key"
            ),
            _tenant_header_for_docs: Optional[str] = Header(
                default=None, alias=cast(str, tenant_header_name)
            ),
        ) -> MappingResponse:
            """
            Map a single detector output to canonical taxonomy.

            Args:
                request: Detector output to map
                http_request: FastAPI request object for metadata

            Returns:
                MappingResponse: Canonical taxonomy mapping

            Raises:
                HTTPException: If mapping fails
            """
            request_id = getattr(http_request.state, "request_id", "unknown")
            start_time = time.time()

            # Size/Privacy enforcement before parsing
            try:
                payload_bytes = json.dumps(request_body).encode("utf-8")
            except Exception:
                payload_bytes = b"{}"
            try:
                max_kb = int(getattr(self.config_manager.serving, "max_payload_kb", 64))
            except Exception:
                max_kb = 64
            if len(payload_bytes) > max_kb * 1024:
                self.metrics_collector.record_payload_rejection("oversize")
                code = "INVALID_REQUEST"
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(request_id, code, f"Payload too large (> {max_kb} KB)").model_dump(),
                )
            try:
                reject_on_raw = bool(
                    getattr(self.config_manager.serving, "reject_on_raw_content", True)
                )
            except Exception:
                reject_on_raw = True
            if reject_on_raw and self._is_raw_content_like(request_body):
                self.metrics_collector.record_payload_rejection("raw_content")
                code = "INVALID_REQUEST"
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(request_id, code, "Raw content detected in payload; send detector outputs only").model_dump(),
                )

            # Parse as MapperPayload first, fallback to legacy DetectorRequest
            request_type = "DetectorRequest"
            normalized_request: DetectorRequest
            try:
                mp = MapperPayload.model_validate(request_body)
                request_type = "MapperPayload"
                normalized_request = DetectorRequest(
                    detector=mp.detector,
                    output=mp.output,
                    metadata=mp.metadata.model_dump() if mp.metadata else None,
                    tenant_id=mp.tenant_id,
                )
            except ValidationError:
                # Try legacy
                try:
                    normalized_request = DetectorRequest.model_validate(request_body)
                except ValidationError:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "error_code": "INVALID_REQUEST",
                            "message": "Request body does not match MapperPayload or DetectorRequest",
                            "request_id": request_id,
                            "retryable": False,
                        },
                    )

            # Enforce tenant context based on auth configuration
            auth_enabled = False
            try:
                _enabled_val = getattr(
                    getattr(self.config_manager, "auth", None), "enabled", False
                )
                auth_enabled = (
                    bool(_enabled_val) if isinstance(_enabled_val, bool) else False
                )
            except Exception:
                auth_enabled = False

            if auth and auth.tenant_id:
                if (
                    normalized_request.tenant_id
                    and normalized_request.tenant_id != auth.tenant_id
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="Tenant mismatch between header and request body",
                    )
                # Populate missing tenant_id from header
                if not normalized_request.tenant_id:
                    normalized_request.tenant_id = auth.tenant_id
            elif auth_enabled:
                # When auth is enabled but no tenant is resolved, require tenant_id
                if not normalized_request.tenant_id:
                    code = "INVALID_REQUEST"
                    raise HTTPException(
                        status_code=http_status_for(code),
                        detail=build_error_body(request_id, code, "tenant_id is required").model_dump(),
                    )

            # Idempotency check
            cache_key = build_idempotency_key(
                auth.tenant_id if auth else None, "/map", idempotency_key
            )
            if cache_key:
                cached = self._idempotency_cache.get(cache_key)
                if cached is not None:
                    response.headers["X-Request-ID"] = request_id
                    response.headers["Idempotency-Key"] = idempotency_key or ""
                    return cast(MappingResponse, cached)

            # Deprecation header/metric for legacy requests
            if request_type == "DetectorRequest":
                response.headers["Deprecation"] = "true"
                # Sunset header (RFC 1123 date)
                response.headers["Sunset"] = "Fri, 31 Oct 2025 00:00:00 GMT"
                response.headers["Link"] = "<https://github.com/your-org/comply-ai/blob/main/docs/release/mapper_migration.md>; rel=\"sunset\""
                self.metrics_collector.record_deprecated_request("DetectorRequest")

            try:
                logger.info(
                    f"Processing mapping request {request_id} for detector {normalized_request.detector}"
                )

                # Attempt mapping with the fine-tuned model
                result = await self._map_single_request(normalized_request, request_id)

                # Track success metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(
                    normalized_request.detector, processing_time, True
                )

                logger.info(
                    f"Successfully processed request {request_id} in {processing_time:.3f}s"
                )
                # Save idempotency result
                if cache_key:
                    self._idempotency_cache.set(cache_key, result)
                    response.headers["Idempotency-Key"] = idempotency_key or ""
                return result

            except asyncio.TimeoutError:
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(
                    normalized_request.detector, processing_time, False
                )
                code = "REQUEST_TIMEOUT"
                self.metrics_collector.record_error(code)
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(request_id, code, "Mapper timeout budget exceeded").model_dump(),
                )
            except Exception as e:
                # Track error metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(
                    normalized_request.detector, processing_time, False
                )

                logger.error(f"Error processing request {request_id}: {str(e)}")
                code = "INTERNAL_ERROR"
                raise HTTPException(
                    status_code=http_status_for(code),
                    detail=build_error_body(request_id, code, f"Failed to process mapping request: {str(e)}").model_dump(),
                )

        @self.app.post(
            "/map/batch",
            response_model=BatchMappingResponse,
            summary="Batch map detector outputs to canonical taxonomy",
            dependencies=[Depends(self._api_key_header_for_docs)],
            responses={
                200: {
                    "description": "Batch mapping succeeded",
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
                403: {
                    "description": "Forbidden (insufficient scope or tenant mismatch)"
                },
                422: {"description": "Validation error"},
                429: {"description": "Too Many Requests (rate limited)"},
                500: {"description": "Internal server error"},
            },
        )
        async def map_detector_outputs_batch(
            http_request: Request,
            response: Response,
            request_body: Dict[str, Any] = Body(...),
            auth: AuthContext = Depends(self._auth_map_write),
            idempotency_key: Optional[str] = Header(
                default=None, alias="Idempotency-Key"
            ),
            _tenant_header_for_docs: Optional[str] = Header(
                default=None, alias=cast(str, tenant_header_name)
            ),
        ) -> BatchMappingResponse:
            """
            Map multiple detector outputs to canonical taxonomy.

            Args:
                request: Batch of detector outputs to map
                http_request: FastAPI request object for metadata

            Returns:
                BatchMappingResponse: Batch of canonical taxonomy mappings
            """
            request_id = getattr(http_request.state, "request_id", "unknown")
            start_time = time.time()

            # Normalize batch items and enforce tenant context
            raw_items = []
            try:
                raw_items = list(request_body.get("requests", []))
            except Exception:
                raise HTTPException(status_code=422, detail="Invalid batch body")
            if not raw_items:
                raise HTTPException(status_code=422, detail="Empty batch")

            normalized_items: list[DetectorRequest] = []
            used_legacy = False
            for idx, item in enumerate(raw_items):
                # Reject oversize or raw content item
                try:
                    max_kb = int(
                        getattr(self.config_manager.serving, "max_payload_kb", 64)
                    )
                except Exception:
                    max_kb = 64
                if len(json.dumps(item).encode("utf-8")) > max_kb * 1024:
                    self.metrics_collector.record_payload_rejection("oversize")
                    code = "INVALID_REQUEST"
                    raise HTTPException(status_code=http_status_for(code), detail=build_error_body(request_id, code, f"Item {idx} too large").model_dump())
                try:
                    reject_on_raw = bool(
                        getattr(
                            self.config_manager.serving, "reject_on_raw_content", True
                        )
                    )
                except Exception:
                    reject_on_raw = True
                if reject_on_raw and self._is_raw_content_like(item):
                    self.metrics_collector.record_payload_rejection("raw_content")
                    code = "INVALID_REQUEST"
                    raise HTTPException(
                        status_code=http_status_for(code),
                        detail=build_error_body(request_id, code, f"Item {idx} contains raw content").model_dump(),
                    )

                # Parse MapperPayload preferred
                parsed: Optional[DetectorRequest] = None
                try:
                    mp = MapperPayload.model_validate(item)
                    parsed = DetectorRequest(
                        detector=mp.detector,
                        output=mp.output,
                        metadata=mp.metadata.model_dump() if mp.metadata else None,
                        tenant_id=mp.tenant_id,
                    )
                except ValidationError:
                    try:
                        parsed = DetectorRequest.model_validate(item)
                        used_legacy = True
                    except ValidationError:
                        raise HTTPException(
                            status_code=422, detail=f"Item {idx} invalid"
                        )

                # Tenant enforcement
                auth_enabled = False
                try:
                    _enabled_val = getattr(getattr(self.config_manager, "auth", None), "enabled", False)
                    auth_enabled = bool(_enabled_val) if isinstance(_enabled_val, bool) else False
                except Exception:
                    auth_enabled = False

                if auth and auth.tenant_id:
                    if parsed.tenant_id and parsed.tenant_id != auth.tenant_id:
                        raise HTTPException(
                            status_code=403, detail=f"Tenant mismatch at index {idx}"
                        )
                    if not parsed.tenant_id:
                        parsed.tenant_id = auth.tenant_id
                elif auth_enabled:
                    if not parsed.tenant_id:
                        code = "INVALID_REQUEST"
                        raise HTTPException(
                            status_code=http_status_for(code),
                            detail=build_error_body(request_id, code, f"Missing tenant_id at index {idx}").model_dump(),
                        )

                normalized_items.append(parsed)

            logger.info(
                f"Processing batch mapping request {request_id} with {len(normalized_items)} items"
            )

            results = []
            errors = []

            for i, single_request in enumerate(normalized_items):
                try:
                    result = await self._map_single_request(
                        single_request, f"{request_id}-{i}"
                    )
                    results.append(result)
                except asyncio.TimeoutError:
                    self.metrics_collector.record_error("REQUEST_TIMEOUT")
                    errors.append(
                        {
                            "index": i,
                            "error": "REQUEST_TIMEOUT",
                            "detector": single_request.detector,
                        }
                    )
                    results.append(
                        self._create_error_response(
                            single_request.detector, "REQUEST_TIMEOUT"
                        )
                    )
                except Exception as e:
                    logger.error(f"Error processing batch item {i}: {str(e)}")
                    errors.append(
                        {
                            "index": i,
                            "error": str(e),
                            "detector": single_request.detector,
                        }
                    )
                    # Add a placeholder result for failed items
                    results.append(
                        self._create_error_response(single_request.detector, str(e))
                    )

            processing_time = time.time() - start_time
            self.metrics_collector.record_histogram(
                "batch_request_duration_seconds", processing_time
            )
            self.metrics_collector.record_batch_request(len(normalized_items))

            logger.info(
                f"Processed batch request {request_id} in {processing_time:.3f}s"
            )

            # Deprecation header if legacy detected in batch
            if used_legacy:
                response.headers["Deprecation"] = "true"
                response.headers["Sunset"] = "Fri, 31 Oct 2025 00:00:00 GMT"
                response.headers["Link"] = "<https://github.com/your-org/comply-ai/blob/main/docs/release/mapper_migration.md>; rel=\"sunset\""
                self.metrics_collector.record_deprecated_request("DetectorRequest")

            # Save idempotency result
            cache_key = build_idempotency_key(
                auth.tenant_id if auth else None, "/map/batch", idempotency_key
            )
            if cache_key:
                self._idempotency_cache.set(
                    cache_key,
                    BatchMappingResponse(
                        results=results, errors=errors if errors else None
                    ),
                )
                response.headers["Idempotency-Key"] = idempotency_key or ""

            return BatchMappingResponse(
                results=results, errors=errors if errors else None
            )

    async def _map_single_request(
        self, request: DetectorRequest, request_id: str
    ) -> MappingResponse:
        """
        Map a single detector request to canonical taxonomy.

        Args:
            request: Single detector request
            request_id: Unique request identifier

        Returns:
            MappingResponse: Canonical taxonomy mapping
        """
        # Create provenance information
        from datetime import datetime, timezone

        provenance = Provenance(
            detector=request.detector,
            tenant_id=request.tenant_id,
            ts=datetime.now(timezone.utc),
            raw_ref=None,
        )

        # Kill-switch: force rule-only mapping when runtime mode is rules_only
        try:
            runtime_mode = getattr(self.config_manager.serving, "mode", "hybrid")
        except Exception:
            runtime_mode = "hybrid"
        if runtime_mode == "rules_only":
            logger.warning("Kill-switch active: forcing rule-only mapping")
            fallback_result = self.fallback_mapper.map(
                request.detector, request.output, reason="kill_switch"
            )
            # Enrich provenance and notes with version tags
            self.version_manager.apply_to_provenance(provenance)
            fallback_result.provenance = provenance
            fallback_result.notes = self.version_manager.annotate_notes_with_versions(
                "Kill-switch active: rule-based mapping enforced"
            )
            self.metrics_collector.record_fallback_usage(
                request.detector, "kill_switch"
            )
            return fallback_result

        try:
            # Generate mapping using the fine-tuned model with timeout budget
            try:
                timeout_ms = getattr(
                    self.config_manager.serving, "mapper_timeout_ms", 500
                )
            except Exception:
                timeout_ms = 500
            model_output = await asyncio.wait_for(
                self.model_server.generate_mapping(
                    detector=request.detector,
                    output=request.output,
                    metadata=request.metadata,
                ),
                timeout=float(timeout_ms) / 1000.0,
            )

            # Validate the model output against JSON schema
            is_valid, validation_errors = self.json_validator.validate(model_output)
            self.metrics_collector.record_schema_validation(request.detector, is_valid)

            if is_valid:
                # Parse the validated output
                parsed_output = self.json_validator.parse_output(model_output)

                # Record confidence score
                self.metrics_collector.record_confidence_score(
                    request.detector, parsed_output.confidence
                )

                # Check confidence threshold
                confidence_threshold = self.config_manager.confidence.threshold
                if parsed_output.confidence >= confidence_threshold:
                    # Use model output
                    self.metrics_collector.record_model_success(request.detector)
                    # Enrich provenance and notes with version tags
                    self.version_manager.apply_to_provenance(provenance)
                    parsed_output.provenance = provenance
                    parsed_output.notes = (
                        self.version_manager.annotate_notes_with_versions(
                            parsed_output.notes
                        )
                    )
                    # Attach version_info (Sec 10)
                    vi = self.version_manager.get_version_info_dict()
                    try:
                        parsed_output.version_info = VersionInfo(**vi)
                    except Exception:
                        pass
                    return parsed_output
                else:
                    # Confidence too low, use fallback
                    logger.warning(
                        f"Model confidence {parsed_output.confidence} below threshold {confidence_threshold}"
                    )
                    self.metrics_collector.record_fallback_usage(
                        request.detector, "low_confidence"
                    )
            else:
                # Schema validation failed
                logger.warning(f"Schema validation failed: {validation_errors}")
                self.metrics_collector.record_fallback_usage(
                    request.detector, "schema_validation_failed"
                )

        except asyncio.TimeoutError:
            # Propagate timeout to route handler; do not fallback here per contract
            raise
        except Exception as e:
            logger.error(f"Model generation failed: {str(e)}")
            self.metrics_collector.record_model_error(
                request.detector, "generation_failed"
            )

        # Fall back to rule-based mapping
        logger.info(f"Using fallback mapping for detector {request.detector}")
        fallback_result = self.fallback_mapper.map(request.detector, request.output)
        # Enrich provenance and notes with version tags
        self.version_manager.apply_to_provenance(provenance)
        fallback_result.provenance = provenance
        fallback_result.notes = self.version_manager.annotate_notes_with_versions(
            "Generated using rule-based fallback mapping"
        )
        # Attach version_info as well for auditability
        vi = self.version_manager.get_version_info_dict()
        try:
            fallback_result.version_info = VersionInfo(**vi)
        except Exception:
            pass

        self.metrics_collector.record_fallback_usage(request.detector, "model_error")
        return fallback_result

    def _create_error_response(
        self, detector: str, error_message: str
    ) -> MappingResponse:
        """
        Create an error response for failed mappings.

        Args:
            detector: Name of the detector
            error_message: Error message

        Returns:
            MappingResponse: Error response with OTHER.Unknown mapping
        """
        vi = self.version_manager.get_version_info_dict()
        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.0},
            confidence=0.0,
            notes=f"Mapping failed: {error_message}",
            provenance=Provenance(detector=detector, raw_ref=None),
            version_info=VersionInfo(**vi),
        )


def create_app(
    model_server: ModelServer,
    json_validator: JSONValidator,
    fallback_mapper: FallbackMapper,
    config_manager: ConfigManager,
    metrics_collector: MetricsCollector,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        model_server: Model serving backend
        json_validator: JSON schema validator
        fallback_mapper: Rule-based fallback mapper
        config_manager: Configuration manager
        metrics_collector: Metrics collection service

    Returns:
        FastAPI: Configured FastAPI application
    """
    mapper_api = MapperAPI(
        model_server=model_server,
        json_validator=json_validator,
        fallback_mapper=fallback_mapper,
        config_manager=config_manager,
        metrics_collector=metrics_collector,
    )
    return mapper_api.app
