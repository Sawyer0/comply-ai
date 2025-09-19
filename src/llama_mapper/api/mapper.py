"""
FastAPI application for the Llama Mapper service.
"""
import asyncio
import logging
from typing import Any, Dict, Callable, cast, AsyncIterator, Awaitable
import time
import uuid
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from ..config.manager import ConfigManager
from ..monitoring.metrics_collector import MetricsCollector
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
    BatchDetectorRequest,
    BatchMappingResponse,
    DetectorRequest,
    MappingResponse,
    Provenance,
)

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

        # Idempotency cache
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
                    "description": "Bad request (missing tenant header when required)"
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
        async def map_detector_output(
            request: DetectorRequest,
            http_request: Request,
            response: Response,
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

            # Enforce tenant context if provided by auth
            if auth and auth.tenant_id:
                if request.tenant_id and request.tenant_id != auth.tenant_id:
                    raise HTTPException(
                        status_code=403,
                        detail="Tenant mismatch between header and request body",
                    )
                # Populate missing tenant_id from header
                if not request.tenant_id:
                    request.tenant_id = auth.tenant_id

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

            try:
                logger.info(
                    f"Processing mapping request {request_id} for detector {request.detector}"
                )

                # Attempt mapping with the fine-tuned model
                result = await self._map_single_request(request, request_id)

                # Track success metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(
                    request.detector, processing_time, True
                )

                logger.info(
                    f"Successfully processed request {request_id} in {processing_time:.3f}s"
                )
                # Save idempotency result
                if cache_key:
                    self._idempotency_cache.set(cache_key, result)
                    response.headers["Idempotency-Key"] = idempotency_key or ""
                return result

            except Exception as e:
                # Track error metrics
                processing_time = time.time() - start_time
                self.metrics_collector.record_request(
                    request.detector, processing_time, False
                )

                logger.error(f"Error processing request {request_id}: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to process mapping request: {str(e)}",
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
                    "description": "Bad request (missing tenant header when required)"
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
            request: BatchDetectorRequest,
            http_request: Request,
            response: Response,
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

            # Enforce tenant context if provided by auth
            if auth and auth.tenant_id:
                # Ensure all items belong to same tenant if tenant_id present in body
                for i, single_request in enumerate(request.requests):
                    if (
                        single_request.tenant_id
                        and single_request.tenant_id != auth.tenant_id
                    ):
                        raise HTTPException(
                            status_code=403, detail=f"Tenant mismatch at index {i}"
                        )
                    if not single_request.tenant_id:
                        single_request.tenant_id = auth.tenant_id

            logger.info(
                f"Processing batch mapping request {request_id} with {len(request.requests)} items"
            )

            results = []
            errors = []

            for i, single_request in enumerate(request.requests):
                try:
                    result = await self._map_single_request(
                        single_request, f"{request_id}-{i}"
                    )
                    results.append(result)
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
            self.metrics_collector.record_batch_request(len(request.requests))

            logger.info(
                f"Processed batch request {request_id} in {processing_time:.3f}s"
            )

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
            # Generate mapping using the fine-tuned model
            model_output = await self.model_server.generate_mapping(
                detector=request.detector,
                output=request.output,
                metadata=request.metadata,
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
        return MappingResponse(
            taxonomy=["OTHER.Unknown"],
            scores={"OTHER.Unknown": 0.0},
            confidence=0.0,
            notes=f"Mapping failed: {error_message}",
            provenance=Provenance(detector=detector, raw_ref=None),
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
