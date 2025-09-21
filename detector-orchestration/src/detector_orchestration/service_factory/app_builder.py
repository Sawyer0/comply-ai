"""FastAPI application builder for the orchestration service."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from detector_orchestration import __version__ as ORCH_VERSION
from detector_orchestration.config import Settings
from detector_orchestration.errors import build_error_body
from detector_orchestration.models import (
    OrchestrationRequest,
    OrchestrationResponse,
    ProcessingMode,
    RoutingDecision,
    RoutingPlan,
)
from detector_orchestration.policy import (
    ApprovalDecision,
    PolicySubmission,
    PolicyVersionStatus,
    ReviewRecord,
    RollbackRequest,
    TenantPolicy,
)
from detector_orchestration.rate_limit import OrchestratorRateLimitMiddleware
from detector_orchestration.registry import DetectorRegistration
from detector_orchestration.security import AuthContext, build_api_key_auth

from .factory import OrchestrationServiceFactory

# pylint: disable=too-many-branches,too-many-statements,too-many-locals,too-few-public-methods

logger = logging.getLogger(__name__)


class OrchestrationAppBuilder:
    """Builder for creating FastAPI orchestration application."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.factory = OrchestrationServiceFactory(settings)
        self.app: Optional[FastAPI] = None

    def _create_lifespan_handler(self) -> Callable:
        """Create lifespan handler for the FastAPI app."""

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            try:
                await self.factory.container.initialize_all()
                yield
            finally:
                await self.factory.container.shutdown_all()

        return lifespan

    def _add_middleware(self, app: FastAPI) -> None:
        """Add middleware to the application."""
        app.add_middleware(
            OrchestratorRateLimitMiddleware,
            settings=self.settings,
            metrics=self.factory.metrics,
        )

    def _add_exception_handlers(self) -> None:
        """Add exception handlers."""

        assert self.app is not None  # FastAPI instance required for handler registration

        async def unhandled_exception_handler(_request, exc: Exception):
            body = build_error_body(
                request_id=None, code="INTERNAL_ERROR", message=str(exc)
            ).model_dump()
            return {"detail": body}

        self.app.add_exception_handler(Exception, unhandled_exception_handler)

    def _add_auth_dependencies(self) -> Dict[str, Callable]:
        """Create authentication dependencies."""
        return {
            "auth_orchestrate_write": build_api_key_auth(
                self.settings,
                required_scopes=["orchestrate:write"],
                metrics=self.factory.metrics,
            ),
            "auth_orchestrate_status": build_api_key_auth(
                self.settings,
                required_scopes=["orchestrate:status"],
                metrics=self.factory.metrics,
            ),
            "auth_registry_read": build_api_key_auth(
                self.settings,
                required_scopes=["registry:read"],
                metrics=self.factory.metrics,
            ),
            "auth_registry_write": build_api_key_auth(
                self.settings,
                required_scopes=["registry:write"],
                metrics=self.factory.metrics,
            ),
            "auth_policy_read": build_api_key_auth(
                self.settings,
                required_scopes=["policy:read"],
                metrics=self.factory.metrics,
            ),
            "auth_policy_write": build_api_key_auth(
                self.settings,
                required_scopes=["policy:write"],
                metrics=self.factory.metrics,
            ),
        }

    def _add_health_endpoints(self, app: FastAPI) -> None:
        """Add health check endpoints."""

        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "version": ORCH_VERSION,
                "environment": self.settings.environment,
                "detectors_total": len(self.factory.detector_clients),
            }

        @app.get("/health/ready")
        async def readiness():
            dependencies = await self.factory.container.health_check_all()
            all_ready = all(dependencies.values())

            redis_healthy = True
            if (
                self.settings.config.cache_backend == "redis"
                and self.settings.config.redis_url
            ):
                try:
                    if hasattr(self.factory.idempotency_cache, "is_healthy"):
                        redis_healthy = (
                            redis_healthy
                            and self.factory.idempotency_cache.is_healthy()
                        )
                    if hasattr(self.factory.response_cache, "is_healthy"):
                        redis_healthy = (
                            redis_healthy and self.factory.response_cache.is_healthy()
                        )
                except Exception:  # pylint: disable=broad-exception-caught
                    logger.exception("Redis health check failed")
                    redis_healthy = False

            min_detectors_ready = (
                len(self.factory.detector_clients) > 0
                and self.factory.health_monitor is not None
                and len(self.factory.health_monitor.healthy_detectors()) > 0
            )

            readiness_status = {
                "status": (
                    "ready"
                    if (all_ready and redis_healthy and min_detectors_ready)
                    else "not_ready"
                ),
                "version": ORCH_VERSION,
                "environment": self.settings.environment,
                "dependencies": dependencies,
                "redis_healthy": redis_healthy,
                "detectors_total": len(self.factory.detector_clients),
                "detectors_healthy": (
                    len(self.factory.health_monitor.healthy_detectors())
                    if self.factory.health_monitor
                    else 0
                ),
                "min_detectors_ready": min_detectors_ready,
            }

            self.factory.metrics.set_orchestrator_ready(
                all_ready and redis_healthy and min_detectors_ready
            )

            if readiness_status["status"] != "ready":
                return JSONResponse(readiness_status, status_code=503)
            return readiness_status

        @app.get("/metrics")
        async def metrics_endpoint():
            data = generate_latest()
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    def _add_service_endpoints(
        self, app: FastAPI, auth_deps: Dict[str, Callable]
    ) -> None:
        """Add registry and service management endpoints."""

        @app.get("/detectors")
        async def list_detectors(
            _auth: AuthContext = Depends(auth_deps["auth_registry_read"]),
        ) -> Dict[str, Any]:
            if not self.factory.registry:
                raise HTTPException(status_code=503, detail="Registry not available")

            detector_names = self.factory.registry.list()
            detectors: List[Dict[str, Any]] = []
            for name in detector_names:
                detector_cfg = self.settings.detectors.get(name)
                if detector_cfg is not None:
                    detector_data = detector_cfg.model_dump(mode="json")
                else:
                    detector_data = {"name": name}
                detector_data.setdefault("name", name)
                detectors.append(detector_data)

            return {"detectors": detectors}

        @app.post("/detectors")
        async def register_detector(
            payload: DetectorRegistration,
            _auth: AuthContext = Depends(auth_deps["auth_registry_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.registry:
                raise HTTPException(status_code=503, detail="Registry not available")

            self.factory.registry.register(payload)
            return {"status": "ok", "detector": payload.name}

        @app.put("/detectors/{name}")
        async def update_detector(
            name: str,
            payload: DetectorRegistration,
            _auth: AuthContext = Depends(auth_deps["auth_registry_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.registry:
                raise HTTPException(status_code=503, detail="Registry not available")

            try:
                self.factory.registry.update(name, payload)
                cache = self.factory.response_cache
                if cache and hasattr(cache, "invalidate_for_detector"):
                    cache.invalidate_for_detector(name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="detector_not_found") from exc

            return {"status": "ok", "name": name}

        @app.delete("/detectors/{name}")
        async def delete_detector(
            name: str,
            _auth: AuthContext = Depends(auth_deps["auth_registry_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.registry:
                raise HTTPException(status_code=503, detail="Registry not available")

            if name not in self.settings.detectors:
                raise HTTPException(status_code=404, detail="detector_not_found")

            self.factory.registry.remove(name)
            cache = self.factory.response_cache
            if cache and hasattr(cache, "invalidate_for_detector"):
                cache.invalidate_for_detector(name)

            return {"status": "ok", "name": name}

    def _add_policy_endpoints(
        self, app: FastAPI, auth_deps: Dict[str, Callable]
    ) -> None:
        """Add policy management endpoints."""

        @app.get("/policies/{tenant_id}")
        async def list_policies(
            tenant_id: str, _auth: AuthContext = Depends(auth_deps["auth_policy_read"])
        ) -> Dict[str, Any]:
            if not self.factory.policy_store:
                raise HTTPException(
                    status_code=503, detail="Policy store not available"
                )

            return {
                "tenant": tenant_id,
                "bundles": self.factory.policy_store.list_policies(tenant_id),
            }

        @app.post("/policies/{tenant_id}/{bundle}")
        async def submit_policy(
            tenant_id: str,
            bundle: str,
            body: Dict[str, Any],
            auth: AuthContext = Depends(auth_deps["auth_policy_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.policy_store:
                raise HTTPException(
                    status_code=503, detail="Policy store not available"
                )

            payload = body or {}
            raw_policy = payload.get("policy", payload)
            try:
                policy = TenantPolicy(**raw_policy)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid_policy: {exc}") from exc

            policy.tenant_id = tenant_id
            policy.bundle = bundle

            submitted_by = payload.get("submitted_by") or getattr(
                auth, "api_key_id", None
            )
            if not submitted_by or submitted_by == "***masked***":
                submitted_by = getattr(auth, "tenant_id", "unknown")

            submission = PolicySubmission(
                policy=policy,
                submitted_by=submitted_by,
                description=payload.get("description"),
                requires_approval=payload.get("requires_approval", True),
            )

            detector_catalog = {name: True for name in self.settings.detectors.keys()}

            try:
                version = self.factory.policy_store.submit_policy(
                    submission,
                    detector_catalog=detector_catalog,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            if (
                version.status == PolicyVersionStatus.APPROVED
                and self.factory.response_cache
                and hasattr(self.factory.response_cache, "invalidate_for_policy")
            ):
                self.factory.response_cache.invalidate_for_policy(bundle)

            return {"version": version.model_dump(mode="json")}

        @app.get("/policies/{tenant_id}/{bundle}/versions")
        async def list_policy_versions(
            tenant_id: str,
            bundle: str,
            _auth: AuthContext = Depends(auth_deps["auth_policy_read"]),
        ) -> Dict[str, Any]:
            if not self.factory.policy_store:
                raise HTTPException(
                    status_code=503, detail="Policy store not available"
                )

            versions = self.factory.policy_store.list_versions(tenant_id, bundle)
            return {
                "tenant": tenant_id,
                "bundle": bundle,
                "versions": [v.model_dump(mode="json") for v in versions],
            }

        @app.post("/policies/{tenant_id}/{bundle}/versions/{version_id}/review")
        async def review_policy_version(
            tenant_id: str,
            bundle: str,
            version_id: str,
            decision: ApprovalDecision,
            auth: AuthContext = Depends(auth_deps["auth_policy_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.policy_store:
                raise HTTPException(
                    status_code=503, detail="Policy store not available"
                )

            reviewer = getattr(auth, "api_key_id", None) or getattr(
                auth, "tenant_id", "unknown"
            )
            review_record = ReviewRecord(
                reviewer=reviewer,
                decision=decision,
                note=None
            )
            version = self.factory.policy_store.record_review(
                tenant_id,
                bundle,
                version_id,
                review_record,
            )

            if (
                version.status == PolicyVersionStatus.APPROVED
                and self.factory.response_cache
                and hasattr(self.factory.response_cache, "invalidate_for_policy")
            ):
                self.factory.response_cache.invalidate_for_policy(bundle)

            return {"version": version.model_dump(mode="json")}

        @app.post("/policies/{tenant_id}/{bundle}/versions/{version_id}/rollback")
        async def rollback_policy_version(
            tenant_id: str,
            bundle: str,
            version_id: str,
            auth: AuthContext = Depends(auth_deps["auth_policy_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.policy_store:
                raise HTTPException(
                    status_code=503, detail="Policy store not available"
                )

            actor = getattr(auth, "api_key_id", None) or getattr(
                auth, "tenant_id", "unknown"
            )
            rollback_request = RollbackRequest(
                actor=actor,
                reason="Manual rollback via API"
            )
            version = self.factory.policy_store.rollback_policy(
                tenant_id,
                bundle,
                version_id,
                rollback_request,
            )

            if self.factory.response_cache and hasattr(
                self.factory.response_cache, "invalidate_for_policy"
            ):
                self.factory.response_cache.invalidate_for_policy(bundle)

            return {"version": version.model_dump(mode="json")}

        @app.delete("/policies/{tenant_id}/{bundle}")
        async def delete_policy(
            tenant_id: str,
            bundle: str,
            _auth: AuthContext = Depends(auth_deps["auth_policy_write"]),
        ) -> Dict[str, Any]:
            if not self.factory.policy_store:
                raise HTTPException(
                    status_code=503, detail="Policy store not available"
                )

            self.factory.policy_store.delete_policy(tenant_id, bundle)
            if self.factory.response_cache and hasattr(
                self.factory.response_cache, "invalidate_for_policy"
            ):
                self.factory.response_cache.invalidate_for_policy(bundle)

            return {"status": "deleted", "bundle": bundle}

    def _add_orchestration_endpoints(
        self, app: FastAPI, auth_deps: Dict[str, Callable]
    ) -> None:
        """Add orchestration endpoints (orchestrate, status, events)."""

        async def _resolve_tenant(
            request_obj: OrchestrationRequest,
            tenant_header: Optional[str],
            auth: AuthContext,
        ) -> None:
            if request_obj.tenant_id:
                return
            if tenant_header:
                request_obj.tenant_id = tenant_header
                return
            if auth and getattr(auth, "tenant_id", None):
                request_obj.tenant_id = auth.tenant_id
                return
            raise HTTPException(status_code=400, detail="tenant_not_provided")

        async def _route_orchestration(
            request_obj: OrchestrationRequest,
        ) -> Tuple[RoutingPlan, RoutingDecision]:
            if not self.factory.router:
                raise HTTPException(status_code=503, detail="router_not_ready")
            routing_plan, decision = await self.factory.router.route_request(
                request_obj
            )
            if not routing_plan.primary_detectors:
                raise HTTPException(status_code=503, detail="no_detectors_available")
            return routing_plan, decision

        def _idempotency_lookup(
            request_obj: OrchestrationRequest,
            idempotency_key: Optional[str],
            fingerprint: Optional[str],
        ) -> Optional[Any]:
            if not idempotency_key or not self.factory.idempotency_cache:
                return None

            cache_key = self.factory.build_idempotency_cache_key(
                request_obj.tenant_id, idempotency_key
            )
            entry = None
            entry = None
            entry_getter = getattr(self.factory.idempotency_cache, "get_entry", None)
            if callable(entry_getter):
                entry = entry_getter(cache_key)  # type: ignore[misc]  # pylint: disable=not-callable
            else:
                cached_value = self.factory.idempotency_cache.get(cache_key)
                if cached_value is not None:
                    entry = type(
                        "_InlineEntry",
                        (),
                        {"value": cached_value, "fingerprint": None},
                    )()
            if entry is None:
                pending = self.factory.pending_idempotent_jobs.get(cache_key)
                if pending:
                    job_id, pending_fp = pending
                    if pending_fp and pending_fp != fingerprint:
                        raise HTTPException(
                            status_code=409, detail="idempotency_conflict"
                        )
                    headers = (
                        {"Idempotency-Key": idempotency_key}
                        if idempotency_key
                        else None
                    )
                    return JSONResponse(
                        {"job_id": job_id, "status": "pending"},
                        status_code=202,
                        headers=headers,
                    )
                return None
            stored_fp = getattr(entry, "fingerprint", None)
            if stored_fp and stored_fp != fingerprint:
                raise HTTPException(status_code=409, detail="idempotency_conflict")
            response_obj = entry.value
            return self.factory.clone_response(
                response_obj,
                idempotency_key=idempotency_key,
            )

        @app.get("/events/stream")
        async def stream_events(
            _auth: AuthContext = Depends(auth_deps["auth_orchestrate_status"]),
        ) -> StreamingResponse:
            queue = self.factory.register_event_subscriber()

            async def event_generator() -> AsyncIterator[str]:
                try:
                    while True:
                        try:
                            event = await asyncio.wait_for(queue.get(), timeout=15.0)
                            yield f"data: {json.dumps(event)}\n\n"
                        except asyncio.TimeoutError:
                            yield ": keep-alive\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    self.factory.unregister_event_subscriber(queue)

            return StreamingResponse(
                event_generator(), media_type="text/event-stream"
            )

        @app.get("/events/incidents")
        async def list_incidents(
            _auth: AuthContext = Depends(auth_deps["auth_orchestrate_status"]),
        ) -> Dict[str, Any]:
            return {"incidents": self.factory.get_recent_incidents()}

        @app.post("/orchestrate")
        async def orchestrate_endpoint(
            request_obj: OrchestrationRequest,
            response: Response,
            idempotency_key: Optional[str] = Header(
                default=None, alias="Idempotency-Key"
            ),
            tenant_header: Optional[str] = Header(
                default=None, alias=self.settings.config.tenant_header
            ),
            auth: AuthContext = Depends(auth_deps["auth_orchestrate_write"]),
        ) -> Any:
            await _resolve_tenant(request_obj, tenant_header, auth)
            fingerprint = self.factory.request_fingerprint(request_obj)

            replay = _idempotency_lookup(request_obj, idempotency_key, fingerprint)
            if isinstance(replay, JSONResponse):
                return replay
            if isinstance(replay, OrchestrationResponse):
                response.headers["X-Orchestrator-Request-ID"] = replay.request_id  # pylint: disable=no-member
                if idempotency_key:
                    response.headers["Idempotency-Key"] = idempotency_key
                return replay

            routing_plan, decision = await _route_orchestration(request_obj)

            cache_hit = None
            cache_key = self.factory.response_cache_key(request_obj, routing_plan)
            if cache_key and self.factory.response_cache:
                cached = self.factory.response_cache.get(cache_key)
                if cached:
                    request_id = str(uuid.uuid4())
                    cache_hit = self.factory.clone_response(
                        cached,
                        request_id=request_id,
                        idempotency_key=idempotency_key,
                    )
                    try:
                        self.factory.metrics.record_request_start(
                            request_id,
                            request_obj.tenant_id,
                            decision.policy_applied,
                            processing_mode=request_obj.processing_mode.value,
                        )
                        self.factory.metrics.record_coverage(
                            request_obj.tenant_id,
                            decision.policy_applied,
                            cache_hit.coverage_achieved,
                        )
                        self.factory.metrics.record_request_end(
                            request_id,
                            success=cache_hit.error_code is None,
                            duration_ms=0.0,
                            tenant=request_obj.tenant_id,
                            policy=decision.policy_applied,
                            processing_mode=request_obj.processing_mode.value,
                        )
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                    response.headers["X-Orchestrator-Request-ID"] = request_id
                    if idempotency_key:
                        response.headers["Idempotency-Key"] = idempotency_key
                    return cache_hit

            should_async = request_obj.processing_mode == ProcessingMode.ASYNC
            if not should_async:
                est = self.factory.estimate_processing_time_ms(routing_plan)
                threshold = int(
                    getattr(
                        self.settings.config.sla, "sync_to_async_threshold_ms", 1500
                    )
                )
                if est > threshold:
                    should_async = True

            request_id = str(uuid.uuid4())

            if should_async:
                if not self.factory.job_manager:
                    raise HTTPException(
                        status_code=503, detail="job_manager_unavailable"
                    )

                cache_key = None
                if idempotency_key:
                    cache_key = self.factory.build_idempotency_cache_key(
                        request_obj.tenant_id, idempotency_key
                    )
                    self.factory.pending_idempotent_jobs[cache_key] = (
                        request_id,
                        fingerprint or "",
                    )

                job_id = await self.factory.job_manager.enqueue(
                    request=request_obj,
                    idempotency_key=idempotency_key,
                    decision=decision,
                    routing_plan=routing_plan,
                )

                response.headers["X-Orchestrator-Request-ID"] = request_id
                if idempotency_key:
                    response.headers["Idempotency-Key"] = idempotency_key
                return JSONResponse(
                    {
                        "job_id": job_id,
                        "estimated_processing_time_ms": self.factory.estimate_processing_time_ms(
                            routing_plan
                        ),
                        "status": "pending",
                    },
                    status_code=202,
                )

            result = await self.factory.run_pipeline(
                request_obj,
                routing_plan=routing_plan,
                decision=decision,
                request_id=request_id,
                raw_idempotency_key=idempotency_key,
                idempotency_cache_key=(
                    self.factory.build_idempotency_cache_key(
                        request_obj.tenant_id, idempotency_key
                    )
                    if idempotency_key
                    else None
                ),
                fingerprint=fingerprint,
            )

            response.headers["X-Orchestrator-Request-ID"] = result.request_id
            if idempotency_key:
                response.headers["Idempotency-Key"] = idempotency_key
            return result

        @app.post("/orchestrate/batch")
        async def orchestrate_batch_endpoint(
            requests: List[OrchestrationRequest],
            response: Response,
            tenant_header: Optional[str] = Header(
                default=None, alias=self.settings.config.tenant_header
            ),
            auth: AuthContext = Depends(auth_deps["auth_orchestrate_write"]),
        ) -> Dict[str, Any]:
            if not requests:
                raise HTTPException(status_code=400, detail="empty_batch")

            results: List[Dict[str, Any]] = []
            errors: List[Dict[str, Any]] = []

            for req in requests:
                await _resolve_tenant(req, tenant_header, auth)
                if req.processing_mode == ProcessingMode.ASYNC:
                    errors.append(
                        {
                            "tenant_id": req.tenant_id,
                            "error": "async_not_supported_in_batch",
                        }
                    )
                    continue
                req.processing_mode = ProcessingMode.SYNC
                routing_plan, decision = await _route_orchestration(req)
                request_id = str(uuid.uuid4())
                result = await self.factory.run_pipeline(
                    req,
                    routing_plan=routing_plan,
                    decision=decision,
                    request_id=request_id,
                    raw_idempotency_key=None,
                    idempotency_cache_key=None,
                    fingerprint=self.factory.request_fingerprint(req),
                )
                results.append(result.model_dump())

            response.headers["X-Orchestrator-Request-ID"] = str(uuid.uuid4())
            return {
                "results": results,
                "errors": errors or None,
            }

        @app.get("/orchestrate/status/{job_id}")
        async def get_job_status(
            job_id: str,
            _auth: AuthContext = Depends(auth_deps["auth_orchestrate_status"]),
        ) -> Any:
            if not self.factory.job_manager:
                raise HTTPException(status_code=503, detail="job_manager_unavailable")
            status = self.factory.job_manager.get_status(job_id)
            if not status:
                raise HTTPException(status_code=404, detail="job_not_found")
            return status.model_dump()

        @app.delete("/orchestrate/status/{job_id}")
        async def cancel_job(
            job_id: str,
            _auth: AuthContext = Depends(auth_deps["auth_orchestrate_status"]),
        ) -> Any:
            if not self.factory.job_manager:
                raise HTTPException(status_code=503, detail="job_manager_unavailable")
            ok = self.factory.job_manager.cancel(job_id)
            if ok:
                return {"job_id": job_id, "status": "cancelled"}
            raise HTTPException(status_code=409, detail="job_not_cancellable")

    def build(self) -> FastAPI:
        """Build the complete FastAPI application."""
        logger.info("Building orchestration FastAPI application...")

        self.factory.initialize_services()

        self.app = FastAPI(
            title="Detector Orchestration Service",
            version=ORCH_VERSION,
            lifespan=self._create_lifespan_handler(),
        )

        self._add_middleware(self.app)
        self._add_exception_handlers()
        auth_deps = self._add_auth_dependencies()
        self._add_health_endpoints(self.app)
        self._add_service_endpoints(self.app, auth_deps)
        self._add_policy_endpoints(self.app, auth_deps)
        self._add_orchestration_endpoints(self.app, auth_deps)

        logger.info("Orchestration FastAPI application built successfully")
        return self.app


def create_orchestration_app(settings: Optional[Settings] = None) -> FastAPI:
    """Factory function to build the orchestration FastAPI application."""
    if settings is None:
        settings = Settings()

    builder = OrchestrationAppBuilder(settings)
    return builder.build()


create_app = create_orchestration_app  # Backward compatibility alias
