from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Response, Request
from contextlib import asynccontextmanager
import json as _json

from ..aggregator import ResponseAggregator
from ..clients import DetectorClient
from ..config import Settings
from ..coordinator import DetectorCoordinator
from ..models import (
    OrchestrationRequest,
    OrchestrationResponse,
    RoutingDecision,
    DetectorResult,
    JobStatus,
    JobStatusResponse,
    MappingResponse,
    RoutingPlan,
)
from ..models import ProcessingMode, PolicyContext
from ..router import ContentRouter
from ..content_analysis import infer_content_type
from ..security import AuthContext, build_api_key_auth
from ..metrics import OrchestrationMetricsCollector
from ..jobs import JobManager
from ..cache import IdempotencyCache, ResponseCache, RedisIdempotencyCache, RedisResponseCache
from ..health_monitor import HealthMonitor
from ..circuit_breaker import CircuitBreakerManager
from ..registry import DetectorRegistry, DetectorRegistration
from ..policy import PolicyStore, OPAPolicyEngine, PolicyManager, TenantPolicy
from .. import __version__ as ORCH_VERSION
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from fastapi.responses import JSONResponse
from ..rate_limit import OrchestratorRateLimitMiddleware
from ..mapper_client import MapperClient
from ..conflict import ConflictResolver
from ..errors import build_error_body


settings = Settings()
@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_clients_initialized()
    if HEALTH_MONITOR:
        await HEALTH_MONITOR.start()
    if JOB_MANAGER:
        try:
            await JOB_MANAGER.start()
        except Exception:
            pass
    try:
        yield
    finally:
        if HEALTH_MONITOR:
            await HEALTH_MONITOR.stop()
        if JOB_MANAGER:
            try:
                await JOB_MANAGER.stop()
            except Exception:
                pass

app = FastAPI(title="Detector Orchestration Service", version="0.1.0", lifespan=lifespan)
# Install simple rate limit middleware (per-tenant)
metrics = OrchestrationMetricsCollector()
app.add_middleware(OrchestratorRateLimitMiddleware, settings=settings, metrics=metrics)

# Canonical error response handlers (Sec 8)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    # Map common statuses to canonical error codes
    status = int(exc.status_code)
    message = exc.detail if isinstance(exc.detail, str) else None
    if status == 401:
        code = "UNAUTHORIZED"
    elif status == 403:
        # Distinguish rate limit vs RBAC if detail provided
        if isinstance(exc.detail, str) and "rate" in exc.detail.lower():
            code = "RATE_LIMITED"
        else:
            code = "INSUFFICIENT_RBAC"
    elif status == 400:
        code = "INVALID_REQUEST"
    elif status == 408:
        code = "REQUEST_TIMEOUT"
    elif status == 502:
        code = "DETECTOR_COMMUNICATION_FAILED"
    else:
        code = "INTERNAL_ERROR"
    body = build_error_body(request_id=None, code=code, message=message).model_dump()
    return JSONResponse(status_code=status, content={"detail": body})

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    body = build_error_body(request_id=None, code="INTERNAL_ERROR", message=str(exc)).model_dump()
    return JSONResponse(status_code=500, content={"detail": body})
# Auth dependencies
auth_orchestrate_write = build_api_key_auth(settings, required_scopes=["orchestrate:write"], metrics=metrics)
auth_orchestrate_status = build_api_key_auth(settings, required_scopes=["orchestrate:status"], metrics=metrics)
auth_registry_read = build_api_key_auth(settings, required_scopes=["registry:read"], metrics=metrics) 
auth_registry_write = build_api_key_auth(settings, required_scopes=["registry:write"], metrics=metrics) 
auth_policy_read = build_api_key_auth(settings, required_scopes=["policy:read"], metrics=metrics) 
auth_policy_write = build_api_key_auth(settings, required_scopes=["policy:write"], metrics=metrics) 


# In-memory stores (MVP)
if settings.config.cache_backend == "redis" and settings.config.redis_url:
    _idem = RedisIdempotencyCache(
        settings.config.redis_url,
        ttl_seconds=60 * 60 * 24,
        key_prefix=f"idem:{settings.config.redis_prefix}",
    )
    _resp = RedisResponseCache(
        settings.config.redis_url,
        ttl_seconds=settings.config.response_cache_ttl_seconds,
        key_prefix=f"resp:{settings.config.redis_prefix}",
    )
    if _idem.is_healthy() and _resp.is_healthy():
        IDEMPOTENCY = _idem
        RESP_CACHE = _resp
        metrics.set_redis_backend_up("idempotency", True)
        metrics.set_redis_backend_up("response", True)
    else:
        IDEMPOTENCY = IdempotencyCache()
        RESP_CACHE = ResponseCache(ttl_seconds=settings.config.response_cache_ttl_seconds)
        metrics.set_redis_backend_up("idempotency", False)
        metrics.set_redis_backend_up("response", False)
        metrics.inc_redis_fallback("idempotency")
        metrics.inc_redis_fallback("response")
else:
    IDEMPOTENCY = IdempotencyCache()
    RESP_CACHE = ResponseCache(ttl_seconds=settings.config.response_cache_ttl_seconds)
DETECTOR_CLIENTS: dict[str, DetectorClient] = {}
HEALTH_MONITOR: HealthMonitor | None = None
BREAKERS: CircuitBreakerManager | None = None
REGISTRY: DetectorRegistry | None = None
POLICY_STORE: PolicyStore | None = None
POLICY_MANAGER: PolicyManager | None = None
MAPPER_CLIENT: MapperClient | None = None
JOB_MANAGER: JobManager | None = None


def _ensure_clients_initialized() -> None:
    global DETECTOR_CLIENTS, HEALTH_MONITOR, BREAKERS
    if not DETECTOR_CLIENTS:
        for name, det in settings.detectors.items():
            DETECTOR_CLIENTS[name] = DetectorClient(
                name=name,
                endpoint=det.endpoint,
                timeout_ms=det.timeout_ms,
                max_retries=det.max_retries,
                auth=det.auth,
            )
    if not BREAKERS:
        BREAKERS = CircuitBreakerManager(
            failure_threshold=settings.config.circuit_breaker_failure_threshold,
            recovery_timeout_seconds=settings.config.circuit_breaker_recovery_timeout_seconds,
        )
    if not HEALTH_MONITOR:
        HEALTH_MONITOR = HealthMonitor(
            DETECTOR_CLIENTS,
            interval_seconds=settings.config.health_check_interval_seconds,
            metrics=metrics,
            unhealthy_threshold=settings.config.unhealthy_threshold,
        )
    global REGISTRY
    if not REGISTRY:
        REGISTRY = DetectorRegistry(settings, DETECTOR_CLIENTS, HEALTH_MONITOR)
    global POLICY_STORE, POLICY_MANAGER
    if not POLICY_STORE:
        POLICY_STORE = PolicyStore(settings.config.policy_dir)
    if not POLICY_MANAGER:
        POLICY_MANAGER = PolicyManager(POLICY_STORE, OPAPolicyEngine(settings))
    global MAPPER_CLIENT
    if not MAPPER_CLIENT:
        MAPPER_CLIENT = MapperClient(settings)
    global JOB_MANAGER
    if not JOB_MANAGER:
        # Defer binding of _process_sync until runtime; function defined below
        JOB_MANAGER = JobManager(run_fn=lambda req, idem, dec, plan, prog: _process_sync(req, Response(), idem, dec, plan, prog))


# Lifespan handles startup/shutdown to avoid deprecation warnings.


@app.get("/health")
async def health():
    _ensure_clients_initialized()
    healthy = sum(1 for n in DETECTOR_CLIENTS if HEALTH_MONITOR and HEALTH_MONITOR.is_healthy(n))
    return {"status": "healthy", "ts": datetime.now(timezone.utc).isoformat(), "detectors_total": len(DETECTOR_CLIENTS), "detectors_healthy": healthy}


@app.get("/detectors")
async def list_detectors(auth: AuthContext = Depends(auth_registry_read)):
    _ensure_clients_initialized()
    return {
        "detectors": [
            {
                "name": name,
                "endpoint": settings.detectors[name].endpoint,
                "timeout_ms": settings.detectors[name].timeout_ms,
                "supported_content_types": settings.detectors[name].supported_content_types,
                "healthy": (HEALTH_MONITOR.is_healthy(name) if HEALTH_MONITOR else True),
            }
            for name in settings.detectors.keys()
        ]
    }


@app.get("/detectors/{name}")
async def get_detector(name: str, auth: AuthContext = Depends(auth_registry_read)):
    _ensure_clients_initialized()
    det = settings.detectors.get(name)
    if not det:
        raise HTTPException(status_code=404, detail="detector_not_found")
    client = DETECTOR_CLIENTS.get(name)
    caps = await (client.get_capabilities() if client else DetectorClient(name, det.endpoint, det.timeout_ms, det.max_retries, det.auth).get_capabilities())
    return {
        "name": name,
        "endpoint": det.endpoint,
        "timeout_ms": det.timeout_ms,
        "supported_content_types": det.supported_content_types,
        "healthy": (HEALTH_MONITOR.is_healthy(name) if HEALTH_MONITOR else True),
        "capabilities": caps.model_dump(),
    }


@app.post("/detectors/register")
async def register_detector(payload: DetectorRegistration, auth: AuthContext = Depends(auth_registry_write)):
    _ensure_clients_initialized()
    assert REGISTRY is not None
    REGISTRY.register(payload)
    return {"status": "ok", "name": payload.name}


@app.post("/detectors/{name}/update")
async def update_detector(name: str, payload: DetectorRegistration, auth: AuthContext = Depends(auth_registry_write)):
    _ensure_clients_initialized()
    assert REGISTRY is not None
    try:
        REGISTRY.update(name, payload)
    except KeyError:
        raise HTTPException(status_code=404, detail="detector_not_found")
    # Invalidate response cache for this detector (policy-aware invalidation)
    try:
        invalidated = getattr(RESP_CACHE, "invalidate_for_detector", None)
        if callable(invalidated):
            RESP_CACHE.invalidate_for_detector(name)
    except Exception:
        pass
    return {"status": "ok", "name": name}


@app.delete("/detectors/{name}")
async def delete_detector(name: str, auth: AuthContext = Depends(auth_registry_write)):
    _ensure_clients_initialized()
    assert REGISTRY is not None
    if name not in settings.detectors:
        raise HTTPException(status_code=404, detail="detector_not_found")
    REGISTRY.remove(name)
    # Invalidate response cache for this detector
    try:
        invalidated = getattr(RESP_CACHE, "invalidate_for_detector", None)
        if callable(invalidated):
            RESP_CACHE.invalidate_for_detector(name)
    except Exception:
        pass
    return {"status": "ok", "name": name}


# Policy endpoints
@app.get("/policies/{tenant_id}")
async def list_policies(tenant_id: str, auth: AuthContext = Depends(auth_policy_read)):
    _ensure_clients_initialized()
    assert POLICY_STORE is not None
    return {"tenant": tenant_id, "bundles": POLICY_STORE.list_policies(tenant_id)}


@app.get("/policies/{tenant_id}/{bundle}")
async def get_policy(tenant_id: str, bundle: str, auth: AuthContext = Depends(auth_policy_read)):
    _ensure_clients_initialized()
    assert POLICY_STORE is not None
    pol = POLICY_STORE.get_policy(tenant_id, bundle)
    if not pol:
        raise HTTPException(status_code=404, detail="policy_not_found")
    return pol.model_dump()


@app.post("/policies/{tenant_id}/{bundle}")
async def put_policy(tenant_id: str, bundle: str, body: dict, auth: AuthContext = Depends(auth_policy_write)):
    _ensure_clients_initialized()
    assert POLICY_STORE is not None
    try:
        pol = TenantPolicy(**body)
        pol.tenant_id = tenant_id
        pol.bundle = bundle
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid_policy: {e}")
    POLICY_STORE.save_policy(pol)
    # Invalidate response cache for this policy bundle
    try:
        invalidated = getattr(RESP_CACHE, "invalidate_for_policy", None)
        if callable(invalidated):
            RESP_CACHE.invalidate_for_policy(bundle)
    except Exception:
        pass
    return {"status": "ok", "tenant": tenant_id, "bundle": bundle}


@app.delete("/policies/{tenant_id}/{bundle}")
async def delete_policy(tenant_id: str, bundle: str, auth: AuthContext = Depends(auth_policy_write)):
    _ensure_clients_initialized()
    assert POLICY_STORE is not None
    ok = POLICY_STORE.delete_policy(tenant_id, bundle)
    if not ok:
        raise HTTPException(status_code=404, detail="policy_not_found")
    # Invalidate response cache for this policy bundle
    try:
        invalidated = getattr(RESP_CACHE, "invalidate_for_policy", None)
        if callable(invalidated):
            RESP_CACHE.invalidate_for_policy(bundle)
    except Exception:
        pass
    return {"status": "ok", "tenant": tenant_id, "bundle": bundle}


@app.get("/metrics")
async def metrics_endpoint():
    data = generate_latest()  # type: ignore[arg-type]
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


async def _process_sync(
    request: OrchestrationRequest,
    response: Response,
    idempotency_key: Optional[str],
    decision: RoutingDecision,
    routing_plan,
    progress_cb: Optional[Callable[[float], None]] = None,
):
    # Assumes tenant context already validated upstream

    # Compute request fingerprint for idempotency (content+tenant+policy+env+type+priority)
    try:
        import hashlib as _hashlib
        import json as _json_local
        _fp_obj = {
            "content_hash": _hashlib.sha256((request.content or "").encode("utf-8")).hexdigest(),
            "tenant_id": request.tenant_id,
            "policy_bundle": request.policy_bundle,
            "environment": request.environment,
            "content_type": request.content_type.value if hasattr(request.content_type, "value") else str(request.content_type),
            "priority": request.priority.value if hasattr(request.priority, "value") else str(request.priority),
        }
        _request_fp = _hashlib.sha256(_json_local.dumps(_fp_obj, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception:
        _request_fp = None

    # Idempotency quick path (bypass for CRITICAL)
    if idempotency_key and request.priority.value != "critical":
        # Check for conflict
        entry = getattr(IDEMPOTENCY, "get_entry", None)
        if callable(entry):
            idem_entry = IDEMPOTENCY.get_entry(idempotency_key)
            if idem_entry is not None:
                # If a fingerprint exists and mismatches, treat as conflict
                if idem_entry.fingerprint and _request_fp and idem_entry.fingerprint != _request_fp:
                    now = datetime.now(timezone.utc)
                    conflict_resp = OrchestrationResponse(
                        request_id="req",
                        processing_mode=request.processing_mode,
                        detector_results=[],
                        aggregated_payload=None,
                        mapping_result=None,
                        total_processing_time_ms=0,
                        detectors_attempted=0,
                        detectors_succeeded=0,
                        detectors_failed=0,
                        coverage_achieved=0.0,
                        routing_decision=decision,
                        fallback_used=False,
                        timestamp=now,
                        idempotency_key=idempotency_key,
                        error_code="IDEMPOTENCY_CONFLICT",
                    )
                    response.status_code = 409
                    return conflict_resp
                # Otherwise, return cached value
                response.headers["Idempotency-Key"] = idempotency_key
                return idem_entry.value
        else:
            cached = IDEMPOTENCY.get(idempotency_key)
            if cached:
                response.headers["Idempotency-Key"] = idempotency_key
                return cached

    # Response cache quick path (content_hash + detector_set + policy_bundle)
    if settings.config.cache_enabled and request.priority.value != "critical":
        key = RESP_CACHE.build_key(request.content, tuple(routing_plan.primary_detectors), request.policy_bundle)
        cached = RESP_CACHE.get(key)
        if cached:
            return cached

    # Short-circuit when no detectors available
    if not routing_plan.primary_detectors:
        now = datetime.now(timezone.utc)
        resp_obj = OrchestrationResponse(
            request_id="req",
            processing_mode=request.processing_mode,
            detector_results=[],
            aggregated_payload=None,
            mapping_result=None,
            total_processing_time_ms=0,
            detectors_attempted=0,
            detectors_succeeded=0,
            detectors_failed=0,
            coverage_achieved=0.0,
            routing_decision=decision,
            fallback_used=True,
            timestamp=now,
            idempotency_key=idempotency_key,
            error_code="ALL_DETECTORS_UNAVAILABLE",
        )
        response.status_code = 502
        return resp_obj

    _ensure_clients_initialized()
    router = ContentRouter(settings, health_monitor=HEALTH_MONITOR, policy_manager=POLICY_MANAGER)
    coordinator = DetectorCoordinator(
        DETECTOR_CLIENTS,
        breakers=BREAKERS,
        metrics=metrics,
        retry_on_timeouts=settings.config.retry_on_timeouts,
        retry_on_failures=settings.config.retry_on_failures,
    )
    start_ts = datetime.now(timezone.utc)
    # Two-phase execution: primary then optional secondary based on coverage
    primary_only_plan = RoutingPlan(
        primary_detectors=routing_plan.primary_detectors,
        parallel_groups=[routing_plan.primary_detectors],
        timeout_config={d: routing_plan.timeout_config.get(d, settings.config.default_timeout_ms) for d in routing_plan.primary_detectors},
        retry_config={d: routing_plan.retry_config.get(d, settings.config.max_retries) for d in routing_plan.primary_detectors},
        coverage_method=routing_plan.coverage_method,
        weights=routing_plan.weights,
        required_taxonomy_categories=routing_plan.required_taxonomy_categories,
    )

    results_primary = await coordinator.execute_detector_group(
        routing_plan.primary_detectors, request.content, primary_only_plan, request.metadata or {}
    )
    # Progress ~30% after primary detectors
    try:
        if callable(progress_cb):
            progress_cb(0.3)
    except Exception:
        pass

    aggregator = ResponseAggregator()
    # Compute coverage after primary phase only
    _, cov_primary = aggregator.aggregate(results_primary, primary_only_plan, tenant_id=request.tenant_id)

    results = list(results_primary)
    if (
        settings.config.secondary_on_coverage_below
        and routing_plan.secondary_detectors
        and cov_primary < (settings.config.secondary_min_coverage or 1.0)
    ):
        secondary_plan = RoutingPlan(
            primary_detectors=routing_plan.secondary_detectors,
            parallel_groups=[routing_plan.secondary_detectors],
            timeout_config={d: routing_plan.timeout_config.get(d, settings.config.default_timeout_ms) for d in routing_plan.secondary_detectors},
            retry_config={d: routing_plan.retry_config.get(d, settings.config.max_retries) for d in routing_plan.secondary_detectors},
            coverage_method=routing_plan.coverage_method,
            weights=routing_plan.weights,
            required_taxonomy_categories=routing_plan.required_taxonomy_categories,
        )
        results_secondary = await coordinator.execute_detector_group(
            routing_plan.secondary_detectors, request.content, secondary_plan, request.metadata or {}
        )
        results.extend(results_secondary)

    # Compose a plan that includes all considered detectors for final coverage
    all_plan = RoutingPlan(
        primary_detectors=routing_plan.primary_detectors + routing_plan.secondary_detectors,
        parallel_groups=[routing_plan.primary_detectors + routing_plan.secondary_detectors],
        timeout_config=routing_plan.timeout_config,
        retry_config=routing_plan.retry_config,
        coverage_method=routing_plan.coverage_method,
        weights=routing_plan.weights,
        required_taxonomy_categories=routing_plan.required_taxonomy_categories,
    )
    # Compute conflict resolution outcome (OPA-backed when enabled)
    resolver = ConflictResolver(OPAPolicyEngine(settings))
    try:
        conflict_outcome = await resolver.resolve(
            tenant_id=request.tenant_id,
            policy_bundle=request.policy_bundle,
            content_type=request.content_type,
            detector_results=results,
            weights=all_plan.weights,
        )
    except Exception:
        conflict_outcome = None
    payload, coverage_achieved = aggregator.aggregate(
        results, all_plan, tenant_id=request.tenant_id, conflict_outcome=conflict_outcome
    )
    # Progress ~60% after aggregation
    try:
        if callable(progress_cb):
            progress_cb(0.6)
    except Exception:
        pass
    # Validate payload size (<=64KB)
    if len(_json.dumps(payload.model_dump()).encode("utf-8")) > 64 * 1024:
        raise HTTPException(status_code=400, detail="payload_too_large")

    mapping_result: Optional[MappingResponse] = None
    fallback_used = False

    # Optional auto-mapping within reserved budget
    if settings.config.auto_map_results:
        try:
            assert MAPPER_CLIENT is not None
            mapping_result_obj, mapped_error_code, status = await MAPPER_CLIENT.map(payload, tenant_id=request.tenant_id, idempotency_key=idempotency_key)
            if mapping_result_obj is not None:
                mapping_result = mapping_result_obj
                # Propagate version info and policy context
                notes_parts = []
                if mapping_result.notes:
                    notes_parts.append(mapping_result.notes)
                notes_parts.append(f"orchestrator={ORCH_VERSION}")
                if POLICY_STORE:
                    pol = POLICY_STORE.get_policy(request.tenant_id, request.policy_bundle)
                    if pol:
                        notes_parts.append(f"policy={pol.bundle}:{pol.version}")
                mapping_result.notes = "; ".join(notes_parts)[:500]
                if mapping_result.provenance is None:
                    mapping_result.provenance = type(mapping_result.provenance)(tenant_id=request.tenant_id, route="orchestrator")
                else:
                    mapping_result.provenance.route = "orchestrator"
                mapping_result.policy_context = PolicyContext(
                    expected_detectors=decision.selected_detectors,
                    environment=request.environment,
                )
            else:
                fallback_used = True
                response.status_code = status
                error_code = mapped_error_code
        except Exception:
            fallback_used = True
            response.status_code = 502
            error_code = "DETECTOR_COMMUNICATION_FAILED"
    # Progress ~90% after mapper integration or fallback
    try:
        if callable(progress_cb):
            progress_cb(0.9)
    except Exception:
        pass

    # Compose response per contract
    detectors_attempted = len(results)
    detectors_succeeded = len([r for r in results if r.status.value == "success"])
    detectors_failed = max(detectors_attempted - detectors_succeeded, 0)
    now = datetime.now(timezone.utc)
    duration_ms = max(int((now - start_ts).total_seconds() * 1000), 0)
    resp_obj = OrchestrationResponse(
        request_id="req",
        processing_mode=request.processing_mode,
        detector_results=results,
        aggregated_payload=payload,
        mapping_result=mapping_result,
        total_processing_time_ms=duration_ms,
        detectors_attempted=detectors_attempted,
        detectors_succeeded=detectors_succeeded,
        detectors_failed=detectors_failed,
        coverage_achieved=coverage_achieved,
        routing_decision=decision,
        fallback_used=fallback_used,
        timestamp=now,
        idempotency_key=idempotency_key,
    )

    # Status code handling for partial coverage
    if coverage_achieved < 0.8:
        response.status_code = 206
        resp_obj.error_code = "PARTIAL_COVERAGE"
        resp_obj.fallback_used = True

    # If no detectors succeeded at all but attempted > 0, mark communication failure
    if detectors_attempted > 0 and detectors_succeeded == 0 and resp_obj.error_code is None:
        response.status_code = 502
        resp_obj.error_code = "DETECTOR_COMMUNICATION_FAILED"

    # Metrics
    for r in results:
        metrics.record_detector_latency(r.detector, r.status.value == "success", r.processing_time_ms)
    metrics.record_coverage(request.tenant_id, request.policy_bundle, coverage_achieved)
    # Record policy enforcement for coverage
    sc = response.status_code if isinstance(response.status_code, int) else 200
    if sc == 206:
        metrics.record_policy_enforcement(request.tenant_id, request.policy_bundle, enforced=False, violation_type="coverage_below_threshold")
    else:
        metrics.record_policy_enforcement(request.tenant_id, request.policy_bundle, enforced=True)
    # Request status labeling
    if sc == 206:
        _req_status = "partial"
    elif sc == 408:
        _req_status = "timeout"
    elif 200 <= sc < 300:
        _req_status = "success"
    else:
        _req_status = "error"
    metrics.record_request(request.tenant_id, request.policy_bundle, _req_status, duration_ms)

    # SLA check
    if duration_ms > settings.config.sla.sync_request_sla_ms:
        response.status_code = 408
        resp_obj.error_code = "REQUEST_TIMEOUT"
        resp_obj.fallback_used = True

    # Write idempotency cache
    if idempotency_key and request.priority.value != "critical":
        # Store with fingerprint when available
        try:
            IDEMPOTENCY.set(idempotency_key, resp_obj, _request_fp)  # type: ignore[arg-type]
        except TypeError:
            # Fallback to legacy signature
            IDEMPOTENCY.set(idempotency_key, resp_obj)
        response.headers["Idempotency-Key"] = idempotency_key
    # Write response cache
    if settings.config.cache_enabled and request.priority.value != "critical":
        key = RESP_CACHE.build_key(request.content, tuple(routing_plan.primary_detectors), request.policy_bundle)
        RESP_CACHE.set(key, resp_obj)

    return resp_obj


@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate(
    request: OrchestrationRequest,
    response: Response,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    auth: AuthContext = Depends(auth_orchestrate_write),
    tenant_header: Optional[str] = Header(default=None, alias=settings.config.tenant_header),
):
    # Validate content size
    try:
        if len(request.content) > settings.config.max_content_length:
            raise HTTPException(status_code=400, detail="content_too_large")
    except Exception:
        pass

    # Enforce/propagate tenant context
    if tenant_header and request.tenant_id and tenant_header != request.tenant_id:
        raise HTTPException(status_code=403, detail="tenant_mismatch")
    if tenant_header and not request.tenant_id:
        request.tenant_id = tenant_header
    if not request.tenant_id:
        raise HTTPException(status_code=400, detail="missing_tenant_id")

    # Build routing plan and decision
    _ensure_clients_initialized()
    router = ContentRouter(settings, health_monitor=HEALTH_MONITOR, policy_manager=POLICY_MANAGER)
    # Optionally infer content type (advisory)
    try:
        inferred = infer_content_type(request.content)
        # If mismatch, keep request value but may be used by policy
    except Exception:
        inferred = request.content_type
    routing_plan, decision = await router.route_request(request)

    # Estimate time budget; convert to async if needed
    detector_time_est = max(routing_plan.timeout_config.values()) if routing_plan.timeout_config else 0
    budget_ms = settings.config.sla.sync_to_async_threshold_ms
    if settings.config.auto_map_results:
        detector_time_est += settings.config.sla.mapper_timeout_budget_ms
    if request.processing_mode.value == "async" or detector_time_est > budget_ms:
        # Spawn async job via JobManager with priority handling
        _ensure_clients_initialized()
        assert JOB_MANAGER is not None
        callback_url = None
        try:
            if isinstance(request.metadata, dict):
                cb = request.metadata.get("callback_url")
                if isinstance(cb, str) and cb:
                    callback_url = cb
        except Exception:
            callback_url = None
        job_id = await JOB_MANAGER.enqueue(request, idempotency_key, decision, routing_plan, callback_url=callback_url)

        # Return async response shell
        response.status_code = 202
        return OrchestrationResponse(
            request_id="req",
            job_id=job_id,
            processing_mode=ProcessingMode.ASYNC,
            detector_results=[],
            aggregated_payload=None,
            mapping_result=None,
            total_processing_time_ms=0,
            detectors_attempted=len(routing_plan.primary_detectors),
            detectors_succeeded=0,
            detectors_failed=0,
            coverage_achieved=0.0,
            routing_decision=decision,
            fallback_used=False,
            timestamp=datetime.now(timezone.utc),
            idempotency_key=idempotency_key,
        )

    # Sync path
    return await _process_sync(request, response, idempotency_key, decision, routing_plan)


@app.post("/orchestrate/batch")
async def orchestrate_batch(
    requests: list[OrchestrationRequest],
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    auth: AuthContext = Depends(auth_orchestrate_write),
    tenant_header: Optional[str] = Header(default=None, alias=settings.config.tenant_header),
):
    # Simple sequential batch for MVP
    out = []
    for r in requests:
        # Ensure tenant header propagated per element if not set
        if tenant_header and not r.tenant_id:
            r.tenant_id = tenant_header
        _ensure_clients_initialized()
        router = ContentRouter(settings, health_monitor=HEALTH_MONITOR)
        plan, decision = await router.route_request(r)
        # Derive per-item idempotency child key to avoid cross-item conflicts
        child_key: Optional[str] = None
        if idempotency_key:
            try:
                import hashlib as _hashlib
                import json as _json_local
                _fp_obj = {
                    "content_hash": _hashlib.sha256((r.content or "").encode("utf-8")).hexdigest(),
                    "tenant_id": r.tenant_id,
                    "policy_bundle": r.policy_bundle,
                    "environment": r.environment,
                    "content_type": r.content_type.value if hasattr(r.content_type, "value") else str(r.content_type),
                    "priority": r.priority.value if hasattr(r.priority, "value") else str(r.priority),
                }
                _fp = _hashlib.sha256(_json_local.dumps(_fp_obj, sort_keys=True).encode("utf-8")).hexdigest()
                child_key = f"{idempotency_key}:{_fp}"
            except Exception:
                child_key = idempotency_key
        o = await _process_sync(r, Response(), child_key, decision, plan)
        out.append(o)
    return {"results": out, "idempotency_key": idempotency_key}


@app.get("/orchestrate/status/{job_id}", response_model=JobStatusResponse)
async def get_status(job_id: str, auth: AuthContext = Depends(auth_orchestrate_status)):
    _ensure_clients_initialized()
    assert JOB_MANAGER is not None
    job = JOB_MANAGER.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return job


@app.delete("/orchestrate/status/{job_id}", response_model=JobStatusResponse)
async def cancel_job(job_id: str):
    """Cancel a pending or running async job.

    Returns current job status. If the job is not found, 404. If the job cannot be cancelled
    (already completed/failed/cancelled), returns 409 with current status.
    """
    _ensure_clients_initialized()
    assert JOB_MANAGER is not None
    job = JOB_MANAGER.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    if job.status.value in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=409, detail="job_not_cancellable")
    ok = JOB_MANAGER.cancel(job_id)
    job2 = JOB_MANAGER.get_status(job_id)
    if ok and job2:
        return job2
    # Fallback: return latest known status
    if job2:
        return job2
    raise HTTPException(status_code=404, detail="job_not_found")
