# Detector Orchestration Service

A microservice that selects and runs detectors in parallel, aggregates raw results, and hands off a unified payload to the existing Llama Mapper `/map` endpoint.

Quick start (dev)
- Install: `pip install -e detector-orchestration/`
- Run API: `uvicorn detector_orchestration.api.main:app --reload --port 8081`
- Health: `GET http://localhost:8081/health`
- Orchestrate: `POST http://localhost:8081/orchestrate`

Notes
- Built-in demo detectors are provided (endpoints like `builtin:toxicity`, `builtin:regex-pii`, `builtin:echo`) for local development.
- Mapper handoff is automatic by default (auto_map_results=true). Configure mapper endpoint via `ORCH_CONFIG__MAPPER_ENDPOINT=http://localhost:8000/map`.
- Do not log raw content. Only metadata should be logged.
- Rate limiting: simple per-tenant token bucket returning 403 with `error_code=RATE_LIMITED`. Configure via `ORCH_CONFIG__rate_limit_enabled`, `ORCH_CONFIG__rate_limit_window_seconds`, and `ORCH_CONFIG__rate_limit_tenant_limit`.
- Redis caches: set `ORCH_CONFIG__cache_backend=redis`, `ORCH_CONFIG__redis_url=redis://localhost:6379/0`, `ORCH_CONFIG__redis_prefix=orch` to enable Redis-backed idempotency/response caches.
