# Rate limiting runbook

This document explains how the service enforces rate limiting, how to configure quotas, how identities are determined, and how to monitor and operate the limiter.

Overview
- Middleware: An ASGI middleware enforces token-bucket limits per identity.
- Identities (precedence):
  1) API key from header (default X-API-Key), hashed with SHA-256 (never stored in plaintext)
  2) Tenant ID (X-Tenant-ID or request.state.auth.tenant_id)
  3) Client IP (X-Forwarded-For respecting trusted_proxies; fallback request.client.host)
- Endpoints covered: /map and /map/batch
- Defaults: 600 requests/min for API key or tenant; 120 requests/min per IP; 60s window

Configuration (config.yaml)
rate_limit:
  enabled: true
  window_seconds: 60
  trusted_proxies: 0
  headers:
    emit_standard: true   # RateLimit-* and Retry-After
    emit_legacy: true     # X-RateLimit-*
  endpoints:
    map:
      api_key_limit: 600
      tenant_limit: 600
      ip_limit: 120
    map_batch:
      api_key_limit: 600
      tenant_limit: 600
      ip_limit: 120

Environment overrides
- RATE_LIMIT_ENABLED=true|false
- RATE_LIMIT_WINDOW_SECONDS=60
- RATE_LIMIT_TRUSTED_PROXIES=0
- RATE_LIMIT_HEADERS_EMIT_STANDARD=true|false
- RATE_LIMIT_HEADERS_EMIT_LEGACY=true|false

Headers returned
- On success (200):
  - RateLimit-Limit: "<limit>;w=<window_seconds>"
  - RateLimit-Remaining: remaining in current window
  - RateLimit-Reset: seconds until reset
  - X-RateLimit-Limit, X-RateLimit-Remaining (legacy)
- On 429:
  - Above headers + Retry-After: ceil(seconds until next token)

Observability
- Prometheus metrics:
  - mapper_rate_limit_requests_total{endpoint,identity_kind,action="allow|block"}
  - mapper_rate_limit_backend_errors_total
  - mapper_rate_limit_reset_seconds{endpoint,identity_kind} histogram
- Logs:
  - "Rate limit allow" at debug level with endpoint, identity_kind, remaining, limit, request_id
  - "Rate limit block" at warning level with endpoint, identity_kind, limit, reset_seconds, request_id
  - "Rate limit backend error" at error level with endpoint, identity_kind, error, request_id

Operational guidance
- Disable quickly: set RATE_LIMIT_ENABLED=false (or rate_limit.enabled: false) and restart
- Raise limits for a specific tenant/api key: adjust per-endpoint limits; for finer-grained control add a dedicated endpoint config if needed
- Trusted proxies: set rate_limit.trusted_proxies to the number of reverse proxies that append to X-Forwarded-For

Rollout plan and safeguards
- Stage with conservative limits and monitor:
  - mapper_rate_limit_requests_total (block fraction)
  - mapper_rate_limit_reset_seconds p95
- Alert on elevated 429 rate for sustained periods per endpoint
- Rollback: toggle RATE_LIMIT_ENABLED=false

OpenAPI regeneration
- Export current OpenAPI: python scripts/export_openapi.py --output docs/openapi.yaml
- CI check: .github/workflows/openapi-consistency.yml fails if docs/openapi.yaml is stale
