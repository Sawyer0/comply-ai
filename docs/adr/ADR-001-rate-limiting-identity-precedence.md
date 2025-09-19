# ADR-001: Rate limiting identity precedence and proxy handling

Date: 2025-09-19

Status: Accepted

Context
- We must protect the /map and /map/batch endpoints from abuse while preserving multi-tenant fairness.
- Requests may be proxied; IP-based identity must account for X-Forwarded-For.

Decision
- Identity precedence for rate limiting:
  1) API key from header (default X-API-Key). We use a SHA-256 hash for storage to avoid keeping secrets in plain text.
  2) Tenant ID (X-Tenant-ID) if present, or request.state.auth.tenant_id if auth dependency populated it.
  3) Client IP from X-Forwarded-For, selecting the first untrusted hop based on configured trusted_proxies. If absent, fallback to request.client.host.
- Limits apply per endpoint key: "map" for /map, "map_batch" for /map/batch.
- Response headers follow the IETF RateLimit fields and include legacy X-RateLimit-* headers for compatibility.

Consequences
- API-key identities get their own independent buckets; tenant-scoped identities apply when no API key is provided; unauthenticated requests fall back to IP-based quotas.
- Operators must set rate_limit.trusted_proxies to the hop count of trusted proxies to ensure correct client IP extraction.

Alternatives considered
- Always use IP identity: unfair in multi-tenant and API key scenarios.
- Only API key identity: does not protect unauthenticated endpoints.
- JWT/OIDC subject identity: Future enhancement; would be added as another high-precedence identity when OIDC is enabled.

Rollout
- Default limits enabled with conservative quotas.
- Monitor mapper_rate_limit_requests_total and mapper_rate_limit_reset_seconds; alert on sustained 429 rate increases.
