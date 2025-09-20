# Threat Model (STRIDE) for Llama Mapper Service

Scope
- FastAPI service handling detector outputs and mapping them to canonical taxonomy
- StorageManager persisting records to DB and S3 with BYOK
- Multi-tenant isolation (TenantIsolationManager)
- Secrets access for API keys/model weights via SecretsManager

STRIDE Analysis
- Spoofing
  - Risk: API key spoofing, tenant header spoofing
  - Mitigations: API key auth with per-tenant scopes; require tenant header; rate limiting; idempotency
  - Actions: Enforce HSTS/TLS; add optional request signing/OIDC in future
- Tampering
  - Risk: Payload tampering, DB tampering, S3 object modification
  - Mitigations: JSON schema validation; WORM/Object Lock; RLS policies; KMS encryption
  - Actions: Add checksum/signature on records; immutable audit trails
- Repudiation
  - Risk: Users deny actions; lack of traceability
  - Mitigations: Privacy-first audit logs with request_id, tenant_id; AuditTrailManager
  - Actions: Time-sync and version tagging; SIEM integration
- Information Disclosure
  - Risk: Logging raw prompts/outputs; secret leaks; cross-tenant data leakage
  - Mitigations: Metadata-only logging; PII redaction; TenantIsolationManager; pre-commit lints; BYOK
  - Actions: Periodic log scanning; secret scanning in CI
- Denial of Service
  - Risk: High-volume requests; expensive prompts; storage abuse
  - Mitigations: RateLimit middleware; batch limits; S3 Object Lock retention limits
  - Actions: Autoscaling with HPA; circuit breakers and backoff
- Elevation of Privilege
  - Risk: Excess IAM/Vault permissions; code injection via payloads
  - Mitigations: Least privilege IAM; SecretsManager audits; strict JSON schema; input validation
  - Actions: Regular IAM review; static analysis for deserialization vulnerabilities

Residual Risks
- Third-party model/serving dependencies
- Optionality of Vault backend configuration

Assumptions
- TLS termination is configured at ingress
- KMS and IAM are properly configured by infrastructure

References
- docs/security/privacy-checklist.md
- src/llama_mapper/security/secrets_manager.py
- src/llama_mapper/security/redaction.py
