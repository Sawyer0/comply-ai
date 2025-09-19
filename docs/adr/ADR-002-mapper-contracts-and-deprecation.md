# ADR-002: Mapper Service Contracts Compliance and Request Schema Deprecation

Status: Accepted
Date: 2025-09-19

Context
- Mapper must comply with .kiro/specs/service-contracts.md (Sections 2–12), including locked handoff schema, error model, SLOs/timeouts, observability, and privacy.
- Legacy DetectorRequest is in use by some clients; we are introducing MapperPayload as the preferred request schema.

Decision
- Adopt MapperPayload as the preferred request schema for /map and /map/batch.
- Continue supporting DetectorRequest during a deprecation window, with headers and metrics to signal migration.
- Enforce contract knobs:
  - mapper_timeout_ms (default 500 ms) for model calls
  - max_payload_kb (default 64 KB) for request payloads
  - reject_on_raw_content (default true) as a safety guardrail
- Standardize error bodies across endpoints using canonical codes and retryable guidance.
- Enrich responses with version_info {taxonomy, frameworks, model} and retain legacy provenance + version in notes.

Consequences
- Clients must migrate to MapperPayload by the sunset date.
- Requests containing raw content or oversize data are rejected with 400 INVALID_REQUEST.
- Timeouts return 408 REQUEST_TIMEOUT and are retryable.
- Observability provides visibility into deprecation usage, payload rejections, and error codes.

Sunset and Removal
- Announce: 2025-09-19
- Sunset (DetectorRequest): 2025-10-31 00:00:00 GMT
- Removal target: 0.3.0 (no earlier than 2025-11-15)

References
- docs/contracts/mapper_compliance.md
- docs/release/mapper_migration.md
- docs/runbook/mapper-operations.md
