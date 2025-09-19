# Service Contracts: Orchestrator ↔ Mapper ↔ Analysis

Status: DRAFT (locked sections explicitly labeled)
Owners: Platform + Compliance Engineering

1. Purpose

This document defines the cross-service contracts and boundaries for:
- Detector Orchestration: selects and executes detectors, aggregates raw results
- Llama Mapper: maps detector outputs to the canonical taxonomy with schema validation and fallbacks
- Analysis Module: produces concise, auditable insights and policy diffs from structured metrics

The goal is to prevent scope creep and ambiguity by locking interfaces where needed and aligning on shared semantics (coverage, errors, versioning, observability).

2. Roles and Responsibilities (authoritative boundaries)

- Orchestrator
  - Selects detectors per tenant policy/content type/capabilities (OPA optional)
  - Executes detectors in parallel with health/circuit-breaker and retries
  - Aggregates raw detector outputs, computes coverage, resolves conflicts
  - Produces a MapperPayload for handoff to the Mapper
  - May optionally call /map (auto_map_results) with strict timeout budget
  - Does NOT assign canonical taxonomy labels

- Mapper
  - Converts detector outputs to canonical taxonomy
  - Enforces strict JSON schema; confidence, fallback mapping; versioning
  - Persists canonical events; integrates compliance framework mappings
  - Exposes /map endpoint; produces MappingResponse

- Analysis Module
  - Consumes structured metrics (e.g., from ClickHouse) to produce reason/remediation and optional OPA diffs
  - Deterministic outputs with schema validation and template fallback
  - Does NOT call detectors or assign canonical taxonomy

3. Handoff Contract: Orchestrator → Mapper (LOCKED)

Endpoint: Mapper /map (see mapper OpenAPI)
Payload schema (MapperPayload) — values and examples align with detector-orchestration design.

Fields
- detector: string (e.g., "orchestrated-multi")
- output: string
  - Aggregated raw indication for mapper prompt context (e.g., "toxic|hate|pii_detected")
  - Orchestrator MUST NOT claim canonical taxonomy here
- metadata: object
  - contributing_detectors: string[]
  - normalized_scores: object { taxonomy_label: float [0,1] } — optional hints; mapper remains source of truth
  - conflict_resolution_applied: boolean
  - aggregation_method: enum [weighted_average, majority_vote, highest_confidence, most_restrictive]
  - coverage_achieved: float [0,1]
  - provenance: array of { detector, confidence?, output, processing_time_ms? }
- tenant_id: string

Constraints
- Max payload size: 64 KB
- No raw customer content; include only detector outputs and derived metadata
- PII: orchestrator must redact any free-text metadata if present

Mapper expectations
- Mapper treats metadata as hints; canonical decision is based on its fine-tuned model + fallback rules
- Mapper enforces schema and may downgrade confidence or fall back when hints conflict

4. Mapper Response Contract (MappingResponse) — reference alignment

The Mapper returns canonical events per its schema (pillars-detectors/schema.json). Orchestrator (when auto_map_results = true) will pass-through these fields in OrchestrationResponse.mapping_result. Key fields referenced across services:
- taxonomy: string[] (canonical labels)
- scores: { taxonomy_label: float }
- confidence: float [0,1]
- notes: string (≤500 chars), may include conflict resolution notes from Orchestrator
- provenance: { vendor?, detector?, detector_version?, route?, model?, tenant_id?, ts? }
- policy_context?: { expected_detectors?: string[], environment?: string }

5. Orchestration External Response Contract (stable to clients)

Fields
- request_id: string
- processing_mode: enum [sync, async]
- detector_results: DetectorResult[]
  - detector: string
  - status: enum [success, failed, timeout, unavailable, skipped]
  - output?: string
  - metadata?: object
  - error?: string
  - processing_time_ms: int
  - confidence?: float
- aggregated_payload?: MapperPayload (locked handoff schema)
- mapping_result?: MappingResponse (present when auto_map_results=true and call succeeded)
- total_processing_time_ms: int
- detectors_attempted: int
- detectors_succeeded: int
- detectors_failed: int
- coverage_achieved: float [0,1]
- routing_decision: { selected_detectors: string[], routing_reason: string, policy_applied: string, coverage_requirements: {detector: float}, health_status: {detector: bool} }
- fallback_used: boolean
- timestamp: RFC3339
- error_code?: string (see Section 8)
- idempotency_key?: string
- job_id?: string (async)

6. Coverage Semantics (LOCKED)

CoverageCalculation defines how achieved coverage is computed. The same semantics apply to Orchestrator KPIs and Analysis inputs.

Methods
- REQUIRED_SET: 100% when ALL required detectors succeed at least once for the request
  - coverage = (# required detectors succeeded) / (total required detectors)
- WEIGHTED_COVERAGE: 100% when weighted sum of successes reaches 1.0
  - coverage = sum_i (w_i * success_i), sum_i w_i = 1.0, success_i ∈ {0,1}
- TAXONOMY_COVERAGE: 100% when required taxonomy categories are covered by the union of detector signals for the request
  - coverage = (# required taxonomy categories hit) / (total required taxonomy categories)

Thresholds
- partial_coverage_threshold: 0.8 (default). Below threshold: HTTP 206 (Partial Content) with error_code=PARTIAL_COVERAGE and fallback_used=true.

7. Timeouts, SLAs, and Sync→Async (LOCKED defaults)

Global SLAs
- sync_request_sla_ms: 2000
- async_request_sla_ms: 30000
- mapper_timeout_budget_ms: 500 (portion reserved for /map when auto_map_results=true)
- sync_to_async_threshold_ms: 1500 (convert to async if exceeded)

Per-priority overrides
- CRITICAL: 1000ms sync SLA budget
- LOW: 5000ms sync SLA budget

Behavior
- If projected completion exceeds threshold, orchestrator returns job_id and converts to async
- Mapper call must respect the reserved timeout budget; if exceeded, return orchestrator response without mapping_result and set fallback_used=true with appropriate error_code

8. Canonical Error Model (LOCKED)

HTTP → error_code
- 400 INVALID_REQUEST
- 400 POLICY_NOT_FOUND
- 401 UNAUTHORIZED
- 403 INSUFFICIENT_RBAC
- 403 RATE_LIMITED
- 206 PARTIAL_COVERAGE (results present; coverage below threshold)
- 408 REQUEST_TIMEOUT (global SLA exceeded)
- 429 DETECTOR_OVERLOADED
- 502 ALL_DETECTORS_UNAVAILABLE
- 502 DETECTOR_COMMUNICATION_FAILED
- 500 AGGREGATION_FAILED
- 500 INTERNAL_ERROR

Retry-safety Guidance
- RATE_LIMITED, REQUEST_TIMEOUT, DETECTOR_OVERLOADED, ALL_DETECTORS_UNAVAILABLE, DETECTOR_COMMUNICATION_FAILED → retryable
- Others → non-retryable without manual intervention

9. Idempotency and Caching (LOCKED behaviors)

- Idempotency-Key header accepted on all orchestrator endpoints
- Retention: 24h for sync requests; async job_id serves as idempotency key
- Cache key components: content_hash, detector_set, policy_bundle
- CRITICAL requests bypass cache
- Cache invalidation on policy or detector registry changes

10. Versioning and Provenance (LOCKED fields)

- All services include version_info in their responses or metadata:
  - taxonomy: string
  - frameworks: string
  - model: string (mapper model or analyst model depending on service)
- Orchestrator MUST propagate relevant version tags into mapping_result.notes/provenance when it triggers /map
- Policy bundles must carry version

11. Security and Privacy (LOCKED posture)

- Never log raw content; redact free-text metadata before logging
- Enforce tenant isolation; include tenant_id in all cross-service calls
- Do not persist raw detector inputs by default; metadata-only logs
- Secrets via Vault/AWS Secrets Manager; no secrets in logs or metrics

12. Observability Contracts (LOCKED metric names)

- orchestrate_requests_total{tenant, policy, status}
- orchestrate_request_duration_ms histogram
- detector_latency_ms{detector, status} histogram
- detector_health_status{detector} gauge
- detector_health_check_duration_ms{detector} histogram
- circuit_breaker_state{detector, state} gauge
- coverage_achieved{tenant, policy} gauge
- policy_enforcement_total{tenant, policy, status, violation_type} counter

Mapper and Analysis retain their existing metric names and SLOs; Analysis will report schema_valid_rate, template_fallback_rate, and OPA compilation success rate.

13. Compatibility and Extension Guidelines

- MapperPayload is minimal by design. Additive metadata keys are allowed but MUST NOT change canonical behavior; taxonomy remains a Mapper decision.
- Coverage semantics are shared and locked; if a tenant requires alternative semantics, define them via policy versioning + OPA evaluation, not by changing computation definitions.
- Error codes are closed-set; request additions via RFC with clear HTTP mapping and retry-safety classification.
- Auto-mapping is a toggle; downstream workers may call /map later using the same MapperPayload.

14. Validation Checklist

- [ ] Orchestrator: validates MapperPayload before invoking /map
- [ ] Mapper: validates MappingResponse against schema.json; rejects malformed payloads
- [ ] Analysis: validates inputs and outputs; PII redaction on free-text fields
- [ ] Idempotency: duplicate requests within TTL return cached response
- [ ] Observability: required metrics emitted with correct labels
- [ ] Security: no raw content in logs or metrics; tenant isolation enforced
