# Mapper Contracts Compliance

This document explains how the Llama Mapper service complies with the cross-service contracts defined in .kiro/specs/service-contracts.md.

Scope
- Applies to Mapper service /map and /map/batch endpoints
- Aligns with Sections 2–12 of the service contracts

1) Handoff Request (Sec 3)
- Preferred schema: MapperPayload with fields detector, output, metadata (contributing_detectors, normalized_scores, conflict_resolution_applied, aggregation_method, coverage_achieved, provenance[]), and tenant_id.
- Legacy support: DetectorRequest continues to be accepted during a deprecation window. The response includes Deprecation: true header and the metric mapper_request_deprecated_total{type="DetectorRequest"} increments.
- Privacy and size enforcement:
  - Max payload size: configurable via serving.max_payload_kb (default 64 KB). Oversize requests are rejected with 400 INVALID_REQUEST.
  - Raw content guardrail: serving.reject_on_raw_content (default true). Requests that appear to include raw free-text or large blobs are rejected with 400 INVALID_REQUEST.

2) Mapper Response (Sec 4)
- Response model extends MappingResponse with version_info {taxonomy, frameworks, model}.
- Provenance.model is populated; notes include a versions: tag for backward compatibility.

3) Canonical Error Model (Sec 8)
- Responses on error include a canonical JSON body: {error_code, message, request_id, retryable}.
- Implemented error codes:
  - INVALID_REQUEST (400)
  - REQUEST_TIMEOUT (408)
  - INTERNAL_ERROR (500)
- Retry guidance:
  - REQUEST_TIMEOUT: retryable = true
  - INVALID_REQUEST, INTERNAL_ERROR: retryable = false/true depending on context; current internal errors default to retryable=true (can be refined).

4) SLOs and Timeout Budget (Sec 7)
- The model call in /map is wrapped with asyncio.wait_for using serving.mapper_timeout_ms (default 500 ms).
- On timeout, returns 408 REQUEST_TIMEOUT (retryable=true) and logs mapper_errors_total{error_code="REQUEST_TIMEOUT"}.
- No automatic mapper-internal fallback on timeout; fallback remains only for model validation/low-confidence errors, consistent with the orchestrator contract.

5) Observability (Sec 12)
- Existing metrics retained (requests, duration, schema validation, fallback, confidence, F1).
- Added counters:
  - mapper_errors_total{error_code}
  - mapper_request_payload_rejected_total{reason}
  - mapper_request_deprecated_total{type}

6) Security & Privacy (Sec 11)
- No raw content in logs. Metadata-only logging. Redaction utilities applied for error logging.
- Tenant isolation is enforced when auth is enabled. When auth is disabled (development), tenant_id is optional but respected if provided.

7) Boundary (Sec 2)
- Mapper does not assign orchestrator responsibilities and remains the source of truth for canonical taxonomy mapping.

8) Versioning and Provenance (Sec 10)
- VersionManager supplies taxonomy/frameworks/model versions for version_info and provenance.model.
- Notes include a textual versions tag for backwards compatibility.

Request examples
- MapperPayload (preferred):
  {
    "detector": "orchestrated-multi",
    "output": "toxic|hate|pii_detected",
    "tenant_id": "tenant-123",
    "metadata": {
      "contributing_detectors": ["deberta-toxicity", "openai-moderation"],
      "aggregation_method": "weighted_average",
      "coverage_achieved": 1.0,
      "provenance": [{"detector":"deberta-toxicity","confidence":0.93}]
    }
  }

- Legacy DetectorRequest (deprecated):
  {
    "detector": "deberta-toxicity",
    "output": "toxic"
  }

Configuration knobs
- MAPPER_SERVING_MAPPER_TIMEOUT_MS (default 500)
- MAPPER_SERVING_MAX_PAYLOAD_KB (default 64)
- MAPPER_SERVING_REJECT_ON_RAW_CONTENT (default true)

OpenAPI
- docs/openapi.yaml regenerated via scripts/export_openapi.py to include response version_info.
- Note: Request oneOf documentation (MapperPayload | DetectorRequest) will be introduced in a follow-up OpenAPI enhancement; current schema shows the body as a generic object while both shapes are accepted.
