# Acceptance Criteria: Mapper Contracts Compliance

Overall status: READY

API behavior
- /map and /map/batch accept MapperPayload and return MappingResponse with version_info populated.
- Legacy DetectorRequest accepted during deprecation window and adds Deprecation, Sunset, and Link headers.
- Canonical error bodies present for 400 INVALID_REQUEST and 408 REQUEST_TIMEOUT with correct retryable flags.

Privacy and size
- Raw-content-like and oversize payloads return 400 INVALID_REQUEST.
- No raw content logged; metadata-only logging policy holds.

Timeouts and SLOs
- Model call wrapped with mapper_timeout_ms (default 500 ms). Exceeded calls return 408 and are counted in metrics.

Observability
- Metrics present:
  - mapper_requests_total{detector, status}
  - mapper_request_duration_seconds
  - mapper_schema_valid_percentage
  - mapper_fallback_percentage
  - mapper_errors_total{error_code}
  - mapper_request_payload_rejected_total{reason}
  - mapper_request_deprecated_total{type}

OpenAPI and docs
- docs/openapi.yaml reflects oneOf request bodies and includes schemas.
- docs/contracts/mapper_compliance.md describes implementation.
- docs/release/mapper_migration.md outlines deprecation plan and timeline.
- docs/runbook/mapper-operations.md provides operational guidance.

Helm and config
- Helm values expose mapper.timeoutMs, mapper.maxPayloadKb, mapper.rejectOnRawContent.

Tests
- Unit and integration tests passing in WSL2 Ubuntu.
- Coverage includes raw-content and oversize rejections, timeout behavior, canonical error bodies, and version tags.
