# Implementation Checklist: Mapper Contracts Compliance

- API models (src/llama_mapper/api/models.py)
  - [x] MapperPayload, HandoffMetadata, VersionInfo, ErrorBody
  - [x] MappingResponse.version_info

- API handlers (src/llama_mapper/api/mapper.py)
  - [x] Accept MapperPayload and legacy DetectorRequest (with deprecation headers/metrics)
  - [x] Privacy guardrails (raw-content, 64KB max) with canonical errors
  - [x] Timeout budget with canonical 408 REQUEST_TIMEOUT
  - [x] Enrich responses with version_info and provenance

- JSON Validator (src/llama_mapper/serving/json_validator.py)
  - [x] Tolerate optional version_info in model output during validation

- Versioning (src/llama_mapper/versioning/version_manager.py)
  - [x] get_version_info_dict helper

- Observability (src/llama_mapper/monitoring/metrics_collector.py)
  - [x] mapper_errors_total{error_code}
  - [x] mapper_request_payload_rejected_total{reason}
  - [x] mapper_request_deprecated_total{type}

- Config (src/llama_mapper/config/manager.py)
  - [x] serving.mapper_timeout_ms, serving.max_payload_kb, serving.reject_on_raw_content
  - [x] Env overrides for these fields

- OpenAPI (scripts/export_openapi.py)
  - [x] oneOf request bodies for /map and /map/batch
  - [x] Ensure components schemas are present

- Helm (charts/llama-mapper)
  - [x] values.yaml mapper tunables
  - [x] deployment env wiring

- Docs
  - [x] docs/contracts/mapper_compliance.md
  - [x] docs/release/mapper_migration.md
  - [x] docs/runbook/mapper-operations.md
  - [x] docs/dev/wsl2.md
  - [x] README examples and deprecation notes

- Tests
  - [x] Unit: error helper, version info helper
  - [x] Integration: API happy path, fallback, deprecation headers
  - [x] Integration: raw-content and oversize rejections
  - [x] Integration: timeout behavior and metrics
