# Llama Mapper Operations Runbook

This runbook covers normal operations and incident response for the Llama Mapper service.

Key endpoints
- GET /health: liveness/readiness probe target
- GET /metrics: Prometheus exposition
- POST /map and POST /map/batch: main APIs

Contracts & boundaries
- Mapper accepts detector outputs via MapperPayload (preferred) and legacy DetectorRequest during a deprecation window.
- Mapper is source of truth for canonical taxonomy mapping and does not assume orchestrator responsibilities.

Configuration (contract knobs)
- MAPPER_SERVING_MAPPER_TIMEOUT_MS (default 500): timeout budget for model mapping calls
- MAPPER_SERVING_MAX_PAYLOAD_KB (default 64): max JSON payload size
- MAPPER_SERVING_REJECT_ON_RAW_CONTENT (default true): reject suspected raw content

Common failure modes
1) Payload rejections
- Symptom: 400 with error body {error_code: INVALID_REQUEST}
- Causes: oversize payloads or suspected raw content
- Action:
  - Verify orchestrator is sending MapperPayload (no raw content) and within size limits
  - If a legitimate workload requires larger metadata, increase maxPayloadKb cautiously and redeploy
  - Do NOT disable raw-content rejection unless in controlled environments

2) Request timeouts
- Symptom: 408 with error body {error_code: REQUEST_TIMEOUT, retryable: true}
- Causes: model call exceeded mapper_timeout_ms budget
- Action:
  - Retry at orchestrator respecting exponential backoff
  - Investigate backend capacity (TGI/vLLM), model settings, or reduce sequence length/generation parameters
  - Scale replicas or move to GPU profile if needed

3) Elevated fallback usage
- Symptom: Increased fallback % and lower confidence scores
- Action:
  - Investigate model quality regressions
  - Review detector configurations and taxonomy changes
  - Re-run quality gates and retrain/calibrate if needed

Monitoring and alerts (Prometheus)
- mapper_requests_total{detector, status}
- mapper_request_duration_seconds histogram (check p95 vs SLO)
- mapper_schema_valid_percentage gauge (target 		95%)
- mapper_fallback_percentage gauge (target < 10%)
- mapper_errors_total{error_code}
- mapper_request_payload_rejected_total{reason}
- mapper_request_deprecated_total{type}

Dashboards
- Create panels for request volume, latency p95, schema valid %, fallback %, errors by error_code, payload rejections by reason, and deprecated request usage.

Deployment notes (Helm)
- values.yaml mapper tunables:
  mapper:
    timeoutMs: 500
    maxPayloadKb: 64
    rejectOnRawContent: true
- Environment variables are injected automatically via templates/deployment.yaml

Escalation
- If payload rejections spike: contact orchestrator team to verify MapperPayload format and metadata size
- If timeouts spike: page on-call for model serving (TGI/vLLM) and consider temporary increase of timeoutMs while investigating
- If schema valid % drops: page model/validation owners and roll back to last known good adapter if needed
