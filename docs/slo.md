# Llama Mapper SLOs and Performance Targets

Owner: Platform/Model Serving
Effective date: 2025-09-01
Review cadence: Quarterly

1) Scope and customer experience
- Endpoint: POST /map and POST /map/batch
- Customers expect fast, reliable normalization of detector outputs with strong correctness guarantees.
- Primary failure modes: schema-invalid outputs, excessive fallback usage (rule-only), elevated latency.

2) Service Level Indicators (SLIs)
- Latency (p95): Histogram-derived p95 per-detector from mapper_request_duration_seconds (seconds)
- Schema-valid percentage: mapper_schema_valid_percentage gauge (Prometheus); calculated from mapper_schema_validation_total
- Fallback percentage: mapper_fallback_percentage gauge (Prometheus); calculated from mapper_fallback_total vs mapper_requests_total
- Availability (observed): percentage of 2xx responses for /map over request count (optional; derived from requests_total{status})

3) Service Level Objectives (SLOs)
Targets are for CPU deployment profile (TGI/CPU) unless stated otherwise.
- Latency, p95: ≤ 250 ms (0.25 s) over 30-day rolling window (CPU)
  - GPU profile (vLLM/GPU) informative target: ≤ 120 ms p95
- Schema-valid rate: ≥ 99.5% over 30-day rolling window
- Fallback usage: < 10.0% over 30-day rolling window
- Availability (observed): ≥ 99.9% 2xx over 30-day rolling window (informational)

4) Measurement & queries
Prometheus (scrapes /metrics). Example queries:
- p95 latency (CPU):
  histogram_quantile(0.95, sum(rate(mapper_request_duration_seconds_bucket[5m])) by (le))
- Schema-valid %:
  100 * (sum(rate(mapper_schema_validation_total{valid="true"}[5m])) / sum(rate(mapper_schema_validation_total[5m])))
- Fallback %:
  100 * (sum(rate(mapper_fallback_total[5m])) / sum(rate(mapper_requests_total[5m])))
- Availability %:
  100 * (sum(rate(mapper_requests_total{status="success"}[5m])) / sum(rate(mapper_requests_total[5m])))

Windows:
- SLO windows: 30 days rolling
- Alert windows: 5m short, 15m sustained, and 1h confirmation

5) Alerting thresholds & escalation
- Latency (CPU):
  - Warning: p95 > 200 ms for 5m
  - Critical: p95 > 250 ms for 10m
- Schema-valid %:
  - Critical: < 99.5% for 15m
- Fallback %:
  - Warning: > 10% for 10m
- Availability (observed):
  - Critical: < 99.0% for 10m

Escalation path:
- Page on-call (P2 for warnings, P1 for critical)
- Runbook: see Operational Runbooks (17.1) for rollback to rule-only mapping, LoRA rollback, and config kill-switch

6) Load model and capacity planning
Terminology:
- RPS_tenant: steady-state requests per second per tenant
- p95: 95th percentile latency (seconds)
- Concurrency ≈ RPS_total × p95
- Pod capacity (CPU profile) target: keep per-pod concurrency <= 100 and CPU utilization <= 70%

Initial sizing (CPU profile):
- Assume 50 RPS total, p95 0.25s => concurrency ≈ 12.5 → allocate 1-2 pods
- Autoscaling: HPA on CPU utilization with bounds (min 2, max N), target 60-70%
- GPU profile guidance: lower latency, smaller fleet; ensure nodeSelector/tolerations and nvidia.com/gpu: 1

7) Error budget
- For schema-valid SLO (99.5%), error budget = 0.5% invalid over 30 days
- For latency SLO (<= 250 ms p95), budget = 0 ms over target; track burn rate via SLO rule violations

8) Dependencies and assumptions
- JSONValidator enforces pillars-detectors/schema.json correctness
- FallbackMapper covers low-confidence or invalid cases; track fallback % to limit rule-only usage
- Rate limiting must not distort SLOs; exclude known 429 test storms from SLO windows

9) Observability mapping
- /metrics: Prometheus exposition (MetricsCollector)
- /metrics/summary: JSON summary for quick debugging
- Alerts endpoint (/metrics/alerts) already maps quality thresholds (95% schema-valid, 10% fallback, 250 ms p95) used for internal checks; SLOs are stricter (99.5% schema-valid)

10) Operational actions
- If schema-valid dips below 99.5%: inspect JSONValidator failures; roll back recent model adapters; temporarily raise confidence threshold to route more to fallback; prioritize fixes
- If fallback exceeds 10%: investigate model confidence calibration; expand rule coverage; review detector YAML mappings
- If p95 latency exceeds 250 ms: scale out (HPA), reduce sequence length or max_new_tokens, consider GPU profile or quantization adjustments

11) Documentation & ownership
- This document is source-of-truth for SLOs. Keep aligned with Helm values and CI quality gates.
- Owners: Platform/Infra + ML Eng (joint)
