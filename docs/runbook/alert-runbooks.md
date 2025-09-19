# Alert Runbooks

Owner: On-call (Platform + ML)
Last updated: 2025-09-19

This guide provides investigation and resolution steps for key alerts emitted by the Llama Mapper service.

Conventions
- Metrics of interest:
  - mapper_schema_valid_percentage (gauge)
  - mapper_fallback_percentage (gauge)
  - mapper_request_duration_seconds (histogram)
  - mapper_requests_total{status} (counter)
- SLO targets (default):
  - Schema-valid ≥ 99.5%
  - Fallback < 10%
  - P95 latency ≤ 250ms (CPU/TGI) or ≤ 150ms (GPU/vLLM)
  - 5xx error rate < 1%

MapperSchemaValidLow
- Symptom: schema-valid % below threshold for sustained period.
- Impact: clients may receive invalid JSON; downstream may reject/repair.
- Likely causes:
  - Model regression or drift
  - Changes to schema or taxonomy not matched in prompts/training
  - Increased traffic profile outside training distribution
- Investigate:
  1) Check recent deploys (image tag, adapters, configs)
  2) Inspect /metrics/summary and logs for JSON validation failures
  3) Review JSONValidator errors and detector-level patterns
  4) Run golden tests in staging (mapper quality validate)
- Mitigate:
  - Enable kill-switch (rules_only) if customer impact is high: mapper runtime kill-switch on
  - Roll back LoRA adapter or config (see operations.md)
  - Increase retry/validation tolerances temporarily if appropriate
- Follow-ups:
  - Improve training data, prompts; adjust generation parameters; add tests

MapperFallbackHigh
- Symptom: fallback percentage above threshold for sustained period.
- Impact: model utility reduced; rule-based mapping dominates.
- Likely causes:
  - Low confidence predictions (confidence < threshold)
  - Model/server degradation or timeouts
  - New detector patterns not well covered
- Investigate:
  1) Check confidence distribution (mapper_confidence_score) if available
  2) Correlate with latency and 5xx errors; look for model server errors
  3) Review recent config/model changes
- Mitigate:
  - Temporarily lower confidence threshold (ConfigManager) with careful monitoring
  - If model unstable, switch to rules_only and triage model
  - Roll back model/config as needed

MapperLatencyP95High
- Symptom: p95 latency above threshold for profile (CPU/GPU).
- Impact: SLO breach; API latency increases.
- Likely causes:
  - Backend contention or resource starvation (CPU/GPU)
  - External model server (TGI) latency spike or network issues
  - Traffic surge beyond capacity
- Investigate:
  1) Compare current profile (.Values.profile) to thresholds
  2) Check HPA scaling, pod resource usage, throttling, OOM restarts
  3) Inspect TGI/vLLM logs and saturation (GPU utilization)
- Mitigate:
  - Scale out (replicas) or up (resources)
  - Switch to GPU/vLLM or enable quantization if appropriate
  - Reduce batch sizes or tune generation parameters

Mapper5xxHigh
- Symptom: 5xx error rate above threshold.
- Impact: API may be failing requests.
- Likely causes:
  - Code exceptions, schema loading failures, or external dependency outages
  - Model server errors/timeouts or service misconfiguration
- Investigate:
  1) Inspect logs for stack traces (privacy-safe)
  2) Check /health, and dependent services (TGI, S3, DB)
  3) Recent changes in Helm values, configs, or secrets
- Mitigate:
  - Roll back most recent change (image or values)
  - Enable rules_only mode if model path failing but rule-path OK
  - Hotfix if clear, else stabilize and escalate

MapperQueueLagHigh (optional)
- Symptom: queue lag metric indicates backlog growth
- Impact: increased latency or timeouts
- Likely causes:
  - Downstream bottlenecks, insufficient consumers, or upstream spikes
- Investigate:
  1) Verify presence of mapper_queue_lag_seconds (or external queue metric)
  2) Check consumer health and scaling
- Mitigate:
  - Scale consumers, reduce load, or purge dead-letter queues as per policy

Escalation
- If critical alerts persist >30m, page secondary on-call and initiate incident process.
- Capture timeline, actions, and attach metrics snapshots to post-mortem.
