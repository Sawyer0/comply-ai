# Troubleshooting Guide

Owner: Platform/ML Ops
Last updated: 2025-09-19

Use this guide to quickly diagnose common issues in Llama Mapper.

1) API returning 5xx errors
- Checks:
  - GET /health on the service
  - Inspect pod logs for exceptions (privacy-safe)
  - Verify external dependencies: model server (TGI/vLLM), S3, DB
- Fixes:
  - Roll back recent deploy or values change
  - Switch to rules_only mode to bypass model temporarily
  - Validate configuration (mapper validate-config)

2) Schema validation failures increased suddenly
- Checks:
  - Look at JSONValidator error messages in logs
  - Confirm pillars-detectors/schema.json and model prompts align
  - Run golden tests (mapper quality validate)
- Fixes:
  - Revert recent schema/mapping changes or adapter version
  - Regenerate examples or adjust generation parameters

3) High fallback percentage
- Checks:
  - Review confidence scores distribution; ensure threshold is appropriate
  - Examine model server latency/timeouts
- Fixes:
  - Tune confidence threshold cautiously; monitor impact
  - Improve training for affected detectors; update rules where needed

4) High p95 latency
- Checks:
  - Pod resource usage (CPU throttling, memory pressure)
  - HPA scaling status; replica counts
  - Backend profile (tgi vs vllm) and GPU utilization
- Fixes:
  - Scale replicas/resources; switch profile when appropriate
  - Optimize generation settings; enable quantization

5) Rate limiting anomalies
- Checks:
  - Review docs/runbook/rate-limits.md
  - Inspect rate_limit_* metrics
- Fixes:
  - Adjust limits or identity precedence per ADR-001

6) Configuration changes not applied
- Checks:
  - Confirm Helm values diff and applied revision
  - Ensure ConfigMaps/Secrets mounted correctly
- Fixes:
  - Redeploy with updated values; avoid manual pod restarts without rollout

7) Storage (S3/DB) issues
- Checks:
  - Verify IAM/creds; KMS keys and region
  - Test connectivity; inspect error rates in logs
- Fixes:
  - Rotate creds via SecretsManager; reapply mounts and restart pods
  - Follow backup-restore runbook to validate data health

8) Multi-tenancy isolation concerns
- Checks:
  - Ensure TenantIsolationManager policies enforced
  - Validate tenant headers and auth scopes
- Fixes:
  - Patch misconfig; add tests for tenant overrides
