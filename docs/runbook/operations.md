# Operations Runbook

Owner: Platform/ML Ops
Last updated: 2025-09-19

Overview
- Purpose: Day-2 operational procedures for Llama Mapper, including rollbacks, kill-switch, and safe config changes.
- Related docs:
  - docs/runbook/migrations.md (taxonomy/framework/model migration + rollback)
  - docs/runbook/backup-restore.md (data backups, restore drills)
  - docs/runbook/alert-runbooks.md (per-alert investigations)
  - docs/runbook/troubleshooting.md (common issues)

1) Rollback procedures
A. LoRA adapter rollback (model)
- When to use: Quality degradation (schema-valid %, F1) or high error/latency after a new adapter release.
- Preconditions:
  - Identify target version (previous good) via versions registry.
  - Confirm you can safely redeploy (no schema/framework breaking changes).
- Steps:
  1. Inspect current versions
     - CLI: mapper versions show
  2. Select previous adapter version (e.g., mapper-lora@vX.Y.Z-1)
  3. Redeploy with previous version
     - If adapter is packaged in the image: update .Values.image.tag and helm upgrade
     - If adapter is external (checkpoint storage): update env/config to point to prior tag and helm upgrade
  4. Verify
     - Check metrics: schema-valid %, fallback %, p95 latency (should recover)
     - Run a subset of golden cases in staging if available
  5. Post-rollback
     - Create an incident note and link to the faulting release

B. Configuration rollback
- When to use: Bad rollout of YAML mappings (taxonomy/framework/detector maps) or service config.
- GitOps first: revert commit and redeploy via pipeline.
- Helm rollback alternative: helm rollback <release> <revision>
- Verify via canary/golden tests if possible.

2) Kill-switch procedures (rules-only mapping)
- Purpose: Force service to bypass the model and rely on rules-only mapping during incidents.
- Options:
  A) CLI (in place) — persistent via config file
     - mapper runtime show-mode
     - mapper runtime kill-switch on    # sets mode=rules_only
     - mapper runtime kill-switch off   # sets mode=hybrid
  B) Kubernetes (declarative via Helm)
     - Set env override to rules_only and redeploy:
       - helm upgrade <release> charts/llama-mapper \
         --set env[2].name=LLAMA_MAPPER_SERVING__MODE \
         --set env[2].value=rules_only
     - Note: adjust index if env array differs; prefer a values file for repeatability.
- Verification:
  - Observe immediate drop in model invocations (fallback % may increase); schema-valid % should remain ≥SLO if rules are sufficient.

3) Safe config changes
- Use PRs + CI quality gates (≥100 golden cases per detector; schema/F1/latency thresholds).
- Stage first with inlineConfigs or externalConfigMaps; promote once quality gates pass.
- Keep taxonomy/framework/detector versions in sync; record a VersionSnapshot.

4) Deployment guardrails
- Health probes: GET /health; HPA on CPU by default.
- Metrics: GET /metrics, ServiceMonitor optional via Helm values.
- Security: non-root container; API key auth (per-tenant scopes).

5) Verification checklist after change/rollback
- Metrics:
  - mapper_schema_valid_percentage ≥ 99.5%
  - mapper_fallback_percentage < 10%
  - p95 latency within profile target (0.25s CPU / 0.15s GPU)
  - 5xx error rate < 1%
- Logs: no raw inputs; privacy filter active.
- Run smoke tests: a few golden cases or conformance tests.

6) Escalation
- If critical alerts persist (5xx errors, sustained low schema-valid), escalate to on-call and initiate incident response per alert-runbooks.
