# Analyst LLM Spec (v1)

## Purpose
Produce concise, auditable explanations and suggested remediations from *structured metrics* — not raw prompts/responses — and (optionally) emit machine-actionable policy diffs (OPA/CEL) to fix coverage/threshold issues.

## Scope (MVP)
- Coverage gap explanations (per endpoint/model/tenant, per time window).
- Incident/anomaly summaries (spikes, detector 5xx, drift).
- Threshold/quorum tuning suggestions (based on false-positive bands and human overrides).
- Optional: OPA policy diffs to implement the remediation.

## Model Choice
- Start zero-shot with **Phi-3 Mini (3.8B)**
- If drift/quality issues appear, fine-tune a small LoRA (300–800 examples).

## Inputs (structured)
Provided by the analytics jobs (ClickHouse queries) for a [period, tenant, app, route]:
- required_detectors: [str]
- observed_coverage: {detector: pct_0_1}
- required_coverage: {detector: target_pct_0_1}
- detector_errors: {detector: {5xx: int, time_buckets:[...]}} 
- high_sev_hits: [{taxonomy, count, p95_score}]
- false_positive_bands: [{detector, score_min, score_max, fp_rate}]
- policy_bundle: str (e.g., "riskpack-1.4.0")
- env: "dev|stage|prod"

## Outputs (schema described in analyst_schema.json)
- reason: <= 20 words (why the issue occurred)
- remediation: <= 20 words (what to change)
- opa_diff: Rego snippet or empty string
- confidence: 0..1
- evidence_refs: [string ids of metrics used]
- notes: short machine note

## Guardrails
- Must reference only provided metrics. No made-up numbers.
- Deterministic style (temperature 0–0.2). 
- Always validate against JSON Schema; drop to template fallback on failure.
- Redact any potential PII in notes (should be none).

## Evaluation
- Golden set: ≥150 examples covering coverage, anomaly, tuning suggestions.
- Acceptance: 
  - JSON schema valid ≥ 98%
  - Human rubric ≥ 4/5 on correctness/clarity
  - OPA snippet compiles (when present)
- Drift monitors: rolling sample eval weekly; alert if rubric < 3.7/5.

## Deployment
- Serve as /analyze (FastAPI), separate from /map.
- Same Docker/Helm pattern; shared secrets/metrics.
- Canary 10%; kill-switch to fallback templates (no LLM).

## Roadmap
- v1: zero-shot + templates.
- v1.2: add “residual risk” roll-up narration (exec summary).
- v1.3: add LoRA on common failure cases.