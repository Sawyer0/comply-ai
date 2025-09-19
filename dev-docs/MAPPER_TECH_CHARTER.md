# Mapper Tech Charter (v1)

## 1. Overview
The Mapper service is the core component for normalizing raw detector outputs into a canonical taxonomy and generating compliance-ready evidence. It is designed to be:
- **Privacy-first**: minimize or avoid storage of raw detector inputs.
- **Deterministic**: all outputs must match a strict schema (`schema.json`).
- **Audit-ready**: normalized outputs can be exported into SOC 2, ISO 27001, and HIPAA evidence packs.

---

## 2. Core Lock-ins
- **Language**: Python (FastAPI + Hugging Face stack).
- **LLM Model**: Llama-3-8B-Instruct (fine-tuned with LoRA).
- **Detectors supported (v1)**: 
  - DeBERTa (toxicity)
  - Detoxify / HateBERT
  - Regex-based PII
  - OpenAI Moderation API
  - Llama Guard
- **Taxonomy**: Locked at `taxonomy.yaml` (version: 2025.09).

---

## 3. Data Contracts
- **Input**: Raw detector outputs (JSONL).  
- **Output**: Canonical JSON matching `schema.json`.  
- **Storage**:  
  - S3 immutable bucket (90-day hot, 1-year cold WORM storage).  
  - ClickHouse/Postgres for hot normalized data (90-day retention).

---

## 4. Evaluation Gates
- **Golden cases**: ≥100 per supported detector.  
- **Acceptance thresholds**:  
  - Schema-valid outputs ≥95%.  
  - Taxonomy F1 ≥90%.  
  - P95 latency ≤250 ms CPU / ≤120 ms GPU.  
- **CI/CD**: GitHub Actions block merges if thresholds not met.

---

## 5. Serving & Deployment
- **Serving runtime**: vLLM (GPU) or TGI (CPU/GPU).  
- **API layer**: FastAPI `/map` endpoint.  
- **Packaging**: Docker image + Helm chart.  
- **Scaling**: Start with 1–2 replicas, enable autoscaling later.  
- **Fallback**: If model confidence <0.6 → fall back to YAML rules mapping.

---

## 6. Security & Privacy
- **Data encryption**: AES256 with KMS (BYOK supported).  
- **Default policy**: Do not persist raw detector inputs.  
- **Secrets management**: Hashicorp Vault or AWS Secrets Manager.  
- **Privacy guarantee**: Logs store only metadata (tenant ID, detector type, taxonomy hit).

---

## 7. Framework Mapping
- **Initial frameworks supported (v1)**:  
  - SOC 2 CC7.2 (monitoring)  
  - ISO 27001 A.12.4.1 (logging & monitoring)  
  - HIPAA §164.308(a) (security safeguards)  
- **Mapping location**: `frameworks.yaml`.  
- Example:  
  ```yaml
  PII.Identifier.SSN: [SOC2_CC7.2, ISO27001_A.12.4.1]
  HARM.SPEECH.Toxicity: [SOC2_CC7.2]
  JAILBREAK.Attempt: [SOC2_CC7.2, HIPAA_164.308a]
  ```

---

## 8. Reporting
- **Supported formats**:  
  - PDF (auditor-facing, generated via WeasyPrint).  
  - CSV (engineer-facing, Pandas export).  
  - JSON (integration-facing).  
- **Report content**: coverage %, incidents, MTTR, control mapping.

---

## 9. Observability
- **Metrics collected**:  
  - Request count  
  - Schema-valid %  
  - Fallback %  
  - Latency P50/P95  
- **Logs**: Only metadata (never raw detector content).  
- **Stack**: Prometheus + Grafana or Datadog.

---

## 10. Versioning
- **Taxonomy**: `taxonomy@YYYY.MM`.  
- **Model**: `mapper-lora@vX.Y.Z`.  
- **Framework mapping**: `frameworks@vX.Y`.  
- **Audit reports**: must include version tags for all of the above.

---

## 11. Runbook Notes
- **Adding a new detector**: drop raw outputs in `data/raw/`, add mapping YAML in `detectors/`, run `prepare_dataset.py`, retrain LoRA.  
- **Rollback**: Canary toggle to disable Mapper LoRA, fallback to YAML-only mapping.  
- **Ops checks**: monitor schema-valid %, fallback %, latency.  
