# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Core dev commands

- Install (dev mode):
  ```bash
  pip install -e ".[dev]"
  ```
- Lint checks (match CI):
  ```bash
  flake8 src/ tests/
  black --check --diff src/ tests/
  isort --check-only --diff src/ tests/
  mypy src/llama_mapper/
  ```
- Auto-format locally (optional):
  ```bash
  black src/ tests/
  isort src/ tests/
  ```
- Run tests with coverage (match CI):
  ```bash
  pytest tests/ -v --cov=src/llama_mapper --cov-report=term-missing --cov-report=xml
  ```
- Run a single test (examples):
  ```bash
  # One test function
  pytest tests/unit/test_config_manager.py::test_default_config
  # By keyword expression
  pytest -k "quality_gates and not integration" -v
  # One file or folder
  pytest tests/integration/test_api_service.py -v
  ```
- Build wheel/sdist (when needed):
  ```bash
  python -m pip install build
  python -m build
  ```

## Running the API locally

- Preferred: run the FastAPI app factory module directly (CLI `serve` is a stub):
  ```bash
  python -m src.llama_mapper.api.main --reload --host 0.0.0.0 --port 8000
  ```
  Endpoints:
  - GET /health
  - POST /map (single)
  - POST /map/batch
  - GET /metrics (Prometheus exposition)

## Quality gates and CI parity

- Generate comprehensive golden cases used by CI:
  ```bash
  python scripts/generate_golden_cases.py
  ```
- Quality coverage check (fails on insufficient coverage):
  ```bash
  python -m llama_mapper.cli quality check-coverage --golden-cases tests/golden_test_cases_comprehensive.json
  ```
- Full quality validation and report (mirrors CI job):
  ```bash
  python -m llama_mapper.cli quality validate \
    --golden-cases tests/golden_test_cases_comprehensive.json \
    --output quality-report.json \
    --fail-on-error
  ```

## CLI tasks

- Show current YAML config and validate:
  ```bash
  mapper show-config
  mapper validate-config
  ```
- Bootstrap a config file (writes defaults):
  ```bash
  mapper init-config -o config.yaml
  ```
  Notes:
  - `mapper serve` and `mapper train` are placeholders; use the API module command above for serving.

## Configuration

Two complementary systems exist:
- YAML + env overrides (ConfigManager):
  - Defaults are created at `config.yaml` on first run.
  - Selected keys can be overridden via env vars (e.g., `MAPPER_MODEL_NAME`, `MAPPER_CONFIDENCE_THRESHOLD`, `MAPPER_SERVING_PORT`).
- Pydantic Settings with nested env using `.env` (Settings):
  - Uses prefix `LLAMA_MAPPER_` with double-underscore for nesting, see `.env.example`.
  - Examples:
    ```bash
    # .env or environment
    LLAMA_MAPPER_MODEL__TEMPERATURE=0.2
    LLAMA_MAPPER_CONFIDENCE__DEFAULT_THRESHOLD=0.7
    LLAMA_MAPPER_SERVING__PORT=8080
    ```

External data dependencies expected at runtime/tests (not included in the repo):
- `pillars-detectors/taxonomy.yaml`
- `pillars-detectors/frameworks.yaml`
- `pillars-detectors/schema.json`
- `pillars-detectors/*.yaml` for detector mappings

## Architecture overview (big picture)

- API layer (FastAPI):
  - Factory creates app with dependencies: model server, JSON schema validator, rule-based fallback, metrics collector, and config manager.
  - Request flow (POST /map): Model infers → JSON schema validated → confidence threshold checked → fallback to rules if invalid/low-confidence; all steps emit Prometheus metrics.
- Serving layer:
  - Abstract `ModelServer` with two implementations:
    - `VLLMModelServer` (GPU/in-process) and `TGIModelServer` (HTTP inference).
  - `GenerationConfig` controls decoding; prompts are instruction-tuned for Llama-3.
- Validation and fallback:
  - `JSONValidator` loads `pillars-detectors/schema.json`, performs schema + custom validations, and can parse to typed response objects.
  - `FallbackMapper` loads detector YAMLs from `pillars-detectors/`, supports exact/case-insensitive/partial matches; emits usage metrics.
- Configuration:
  - `ConfigManager` (YAML + env overrides) and `Settings` (Pydantic, .env) coexist. Confidence thresholds and serving/backend are controlled here.
- Monitoring:
  - `MetricsCollector` exposes counters/gauges/histograms and computes schema-valid %, fallback %, and p95 latency; `/metrics` endpoint returns exposition.
  - `QualityGateValidator` executes golden cases, computes F1/latency/validation rates; CLI wraps these checks for local/CI use.
- Storage and privacy:
  - `StorageManager` (S3 WORM + Postgres/ClickHouse hot storage, optional KMS/FERNET encryption).
  - `TenantIsolationManager` enforces scoped IDs and row-level isolation helpers; `PrivacyLogger` logs metadata-only events (no raw inputs).
- Training and data:
  - `LoRATrainer` (HF Transformers + PEFT) with `ModelLoader`; `TrainingDataGenerator` and `SyntheticDataGenerator` build instruction/JSON targets aligned to the schema.
- Reporting:
  - `AuditTrailManager` and `ReportGenerator` map taxonomy labels to compliance frameworks (SOC2/ISO27001/HIPAA) and emit PDF/CSV/JSON reports.

## CI expectations (from .github/workflows/quality-gates.yml)

- Lint/type: flake8, black --check, isort --check-only, mypy.
- Tests + coverage: pytest with XML + term reports; Codecov upload.
- Quality: generate golden cases → coverage check → `quality validate` produces `quality-report.json` and comments on PRs.
- Security: Bandit scan and TruffleHog secret scan.

## Notes for Warp

- Prefer running the API via `python -m src.llama_mapper.api.main` rather than `mapper serve` (stub).
- Many features require the external `pillars-detectors` data; set `LLAMA_MAPPER_*` paths (see `.env.example`) or place files accordingly.
- Python 3.11+ is required by pyproject.
