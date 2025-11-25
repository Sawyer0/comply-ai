# WSL2 + Ubuntu developer workflow for Comply-AI

These steps assume the project is checked out in your home directory and properly mounted in WSL2 Ubuntu.

1. Create and activate a virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

2. Install dev dependencies

```bash
# Using extras (recommended)
pip install -e .[dev]

# Or minimal set for tests and scripts
pip install fastapi pydantic pydantic-settings pyyaml jsonschema prometheus-client \
  structlog click httpx aiohttp pytest pytest-asyncio uvicorn starlette
```

3. Run tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/path/to/test_file.py
```

4. Regenerate OpenAPI

```bash
python scripts/export_openapi.py --output docs/openapi.yaml
```

5. Contract knobs for local testing

- MAPPER_SERVING_MAPPER_TIMEOUT_MS=500 (default)
- MAPPER_SERVING_MAX_PAYLOAD_KB=64 (default)
- MAPPER_SERVING_REJECT_ON_RAW_CONTENT=true (default)

6. Health and metrics (when running server)

- GET /health
- GET /metrics
