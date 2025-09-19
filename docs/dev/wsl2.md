# WSL2 + Ubuntu developer workflow for Comply-AI

These steps assume the project is checked out at C:\\Users\\Dawan\\comply-ai on Windows and accessible under /mnt/c/Users/Dawan/comply-ai in WSL2 Ubuntu.

1) Create and activate a virtualenv
- wsl bash -lc "cd /mnt/c/Users/Dawan/comply-ai && python3 -m venv .venv && source .venv/bin/activate && python -m pip install -U pip"

2) Install dev dependencies
- If using extras:
  - wsl bash -lc "cd /mnt/c/Users/Dawan/comply-ai && source .venv/bin/activate && pip install -e .[dev]"
- Or minimal set for tests and scripts:
  - wsl bash -lc "cd /mnt/c/Users/Dawan/comply-ai && source .venv/bin/activate && pip install fastapi pydantic pydantic-settings pyyaml jsonschema prometheus-client structlog click httpx aiohttp pytest pytest-asyncio uvicorn starlette"

3) Run tests
- Unit + integration subset:
  - wsl bash -lc "cd /mnt/c/Users/Dawan/comply-ai && source .venv/bin/activate && export PYTHONPATH=/mnt/c/Users/Dawan/comply-ai/src:$PYTHONPATH && python -m pytest -q"

4) Regenerate OpenAPI
- wsl bash -lc "cd /mnt/c/Users/Dawan/comply-ai && source .venv/bin/activate && export PYTHONPATH=/mnt/c/Users/Dawan/comply-ai/src:$PYTHONPATH && python scripts/export_openapi.py --output docs/openapi.yaml"

5) Contract knobs for local testing
- MAPPER_SERVING_MAPPER_TIMEOUT_MS=500 (default)
- MAPPER_SERVING_MAX_PAYLOAD_KB=64 (default)
- MAPPER_SERVING_REJECT_ON_RAW_CONTENT=true (default)

6) Health and metrics (when running server)
- GET /health
- GET /metrics
