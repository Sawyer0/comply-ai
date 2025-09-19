# syntax=docker/dockerfile:1

# -------- Base image with system dependencies --------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_WORKERS=1

# Install minimal OS deps needed for runtime and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Builder image to install dependencies --------
FROM base AS builder

# Build tools sometimes required for scientific deps; keep only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Leverage Docker layer caching by copying only files that impact deps first
COPY pyproject.toml README.md /app/

# Install project (and dependencies) into a staging dir
# You can toggle extras via --build-arg EXTRAS=dev
ARG EXTRAS=
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ -n "$EXTRAS" ]; then \
      pip install --prefix=/install ".[${EXTRAS}]" ; \
    else \
      pip install --prefix=/install "." ; \
    fi

# Copy source last to keep cache hits for deps
COPY src/ /app/src/

# Re-install the local package to ensure latest code is captured
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -e "."

# -------- Final runtime image --------
FROM base AS runtime

# Create non-root user
RUN useradd -u 10001 -ms /bin/bash appuser

# Copy installed site-packages and entrypoints from builder
COPY --from=builder /install /usr/local
COPY --from=builder /app /app

# Switch to non-root
USER appuser

# Expose default port
EXPOSE 8000

# Healthcheck hits the FastAPI /health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default command uses the application module entry point
# Use --env LLAMA_MAPPER_SERVING__BACKEND to toggle vLLM vs TGI
CMD ["python", "-m", "src.llama_mapper.api.main", "--host", "0.0.0.0", "--port", "8000"]
