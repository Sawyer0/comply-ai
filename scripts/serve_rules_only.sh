#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PID_FILE="/tmp/mapper_server.pid"
LOG_FILE="/tmp/mapper_server.log"
HOST="127.0.0.1"
PORT="8000"

ensure_venv() {
  if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
  fi
  # shellcheck source=/dev/null
  . "$VENV/bin/activate"
  python -m pip install -U pip setuptools wheel >/dev/null 2>&1 || true
  # Minimal lightweight deps (avoid heavy ML)
  python -m pip install fastapi uvicorn[standard] pydantic pydantic-settings pyyaml jsonschema prometheus-client structlog click httpx aiohttp >/dev/null 2>&1 || true
}

start() {
  ensure_venv

  export PYTHONPATH="$ROOT/src"
  export MAPPER_SERVING_MODE=rules_only
  export CUDA_VISIBLE_DEVICES=

  # Validate schema presence used by CLI serve
  if [ ! -f "$ROOT/.kiro/pillars-detectors/schema.json" ]; then
    echo "ERROR: Missing $ROOT/.kiro/pillars-detectors/schema.json" >&2
    exit 1
  fi

  rm -f "$LOG_FILE" "$PID_FILE" || true

  # Use uvicorn in factory mode with import string to support multiple workers
  nohup uvicorn --factory "llama_mapper.api.rules_only_factory:create_rules_only_app" \
    --host "$HOST" --port "$PORT" --workers 2 --loop uvloop --http httptools \
    --no-access-log >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  # Wait for health up to ~15s
  for i in $(seq 1 60); do
    if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
      echo "READY http://$HOST:$PORT"
      exit 0
    fi
    sleep 0.25
  done

  echo "FAILED to become healthy. Recent log:" >&2
  tail -n 200 "$LOG_FILE" 2>/dev/null || echo "(no log)" >&2
  exit 1
}

stop() {
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE" || true)
    if [ -n "$pid" ]; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" || true
  fi
  echo "STOPPED"
}

status() {
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "RUNNING http://$HOST:$PORT"
    exit 0
  fi
  if [ -f "$PID_FILE" ]; then
    pid=$(cat "$PID_FILE" || true)
    if [ -n "$pid" ] && ps -p "$pid" >/dev/null 2>&1; then
      echo "STARTING (pid=$pid)"
      exit 0
    fi
  fi
  echo "STOPPED"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "Usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
