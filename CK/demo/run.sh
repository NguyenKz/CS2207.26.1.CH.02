#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5500}"

cd "$DEMO_DIR"

pick_python() {
  local candidates=()
  [[ -n "${PYTHON:-}" ]] && candidates+=("$PYTHON")
  [[ -x "$PROJECT_ROOT/.venv/bin/python" ]] && candidates+=("$PROJECT_ROOT/.venv/bin/python")
  candidates+=(python3 python)

  local candidate
  for candidate in "${candidates[@]}"; do
    if "$candidate" - <<'PY' >/dev/null 2>&1; then
import fastapi, networkx, numpy, sklearn, uvicorn  # noqa: F401
PY
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python)" || {
  echo "Missing deps. Need fastapi, uvicorn, scikit-learn, networkx, numpy."
  echo "Try: python3 -m pip install fastapi uvicorn scikit-learn networkx numpy"
  exit 1
}

check_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop the existing process or set BACKEND_PORT / FRONTEND_PORT."
    exit 1
  fi
}

check_port "$BACKEND_PORT"
check_port "$FRONTEND_PORT"

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "Using:    $PYTHON_BIN"
echo "Backend:  http://127.0.0.1:$BACKEND_PORT"
echo "Frontend: http://127.0.0.1:$FRONTEND_PORT/index.html"
echo "Canvas:   http://127.0.0.1:$FRONTEND_PORT/demo.html"
echo "Press Ctrl+C to stop both servers."

"$PYTHON_BIN" -m uvicorn backend.server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

"$PYTHON_BIN" -m http.server "$FRONTEND_PORT" --bind 127.0.0.1 &
FRONTEND_PID=$!

if command -v open >/dev/null 2>&1; then
  open "http://127.0.0.1:$FRONTEND_PORT/index.html" >/dev/null 2>&1 || true
fi

wait "$BACKEND_PID" "$FRONTEND_PID"
