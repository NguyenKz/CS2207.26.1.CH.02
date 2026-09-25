#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/gk/web/frontend"

cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run ./setup_venv.sh first."
  exit 1
fi

if [[ ! -f "$FRONTEND_DIR/package.json" ]]; then
  echo "Missing frontend package.json: $FRONTEND_DIR/package.json"
  exit 1
fi

source ".venv/bin/activate"

if [[ ! -x ".venv/bin/uvicorn" ]]; then
  echo "Missing uvicorn. Run ./setup_venv.sh or source .venv/bin/activate && python -m pip install -r requirements.txt."
  exit 1
fi

check_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Port $port is already in use. Stop the existing process or choose another port."
    exit 1
  fi
}

check_port 6788
check_port 5113

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm install --prefix "$FRONTEND_DIR"
fi

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "Backend:  http://localhost:6788"
echo "Frontend: http://localhost:5113"
echo "Press Ctrl+C to stop both servers."

uvicorn gk.web.backend.server:app --port 6788 &
BACKEND_PID=$!

npm --prefix "$FRONTEND_DIR" run dev -- --host 0.0.0.0 --port 5113 &
FRONTEND_PID=$!

wait "$BACKEND_PID" "$FRONTEND_PID"
