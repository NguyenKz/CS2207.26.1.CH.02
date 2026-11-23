#!/usr/bin/env bash
# Start the demo on one origin (FastAPI serves UI + /api) and expose it with ngrok.
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"

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

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok not found on PATH."
  echo "Install: https://ngrok.com/download  (then run: ngrok config add-authtoken <token>)"
  exit 1
fi

PYTHON_BIN="$(pick_python)" || {
  echo "Missing deps. Need fastapi, uvicorn, scikit-learn, networkx, numpy."
  echo "Try: python3 -m pip install fastapi uvicorn scikit-learn networkx numpy"
  exit 1
}

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$BACKEND_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $BACKEND_PORT is already in use. Stop the existing process or set BACKEND_PORT."
  exit 1
fi

NGROK_LOG="$(mktemp -t ssmote-ngrok.XXXXXX)"
BACKEND_PID=""
NGROK_PID=""

cleanup() {
  trap - INT TERM EXIT
  [[ -n "${NGROK_PID:-}" ]] && kill "$NGROK_PID" 2>/dev/null || true
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  rm -f "$NGROK_LOG"
}

trap cleanup INT TERM EXIT

echo "Using:   $PYTHON_BIN"
echo "Local:   http://127.0.0.1:$BACKEND_PORT/index.html"
echo "Starting uvicorn + ngrok…"

"$PYTHON_BIN" -m uvicorn backend.server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!

ngrok http "$BACKEND_PORT" --log=stdout --log-format=logfmt >"$NGROK_LOG" 2>&1 &
NGROK_PID=$!

PUBLIC_URL=""
for _ in $(seq 1 40); do
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "uvicorn exited early."
    exit 1
  fi
  if ! kill -0 "$NGROK_PID" 2>/dev/null; then
    echo "ngrok exited early. Last log:"
    tail -n 30 "$NGROK_LOG" || true
    exit 1
  fi
  PUBLIC_URL="$(
    "$PYTHON_BIN" - <<'PY' 2>/dev/null || true
import json, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1) as r:
        data = json.load(r)
    for t in data.get("tunnels", []):
        url = t.get("public_url") or ""
        if url.startswith("https://"):
            print(url)
            break
    else:
        for t in data.get("tunnels", []):
            url = t.get("public_url") or ""
            if url:
                print(url)
                break
except Exception:
    pass
PY
  )"
  if [[ -n "$PUBLIC_URL" ]]; then
    break
  fi
  sleep 0.25
done

if [[ -z "$PUBLIC_URL" ]]; then
  echo "Timed out waiting for ngrok public URL. Log:"
  tail -n 40 "$NGROK_LOG" || true
  exit 1
fi

echo
echo "Public:  $PUBLIC_URL/index.html"
echo "Canvas:  $PUBLIC_URL/demo.html"
echo "Press Ctrl+C to stop."
echo

if command -v open >/dev/null 2>&1; then
  open "$PUBLIC_URL/index.html" >/dev/null 2>&1 || true
fi

wait "$BACKEND_PID" "$NGROK_PID"
