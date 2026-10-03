#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run ./setup_venv.sh first." >&2
  exit 1
fi

source ".venv/bin/activate"

# One process per model: keep BLAS/OpenMP from oversubscribing the CPU.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODEL_MODULE="gk.web.backend.train_digit_model"
DEFAULT_CONFIG="gk/web/backend/model_configs/mnist_28x28.json"

usage() {
  echo "Usage: $0 all [--run-id ID] [--config PATH]"
  echo "       $0 MODEL_ID --run-id ID [--config PATH]"
  echo "       $0 merge --run-id ID [--config PATH]"
  echo "       $0 list"
  echo "       $0 use RUN_ID MODEL_ID"
  echo
  echo "  all       Train every model in a new versioned run in parallel."
  echo "  MODEL_ID  Train one model shard for an existing/new run."
  echo "  merge     Merge completed shards and register the run."
  echo "  list      Show saved runs and validation/test metrics."
  echo "  use       Select the active run and primary demo model."
}

if [[ "$#" -lt 1 ]]; then
  usage >&2
  exit 2
fi

ACTION="$1"
shift
CONFIG="$DEFAULT_CONFIG"
RUN_ID=""
POSITIONAL=()

if [[ "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      [[ "$#" -ge 2 ]] || { echo "--config needs a path" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --run-id)
      [[ "$#" -ge 2 ]] || { echo "--run-id needs an id" >&2; exit 2; }
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ "$ACTION" == "list" ]]; then
  python -u -m "$MODEL_MODULE" list
  exit 0
fi

if [[ "$ACTION" == "use" ]]; then
  if [[ "${#POSITIONAL[@]}" -ne 2 ]]; then
    usage >&2
    exit 2
  fi
  python -u -m "$MODEL_MODULE" use "${POSITIONAL[0]}" "${POSITIONAL[1]}"
  exit 0
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing model config: $CONFIG" >&2
  exit 1
fi

model_list="$(python -u -m "$MODEL_MODULE" --list-models --config "$CONFIG")"
MODEL_IDS=()
while IFS= read -r model_id; do
  [[ -n "$model_id" ]] && MODEL_IDS+=("$model_id")
done <<< "$model_list"

is_model_id() {
  local candidate="$1"
  for model_id in "${MODEL_IDS[@]}"; do
    [[ "$candidate" == "$model_id" ]] && return 0
  done
  return 1
}

if [[ "$ACTION" == "all" ]]; then
  [[ "${#POSITIONAL[@]}" -eq 0 ]] || { usage >&2; exit 2; }
  if [[ -z "$RUN_ID" ]]; then
    RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
  fi
  log_dir="gk/web/backend/model_runs/$RUN_ID/logs"
  mkdir -p "$log_dir"
  pids=()
  cleanup() {
    trap - INT TERM
    for pid in "${pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
  }
  trap cleanup INT TERM
  for model_id in "${MODEL_IDS[@]}"; do
    echo "Starting $model_id (run=$RUN_ID)"
    python -u -m "$MODEL_MODULE" "$model_id" \
      --config "$CONFIG" --run-id "$RUN_ID" \
      >"$log_dir/$model_id.log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "Training failed for ${MODEL_IDS[$index]}; see $log_dir/${MODEL_IDS[$index]}.log" >&2
      failed=1
    fi
  done
  trap - INT TERM
  [[ "$failed" -eq 0 ]] || exit 1
  python -u -m "$MODEL_MODULE" merge --config "$CONFIG" --run-id "$RUN_ID"
  exit 0
fi

if [[ "$ACTION" == "merge" ]]; then
  [[ "${#POSITIONAL[@]}" -eq 0 && -n "$RUN_ID" ]] || { usage >&2; exit 2; }
  python -u -m "$MODEL_MODULE" merge --config "$CONFIG" --run-id "$RUN_ID"
  exit 0
fi

if ! is_model_id "$ACTION" || [[ "${#POSITIONAL[@]}" -ne 0 || -z "$RUN_ID" ]]; then
  usage >&2
  exit 2
fi
python -u -m "$MODEL_MODULE" "$ACTION" --config "$CONFIG" --run-id "$RUN_ID"
