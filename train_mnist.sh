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

MODEL_IDS=(logistic compact_sigmoid compact_tanh compact_relu)
MODEL_MODULE="gk.web.backend.train_digit_model"

usage() {
  echo "Usage: $0 {logistic|compact_sigmoid|compact_tanh|compact_relu|all|merge}"
  echo
  echo "  model       Train one model and write its shard."
  echo "  all         Train all four models in parallel, then merge the artifact."
  echo "  merge       Merge completed shards without retraining."
}

is_model_id() {
  local candidate="$1"
  for model_id in "${MODEL_IDS[@]}"; do
    [[ "$candidate" == "$model_id" ]] && return 0
  done
  return 1
}

if [[ "$#" -ne 1 ]]; then
  usage >&2
  exit 2
fi

case "$1" in
  all)
    log_dir="gk/web/backend/artifacts/mnist_dataset/28x28/model_shards/logs"
    mkdir -p "$log_dir"
    pids=()
    for model_id in "${MODEL_IDS[@]}"; do
      echo "Starting $model_id"
      python -u -m "$MODEL_MODULE" "$model_id" >"$log_dir/$model_id.log" 2>&1 &
      pids+=("$!")
    done

    failed=0
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        echo "Training failed for ${MODEL_IDS[$index]}; see $log_dir/${MODEL_IDS[$index]}.log" >&2
        failed=1
      fi
    done
    [[ "$failed" -eq 0 ]] || exit 1
    python -u -m "$MODEL_MODULE" --merge
    ;;
  merge)
    python -u -m "$MODEL_MODULE" --merge
    ;;
  *)
    if ! is_model_id "$1"; then
      usage >&2
      exit 2
    fi
    python -u -m "$MODEL_MODULE" "$1"
    ;;
esac
