#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv"

cd "$PROJECT_ROOT"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
    echo "Creating virtual environment at .venv"
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user \
    --name cs2207-ann \
    --display-name "Python (CS2207 ANN)"

echo
echo "Environment ready. Activate it with:"
echo "source .venv/bin/activate"
