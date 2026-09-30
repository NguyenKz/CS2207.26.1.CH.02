"""Shared runtime configuration for the MNIST Predict pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = Path(__file__).with_name("artifacts")
VALID_PIXEL_SIZES = (8, 16, 24)
DEFAULT_PIXEL_SIZE = 8


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _load_pixel_size() -> int:
    dotenv_value = _read_dotenv(PROJECT_ROOT / ".env").get("ANN_DIGIT_SIZE")
    configured_value = os.environ.get("ANN_DIGIT_SIZE", dotenv_value)
    if configured_value in (None, ""):
        return DEFAULT_PIXEL_SIZE
    try:
        pixel_size = int(configured_value)
    except ValueError as error:
        raise ValueError(
            "ANN_DIGIT_SIZE must be one of: 8, 16, 24."
        ) from error
    if pixel_size not in VALID_PIXEL_SIZES:
        raise ValueError("ANN_DIGIT_SIZE must be one of: 8, 16, 24.")
    return pixel_size


PIXEL_SIZE = _load_pixel_size()
PIXEL_COUNT = PIXEL_SIZE * PIXEL_SIZE
SIZE_LABEL = f"{PIXEL_SIZE}x{PIXEL_SIZE}"
DATASET_DIR = ARTIFACT_DIR / "mnist_dataset" / SIZE_LABEL
DATASET_PATH = DATASET_DIR / "mnist_normalized.npz"
RAW_DATASET_PATH = DATASET_DIR / "mnist_raw.npz"
DATASET_META_PATH = DATASET_DIR / "dataset_meta.json"
SPLIT_PATH = DATASET_DIR / "split.json"
MODEL_CONFIG_PATH = DATASET_DIR / "mnist_model_config.json"
WEIGHTS_PATH = DATASET_DIR / "mnist_weights.json"
MODEL_ARTIFACT_PATH = ARTIFACT_DIR / f"digits_models_{SIZE_LABEL}.json"
LEGACY_MODEL_ARTIFACT_PATH = ARTIFACT_DIR / "digits_models.json"
# A fixed cap keeps the notebook practical on the augmented train set. This
# is not early stopping; MLPs run until convergence or this explicit cap.
MAX_ITER = 350
EARLY_STOPPING :bool= _read_dotenv(PROJECT_ROOT / ".env").get("EARLY_STOPPING") in ("True", "true", "1", "yes", "y")
