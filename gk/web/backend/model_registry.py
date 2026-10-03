"""Registry for versioned MNIST model runs and the active demo model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .digits_config import MODEL_ARTIFACT_PATH, PIXEL_SIZE, PROJECT_ROOT


BACKEND_DIR = Path(__file__).resolve().parent
MODEL_REGISTRY_PATH = BACKEND_DIR / "model_registry.json"
MODEL_RUNS_DIR = BACKEND_DIR / "model_runs"


def load_registry() -> dict[str, Any]:
    if not MODEL_REGISTRY_PATH.exists():
        return {
            "version": 1,
            "active_run_id": None,
            "primary_model_id": None,
            "runs": [],
        }
    registry = json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    if not isinstance(registry.get("runs"), list):
        raise ValueError("Model registry must contain a runs list.")
    return registry


def _run_entry(registry: dict[str, Any], run_id: str) -> dict[str, Any]:
    for run in registry["runs"]:
        if run.get("id") == run_id:
            return run
    raise KeyError(f"Unknown model run: {run_id}")


def resolve_artifact_path(registry: dict[str, Any] | None = None) -> Path:
    registry = load_registry() if registry is None else registry
    run_id = registry.get("active_run_id")
    if not run_id:
        return MODEL_ARTIFACT_PATH
    artifact = _run_entry(registry, run_id).get("artifact")
    if not artifact:
        return MODEL_ARTIFACT_PATH
    return PROJECT_ROOT / artifact


def active_primary_model_id(registry: dict[str, Any] | None = None) -> str | None:
    registry = load_registry() if registry is None else registry
    return registry.get("primary_model_id")


def relative_project_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def run_directory(run_id: str) -> Path:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
        raise ValueError("run_id must be a simple directory name.")
    return MODEL_RUNS_DIR / run_id


def save_registry(registry: dict[str, Any]) -> None:
    MODEL_REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def register_run(run: dict[str, Any], primary_model_id: str) -> dict[str, Any]:
    registry = load_registry()
    registry["version"] = 1
    registry["active_run_id"] = run["id"]
    registry["primary_model_id"] = primary_model_id
    registry["runs"] = [
        existing for existing in registry["runs"] if existing.get("id") != run["id"]
    ]
    registry["runs"].append(run)
    save_registry(registry)
    return registry


def select_run(run_id: str, primary_model_id: str) -> dict[str, Any]:
    registry = load_registry()
    run = _run_entry(registry, run_id)
    model_ids = {model.get("id") for model in run.get("models", [])}
    if primary_model_id not in model_ids:
        raise ValueError(f"Model {primary_model_id} is not present in run {run_id}.")
    registry["active_run_id"] = run_id
    registry["primary_model_id"] = primary_model_id
    save_registry(registry)
    return registry


def list_runs() -> list[dict[str, Any]]:
    return load_registry().get("runs", [])


def current_dataset_metadata() -> dict[str, int]:
    return {"pixel_size": PIXEL_SIZE, "feature_count": PIXEL_SIZE * PIXEL_SIZE}
