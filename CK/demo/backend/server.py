"""Interactive fraud-detection demo using the paper's official samplers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import kneighbors_graph

DEMO_DIR = Path(__file__).resolve().parents[1]
VENDORED_REPO = DEMO_DIR / "vendor" / "simplical-smote"
sys.path.insert(0, str(VENDORED_REPO))

from ssmote import SMOTE, SimplicialSMOTE  # noqa: E402

DatasetId = Literal["fraud", "moons", "ring"]

DATASET_META: dict[str, dict[str, str]] = {
    "fraud": {
        "name": "Fraud amount × risk",
        "blurb": "Normal transactions cluster around the origin; fraud forms two clusters.",
        "howto": "Synthetic 2D: Normal around the origin (majority) plus two Normal clusters (minority). Not real transaction data.",
        "feature_x": "transaction amount",
        "feature_y": "merchant risk",
    },
    "moons": {
        "name": "Two moons",
        "blurb": "Two moon-shaped arcs; the minority class occupies one arc.",
        "howto": "Synthetic 2D: two arcs (angle ~Uniform[0,π], radius with Normal noise) — one majority arc and one minority arc.",
        "feature_x": "x",
        "feature_y": "y",
    },
    "ring": {
        "name": "Concentric ring",
        "blurb": "The majority class is in the center; the minority class is on the outer ring.",
        "howto": "Synthetic 2D: inner disk (r ∝ √U) = majority; outer ring (angle Uniform[0,2π] plus thickness noise) = minority.",
        "feature_x": "x",
        "feature_y": "y",
    },
}


app = FastAPI(title="Simplicial SMOTE Fraud Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    method: Literal["original", "smote", "simplicial"] = "original"
    dataset: DatasetId = "fraud"
    seed: int = Field(default=2207, ge=1, le=999999)
    k: int = Field(default=5, ge=3, le=11)
    p: int = Field(default=3, ge=3, le=7)
    budget: int = Field(default=1, ge=1, le=8)


def make_fraud_data(seed: int = 2207) -> dict[str, np.ndarray]:
    """Make deterministic, interpretable amount/risk features and a fixed test set."""
    random = np.random.default_rng(seed)
    normal_train = np.column_stack((
        random.normal(0.0, 0.72, 180),
        random.normal(0.0, 0.68, 180),
    ))
    fraud_train = np.column_stack((
        random.normal(1.15, 0.36, 24),
        random.normal(1.08, 0.32, 24),
    ))
    fraud_train = np.concatenate((
        fraud_train,
        np.column_stack((
            random.normal(-0.25, 0.2, 10),
            random.normal(1.45, 0.22, 10),
        )),
    ))
    normal_test = np.column_stack((
        random.normal(0.0, 0.72, 100),
        random.normal(0.0, 0.68, 100),
    ))
    fraud_test = np.concatenate((
        np.column_stack((
            random.normal(1.15, 0.36, 24),
            random.normal(1.08, 0.32, 24),
        )),
        np.column_stack((
            random.normal(-0.25, 0.2, 10),
            random.normal(1.45, 0.22, 10),
        )),
    ))
    return {
        "train_x": np.concatenate((normal_train, fraud_train)),
        "train_y": np.concatenate((np.zeros(len(normal_train)), np.ones(len(fraud_train)))),
        "test_x": np.concatenate((normal_test, fraud_test)),
        "test_y": np.concatenate((np.zeros(len(normal_test)), np.ones(len(fraud_test)))),
    }


def _moon_arc(random: np.random.Generator, n: int, shift: float, flip: bool) -> np.ndarray:
    angles = random.uniform(0.0, np.pi, n)
    radius = 1.0 + random.normal(0.0, 0.08, n)
    x = radius * np.cos(angles) + shift
    y = radius * np.sin(angles)
    if flip:
        x = 1.0 - x
        y = -y + 0.35
    y = y + random.normal(0.0, 0.06, n)
    return np.column_stack((x, y))


def make_moons_data(seed: int = 2207) -> dict[str, np.ndarray]:
    random = np.random.default_rng(seed)
    # Distinct counts from fraud/ring so landing n⁺/n⁻/m visibly change.
    majority_train = _moon_arc(random, 150, shift=0.0, flip=False)
    minority_train = _moon_arc(random, 50, shift=0.0, flip=True)
    majority_test = _moon_arc(random, 90, shift=0.0, flip=False)
    minority_test = _moon_arc(random, 40, shift=0.0, flip=True)
    return {
        "train_x": np.concatenate((majority_train, minority_train)),
        "train_y": np.concatenate((np.zeros(len(majority_train)), np.ones(len(minority_train)))),
        "test_x": np.concatenate((majority_test, minority_test)),
        "test_y": np.concatenate((np.zeros(len(majority_test)), np.ones(len(minority_test)))),
    }


def _disk(random: np.random.Generator, n: int, radius: float, thickness: float = 0.0) -> np.ndarray:
    if thickness <= 0:
        r = radius * np.sqrt(random.uniform(0.0, 1.0, n))
    else:
        r = radius + random.uniform(-thickness, thickness, n)
    angles = random.uniform(0.0, 2 * np.pi, n)
    return np.column_stack((r * np.cos(angles), r * np.sin(angles)))


def make_ring_data(seed: int = 2207) -> dict[str, np.ndarray]:
    random = np.random.default_rng(seed)
    # Distinct counts from fraud/moons so landing n⁺/n⁻/m visibly change.
    majority_train = _disk(random, 200, radius=0.55)
    minority_train = _disk(random, 25, radius=1.45, thickness=0.12)
    majority_test = _disk(random, 110, radius=0.55)
    minority_test = _disk(random, 30, radius=1.45, thickness=0.12)
    return {
        "train_x": np.concatenate((majority_train, minority_train)),
        "train_y": np.concatenate((np.zeros(len(majority_train)), np.ones(len(minority_train)))),
        "test_x": np.concatenate((majority_test, minority_test)),
        "test_y": np.concatenate((np.zeros(len(majority_test)), np.ones(len(minority_test)))),
    }


def load_dataset(dataset: DatasetId, seed: int = 2207) -> dict[str, np.ndarray]:
    if dataset == "fraud":
        return make_fraud_data(seed)
    if dataset == "moons":
        return make_moons_data(seed)
    if dataset == "ring":
        return make_ring_data(seed)
    raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")


def dataset_summary(dataset: DatasetId, data: dict[str, np.ndarray]) -> dict:
    meta = DATASET_META[dataset]
    train_y = data["train_y"]
    test_y = data["test_y"]
    n_minority = int((train_y == 1).sum())
    n_majority = int((train_y == 0).sum())
    return {
        "id": dataset,
        "name": meta["name"],
        "blurb": meta["blurb"],
        "howto": meta.get("howto", ""),
        "features": [meta["feature_x"], meta["feature_y"]],
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "n_minority": n_minority,
        "n_majority": n_majority,
        "imbalance_ratio": float(n_majority / max(n_minority, 1)),
    }


def graph_edges(points: np.ndarray, k: int) -> list[list[int]]:
    graph = kneighbors_graph(points, n_neighbors=k, mode="connectivity", include_self=False)
    matrix = ((graph + graph.T) > 0).toarray()
    return [[int(left), int(right)] for left in range(len(points)) for right in range(left + 1, len(points)) if matrix[left, right]]


def minority_graph(points: np.ndarray, k: int):
    graph = kneighbors_graph(points, n_neighbors=k, mode="connectivity", include_self=False)
    matrix = ((graph + graph.T) > 0).astype(int)
    return nx.from_numpy_array(matrix)


def clique_pool(points: np.ndarray, k: int) -> list[list[int]]:
    """Maximal cliques on the minority k-NN graph (= candidate cluster / simplex pool)."""
    return [sorted(int(node) for node in clique) for clique in nx.find_cliques(minority_graph(points, k))]


def attach_pool_choice(metadata: list[dict], pool_size: int) -> list[dict]:
    size = max(int(pool_size), 1)
    for index, item in enumerate(metadata):
        item["pool_size"] = int(pool_size)
        item["pool_index"] = index % size
    return metadata


def train_model(points: np.ndarray, labels: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(C=1.2, class_weight=None, max_iter=2000, random_state=seed)
    model.fit(points, labels)
    return model


def boundary_grid(model: LogisticRegression, points: np.ndarray) -> list[dict[str, float]]:
    pad = 0.45
    x_min, y_min = points.min(axis=0) - pad
    x_max, y_max = points.max(axis=0) + pad
    xs = np.linspace(float(x_min), float(x_max), 25)
    ys = np.linspace(float(y_min), float(y_max), 25)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    probabilities = model.predict_proba(grid)[:, 1]
    return [
        {"x": float(point[0]), "y": float(point[1]), "probability": float(probabilities[index])}
        for index, point in enumerate(grid)
    ]


def serialize_points(points: np.ndarray, labels: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {"x": float(point[0]), "y": float(point[1]), "label": int(labels[index])}
        for index, point in enumerate(points)
    ]


def animation_timeline(
    synthetic: list[dict],
    minority: np.ndarray,
) -> list[dict]:
    """Describe the explainable phases used by the single-canvas renderer."""
    timeline = []
    for index, item in enumerate(synthetic):
        parents = [int(parent) for parent in item["parents"]]
        parent_points = [minority[parent].tolist() for parent in parents]
        timeline.append(
            {
                "index": index,
                "parents": parents,
                "parent_points": [
                    {"x": float(point[0]), "y": float(point[1])}
                    for point in parent_points
                ],
                "weights": [float(weight) for weight in item["weights"]],
                "exp_draws": [float(value) for value in item.get("exp_draws", [])],
                "dimension": int(item["dimension"]),
                "pool_size": int(item.get("pool_size", 0)),
                "pool_index": int(item.get("pool_index", index)),
                "point": {
                    "x": float(item["point"][0]),
                    "y": float(item["point"][1]),
                },
                "phases": [
                    "select-parent",
                    "connect-neighborhood",
                    "show-weights",
                    "interpolate",
                    "commit-sample",
                ],
            }
        )
    return timeline


def metrics(model: LogisticRegression, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float | int]:
    predicted = model.predict(test_x)
    return {
        "recall": float(recall_score(test_y, predicted, zero_division=0)),
        "precision": float(precision_score(test_y, predicted, zero_division=0)),
        "f1": float(f1_score(test_y, predicted, zero_division=0)),
        "false_negatives": int(((test_y == 1) & (predicted == 0)).sum()),
        "false_positives": int(((test_y == 0) & (predicted == 1)).sum()),
    }


def run_sampler(request: RunRequest, data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    train_x, train_y = data["train_x"], data["train_y"]
    if request.method == "original":
        return train_x, train_y, []

    minority = train_x[train_y == 1]
    sampler = SMOTE(k=request.k, random_state=request.seed) if request.method == "smote" else SimplicialSMOTE(
        k=request.k,
        p=request.p,
        random_state=request.seed,
    )
    sampled_x, sampled_y = sampler.fit_resample(train_x, train_y)
    synthetic = sampled_x[len(train_x):]
    simplices = getattr(sampler, "simplices", {})
    if request.method == "smote" and hasattr(sampler, "sampled_simplices"):
        parent_rows = np.asarray(sampler.sampled_simplices)
        pool_size = len(getattr(sampler, "simplices", parent_rows))
    elif isinstance(simplices, dict):
        parent_rows = np.concatenate(list(simplices.values())) if simplices else np.empty((0, request.p), dtype=int)
        pool_size = len(clique_pool(minority, request.k))
    else:
        parent_rows = np.asarray(simplices)
        pool_size = len(parent_rows)
    # Animation λ: sample e~Exp(1) then normalize (Dir(1..1)), so the UI can show the arithmetic.
    # Train/metrics still use `synthetic` from the official sampler above.
    random = np.random.default_rng(request.seed)
    metadata = []
    for index, point in enumerate(synthetic):
        parents = [int(parent) for parent in (parent_rows[index % len(parent_rows)].tolist() if len(parent_rows) else [])]
        if not parents:
            metadata.append({
                "point": point.tolist(),
                "parents": [],
                "weights": [],
                "exp_draws": [],
                "dimension": 0,
            })
            continue
        exp_draws = random.exponential(1.0, size=len(parents))
        weights = exp_draws / exp_draws.sum()
        viz_point = minority[parents].T @ weights
        metadata.append({
            "point": viz_point.tolist(),
            "parents": parents,
            "weights": [float(weight) for weight in weights],
            "exp_draws": [float(value) for value in exp_draws],
            "dimension": max(len(parents) - 1, 0),
        })
    metadata = attach_pool_choice(metadata, pool_size if pool_size else len(parent_rows))
    return np.concatenate((train_x, synthetic)), np.concatenate((train_y, np.ones(len(synthetic)))), metadata


@app.get("/api/fraud/datasets")
def list_datasets() -> dict:
    return {
        "datasets": [
            {**dataset_summary(dataset_id, load_dataset(dataset_id)), "id": dataset_id}
            for dataset_id in DATASET_META
        ]
    }


@app.post("/api/fraud/run")
def run(request: RunRequest) -> dict:
    data = load_dataset(request.dataset, request.seed)
    summary = dataset_summary(request.dataset, data)
    sampled_x, sampled_y, synthetic = run_sampler(request, data)
    model = train_model(sampled_x, sampled_y, request.seed)
    minority = data["train_x"][data["train_y"] == 1]
    all_points = np.concatenate((data["train_x"], data["test_x"]))
    return {
        "method": request.method,
        "dataset": summary,
        "features": summary["features"],
        "train": serialize_points(sampled_x, sampled_y),
        "test": serialize_points(data["test_x"], data["test_y"]),
        "synthetic": synthetic,
        "minority": serialize_points(minority, np.ones(len(minority))),
        "timeline": animation_timeline(synthetic, minority),
        "edges": graph_edges(data["train_x"][data["train_y"] == 1], request.k),
        "boundary": boundary_grid(model, all_points),
        "metrics": metrics(model, data["test_x"], data["test_y"]),
        "seed": request.seed,
        "test_size": len(data["test_y"]),
    }


@app.get("/api/fraud/scenario")
def scenario(dataset: DatasetId = Query(default="fraud")) -> dict:
    data = load_dataset(dataset)
    summary = dataset_summary(dataset, data)
    return {
        **summary,
        "train": serialize_points(data["train_x"], data["train_y"]),
        "test": serialize_points(data["test_x"], data["test_y"]),
    }
