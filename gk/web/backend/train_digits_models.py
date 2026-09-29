"""Train and export the four offline models used by the Predict tab."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).with_name("artifacts") / "digits_models.json"
PIXEL_COUNT = 64
CLASS_COUNT = 10
TUNED_CANDIDATES = ((8,), (16,), (8, 8), (16, 8))

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parameter_count(layers: list[dict[str, Any]]) -> int:
    return sum(
        len(layer["weights"]) * len(layer["weights"][0]) + len(layer["biases"])
        for layer in layers
    )


def export_layers(model: Any, hidden_activation: str | None) -> list[dict[str, Any]]:
    if not hasattr(model, "coefs_"):
        return [
            {
                "name": "Output",
                "kind": "output",
                "activation": "softmax",
                "weights": model.coef_.T.astype(float).tolist(),
                "biases": model.intercept_.astype(float).tolist(),
            }
        ]

    exported: list[dict[str, Any]] = []
    for index, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        is_output = index == len(model.coefs_) - 1
        exported.append(
            {
                "name": "Output" if is_output else f"Hidden {index + 1}",
                "kind": "output" if is_output else "hidden",
                "activation": "softmax" if is_output else hidden_activation,
                "weights": weights.astype(float).tolist(),
                "biases": biases.astype(float).tolist(),
            }
        )
    return exported


def model_metadata(
    model_id: str,
    name: str,
    kind: str,
    model: Any,
    architecture: list[int],
    activations: list[str],
    test_accuracy: float,
    validation_accuracy: float | None,
    hidden_activation: str | None,
    source: str | None = None,
    source_url: str | None = None,
) -> dict[str, Any]:
    layers = export_layers(model, hidden_activation)
    metadata = {
        "id": model_id,
        "name": name,
        "kind": kind,
        "architecture": architecture,
        "activations": activations,
        "parameter_count": parameter_count(layers),
        "test_accuracy": float(test_accuracy),
        "validation_accuracy": (
            None if validation_accuracy is None else float(validation_accuracy)
        ),
        "layers": layers,
    }
    if source is not None:
        metadata["source"] = source
    if source_url is not None:
        metadata["source_url"] = source_url
    return metadata


def build_mlp(hidden_layers: tuple[int, ...], activation: str = "tanh") -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver="lbfgs",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )


def main() -> None:
    dataset = load_digits()
    features = dataset.data.astype(float)
    labels = dataset.target.astype(int)
    indices = np.arange(len(labels))
    fit_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    train_indices, validation_indices = train_test_split(
        fit_indices,
        test_size=0.2,
        stratify=labels[fit_indices],
        random_state=RANDOM_SEED + 1,
    )

    tuning_scaler = StandardScaler().fit(features[train_indices])
    tuning_train = tuning_scaler.transform(features[train_indices])
    tuning_validation = tuning_scaler.transform(features[validation_indices])

    tuned_scores: list[tuple[float, int, tuple[int, ...]]] = []
    for hidden_layers in TUNED_CANDIDATES:
        candidate = build_mlp(hidden_layers, activation="relu")
        candidate.fit(tuning_train, labels[train_indices])
        score = candidate.score(tuning_validation, labels[validation_indices])
        candidate_layers = export_layers(candidate, "relu")
        tuned_scores.append((float(score), parameter_count(candidate_layers), hidden_layers))
    tuned_scores.sort(key=lambda item: (-item[0], item[1], len(item[2])))
    tuned_score, _, tuned_architecture = tuned_scores[0]

    scaler = StandardScaler().fit(features[fit_indices])
    fit_features = scaler.transform(features[fit_indices])
    test_features = scaler.transform(features[test_indices])

    models: list[dict[str, Any]] = []

    logistic = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    logistic.fit(fit_features, labels[fit_indices])
    models.append(
        model_metadata(
            "logistic",
            "Logistic Regression",
            "linear",
            logistic,
            [PIXEL_COUNT, CLASS_COUNT],
            [],
            logistic.score(test_features, labels[test_indices]),
            None,
            None,
            "scikit-learn LogisticRegression",
            "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        )
    )

    small_one_layer = build_mlp((4,), activation="tanh")
    small_one_layer.fit(fit_features, labels[fit_indices])
    models.append(
        model_metadata(
            "mlp-4-one-layer",
            "MLP · 4 neurons · 1 layer",
            "ann",
            small_one_layer,
            [PIXEL_COUNT, 4, CLASS_COUNT],
            ["tanh", "softmax"],
            small_one_layer.score(test_features, labels[test_indices]),
            None,
            "tanh",
        )
    )

    small_two_layers = build_mlp((4, 4), activation="tanh")
    small_two_layers.fit(fit_features, labels[fit_indices])
    models.append(
        model_metadata(
            "mlp-4-two-layer",
            "MLP · 4 neurons · 2 layers",
            "ann",
            small_two_layers,
            [PIXEL_COUNT, 4, 4, CLASS_COUNT],
            ["tanh", "tanh", "softmax"],
            small_two_layers.score(test_features, labels[test_indices]),
            None,
            "tanh",
        )
    )

    tuned = build_mlp(tuned_architecture, activation="relu")
    tuned.fit(fit_features, labels[fit_indices])
    baseline_test_predictions = logistic.predict(test_features)
    tuned_test_predictions = tuned.predict(test_features)
    demo_sample_index = int(test_indices[0])
    for index, baseline_prediction, tuned_prediction, target in zip(
        test_indices,
        baseline_test_predictions,
        tuned_test_predictions,
        labels[test_indices],
    ):
        if baseline_prediction != target and tuned_prediction == target:
            demo_sample_index = int(index)
            break
    models.append(
        model_metadata(
            "compact-tuned",
            "Compact tuned MLP",
            "ann",
            tuned,
            [PIXEL_COUNT, *tuned_architecture, CLASS_COUNT],
            ["relu"] * len(tuned_architecture) + ["softmax"],
            tuned.score(test_features, labels[test_indices]),
            tuned_score,
            "relu",
        )
    )

    artifact = {
        "version": 1,
        "random_seed": RANDOM_SEED,
        "dataset": {
            "name": "sklearn.datasets.load_digits",
            "description": "Handwritten digits represented as 8x8 grayscale pixels.",
            "sample_count": int(len(features)),
            "input_shape": [8, 8],
            "feature_count": PIXEL_COUNT,
            "class_count": CLASS_COUNT,
            "pixel_min": 0,
            "pixel_max": 16,
            "source_url": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
        },
        "preprocessing": {
            "name": "StandardScaler",
            "mean": scaler.mean_.astype(float).tolist(),
            "std": scaler.scale_.astype(float).tolist(),
        },
        "fit_indices": fit_indices.astype(int).tolist(),
        "test_indices": test_indices.astype(int).tolist(),
        "demo_sample_index": demo_sample_index,
        "tuning": {
            "candidates": [list(candidate) for candidate in TUNED_CANDIDATES],
            "selected": list(tuned_architecture),
            "validation_accuracy": tuned_score,
        },
        "models": models,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for model in models:
        print(
            f"{model['name']}: test_accuracy={model['test_accuracy']:.4f}, "
            f"parameters={model['parameter_count']}"
        )


if __name__ == "__main__":
    main()
