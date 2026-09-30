"""Train and export the four offline models used by the Predict tab."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .digits_config import (
    DATASET_DIR,
    DATASET_PATH,
    MODEL_ARTIFACT_PATH,
    PIXEL_SIZE,
    RAW_DATASET_PATH,
)
from .digits_augmentation import build_augmented_variants, repeat_labels
from .digits_models import (
    CLASS_COUNT,
    LOGISTIC_CANDIDATES,
    MODEL_CONFIGS,
    MLP_SEARCH_CONFIGS,
    PIXEL_COUNT,
    RANDOM_SEED,
    SEARCH_MAX_ITER,
    SEARCH_ROUNDS,
    build_logistic,
    build_mlp,
    choose_search_subset,
    export_layers,
    model_metadata,
    parameter_count,
)
from .model_config import (
    AUGMENT_FACTOR,
    AUGMENT_SCALE_RANGE,
    AUGMENT_SHIFT_PIXELS,
    AUGMENT_STROKE_VARIANTS,
    AUGMENT_TRAINING,
)


OUTPUT_PATH = MODEL_ARTIFACT_PATH

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {DATASET_PATH}. "
            "Run mnist_dataset.ipynb first with the current ANN_DIGIT_SIZE."
        )
    data = np.load(DATASET_PATH)
    raw_data = np.load(RAW_DATASET_PATH)
    features = np.asarray(data["X_train_pixels"], dtype=float).reshape(len(data["y_train"]), -1)
    labels = np.asarray(data["y_train"], dtype=int)
    test_features = np.asarray(data["X_test_pixels"], dtype=float).reshape(len(data["y_test"]), -1)
    test_labels = np.asarray(data["y_test"], dtype=int)
    if features.shape[1] != PIXEL_SIZE * PIXEL_SIZE:
        raise ValueError(
            f"Prepared dataset has {features.shape[1]} features, expected {PIXEL_SIZE * PIXEL_SIZE}."
        )
    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_SEED + 1,
    )

    original_training_features = features[train_indices]
    original_training_labels = labels[train_indices]
    if AUGMENT_TRAINING:
        augmented_features = build_augmented_variants(
            raw_data['X_train_images'][train_indices],
            factor=AUGMENT_FACTOR,
            shift_pixels=AUGMENT_SHIFT_PIXELS,
            stroke_variants=AUGMENT_STROKE_VARIANTS,
            scale_range=AUGMENT_SCALE_RANGE,
            seed=RANDOM_SEED,
        )
        training_features = np.vstack((original_training_features, augmented_features))
        training_labels = repeat_labels(original_training_labels, AUGMENT_FACTOR)
    else:
        augmented_features = np.empty((0, PIXEL_COUNT), dtype=np.float32)
        training_features = original_training_features
        training_labels = original_training_labels
    tuning_scaler = StandardScaler().fit(training_features)
    tuning_train = tuning_scaler.transform(training_features)
    tuning_validation = tuning_scaler.transform(features[validation_indices])
    search_train, search_labels, search_count = choose_search_subset(tuning_train, training_labels)

    logistic_scores: list[tuple[float, float]] = []
    for C in LOGISTIC_CANDIDATES:
        candidate = build_logistic(C)
        candidate.fit(search_train, search_labels)
        logistic_scores.append((float(candidate.score(tuning_validation, labels[validation_indices])), C))
    logistic_scores.sort(key=lambda item: (-item[0], item[1]))
    logistic_validation_score, logistic_C = logistic_scores[0]

    tuned_scores: list[tuple[float, int, tuple[int, ...], str, float]] = []
    for hidden_layers, activation, learning_rate in MLP_SEARCH_CONFIGS:
        candidate = build_mlp(hidden_layers, activation, learning_rate, max_iter=SEARCH_MAX_ITER)
        candidate.fit(search_train, search_labels)
        score = candidate.score(tuning_validation, labels[validation_indices])
        candidate_layers = export_layers(candidate, activation)
        tuned_scores.append(
            (float(score), parameter_count(candidate_layers), hidden_layers, activation, learning_rate)
        )
    tuned_scores.sort(key=lambda item: (-item[0], item[1], len(item[2]), item[4]))
    tuned_score, _, tuned_architecture, tuned_activation, tuned_learning_rate = tuned_scores[0]

    scaler = tuning_scaler
    fit_features = tuning_train
    fit_labels = training_labels
    scaled_test_features = scaler.transform(test_features)

    models: list[dict[str, Any]] = []

    logistic = build_logistic(logistic_C)
    logistic.fit(fit_features, fit_labels)
    models.append(
        model_metadata(
            "logistic",
            "Logistic Regression",
            "linear",
            logistic,
            [PIXEL_COUNT, CLASS_COUNT],
            [],
            logistic.score(scaled_test_features, test_labels),
            logistic_validation_score,
            None,
            "scikit-learn LogisticRegression",
            "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        )
    )

    one_layer_config = MODEL_CONFIGS["mlp_4_one_layer"]
    one_layer_hidden = tuple(one_layer_config["hidden_layer_sizes"])
    small_one_layer = build_mlp(
        one_layer_config["hidden_layer_sizes"],
        one_layer_config["activation"],
        one_layer_config["learning_rate_init"],
        one_layer_config["max_iter"],
    )
    small_one_layer.fit(fit_features, fit_labels)
    models.append(
        model_metadata(
            "mlp-4-one-layer",
            f"MLP · {one_layer_hidden[0]} neurons · 1 layer",
            "ann",
            small_one_layer,
            [PIXEL_COUNT, *one_layer_hidden, CLASS_COUNT],
            [one_layer_config["activation"], "softmax"],
            small_one_layer.score(scaled_test_features, test_labels),
            None,
            "tanh",
        )
    )

    two_layer_config = MODEL_CONFIGS["mlp_4_two_layers"]
    two_layer_hidden = tuple(two_layer_config["hidden_layer_sizes"])
    small_two_layers = build_mlp(
        two_layer_config["hidden_layer_sizes"],
        two_layer_config["activation"],
        two_layer_config["learning_rate_init"],
        two_layer_config["max_iter"],
    )
    small_two_layers.fit(fit_features, fit_labels)
    models.append(
        model_metadata(
            "mlp-4-two-layer",
            f"MLP · {two_layer_hidden[0]} neurons · 2 layers",
            "ann",
            small_two_layers,
            [PIXEL_COUNT, *two_layer_hidden, CLASS_COUNT],
            [two_layer_config["activation"]] * len(two_layer_hidden) + ["softmax"],
            small_two_layers.score(scaled_test_features, test_labels),
            None,
            "tanh",
        )
    )

    tuned = build_mlp(tuned_architecture, tuned_activation, tuned_learning_rate)
    tuned.fit(fit_features, fit_labels)
    baseline_test_predictions = logistic.predict(scaled_test_features)
    tuned_test_predictions = tuned.predict(scaled_test_features)
    demo_sample_index = 0
    for index, baseline_prediction, tuned_prediction, target in zip(
        np.arange(len(test_labels)),
        baseline_test_predictions,
        tuned_test_predictions,
        test_labels,
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
            [tuned_activation] * len(tuned_architecture) + ["softmax"],
            tuned.score(scaled_test_features, test_labels),
            tuned_score,
            tuned_activation,
        )
    )

    artifact = {
        "version": 4,
        "random_seed": RANDOM_SEED,
        "pixel_size": PIXEL_SIZE,
        "dataset": {
            "name": "MNIST handwritten digits",
            "description": f"Real handwritten digit images from MNIST, normalized to {PIXEL_SIZE}x{PIXEL_SIZE} grayscale pixels for this demo.",
            "sample_count": int(len(labels) + len(test_labels)),
            "training_sample_count": int(len(labels)),
            "test_sample_count": int(len(test_labels)),
            "input_shape": [PIXEL_SIZE, PIXEL_SIZE],
            "feature_count": PIXEL_COUNT,
            "class_count": CLASS_COUNT,
            "pixel_min": 0,
            "pixel_max": 16,
            "original_input_shape": [28, 28],
            "source_url": "https://yann.lecun.com/exdb/mnist/",
        },
        "preprocessing": {
            "name": f"CropSquareResize{PIXEL_SIZE} + StandardScaler",
            "feature_transform": f"denoise at 50% of max ink, keep largest component, crop foreground, resize to square, area-average resize to {PIXEL_SIZE}x{PIXEL_SIZE}, scale intensity to 0..16",
            "mean": scaler.mean_.astype(float).tolist(),
            "std": scaler.scale_.astype(float).tolist(),
        },
        "fit_indices": train_indices.astype(int).tolist(),
        "validation_indices": validation_indices.astype(int).tolist(),
        "test_indices": np.arange(len(test_labels)).astype(int).tolist(),
        "test_samples": np.rint(test_features).astype(int).tolist(),
        "test_labels": test_labels.astype(int).tolist(),
        "demo_sample_index": demo_sample_index,
        "augmentation": {
            "enabled": AUGMENT_TRAINING,
            "factor": AUGMENT_FACTOR,
            "source_count": int(len(original_training_features)),
            "generated_count": int(len(training_features)),
            "variant_count": int(len(augmented_features)),
            "shift_pixels": AUGMENT_SHIFT_PIXELS,
            "stroke_variants": AUGMENT_STROKE_VARIANTS,
            "scale_range": list(AUGMENT_SCALE_RANGE),
            "applied_to": "fit split only; validation and test remain unchanged",
        },
        "baseline_tuning": {
            "rounds": SEARCH_ROUNDS,
            "C_candidates": list(LOGISTIC_CANDIDATES),
            "selected_C": logistic_C,
            "validation_accuracy": logistic_validation_score,
        },
        "tuning": {
            "rounds": len(MLP_SEARCH_CONFIGS),
            "search_sample_count": int(search_count),
            "solver": "adam",
            "learning_rate_init": 0.001,
            "search_max_iter": SEARCH_MAX_ITER,
            "final_max_iter": 350,
            "early_stopping": False,
            "candidates": [
                {
                    "architecture": list(architecture),
                    "activation": activation,
                    "learning_rate_init": learning_rate,
                }
                for architecture, activation, learning_rate in MLP_SEARCH_CONFIGS
            ],
            "selected": list(tuned_architecture),
            "selected_activation": tuned_activation,
            "selected_learning_rate_init": tuned_learning_rate,
            "validation_accuracy": tuned_score,
        },
        "models": models,
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
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
