"""Train and export the offline models used by the Predict tab."""

from __future__ import annotations

import json
import warnings
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
    PIXEL_COUNT,
    RANDOM_SEED,
    SEARCH_ROUNDS,
    build_logistic,
    build_mlp,
    choose_search_subset,
    model_metadata,
)
from .model_config import (
    AUGMENT_FACTOR,
    AUGMENT_ROTATE_DEGREES,
    AUGMENT_SCALE_RANGE,
    AUGMENT_SHARPEN,
    AUGMENT_SHIFT_PIXELS,
    AUGMENT_STROKE_VARIANTS,
    AUGMENT_TRAINING,
    architecture_sizes,
    hidden_activation,
    hidden_layer_sizes,
    layer_activations,
)


OUTPUT_PATH = MODEL_ARTIFACT_PATH

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _ann_display_name(model_id: str, config: dict[str, Any]) -> str:
    hidden = hidden_layer_sizes(config)
    activation = hidden_activation(config)
    if len(hidden) == 1:
        return f"MLP · {hidden[0]} · {activation}"
    return f"MLP · {'×'.join(str(size) for size in hidden)} · {activation}"


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
            raw_data["X_train_images"][train_indices],
            factor=AUGMENT_FACTOR,
            shift_pixels=AUGMENT_SHIFT_PIXELS,
            stroke_variants=AUGMENT_STROKE_VARIANTS,
            scale_range=AUGMENT_SCALE_RANGE,
            rotate_degrees=AUGMENT_ROTATE_DEGREES,
            sharpen=AUGMENT_SHARPEN,
            seed=RANDOM_SEED,
        )
        training_features = np.vstack((original_training_features, augmented_features))
        training_labels = repeat_labels(original_training_labels, AUGMENT_FACTOR)
    else:
        augmented_features = np.empty((0, PIXEL_COUNT), dtype=np.float32)
        training_features = original_training_features
        training_labels = original_training_labels
    scaler = StandardScaler().fit(training_features)
    fit_features = scaler.transform(training_features)
    validation_features = scaler.transform(features[validation_indices])
    scaled_test_features = scaler.transform(test_features)
    search_train, search_labels, search_count = choose_search_subset(fit_features, training_labels)

    logistic_scores: list[tuple[float, float]] = []
    for C in LOGISTIC_CANDIDATES:
        candidate = build_logistic(C)
        candidate.fit(search_train, search_labels)
        logistic_scores.append(
            (float(candidate.score(validation_features, labels[validation_indices])), C)
        )
    logistic_scores.sort(key=lambda item: (-item[0], item[1]))
    logistic_validation_score, logistic_C = logistic_scores[0]

    models: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    ann_validation: dict[str, float] = {}

    for model_id, model_config in MODEL_CONFIGS.items():
        if model_config["kind"] == "linear":
            model = build_logistic(
                logistic_C,
                model_config.get("solver", "lbfgs"),
                model_config.get("max_iter"),
            )
            model.fit(fit_features, training_labels)
            trained[model_id] = model
            models.append(
                model_metadata(
                    model_id,
                    "Logistic Regression",
                    "linear",
                    model,
                    architecture_sizes(model_config),
                    [],
                    model.score(scaled_test_features, test_labels),
                    logistic_validation_score,
                    None,
                    "scikit-learn LogisticRegression",
                    "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
                )
            )
            continue

        hidden = hidden_layer_sizes(model_config)
        activation = hidden_activation(model_config)
        assert activation is not None
        model = build_mlp(
            hidden,
            activation,
            model_config["learning_rate_init"],
            model_config["max_iter"],
            model_config.get("early_stopping", False),
            model_config.get("n_iter_no_change"),
            model_config.get("batch_size"),
        )
        model.fit(fit_features, training_labels)
        trained[model_id] = model
        validation_accuracy = float(model.score(validation_features, labels[validation_indices]))
        ann_validation[model_id] = validation_accuracy
        models.append(
            model_metadata(
                model_id,
                _ann_display_name(model_id, model_config),
                "ann",
                model,
                architecture_sizes(model_config),
                layer_activations(model_config),
                model.score(scaled_test_features, test_labels),
                validation_accuracy,
                activation,
            )
        )

    logistic = trained["logistic"]
    primary_ann_id = next(
        model_id for model_id, config in MODEL_CONFIGS.items() if config["kind"] == "ann"
    )
    # Prefer the last ANN as the "strong" model for demo sample picking.
    strong_ann_id = [model_id for model_id, config in MODEL_CONFIGS.items() if config["kind"] == "ann"][-1]
    baseline_test_predictions = logistic.predict(scaled_test_features)
    strong_test_predictions = trained[strong_ann_id].predict(scaled_test_features)
    demo_sample_index = 0
    for index, baseline_prediction, strong_prediction, target in zip(
        np.arange(len(test_labels)),
        baseline_test_predictions,
        strong_test_predictions,
        test_labels,
    ):
        if baseline_prediction != target and strong_prediction == target:
            demo_sample_index = int(index)
            break

    artifact = {
        "version": 4,
        "random_seed": RANDOM_SEED,
        "pixel_size": PIXEL_SIZE,
        "dataset": {
            "name": "MNIST handwritten digits",
            "description": (
                f"Real handwritten digit images from MNIST, normalized to "
                f"{PIXEL_SIZE}x{PIXEL_SIZE} grayscale pixels for this demo."
            ),
            "sample_count": int(len(labels) + len(test_labels)),
            "training_sample_count": int(len(labels)),
            "test_sample_count": int(len(test_labels)),
            "input_shape": [PIXEL_SIZE, PIXEL_SIZE],
            "feature_count": PIXEL_COUNT,
            "class_count": CLASS_COUNT,
            "pixel_min": 0,
            "pixel_max": 1,
            "original_input_shape": [28, 28],
            "source_url": "https://yann.lecun.com/exdb/mnist/",
        },
        "preprocessing": {
            "name": f"CropSquareResize{PIXEL_SIZE} + StandardScaler",
            "feature_transform": (
                f"denoise at 50% of max ink, keep largest component, crop foreground, "
                f"resize to square, area-average resize to {PIXEL_SIZE}x{PIXEL_SIZE}, "
                f"scale intensity to 0..1"
            ),
            "mean": scaler.mean_.astype(float).tolist(),
            "std": scaler.scale_.astype(float).tolist(),
        },
        "fit_indices": train_indices.astype(int).tolist(),
        "validation_indices": validation_indices.astype(int).tolist(),
        "test_indices": np.arange(len(test_labels)).astype(int).tolist(),
        "test_samples": test_features.astype(float).tolist(),
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
            "rotate_degrees": AUGMENT_ROTATE_DEGREES,
            "sharpen": AUGMENT_SHARPEN,
            "applied_to": "fit split only; validation and test remain unchanged",
        },
        "baseline_tuning": {
            "rounds": SEARCH_ROUNDS,
            "C_candidates": list(LOGISTIC_CANDIDATES),
            "selected_C": logistic_C,
            "validation_accuracy": logistic_validation_score,
            "search_sample_count": int(search_count),
        },
        "models": models,
        "model_configs": {
            model_id: {
                "kind": config["kind"],
                "architecture": architecture_sizes(config),
                "activations": layer_activations(config),
                "validation_accuracy": (
                    logistic_validation_score
                    if config["kind"] == "linear"
                    else ann_validation.get(model_id)
                ),
            }
            for model_id, config in MODEL_CONFIGS.items()
        },
        "primary_ann_id": primary_ann_id,
        "strong_ann_id": strong_ann_id,
    }

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for model in models:
        print(
            f"{model['id']}: test_accuracy={model['test_accuracy']:.4f}, "
            f"parameters={model['parameter_count']}"
        )


if __name__ == "__main__":
    main()
