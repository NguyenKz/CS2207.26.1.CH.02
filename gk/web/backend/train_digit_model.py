"""Train one MNIST model and merge per-model artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .digits_augmentation import build_augmented_variants, repeat_labels
from .digits_config import (
    DATASET_DIR,
    DATASET_PATH,
    MODEL_ARTIFACT_PATH,
    PIXEL_SIZE,
    RAW_DATASET_PATH,
)
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


warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pid=%(process)d] %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

MODEL_IDS = tuple(MODEL_CONFIGS)
SHARD_DIR = DATASET_DIR / "model_shards"


def _prepare_data() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset is missing: {DATASET_PATH}. "
            "Run mnist_dataset.ipynb first with the current ANN_DIGIT_SIZE."
        )

    logger.info("Loading normalized dataset: %s", DATASET_PATH)
    data = np.load(DATASET_PATH)
    raw_data = np.load(RAW_DATASET_PATH)
    features = np.asarray(data["X_train_pixels"], dtype=float).reshape(len(data["y_train"]), -1)
    labels = np.asarray(data["y_train"], dtype=int)
    test_features = np.asarray(data["X_test_pixels"], dtype=float).reshape(len(data["y_test"]), -1)
    test_labels = np.asarray(data["y_test"], dtype=int)
    if features.shape[1] != PIXEL_COUNT:
        raise ValueError(
            f"Prepared dataset has {features.shape[1]} features, expected {PIXEL_COUNT}."
        )

    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_SEED + 1,
    )
    logger.info(
        "Split: train=%d, validation=%d, test=%d",
        len(train_indices),
        len(validation_indices),
        len(test_labels),
    )
    original_training_features = features[train_indices]
    original_training_labels = labels[train_indices]
    if AUGMENT_TRAINING:
        logger.info("Building augmented variants (factor=%d)", AUGMENT_FACTOR)
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
        logger.info(
            "Augmentation complete: original=%d, variants=%d, fit=%d",
            len(original_training_features),
            len(augmented_features),
            len(training_features),
        )
    else:
        augmented_features = np.empty((0, PIXEL_COUNT), dtype=np.float32)
        training_features = original_training_features
        training_labels = original_training_labels

    logger.info("Fitting StandardScaler on %d samples", len(training_features))
    scaler = StandardScaler().fit(training_features)
    fit_features = scaler.transform(training_features)
    validation_features = scaler.transform(features[validation_indices])
    scaled_test_features = scaler.transform(test_features)
    search_train, search_labels, search_count = choose_search_subset(
        fit_features, training_labels
    )
    logger.info("Prepared data; logistic search subset=%d", search_count)
    return {
        "features": features,
        "labels": labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "original_training_features": original_training_features,
        "training_features": training_features,
        "training_labels": training_labels,
        "augmented_features": augmented_features,
        "scaler": scaler,
        "fit_features": fit_features,
        "validation_features": validation_features,
        "scaled_test_features": scaled_test_features,
        "search_train": search_train,
        "search_labels": search_labels,
        "search_count": search_count,
    }


def _tune_logistic(data: dict[str, Any]) -> tuple[float, float]:
    logger.info("Tuning Logistic Regression over %d C candidates", len(LOGISTIC_CANDIDATES))
    scores: list[tuple[float, float]] = []
    started = time.perf_counter()
    for candidate_index, C in enumerate(LOGISTIC_CANDIDATES, start=1):
        candidate = build_logistic(C)
        candidate.fit(data["search_train"], data["search_labels"])
        score = candidate.score(
            data["validation_features"], data["labels"][data["validation_indices"]]
        )
        scores.append((float(score), C))
        logger.info(
            "Logistic tuning %d/%d: C=%g validation=%.4f elapsed=%.1fs",
            candidate_index,
            len(LOGISTIC_CANDIDATES),
            C,
            score,
            time.perf_counter() - started,
        )
    scores.sort(key=lambda item: (-item[0], item[1]))
    logger.info("Best Logistic candidate: C=%g validation=%.4f", scores[0][1], scores[0][0])
    return scores[0]


def _train_model(
    model_id: str,
    data: dict[str, Any],
    logistic_tuning: tuple[float, float] | None,
) -> dict[str, Any]:
    config = MODEL_CONFIGS[model_id]
    started = time.perf_counter()
    if config["kind"] == "linear":
        if logistic_tuning is None:
            raise ValueError("Logistic tuning is required for the logistic model.")
        validation_accuracy, logistic_C = logistic_tuning
        model = build_logistic(
            logistic_C,
            config.get("solver", "lbfgs"),
            config.get("max_iter"),
        )
        model.fit(data["fit_features"], data["training_labels"])
        metadata = model_metadata(
            model_id,
            "Logistic Regression",
            "linear",
            model,
            architecture_sizes(config),
            [],
            model.score(data["scaled_test_features"], data["test_labels"]),
            validation_accuracy,
            None,
            "scikit-learn LogisticRegression",
            "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
        )
        logger.info(
            "Finished %s: test=%.4f validation=%.4f elapsed=%.1fs",
            model_id,
            metadata["test_accuracy"],
            metadata["validation_accuracy"],
            time.perf_counter() - started,
        )
        return metadata

    hidden = hidden_layer_sizes(config)
    activation = hidden_activation(config)
    assert activation is not None
    model = build_mlp(
        hidden,
        activation,
        config["learning_rate_init"],
        config["max_iter"],
        config.get("early_stopping", False),
        config.get("n_iter_no_change"),
        config.get("batch_size"),
        verbose=True,
    )
    model.fit(data["fit_features"], data["training_labels"])
    validation_accuracy = model.score(
        data["validation_features"], data["labels"][data["validation_indices"]]
    )
    hidden_name = activation if len(hidden) == 1 else f"{'×'.join(str(size) for size in hidden)} · {activation}"
    metadata = model_metadata(
        model_id,
        f"MLP · {hidden_name}",
        "ann",
        model,
        architecture_sizes(config),
        layer_activations(config),
        model.score(data["scaled_test_features"], data["test_labels"]),
        validation_accuracy,
        activation,
    )
    logger.info(
        "Finished %s: test=%.4f validation=%.4f iterations=%s elapsed=%.1fs",
        model_id,
        metadata["test_accuracy"],
        metadata["validation_accuracy"],
        getattr(model, "n_iter_", "n/a"),
        time.perf_counter() - started,
    )
    return metadata


def _artifact_common(data: dict[str, Any], include_test_samples: bool) -> dict[str, Any]:
    scaler: StandardScaler = data["scaler"]
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
            "sample_count": int(len(data["labels"]) + len(data["test_labels"])),
            "training_sample_count": int(len(data["labels"])),
            "test_sample_count": int(len(data["test_labels"])),
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
        "fit_indices": data["train_indices"].astype(int).tolist(),
        "validation_indices": data["validation_indices"].astype(int).tolist(),
        "test_indices": np.arange(len(data["test_labels"])).astype(int).tolist(),
        "test_labels": data["test_labels"].astype(int).tolist(),
        "augmentation": {
            "enabled": AUGMENT_TRAINING,
            "factor": AUGMENT_FACTOR,
            "source_count": int(len(data["original_training_features"])),
            "generated_count": int(len(data["training_features"])),
            "variant_count": int(len(data["augmented_features"])),
            "shift_pixels": AUGMENT_SHIFT_PIXELS,
            "stroke_variants": AUGMENT_STROKE_VARIANTS,
            "scale_range": list(AUGMENT_SCALE_RANGE),
            "rotate_degrees": AUGMENT_ROTATE_DEGREES,
            "sharpen": AUGMENT_SHARPEN,
            "applied_to": "fit split only; validation and test remain unchanged",
        },
    }
    if include_test_samples:
        artifact["test_samples"] = data["test_features"].astype(float).tolist()
    return artifact


def train_one(model_id: str) -> Path:
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_id}'. Choose one of: {', '.join(MODEL_IDS)}")

    logger.info("[%s] preparing dataset and scaler", model_id)
    data = _prepare_data()
    logistic_tuning = _tune_logistic(data) if model_id == "logistic" else None
    if logistic_tuning is not None:
        logger.info(
            "[%s] selected C=%g (validation=%.4f)",
            model_id,
            logistic_tuning[1],
            logistic_tuning[0],
        )
    logger.info("[%s] fitting model", model_id)
    model = _train_model(model_id, data, logistic_tuning)
    artifact = _artifact_common(data, include_test_samples=model_id == "logistic")
    artifact["models"] = [model]
    artifact["model_configs"] = {
        model_id: {
            "kind": MODEL_CONFIGS[model_id]["kind"],
            "architecture": architecture_sizes(MODEL_CONFIGS[model_id]),
            "activations": layer_activations(MODEL_CONFIGS[model_id]),
            "validation_accuracy": model["validation_accuracy"],
        }
    }
    artifact["baseline_tuning"] = None
    if logistic_tuning is not None:
        artifact["baseline_tuning"] = {
            "rounds": SEARCH_ROUNDS,
            "C_candidates": list(LOGISTIC_CANDIDATES),
            "selected_C": logistic_tuning[1],
            "validation_accuracy": logistic_tuning[0],
            "search_sample_count": int(data["search_count"]),
        }
    artifact["primary_ann_id"] = next(
        model_name
        for model_name, config in MODEL_CONFIGS.items()
        if config["kind"] == "ann"
    )
    artifact["strong_ann_id"] = [
        model_name for model_name, config in MODEL_CONFIGS.items() if config["kind"] == "ann"
    ][-1]
    output_path = SHARD_DIR / f"{model_id}.json"
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("[%s] wrote %s", model_id, output_path)
    logger.info("[%s] test_accuracy=%.4f", model_id, model["test_accuracy"])
    return output_path


def _forward_exported(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    for layer in model["layers"]:
        values = values @ np.asarray(layer["weights"], dtype=float) + np.asarray(
            layer["biases"], dtype=float
        )
        if layer["kind"] != "output":
            activation = layer["activation"]
            if activation == "sigmoid":
                values = 1.0 / (1.0 + np.exp(-np.clip(values, -500, 500)))
            elif activation == "tanh":
                values = np.tanh(values)
            elif activation == "relu":
                values = np.maximum(values, 0.0)
    return np.argmax(values, axis=1)


def merge_shards() -> Path:
    shards: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        path = SHARD_DIR / f"{model_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing shard for {model_id}: {path}. "
                "Train all models before merging."
            )
        shard = json.loads(path.read_text(encoding="utf-8"))
        if len(shard.get("models", [])) != 1 or shard["models"][0].get("id") != model_id:
            raise ValueError(f"Shard {path} does not contain exactly model {model_id}.")
        if shard.get("pixel_size") != PIXEL_SIZE:
            raise ValueError(f"Shard {path} was trained for another pixel size.")
        shards[model_id] = shard

    base = shards["logistic"]
    if "test_samples" not in base:
        raise ValueError("The logistic shard must contain test_samples.")
    models = [shards[model_id]["models"][0] for model_id in MODEL_IDS]
    artifact = dict(base)
    artifact["models"] = models
    artifact["model_configs"] = {
        model_id: {
            "kind": MODEL_CONFIGS[model_id]["kind"],
            "architecture": architecture_sizes(MODEL_CONFIGS[model_id]),
            "activations": layer_activations(MODEL_CONFIGS[model_id]),
            "validation_accuracy": shards[model_id]["models"][0]["validation_accuracy"],
        }
        for model_id in MODEL_IDS
    }
    artifact["baseline_tuning"] = base["baseline_tuning"]
    artifact["primary_ann_id"] = next(
        model_id for model_id, config in MODEL_CONFIGS.items() if config["kind"] == "ann"
    )
    artifact["strong_ann_id"] = [
        model_id for model_id, config in MODEL_CONFIGS.items() if config["kind"] == "ann"
    ][-1]

    test_features = np.asarray(artifact["test_samples"], dtype=float)
    mean = np.asarray(artifact["preprocessing"]["mean"], dtype=float)
    std = np.asarray(artifact["preprocessing"]["std"], dtype=float)
    normalized = (test_features - mean) / std
    baseline_predictions = _forward_exported(models[0], normalized)
    strong_predictions = _forward_exported(models[-1], normalized)
    artifact["demo_sample_index"] = 0
    for index, baseline_prediction, strong_prediction, target in zip(
        range(len(artifact["test_labels"])),
        baseline_predictions,
        strong_predictions,
        artifact["test_labels"],
    ):
        if baseline_prediction != target and strong_prediction == target:
            artifact["demo_sample_index"] = int(index)
            break

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("Wrote %s", MODEL_ARTIFACT_PATH)
    for model in models:
        logger.info(
            "%s: test_accuracy=%.4f, parameters=%d",
            model["id"],
            model["test_accuracy"],
            model["parameter_count"],
        )
    return MODEL_ARTIFACT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", nargs="?", choices=MODEL_IDS)
    parser.add_argument("--merge", action="store_true", help="Merge all model shards.")
    args = parser.parse_args()
    if args.merge:
        merge_shards()
        return
    if args.model_id is None:
        parser.error(f"provide a model id ({', '.join(MODEL_IDS)}) or --merge")
    train_one(args.model_id)


if __name__ == "__main__":
    main()
