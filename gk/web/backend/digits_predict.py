"""Offline-trained digits models and the NumPy forward pass for Predict."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .digits_config import (
    LEGACY_MODEL_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    PIXEL_COUNT,
    PIXEL_SIZE,
)

ARTIFACT_PATH = (
    MODEL_ARTIFACT_PATH
    if MODEL_ARTIFACT_PATH.exists()
    else LEGACY_MODEL_ARTIFACT_PATH
    if PIXEL_SIZE == 8
    else MODEL_ARTIFACT_PATH
)
CLASS_COUNT = 10
DRAWING_SIZE = 128
DRAWING_COUNT = DRAWING_SIZE * DRAWING_SIZE


class DigitsArtifactError(RuntimeError):
    """Raised when the offline model artifact is missing or malformed."""


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


def _display_softmax(logits: np.ndarray, target_top: float = 0.90) -> np.ndarray:
    """Softmax for Predict UI bars/confidence.

    Large MLPs often leave a huge logit gap, so true softmax is numerically 1.0
    and the UI prints 100.0%. Soften only then; argmax stays identical.
    """
    logits = np.asarray(logits, dtype=float).reshape(-1)
    true_probabilities = _softmax(logits)
    if logits.size < 2 or float(true_probabilities.max()) < 0.99:
        return true_probabilities
    margin = float(np.sort(logits)[-1] - np.sort(logits)[-2])
    target_logit = float(np.log(target_top / (1.0 - target_top)))
    temperature = max(1.0, margin / target_logit)
    return _softmax(logits / temperature)


def _apply_activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(values)
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "sigmoid":
        clipped = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-clipped))
    raise DigitsArtifactError(f"Unsupported activation in artifact: {name}")


def _as_matrix(values: Any, shape: tuple[int, int], label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape:
        raise DigitsArtifactError(
            f"Invalid {label} shape: expected {shape}, received {matrix.shape}"
        )
    return matrix


def _as_vector(values: Any, size: int, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (size,):
        raise DigitsArtifactError(
            f"Invalid {label} shape: expected {(size,)}, received {vector.shape}"
        )
    return vector


@lru_cache(maxsize=1)
def load_digits_artifact() -> dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise DigitsArtifactError(
            f"Digits model artifact for {PIXEL_SIZE}x{PIXEL_SIZE} is missing. "
            "Run mnist_dataset.ipynb and mnist_train.ipynb with the current "
            "ANN_DIGIT_SIZE, then restart the backend."
        )
    try:
        artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DigitsArtifactError(f"Could not read digits artifact: {error}") from error

    if artifact.get("version") not in (3, 4):
        raise DigitsArtifactError("Unsupported digits artifact version.")
    if artifact.get("version") != 4 and PIXEL_SIZE != 8:
        raise DigitsArtifactError("Legacy digits artifacts are supported only for 8x8.")
    if len(artifact.get("models", [])) != 4:
        raise DigitsArtifactError("Digits artifact must contain exactly four models.")
    dataset = artifact.get("dataset", {})
    if dataset.get("input_shape") != [PIXEL_SIZE, PIXEL_SIZE]:
        raise DigitsArtifactError(
            f"Artifact input shape does not match configured {PIXEL_SIZE}x{PIXEL_SIZE}. "
            "Regenerate the dataset and model artifact with ANN_DIGIT_SIZE."
        )
    if dataset.get("feature_count") != PIXEL_COUNT:
        raise DigitsArtifactError(
            f"Artifact feature count does not match configured {PIXEL_COUNT} features."
        )
    preprocessing = artifact.get("preprocessing", {})
    _as_vector(preprocessing.get("mean"), PIXEL_COUNT, "preprocessing mean")
    standard_deviation = _as_vector(
        preprocessing.get("std"), PIXEL_COUNT, "preprocessing std"
    )
    if np.any(standard_deviation <= 0):
        raise DigitsArtifactError("Preprocessing standard deviations must be positive.")
    samples = np.asarray(artifact.get("test_samples"), dtype=float)
    labels = np.asarray(artifact.get("test_labels"), dtype=int)
    if samples.ndim != 2 or samples.shape[1] != PIXEL_COUNT:
        raise DigitsArtifactError(
            f"MNIST test samples must have shape (n, {PIXEL_COUNT})."
        )
    if labels.shape != (samples.shape[0],):
        raise DigitsArtifactError("MNIST test labels do not match test samples.")

    for model in artifact["models"]:
        if not model.get("id") or not model.get("layers"):
            raise DigitsArtifactError("Every digits model needs an id and layers.")
        previous_size = PIXEL_COUNT
        for layer_index, layer in enumerate(model["layers"]):
            biases = np.asarray(layer.get("biases"), dtype=float)
            if biases.ndim != 1:
                raise DigitsArtifactError(
                    f"Model {model['id']} layer {layer_index} has invalid biases."
                )
            _as_matrix(
                layer.get("weights"),
                (previous_size, biases.size),
                f"model {model['id']} layer {layer_index} weights",
            )
            previous_size = biases.size
        if previous_size != CLASS_COUNT:
            raise DigitsArtifactError(
                f"Model {model['id']} must end with {CLASS_COUNT} outputs."
            )
    return artifact


def public_model_metadata(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: model[key]
            for key in (
                "id",
                "name",
                "kind",
                "architecture",
                "activations",
                "parameter_count",
                "test_accuracy",
                "validation_accuracy",
                "source",
                "source_url",
            )
            if key in model
        }
        for model in artifact["models"]
    ]


def get_test_indices(artifact: dict[str, Any]) -> list[int]:
    indices = artifact.get("test_indices")
    if not isinstance(indices, list) or not indices:
        raise DigitsArtifactError("Digits artifact has no test sample indices.")
    return [int(index) for index in indices]


def serialize_sample(artifact: dict[str, Any], index: int) -> dict[str, Any]:
    samples = np.asarray(artifact.get("test_samples"), dtype=float)
    labels = np.asarray(artifact.get("test_labels"), dtype=int)
    if index < 0 or index >= len(samples):
        raise IndexError("Sample index is outside the digits dataset.")
    pixels = samples[index].reshape(PIXEL_SIZE, PIXEL_SIZE).astype(int).tolist()
    return {
        "index": index,
        "pixels": pixels,
        "label": int(labels[index]),
        "is_test_sample": index in set(get_test_indices(artifact)),
    }


def _largest_component(image: np.ndarray, threshold: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = image >= threshold
    visited = np.zeros(mask.shape, dtype=bool)
    best_pixels: list[tuple[int, int]] = []
    best_bounds = (0, 0, 0, 0)

    for start_y, start_x in zip(*np.where(mask)):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        min_y = max_y = int(start_y)
        min_x = max_x = int(start_x)

        while stack:
            current_y, current_x = stack.pop()
            component.append((current_y, current_x))
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_y == 0 and offset_x == 0:
                        continue
                    next_y = current_y + offset_y
                    next_x = current_x + offset_x
                    if (
                        0 <= next_y < mask.shape[0]
                        and 0 <= next_x < mask.shape[1]
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        stack.append((next_y, next_x))

        if len(component) > len(best_pixels):
            best_pixels = component
            best_bounds = (min_x, min_y, max_x + 1, max_y + 1)

    if not best_pixels:
        raise ValueError("Draw a digit before predicting.")

    cleaned = np.zeros_like(image)
    for pixel_y, pixel_x in best_pixels:
        cleaned[pixel_y, pixel_x] = image[pixel_y, pixel_x]
    return cleaned, best_bounds


def _resize_area_average(
    image: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Resize by averaging the source area covered by each target pixel.

    A point-sampled resize can miss a thin stroke completely when a 128x128
    drawing is reduced to the configured N×N grid. Area averaging preserves the ink coverage, so
    a thick hand-drawn line becomes a continuous grayscale digit instead of a
    few isolated dark cells.
    """
    source_height, source_width = image.shape
    y_scale = source_height / target_height
    x_scale = source_width / target_width

    def overlap_weights(source_size: int, target_size: int) -> np.ndarray:
        source_positions = np.arange(source_size, dtype=float)[None, :]
        target_starts = (np.arange(target_size, dtype=float) * source_size / target_size)[:, None]
        target_ends = ((np.arange(target_size, dtype=float) + 1) * source_size / target_size)[:, None]
        return np.clip(
            np.minimum(target_ends, source_positions + 1)
            - np.maximum(target_starts, source_positions),
            0.0,
            None,
        )

    y_weights = overlap_weights(source_height, target_height)
    x_weights = overlap_weights(source_width, target_width)
    return (y_weights @ np.asarray(image, dtype=float) @ x_weights.T) / (y_scale * x_scale)


def preprocess_dataset_pixels(pixels: np.ndarray) -> np.ndarray:
    """Normalize one grayscale source image with the same geometry as freehand input."""
    values = np.asarray(pixels, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("The source image is empty.")
    if values.ndim != 2:
        side = int(np.sqrt(values.size))
        if side * side != values.size:
            raise ValueError("The source image must be a square grayscale image.")
        values = values.reshape(side, side)
    normalized, _ = _normalize_source_image(values)
    return normalized.reshape(-1)


def _center_in_square(cropped: np.ndarray) -> np.ndarray:
    """Keep a cropped digit's aspect ratio while centering it in a square canvas."""
    height, width = cropped.shape
    square_size = max(height, width)
    square = np.zeros((square_size, square_size), dtype=float)
    top = (square_size - height) // 2
    left = (square_size - width) // 2
    square[top:top + height, left:left + width] = cropped
    return square


def _normalize_source_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Binarize, denoise, crop, center in a square, then resize to N×N."""
    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not np.isfinite(image).all() or np.any(image < 0):
        raise ValueError("The source image must be a finite non-negative grayscale matrix.")
    maximum = float(image.max())
    if maximum <= 0:
        raise ValueError("The source image does not contain a visible digit.")
    threshold = max(1.0, maximum * 0.5)
    binary = np.where(image >= threshold, maximum, 0.0)
    cleaned, (left, top, right, bottom) = _largest_component(binary, threshold)
    cropped = cleaned[top:bottom, left:right]
    content_height, content_width = cropped.shape
    square = _center_in_square(cropped)
    resized = _resize_area_average(square, PIXEL_SIZE, PIXEL_SIZE)
    resized_maximum = float(resized.max())
    if resized_maximum <= 0:
        raise ValueError("The source image does not contain a visible digit.")
    normalized = np.rint(np.clip(resized * (16.0 / resized_maximum), 0.0, 16.0)).astype(float)
    info = {
        "threshold": threshold,
        "bounding_box": {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
        },
        "cropped_size": [content_height, content_width],
        "square_size": int(square.shape[0]),
        "normalized_pixels": normalized.astype(int).tolist(),
    }
    return normalized, info


def preprocess_drawing(drawing: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(drawing, dtype=float).reshape(-1)
    if values.size != DRAWING_COUNT:
        raise ValueError(f"Expected {DRAWING_COUNT} drawing values.")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 16):
        raise ValueError("Drawing values must be finite and between 0 and 16.")

    image = values.reshape(DRAWING_SIZE, DRAWING_SIZE)
    normalized, info = _normalize_source_image(image)
    metadata = {
        "source_size": [DRAWING_SIZE, DRAWING_SIZE],
        **info,
    }
    return normalized.reshape(-1).astype(float), metadata


def _forward_model(
    model: dict[str, Any],
    normalized_features: np.ndarray,
) -> dict[str, Any]:
    current = normalized_features
    trace_layers: list[dict[str, Any]] = []
    predicted_class: int | None = None
    display_probabilities: np.ndarray | None = None
    for layer_index, layer in enumerate(model["layers"]):
        weights = np.asarray(layer["weights"], dtype=float)
        biases = np.asarray(layer["biases"], dtype=float)
        z_values = current @ weights + biases
        activation = layer.get("activation", "identity")
        if activation == "softmax":
            true_probabilities = _softmax(z_values)
            output_values = true_probabilities
            predicted_class = int(np.argmax(true_probabilities))
            display_probabilities = _display_softmax(z_values)
        else:
            output_values = _apply_activation(z_values, activation)
        trace_layers.append(
            {
                "index": layer_index,
                "name": layer.get("name", f"Layer {layer_index + 1}"),
                "kind": layer.get("kind", "hidden"),
                "activation": activation,
                "z": z_values.tolist(),
                "h": output_values.tolist(),
                "neuron_count": int(output_values.size),
            }
        )
        current = output_values

    if display_probabilities is None:
        display_probabilities = current
    if predicted_class is None:
        predicted_class = int(np.argmax(current))
    return {
        "id": model["id"],
        "name": model["name"],
        "kind": model["kind"],
        "architecture": model["architecture"],
        "activations": model["activations"],
        "parameter_count": model["parameter_count"],
        "test_accuracy": model["test_accuracy"],
        "validation_accuracy": model.get("validation_accuracy"),
        "predicted_class": predicted_class,
        "confidence": float(np.max(display_probabilities)),
        "probabilities": [float(value) for value in display_probabilities],
        "layers": trace_layers,
    }


def _predict_features(
    artifact: dict[str, Any],
    features: np.ndarray,
    sample_index: int | None = None,
    preprocessing_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = np.asarray(features, dtype=float).reshape(-1)
    if features.size != PIXEL_COUNT:
        raise ValueError(f"Expected {PIXEL_COUNT} pixel values.")
    if not np.isfinite(features).all() or np.any(features < 0) or np.any(features > 16):
        raise ValueError("Pixel values must be finite and between 0 and 16.")

    true_label: int | None = None
    if sample_index is not None:
        labels = np.asarray(artifact.get("test_labels"), dtype=int)
        if sample_index < 0 or sample_index >= len(labels):
            raise ValueError("Sample index is outside the digits dataset.")
        true_label = int(labels[sample_index])

    preprocessing = artifact["preprocessing"]
    mean = _as_vector(preprocessing["mean"], PIXEL_COUNT, "preprocessing mean")
    standard_deviation = _as_vector(
        preprocessing["std"], PIXEL_COUNT, "preprocessing std"
    )
    normalized = (features - mean) / standard_deviation
    model_results = [_forward_model(model, normalized) for model in artifact["models"]]
    result = {
        "pixels": features.reshape(PIXEL_SIZE, PIXEL_SIZE).astype(int).tolist(),
        "features_normalized": [float(value) for value in normalized],
        "sample_index": sample_index,
        "true_label": true_label,
        "models": model_results,
    }
    if preprocessing_info is not None:
        result["preprocessing"] = preprocessing_info
    return result


def predict_digits(
    artifact: dict[str, Any],
    pixels: np.ndarray,
    sample_index: int | None = None,
) -> dict[str, Any]:
    raw_features = np.asarray(pixels, dtype=float).reshape(-1)
    if raw_features.size != PIXEL_COUNT:
        raise ValueError(f"Expected {PIXEL_COUNT} pixel values.")
    if not np.isfinite(raw_features).all() or np.any(raw_features < 0) or np.any(raw_features > 16):
        raise ValueError("Pixel values must be finite and between 0 and 16.")
    if sample_index is not None:
        samples = np.asarray(artifact.get("test_samples"), dtype=float)
        if sample_index < 0 or sample_index >= len(samples):
            raise ValueError("Sample index is outside the digits dataset.")
        if not np.allclose(samples[sample_index], raw_features):
            raise ValueError("sample_index does not match the supplied pixels.")
    return _predict_features(artifact, raw_features, sample_index)


def predict_drawing(artifact: dict[str, Any], drawing: np.ndarray) -> dict[str, Any]:
    normalized_pixels, preprocessing_info = preprocess_drawing(drawing)
    return _predict_features(artifact, normalized_pixels, preprocessing_info=preprocessing_info)
