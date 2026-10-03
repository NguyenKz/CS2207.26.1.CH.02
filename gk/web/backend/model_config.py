"""Single source of truth for the four MNIST demo model configurations.

The notebooks copy ``MODEL_CONFIGS`` before tuning so that a tuning run can
update the size-specific JSON config without changing these defaults.

ANN configs declare every layer as ``{input, output, activation}``. Sklearn's
``MLPClassifier`` still trains with one shared hidden activation — use
``hidden_activation`` / ``hidden_layer_sizes`` when building the estimator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .digits_config import EARLY_STOPPING, MAX_ITER, PIXEL_COUNT, PIXEL_SIZE


RANDOM_SEED = 42
CLASS_COUNT = 10
SEARCH_ROUNDS = 30
SEARCH_MAX_ITER = 120
TUNING_SEARCH_SAMPLE_LIMIT = 24000
AUGMENT_TRAINING = True
AUGMENT_FACTOR = 5
AUGMENT_SHIFT_PIXELS = 3
AUGMENT_STROKE_VARIANTS = True
AUGMENT_SCALE_RANGE = (0.85, 1.15)
AUGMENT_ROTATE_DEGREES = 15.0
AUGMENT_SHARPEN = True
BATCH_SIZE = 256
LOGISTIC_MAX_ITER = 300
LOGISTIC_CANDIDATES = tuple(float(value) for value in np.logspace(-4, 2, SEARCH_ROUNDS))
MLP_SEARCH_CONFIGS = tuple(
    (architecture, "relu", learning_rate)
    for architecture in ((16,), (24,), (32,), (40,), (48,), (32, 16), (48, 24))
    for learning_rate in (0.0005, 0.0007, 0.001, 0.002)
) + (((48, 24), "tanh", 0.001), ((32,), "tanh", 0.001))


def layers_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer ``layers``; upgrade legacy ``hidden_layer_sizes`` + ``activation``."""
    if "layers" in config:
        return list(config["layers"])
    sizes = [int(size) for size in config.get("hidden_layer_sizes", ())]
    activation = config.get("activation", "tanh")
    layers: list[dict[str, Any]] = []
    previous = PIXEL_COUNT
    for size in sizes:
        layers.append({"input": previous, "output": size, "activation": activation})
        previous = size
    layers.append({"input": previous, "output": CLASS_COUNT, "activation": "softmax"})
    return layers


def architecture_sizes(config: dict[str, Any]) -> list[int]:
    layers = layers_from_config(config)
    return [layers[0]["input"], *(layer["output"] for layer in layers)]


def hidden_layer_sizes(config: dict[str, Any]) -> tuple[int, ...]:
    return tuple(layer["output"] for layer in layers_from_config(config)[:-1])


def layer_activations(config: dict[str, Any]) -> list[str]:
    return [layer["activation"] for layer in layers_from_config(config)]


def hidden_activation(config: dict[str, Any]) -> str | None:
    """Activation sklearn MLPClassifier applies to every hidden layer."""
    hidden = layers_from_config(config)[:-1]
    return None if not hidden else hidden[0]["activation"]


MODEL_CONFIG_SOURCE_PATH = Path(__file__).with_name("model_configs") / "mnist_28x28.json"


def load_model_configs(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate a JSON model configuration for the active 28x28 pipeline."""
    config_path = MODEL_CONFIG_SOURCE_PATH if path is None else Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("pixel_size") != PIXEL_SIZE:
        raise ValueError(
            f"Model config pixel_size must be {PIXEL_SIZE}: {config_path}"
        )
    configs = payload.get("models")
    if not isinstance(configs, dict) or not configs:
        raise ValueError(f"Model config must contain a non-empty models object: {config_path}")

    for model_id, config in configs.items():
        if not isinstance(config, dict):
            raise ValueError(f"Model config {model_id} must be an object.")
        layers = layers_from_config(config)
        if layers[0]["input"] != PIXEL_COUNT:
            raise ValueError(f"Model {model_id} must start with {PIXEL_COUNT} features.")
        if layers[-1]["output"] != CLASS_COUNT:
            raise ValueError(f"Model {model_id} must end with {CLASS_COUNT} outputs.")
        for left, right in zip(layers, layers[1:]):
            if left["output"] != right["input"]:
                raise ValueError(f"Model {model_id} has disconnected layers.")
        if config.get("kind") not in {"linear", "ann"}:
            raise ValueError(f"Model {model_id} has an unsupported kind.")
        if config["kind"] == "ann":
            if not hidden_layer_sizes(config) or hidden_activation(config) is None:
                raise ValueError(f"ANN model {model_id} needs at least one hidden layer.")
            hidden_activations = [layer["activation"] for layer in layers[:-1]]
            if len(set(hidden_activations)) != 1:
                raise ValueError(
                    f"MLPClassifier requires one shared hidden activation: {model_id}"
                )
    return configs


MODEL_CONFIGS = load_model_configs()


if __name__ == "__main__":
    for name, config in MODEL_CONFIGS.items():
        layers = config["layers"]
        assert layers[0]["input"] == PIXEL_COUNT, name
        assert layers[-1]["output"] == CLASS_COUNT, name
        for left, right in zip(layers, layers[1:]):
            assert left["output"] == right["input"], name
        assert architecture_sizes(config)[0] == PIXEL_COUNT, name
        if config["kind"] == "ann":
            assert hidden_layer_sizes(config)
            assert hidden_activation(config) is not None
            hidden_acts = [layer["activation"] for layer in layers[:-1]]
            assert len(set(hidden_acts)) == 1, name
    print("ok", {name: architecture_sizes(cfg) for name, cfg in MODEL_CONFIGS.items()})
