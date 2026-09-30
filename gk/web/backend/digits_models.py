"""Configurable scikit-learn models used by the MNIST training export."""

from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
from .digits_config import EARLY_STOPPING, MAX_ITER, PIXEL_COUNT
from .model_config import (
    BATCH_SIZE,
    CLASS_COUNT,
    LOGISTIC_CANDIDATES,
    MLP_SEARCH_CONFIGS,
    MODEL_CONFIGS,
    RANDOM_SEED,
    SEARCH_MAX_ITER,
    SEARCH_ROUNDS,
    TUNING_SEARCH_SAMPLE_LIMIT,
)


def parameter_count(layers: list[dict[str, Any]]) -> int:
    return sum(
        len(layer['weights']) * len(layer['weights'][0]) + len(layer['biases'])
        for layer in layers
    )


def export_layers(model: Any, hidden_activation: str | None) -> list[dict[str, Any]]:
    if not hasattr(model, 'coefs_'):
        return [{
            'name': 'Output',
            'kind': 'output',
            'activation': 'softmax',
            'weights': model.coef_.T.astype(float).tolist(),
            'biases': model.intercept_.astype(float).tolist(),
        }]

    exported: list[dict[str, Any]] = []
    for index, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        is_output = index == len(model.coefs_) - 1
        exported.append({
            'name': 'Output' if is_output else f'Hidden {index + 1}',
            'kind': 'output' if is_output else 'hidden',
            'activation': 'softmax' if is_output else hidden_activation,
            'weights': weights.astype(float).tolist(),
            'biases': biases.astype(float).tolist(),
        })
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
        'id': model_id,
        'name': name,
        'kind': kind,
        'architecture': architecture,
        'activations': activations,
        'parameter_count': parameter_count(layers),
        'test_accuracy': float(test_accuracy),
        'validation_accuracy': None if validation_accuracy is None else float(validation_accuracy),
        'layers': layers,
    }
    if source is not None:
        metadata['source'] = source
    if source_url is not None:
        metadata['source_url'] = source_url
    return metadata


def build_logistic(
    C: float,
    solver: str = 'lbfgs',
    max_iter: int | None = None,
    verbose: int = 0,
) -> LogisticRegression:
    configured_max_iter = MODEL_CONFIGS['logistic']['max_iter'] if max_iter is None else max_iter
    return LogisticRegression(
        C=C,
        max_iter=int(configured_max_iter),
        solver=solver,
        random_state=RANDOM_SEED,
        verbose=verbose,
    )


def build_mlp(
    hidden_layers: tuple[int, ...],
    activation: str = 'tanh',
    learning_rate_init: float = 0.001,
    max_iter: int | None = None,
    early_stopping: bool | None = None,
    n_iter_no_change: int | None = None,
    batch_size: int | None = None,
    verbose: bool = False,
) -> MLPClassifier:
    configured_max_iter = MAX_ITER if max_iter is None else max_iter
    configured_early_stopping = EARLY_STOPPING if early_stopping is None else early_stopping
    configured_batch_size = BATCH_SIZE if batch_size is None else batch_size
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver='adam',
        learning_rate_init=learning_rate_init,
        alpha=0.0001,
        batch_size=int(configured_batch_size),
        max_iter=int(configured_max_iter),
        early_stopping=bool(configured_early_stopping),
        tol=0.0,
        n_iter_no_change=(
            int(configured_max_iter)
            if n_iter_no_change is None
            else n_iter_no_change
        ),
        random_state=RANDOM_SEED,
        verbose=verbose,
    )


def choose_search_subset(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    count = min(TUNING_SEARCH_SAMPLE_LIMIT, len(features))
    indices = np.random.default_rng(RANDOM_SEED).choice(len(features), count, replace=False)
    return features[indices], labels[indices], count
