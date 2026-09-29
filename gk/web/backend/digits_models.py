"""Configurable scikit-learn models used by the MNIST training export."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


RANDOM_SEED = 42
PIXEL_COUNT = 64
CLASS_COUNT = 10
SEARCH_ROUNDS = 30
SEARCH_MAX_ITER = 120
TUNING_SEARCH_SAMPLE_LIMIT = 24000
LOGISTIC_CANDIDATES = tuple(float(value) for value in np.logspace(-4, 2, SEARCH_ROUNDS))
MLP_SEARCH_CONFIGS = tuple(
    (architecture, 'relu', learning_rate)
    for architecture in ((16,), (24,), (32,), (40,), (48,), (32, 16), (48, 24))
    for learning_rate in (0.0005, 0.0007, 0.001, 0.002)
) + (((48, 24), 'tanh', 0.001), ((32,), 'tanh', 0.001))
MODEL_CONFIGS = {
    'logistic': {'kind': 'linear', 'C': 1.0, 'solver': 'lbfgs', 'max_iter': 2000},
    'mlp_4_one_layer': {
        'kind': 'ann', 'hidden_layer_sizes': (4,), 'activation': 'tanh',
        'learning_rate_init': 0.001, 'max_iter': 350, 'early_stopping': False,
    },
    'mlp_4_two_layers': {
        'kind': 'ann', 'hidden_layer_sizes': (4, 4), 'activation': 'tanh',
        'learning_rate_init': 0.001, 'max_iter': 350, 'early_stopping': False,
    },
    'compact_tuned': {
        'kind': 'ann', 'hidden_layer_sizes': (48,), 'activation': 'relu',
        'learning_rate_init': 0.002, 'max_iter': 350, 'early_stopping': False,
    },
}


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
    max_iter: int = 2000,
    verbose: int = 0,
) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=RANDOM_SEED, verbose=verbose)


def build_mlp(
    hidden_layers: tuple[int, ...],
    activation: str = 'tanh',
    learning_rate_init: float = 0.001,
    max_iter: int = 350,
    early_stopping: bool = False,
    n_iter_no_change: int | None = None,
    verbose: bool = False,
) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver='adam',
        learning_rate_init=learning_rate_init,
        alpha=0.0001,
        batch_size=64,
        max_iter=max_iter,
        early_stopping=early_stopping,
        tol=0.0,
        n_iter_no_change=max_iter if n_iter_no_change is None else n_iter_no_change,
        random_state=RANDOM_SEED,
        verbose=verbose,
    )


def choose_search_subset(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    count = min(TUNING_SEARCH_SAMPLE_LIMIT, len(features))
    indices = np.random.default_rng(RANDOM_SEED).choice(len(features), count, replace=False)
    return features[indices], labels[indices], count
