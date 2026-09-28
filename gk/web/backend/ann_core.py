"""Small, framework-free ANN core shared by the realtime demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

ActivationName = Literal[
    "tanh",
    "sigmoid",
    "relu",
    "leaky_relu",
    "softplus",
    "identity",
]

SUPPORTED_ACTIVATIONS: tuple[ActivationName, ...] = (
    "tanh",
    "sigmoid",
    "relu",
    "leaky_relu",
    "softplus",
    "identity",
)
CLASS_COUNT = 3
SAMPLES_PER_CLASS = 1000
DEFAULT_SAMPLE_COUNT = CLASS_COUNT * SAMPLES_PER_CLASS


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - logits.max(axis=1, keepdims=True)
    exponentiated_logits = np.exp(shifted_logits)
    return exponentiated_logits / exponentiated_logits.sum(axis=1, keepdims=True)


def apply_activation(values: np.ndarray, activation_name: str) -> np.ndarray:
    if activation_name == "tanh":
        return np.tanh(values)
    if activation_name == "sigmoid":
        clipped_values = np.clip(values, -500, 500)
        return 1 / (1 + np.exp(-clipped_values))
    if activation_name == "relu":
        return np.maximum(0, values)
    if activation_name == "leaky_relu":
        return np.where(values > 0, values, 0.01 * values)
    if activation_name == "softplus":
        return np.logaddexp(0, values)
    if activation_name == "identity":
        return values
    raise ValueError(f"Unsupported activation: {activation_name}")


def activation_derivative(values: np.ndarray, activation_name: str) -> np.ndarray:
    if activation_name == "tanh":
        output_values = np.tanh(values)
        return 1 - output_values**2
    if activation_name == "sigmoid":
        output_values = apply_activation(values, "sigmoid")
        return output_values * (1 - output_values)
    if activation_name == "relu":
        return (values > 0).astype(float)
    if activation_name == "leaky_relu":
        return np.where(values > 0, 1.0, 0.01)
    if activation_name == "softplus":
        return apply_activation(values, "sigmoid")
    if activation_name == "identity":
        return np.ones_like(values)
    raise ValueError(f"Unsupported activation: {activation_name}")


def one_hot_encode(class_labels: np.ndarray, number_of_classes: int) -> np.ndarray:
    encoded_labels = np.zeros((class_labels.size, number_of_classes))
    encoded_labels[np.arange(class_labels.size), class_labels] = 1
    return encoded_labels


@dataclass(frozen=True)
class ClassificationData:
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    training_features_raw: np.ndarray
    training_features: np.ndarray
    training_labels: np.ndarray
    training_one_hot_labels: np.ndarray
    validation_features: np.ndarray
    validation_labels: np.ndarray
    validation_one_hot_labels: np.ndarray
    testing_features: np.ndarray
    testing_labels: np.ndarray
    holdout_features: np.ndarray
    holdout_labels: np.ndarray
    holdout_one_hot_labels: np.ndarray


def prepare_classification_data(
    random_seed: int = 42,
    difficulty: float = 0.5,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    train_percentage: float = 60.0,
    validation_percentage: float = 15.0,
    test_percentage: float = 15.0,
    holdout_percentage: float = 10.0,
) -> ClassificationData:
    if not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be between 0.0 and 1.0")
    split_percentages = (
        train_percentage,
        validation_percentage,
        test_percentage,
        holdout_percentage,
    )
    if any(percentage <= 0 for percentage in split_percentages):
        raise ValueError("all dataset percentages must be positive")
    if not np.isclose(sum(split_percentages), 100.0):
        raise ValueError("all dataset percentages must sum to 100")
    if sample_count < 30:
        raise ValueError("sample_count must be at least 30")

    difficulty_level = float(difficulty)
    feature_matrix, target_labels = make_classification(
        n_samples=sample_count,
        n_features=4,
        n_informative=3 if difficulty_level < 0.5 else 4,
        n_redundant=0,
        n_repeated=0,
        n_classes=CLASS_COUNT,
        n_clusters_per_class=1 if difficulty_level < 0.5 else 2,
        weights=[1 / 3, 1 / 3, 1 / 3],
        class_sep=1.4 - 1.2 * difficulty_level,
        flip_y=0.02 + 0.28 * difficulty_level,
        random_state=random_seed,
    )

    all_indices = np.arange(sample_count)
    training_sample_count = round(sample_count * train_percentage / 100)
    validation_sample_count = round(sample_count * validation_percentage / 100)
    testing_sample_count = round(sample_count * test_percentage / 100)
    holdout_sample_count = sample_count - training_sample_count - validation_sample_count - testing_sample_count
    if min(training_sample_count, validation_sample_count, testing_sample_count, holdout_sample_count) < 3:
        raise ValueError("each dataset split must contain at least 3 samples")
    training_indices, remaining_indices = train_test_split(
        all_indices,
        train_size=training_sample_count,
        stratify=target_labels,
        random_state=random_seed,
    )
    validation_indices, remaining_indices = train_test_split(
        remaining_indices,
        train_size=validation_sample_count,
        test_size=testing_sample_count + holdout_sample_count,
        stratify=target_labels[remaining_indices],
        random_state=random_seed + 1,
    )
    testing_indices, holdout_indices = train_test_split(
        remaining_indices,
        train_size=testing_sample_count,
        test_size=holdout_sample_count,
        stratify=target_labels[remaining_indices],
        random_state=random_seed + 2,
    )

    training_features_raw = feature_matrix[training_indices]
    training_labels = target_labels[training_indices]
    validation_features_raw = feature_matrix[validation_indices]
    validation_labels = target_labels[validation_indices]
    testing_features_raw = feature_matrix[testing_indices]
    testing_labels = target_labels[testing_indices]
    holdout_features_raw = feature_matrix[holdout_indices]
    holdout_labels = target_labels[holdout_indices]

    training_feature_means = training_features_raw.mean(axis=0)
    training_feature_stds = training_features_raw.std(axis=0)
    training_features = (
        training_features_raw - training_feature_means
    ) / training_feature_stds
    validation_features = (
        validation_features_raw - training_feature_means
    ) / training_feature_stds
    testing_features = (
        testing_features_raw - training_feature_means
    ) / training_feature_stds
    holdout_features = (
        holdout_features_raw - training_feature_means
    ) / training_feature_stds

    return ClassificationData(
        feature_names=tuple(f"feature_{index + 1}" for index in range(4)),
        class_names=tuple(f"class_{index}" for index in range(3)),
        training_features_raw=training_features_raw,
        training_features=training_features,
        training_labels=training_labels,
        training_one_hot_labels=one_hot_encode(training_labels, CLASS_COUNT),
        validation_features=validation_features,
        validation_labels=validation_labels,
        validation_one_hot_labels=one_hot_encode(validation_labels, CLASS_COUNT),
        testing_features=testing_features,
        testing_labels=testing_labels,
        holdout_features=holdout_features,
        holdout_labels=holdout_labels,
        holdout_one_hot_labels=one_hot_encode(holdout_labels, CLASS_COUNT),
    )


class SimpleANN:
    """A configurable MLP classifier with explicit forward and gradient steps."""

    def __init__(
        self,
        input_feature_count: int = 4,
        hidden_neuron_count: int = 8,
        output_class_count: int = CLASS_COUNT,
        learning_rate: float = 0.05,
        random_seed: int = 42,
        hidden_activation_name: ActivationName = "tanh",
        hidden_layers: list[tuple[int, ActivationName]] | None = None,
    ) -> None:
        self.random_generator = np.random.default_rng(random_seed)
        configured_layers = hidden_layers or [(hidden_neuron_count, hidden_activation_name)]
        if not configured_layers:
            raise ValueError("At least one hidden layer is required")
        for neuron_count, activation_name in configured_layers:
            if neuron_count < 1:
                raise ValueError("hidden layer size must be positive")
            if activation_name not in SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Unsupported activation: {activation_name}")

        self.hidden_layer_sizes = [neuron_count for neuron_count, _ in configured_layers]
        self.hidden_activation_names = [activation_name for _, activation_name in configured_layers]
        self.hidden_layer_weights: list[np.ndarray] = []
        self.hidden_layer_biases_list: list[np.ndarray] = []
        previous_size = input_feature_count
        for neuron_count, _ in configured_layers:
            self.hidden_layer_weights.append(
                self.random_generator.normal(0, 0.5, size=(previous_size, neuron_count))
            )
            self.hidden_layer_biases_list.append(np.zeros((1, neuron_count)))
            previous_size = neuron_count

        self.output_layer_weights = self.random_generator.normal(
            0, 0.5, size=(previous_size, output_class_count)
        )
        self.output_layer_biases = np.zeros((1, output_class_count))
        self.learning_rate = learning_rate
        self.hidden_activation_name = self.hidden_activation_names[0]

        # Keep the original attribute names available to the existing Train tab
        # and educational notebook while the studio uses the layer lists above.
        self.input_to_hidden_weights = self.hidden_layer_weights[0]
        self.hidden_layer_biases = self.hidden_layer_biases_list[0]
        self.hidden_to_output_weights = self.output_layer_weights
        self.loss_history: list[float] = []
        self.validation_loss_history: list[float] = []
        self.validation_accuracy_history: list[float] = []

    def forward(self, input_features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        hidden_layer_pre_activations: list[np.ndarray] = []
        hidden_layer_outputs: list[np.ndarray] = []
        current_output = input_features
        for weights, biases, activation_name in zip(
            self.hidden_layer_weights,
            self.hidden_layer_biases_list,
            self.hidden_activation_names,
        ):
            hidden_layer_pre_activation = current_output @ weights + biases
            current_output = apply_activation(hidden_layer_pre_activation, activation_name)
            hidden_layer_pre_activations.append(hidden_layer_pre_activation)
            hidden_layer_outputs.append(current_output)

        output_layer_pre_activation = (
            current_output @ self.output_layer_weights
            + self.output_layer_biases
        )
        output_probabilities = softmax(output_layer_pre_activation)
        forward_cache = {
            "hidden_layer_pre_activations": hidden_layer_pre_activations,
            "hidden_layer_outputs": hidden_layer_outputs,
            "output_layer_pre_activation": output_layer_pre_activation,
        }
        if len(hidden_layer_outputs) == 1:
            forward_cache["hidden_layer_pre_activation"] = hidden_layer_pre_activations[0]
            forward_cache["hidden_layer_output"] = hidden_layer_outputs[0]
        return output_probabilities, forward_cache

    @staticmethod
    def cross_entropy_loss(
        output_probabilities: np.ndarray,
        one_hot_targets: np.ndarray,
    ) -> float:
        clipped_probabilities = np.clip(output_probabilities, 1e-12, 1.0)
        return float(
            -np.mean(np.sum(one_hot_targets * np.log(clipped_probabilities), axis=1))
        )

    def train_epoch(
        self,
        input_features: np.ndarray,
        one_hot_targets: np.ndarray,
        batch_size: int | None = None,
    ) -> float:
        training_sample_count = input_features.shape[0]
        effective_batch_size = training_sample_count if batch_size is None else max(
            1, min(batch_size, training_sample_count)
        )
        shuffled_indices = self.random_generator.permutation(training_sample_count)

        for batch_start in range(0, training_sample_count, effective_batch_size):
            batch_indices = shuffled_indices[batch_start : batch_start + effective_batch_size]
            self._train_batch(input_features[batch_indices], one_hot_targets[batch_indices])

        training_loss = self.cross_entropy_loss(
            self.predict_proba(input_features),
            one_hot_targets,
        )
        self.loss_history.append(training_loss)
        return training_loss

    def _train_batch(
        self,
        input_features: np.ndarray,
        one_hot_targets: np.ndarray,
    ) -> None:
        training_sample_count = input_features.shape[0]
        output_probabilities, forward_cache = self.forward(input_features)

        output_layer_pre_activation_gradients = (
            output_probabilities - one_hot_targets
        ) / training_sample_count
        hidden_layer_outputs = forward_cache["hidden_layer_outputs"]
        hidden_layer_pre_activations = forward_cache["hidden_layer_pre_activations"]
        output_layer_input = hidden_layer_outputs[-1]
        output_layer_weight_gradients = output_layer_input.T @ output_layer_pre_activation_gradients
        output_layer_bias_gradients = output_layer_pre_activation_gradients.sum(
            axis=0, keepdims=True
        )
        hidden_layer_output_gradients = (
            output_layer_pre_activation_gradients
            @ self.output_layer_weights.T
        )

        hidden_weight_gradients: list[np.ndarray] = [
            np.zeros_like(weights) for weights in self.hidden_layer_weights
        ]
        hidden_bias_gradients: list[np.ndarray] = [
            np.zeros_like(biases) for biases in self.hidden_layer_biases_list
        ]
        for layer_index in range(len(self.hidden_layer_weights) - 1, -1, -1):
            hidden_layer_pre_activation_gradients = hidden_layer_output_gradients * activation_derivative(
                hidden_layer_pre_activations[layer_index],
                self.hidden_activation_names[layer_index],
            )
            previous_output = (
                input_features
                if layer_index == 0
                else hidden_layer_outputs[layer_index - 1]
            )
            hidden_weight_gradients[layer_index] = (
                previous_output.T @ hidden_layer_pre_activation_gradients
            )
            hidden_bias_gradients[layer_index] = hidden_layer_pre_activation_gradients.sum(
                axis=0, keepdims=True
            )
            hidden_layer_output_gradients = (
                hidden_layer_pre_activation_gradients
                @ self.hidden_layer_weights[layer_index].T
            )

        self.output_layer_weights -= self.learning_rate * output_layer_weight_gradients
        self.output_layer_biases -= self.learning_rate * output_layer_bias_gradients
        for layer_index in range(len(self.hidden_layer_weights)):
            self.hidden_layer_weights[layer_index] -= (
                self.learning_rate * hidden_weight_gradients[layer_index]
            )
            self.hidden_layer_biases_list[layer_index] -= (
                self.learning_rate * hidden_bias_gradients[layer_index]
            )

    def evaluate(
        self,
        input_features: np.ndarray,
        one_hot_targets: np.ndarray,
    ) -> tuple[float, float]:
        output_probabilities = self.predict_proba(input_features)
        loss = self.cross_entropy_loss(output_probabilities, one_hot_targets)
        accuracy = float(
            np.mean(
                output_probabilities.argmax(axis=1)
                == one_hot_targets.argmax(axis=1)
            )
        )
        return loss, accuracy

    def predict_proba(self, input_features: np.ndarray) -> np.ndarray:
        output_probabilities, _ = self.forward(input_features)
        return output_probabilities

    def predict(self, input_features: np.ndarray) -> np.ndarray:
        return self.predict_proba(input_features).argmax(axis=1)

    def trace(self, input_features: np.ndarray) -> dict[str, np.ndarray]:
        """Forward pass with intermediates for the Inspect UI."""
        features = np.asarray(input_features, dtype=float)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        output_probabilities, forward_cache = self.forward(features)
        return {
            "input": features,
            "hidden_pre_activation": forward_cache["hidden_layer_pre_activations"][0],
            "hidden_output": forward_cache["hidden_layer_outputs"][0],
            "hidden_pre_activations": forward_cache["hidden_layer_pre_activations"],
            "hidden_outputs": forward_cache["hidden_layer_outputs"],
            "output_pre_activation": forward_cache["output_layer_pre_activation"],
            "output_probabilities": output_probabilities,
        }
