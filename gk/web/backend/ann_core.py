"""Small, framework-free ANN core shared by the realtime demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.datasets import load_iris

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
class IrisData:
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


def prepare_iris_data(random_seed: int = 42) -> IrisData:
    iris_dataset = load_iris()
    feature_matrix = iris_dataset.data
    target_labels = iris_dataset.target
    random_generator = np.random.default_rng(random_seed)

    training_indices_by_class: list[np.ndarray] = []
    validation_indices_by_class: list[np.ndarray] = []
    testing_indices_by_class: list[np.ndarray] = []

    for class_label in np.unique(target_labels):
        class_indices = np.flatnonzero(target_labels == class_label)
        random_generator.shuffle(class_indices)
        training_sample_count = int(0.60 * len(class_indices))
        validation_sample_count = int(0.20 * len(class_indices))
        training_indices_by_class.append(class_indices[:training_sample_count])
        validation_indices_by_class.append(
            class_indices[
                training_sample_count : training_sample_count + validation_sample_count
            ]
        )
        testing_indices_by_class.append(
            class_indices[training_sample_count + validation_sample_count :]
        )

    training_indices = np.concatenate(training_indices_by_class)
    validation_indices = np.concatenate(validation_indices_by_class)
    testing_indices = np.concatenate(testing_indices_by_class)
    random_generator.shuffle(training_indices)
    random_generator.shuffle(validation_indices)
    random_generator.shuffle(testing_indices)

    training_features_raw = feature_matrix[training_indices]
    training_labels = target_labels[training_indices]
    validation_features_raw = feature_matrix[validation_indices]
    validation_labels = target_labels[validation_indices]
    testing_features_raw = feature_matrix[testing_indices]
    testing_labels = target_labels[testing_indices]

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

    return IrisData(
        feature_names=tuple(iris_dataset.feature_names),
        class_names=tuple(iris_dataset.target_names),
        training_features_raw=training_features_raw,
        training_features=training_features,
        training_labels=training_labels,
        training_one_hot_labels=one_hot_encode(training_labels, 3),
        validation_features=validation_features,
        validation_labels=validation_labels,
        validation_one_hot_labels=one_hot_encode(validation_labels, 3),
        testing_features=testing_features,
        testing_labels=testing_labels,
    )


class SimpleANN:
    """A two-layer classifier with explicit forward and gradient steps."""

    def __init__(
        self,
        input_feature_count: int = 4,
        hidden_neuron_count: int = 8,
        output_class_count: int = 3,
        learning_rate: float = 0.05,
        random_seed: int = 42,
        hidden_activation_name: ActivationName = "tanh",
    ) -> None:
        if hidden_activation_name not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {hidden_activation_name}")

        random_generator = np.random.default_rng(random_seed)
        self.input_to_hidden_weights = random_generator.normal(
            0, 0.5, size=(input_feature_count, hidden_neuron_count)
        )
        self.hidden_layer_biases = np.zeros((1, hidden_neuron_count))
        self.hidden_to_output_weights = random_generator.normal(
            0, 0.5, size=(hidden_neuron_count, output_class_count)
        )
        self.output_layer_biases = np.zeros((1, output_class_count))
        self.learning_rate = learning_rate
        self.hidden_activation_name = hidden_activation_name
        self.loss_history: list[float] = []
        self.validation_loss_history: list[float] = []
        self.validation_accuracy_history: list[float] = []

    def forward(self, input_features: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        hidden_layer_pre_activation = (
            input_features @ self.input_to_hidden_weights
            + self.hidden_layer_biases
        )
        hidden_layer_output = apply_activation(
            hidden_layer_pre_activation,
            self.hidden_activation_name,
        )
        output_layer_pre_activation = (
            hidden_layer_output @ self.hidden_to_output_weights
            + self.output_layer_biases
        )
        output_probabilities = softmax(output_layer_pre_activation)
        forward_cache = {
            "hidden_layer_pre_activation": hidden_layer_pre_activation,
            "hidden_layer_output": hidden_layer_output,
            "output_layer_pre_activation": output_layer_pre_activation,
        }
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
    ) -> float:
        training_sample_count = input_features.shape[0]
        output_probabilities, forward_cache = self.forward(input_features)
        training_loss = self.cross_entropy_loss(output_probabilities, one_hot_targets)

        output_layer_pre_activation_gradients = (
            output_probabilities - one_hot_targets
        ) / training_sample_count
        hidden_to_output_weight_gradients = (
            forward_cache["hidden_layer_output"].T
            @ output_layer_pre_activation_gradients
        )
        output_layer_bias_gradients = output_layer_pre_activation_gradients.sum(
            axis=0, keepdims=True
        )
        hidden_layer_output_gradients = (
            output_layer_pre_activation_gradients
            @ self.hidden_to_output_weights.T
        )
        hidden_layer_pre_activation_gradients = (
            hidden_layer_output_gradients
            * activation_derivative(
                forward_cache["hidden_layer_pre_activation"],
                self.hidden_activation_name,
            )
        )
        input_to_hidden_weight_gradients = (
            input_features.T @ hidden_layer_pre_activation_gradients
        )
        hidden_layer_bias_gradients = hidden_layer_pre_activation_gradients.sum(
            axis=0, keepdims=True
        )

        self.hidden_to_output_weights -= (
            self.learning_rate * hidden_to_output_weight_gradients
        )
        self.output_layer_biases -= self.learning_rate * output_layer_bias_gradients
        self.input_to_hidden_weights -= (
            self.learning_rate * input_to_hidden_weight_gradients
        )
        self.hidden_layer_biases -= self.learning_rate * hidden_layer_bias_gradients
        self.loss_history.append(training_loss)
        return training_loss

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
