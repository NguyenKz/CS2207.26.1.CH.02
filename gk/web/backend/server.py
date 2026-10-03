"""Realtime FastAPI server for the ANN training lesson."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.websockets import WebSocketDisconnected, WebSocketState

from .ann_core import (
    CLASS_COUNT,
    DEFAULT_SAMPLE_COUNT,
    SAMPLES_PER_CLASS,
    SUPPORTED_ACTIVATIONS,
    SUPPORTED_INPUT_FEATURE_COUNTS,
    SimpleANN,
    prepare_classification_data,
)
from .digits_predict import (
    DRAWING_COUNT,
    DigitsArtifactError,
    PIXEL_COUNT,
    get_test_indices,
    load_digits_artifact,
    predict_digits,
    predict_drawing,
    public_model_metadata,
    serialize_sample,
)

app = FastAPI(title="ANN Training Lab")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5113", "http://127.0.0.1:5113"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainConfig(BaseModel):
    epochs: int = Field(default=2000, ge=1, le=500000)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    sample_count: int = Field(default=DEFAULT_SAMPLE_COUNT, ge=30, le=10000)
    batch_size: int = Field(default=32, ge=1, le=10000)
    train_percentage: float = Field(default=70.0, gt=0.0, lt=100.0)
    validation_percentage: float = Field(default=15.0, gt=0.0, lt=100.0)
    test_percentage: float = Field(default=15.0, gt=0.0, lt=100.0)
    input_feature_count: int = Field(default=8)
    hidden_neuron_count: int = Field(default=8, ge=1, le=128)
    random_seed: int = 42
    early_stopping: bool = False
    early_stopping_patience: int = Field(default=40, ge=1, le=500)
    early_stopping_min_delta: float = Field(default=0.001, ge=0.0, le=1.0)
    activations: list[str] = Field(default_factory=lambda: list(SUPPORTED_ACTIVATIONS))

    @field_validator("input_feature_count")
    @classmethod
    def validate_input_feature_count(cls, value: int) -> int:
        if value not in SUPPORTED_INPUT_FEATURE_COUNTS:
            raise ValueError(
                f"input_feature_count must be one of: {list(SUPPORTED_INPUT_FEATURE_COUNTS)}"
            )
        return value

    @field_validator("activations")
    @classmethod
    def validate_activations(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one activation is required")
        unsupported = set(value) - set(SUPPORTED_ACTIVATIONS)
        if unsupported:
            raise ValueError(f"Unsupported activations: {sorted(unsupported)}")
        return value

    @model_validator(mode="after")
    def validate_dataset_split(self) -> "TrainConfig":
        _validate_dataset_split(
            self.train_percentage,
            self.validation_percentage,
            self.test_percentage,
        )
        return self


def _validate_dataset_split(
    train_percentage: float,
    validation_percentage: float,
    test_percentage: float,
) -> None:
    split_total = train_percentage + validation_percentage + test_percentage
    if abs(split_total - 100.0) > 1e-6:
        raise ValueError("all dataset percentages must sum to 100")


class DatasetTrainConfig(BaseModel):
    difficulty: float = Field(default=0.7, ge=0.0, le=1.0)
    sample_count: int = Field(default=DEFAULT_SAMPLE_COUNT, ge=30, le=10000)
    train_percentage: float = Field(default=70.0, gt=0.0, lt=100.0)
    validation_percentage: float = Field(default=15.0, gt=0.0, lt=100.0)
    test_percentage: float = Field(default=15.0, gt=0.0, lt=100.0)
    input_feature_count: int = Field(default=32)
    random_seed: int = 42

    @field_validator("input_feature_count")
    @classmethod
    def validate_input_feature_count(cls, value: int) -> int:
        if value not in SUPPORTED_INPUT_FEATURE_COUNTS:
            raise ValueError(
                f"input_feature_count must be one of: {list(SUPPORTED_INPUT_FEATURE_COUNTS)}"
            )
        return value

    @model_validator(mode="after")
    def validate_dataset_split(self) -> "DatasetTrainConfig":
        _validate_dataset_split(
            self.train_percentage,
            self.validation_percentage,
            self.test_percentage,
        )
        return self


class DatasetTrainPreviewRequest(DatasetTrainConfig):
    feature_x: int = Field(default=0, ge=0)
    feature_y: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def validate_feature_axes(self) -> "DatasetTrainPreviewRequest":
        if self.feature_x >= self.input_feature_count or self.feature_y >= self.input_feature_count:
            raise ValueError("feature axes must be inside input_feature_count")
        if self.feature_x == self.feature_y:
            raise ValueError("feature axes must be different")
        return self


class DatasetTrainSamplesRequest(DatasetTrainConfig):
    class_label: int | None = Field(default=None, ge=0, lt=CLASS_COUNT)
    split: Literal["all", "training", "validation", "testing"] = "all"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=40, ge=1, le=40)


def validation_error_message(error: Exception) -> str:
    return f"Invalid training configuration: {error}"


_WS_CLOSED = (WebSocketDisconnect, WebSocketDisconnected)


async def _ws_send_json(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    """Send JSON if the client is still connected; return False on disconnect/reset."""
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except _WS_CLOSED:
        return False


async def train_activation(
    activation_name: str,
    config: TrainConfig,
    data: Any,
    event_queue: asyncio.Queue[dict[str, Any]],
    cancel_event: asyncio.Event,
) -> dict[str, Any]:
    model = SimpleANN(
        input_feature_count=config.input_feature_count,
        hidden_neuron_count=config.hidden_neuron_count,
        output_class_count=CLASS_COUNT,
        learning_rate=config.learning_rate,
        random_seed=config.random_seed,
        hidden_activation_name=activation_name,
    )

    completed_epochs = 0
    last_metrics = {
        "training_loss": None,
        "validation_loss": None,
        "validation_accuracy": None,
    }
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    stopped_early = False

    for epoch_number in range(1, config.epochs + 1):
        if cancel_event.is_set():
            break

        training_loss = model.train_epoch(
            data.training_features,
            data.training_one_hot_labels,
            batch_size=config.batch_size,
        )
        validation_loss, validation_accuracy = model.evaluate(
            data.validation_features,
            data.validation_one_hot_labels,
        )
        completed_epochs = epoch_number
        last_metrics = {
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        }
        if validation_loss < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
        elif config.early_stopping:
            epochs_without_improvement += 1

        should_stop_early = (
            config.early_stopping
            and epochs_without_improvement >= config.early_stopping_patience
        )
        await event_queue.put(
            {
                "type": "epoch_update",
                "activation": activation_name,
                "epoch": epoch_number,
                "total_epochs": config.epochs,
                **last_metrics,
                "status": "early_stopped" if should_stop_early else "running",
            }
        )
        await asyncio.sleep(0)
        if should_stop_early:
            stopped_early = True
            break

    test_accuracy = float(
        (model.predict(data.testing_features) == data.testing_labels).mean()
    )
    return {
        "activation": activation_name,
        "epochs_completed": completed_epochs,
        "test_accuracy": test_accuracy,
        "stopped_early": stopped_early,
        **last_metrics,
    }


async def stream_training(
    websocket: WebSocket,
    config: TrainConfig,
    cancel_event: asyncio.Event,
) -> None:
    run_id = str(uuid.uuid4())
    started_at = time.perf_counter()
    data = prepare_classification_data(
        random_seed=config.random_seed,
        difficulty=config.difficulty,
        sample_count=config.sample_count,
        train_percentage=config.train_percentage,
        validation_percentage=config.validation_percentage,
        test_percentage=config.test_percentage,
        input_feature_count=config.input_feature_count,
    )
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    if not await _ws_send_json(
        websocket,
        {
            "type": "run_started",
            "run_id": run_id,
            "activations": config.activations,
            "total_epochs": config.epochs,
            "difficulty": config.difficulty,
            "sample_count": config.sample_count,
            "class_count": CLASS_COUNT,
            "samples_per_class": SAMPLES_PER_CLASS,
            "input_feature_count": config.input_feature_count,
            "hidden_neuron_count": config.hidden_neuron_count,
            "batch_size": config.batch_size,
            "train_percentage": config.train_percentage,
            "validation_percentage": config.validation_percentage,
            "test_percentage": config.test_percentage,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "dataset": {
                "training": list(data.training_features.shape),
                "validation": list(data.validation_features.shape),
                "testing": list(data.testing_features.shape),
            },
        },
    ):
        return

    workers = [
        asyncio.create_task(
            train_activation(
                activation_name,
                config,
                data,
                event_queue,
                cancel_event,
            )
        )
        for activation_name in config.activations
    ]

    async def mark_worker_done(worker: asyncio.Task[dict[str, Any]]) -> None:
        try:
            result = await worker
            await event_queue.put({"type": "worker_done", "result": result})
        except Exception as error:  # pragma: no cover - surfaced to the client
            await event_queue.put({"type": "worker_error", "error": str(error)})

    completion_tasks = [asyncio.create_task(mark_worker_done(worker)) for worker in workers]
    results: list[dict[str, Any]] = []
    completed_workers = 0

    while completed_workers < len(completion_tasks):
        event = await event_queue.get()
        event_type = event.get("type")
        if event_type == "worker_done":
            results.append(event["result"])
            completed_workers += 1
        elif event_type == "worker_error":
            cancel_event.set()
            for worker in workers:
                worker.cancel()
            raise RuntimeError(event["error"])
        else:
            if not await _ws_send_json(websocket, {"run_id": run_id, **event}):
                cancel_event.set()
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*completion_tasks, return_exceptions=True)
                return

    await asyncio.gather(*completion_tasks)
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    status = "cancelled" if cancel_event.is_set() else "completed"
    await _ws_send_json(
        websocket,
        {
            "type": "run_cancelled" if cancel_event.is_set() else "run_completed",
            "run_id": run_id,
            "status": status,
            "duration_ms": duration_ms,
            "results": sorted(results, key=lambda result: result["activation"]),
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


class PredictRequest(BaseModel):
    pixels: list[float] | None = None
    drawing: list[float] | None = None
    sample_index: int | None = Field(default=None, ge=0)

    @field_validator("pixels")
    @classmethod
    def validate_pixels(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            if len(value) != PIXEL_COUNT:
                raise ValueError(f"pixels must contain exactly {PIXEL_COUNT} values")
            if any(not np.isfinite(pixel) or pixel < 0 or pixel > 16 for pixel in value):
                raise ValueError("pixel values must be between 0 and 16")
        return value

    @field_validator("drawing")
    @classmethod
    def validate_drawing(cls, value: list[float] | None) -> list[float] | None:
        if value is not None:
            if len(value) != DRAWING_COUNT:
                raise ValueError(f"drawing must contain exactly {DRAWING_COUNT} values")
            if any(not np.isfinite(pixel) or pixel < 0 or pixel > 16 for pixel in value):
                raise ValueError("drawing values must be finite and between 0 and 16")
        return value

    @model_validator(mode="after")
    def validate_input(self) -> "PredictRequest":
        if (self.pixels is None) == (self.drawing is None):
            raise ValueError("Provide exactly one of pixels or drawing")
        if self.sample_index is not None and self.drawing is not None:
            raise ValueError("sample_index can only be used with pixels")
        return self


def _get_digits_artifact() -> dict[str, Any]:
    try:
        return load_digits_artifact()
    except DigitsArtifactError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@lru_cache(maxsize=1)
def _get_dataset_train_data(
    difficulty: float,
    sample_count: int,
    train_percentage: float,
    validation_percentage: float,
    test_percentage: float,
    input_feature_count: int,
    random_seed: int,
) -> Any:
    return prepare_classification_data(
        random_seed=random_seed,
        difficulty=difficulty,
        sample_count=sample_count,
        train_percentage=train_percentage,
        validation_percentage=validation_percentage,
        test_percentage=test_percentage,
        input_feature_count=input_feature_count,
    )


def _train_data_cache_key(config: DatasetTrainConfig) -> tuple[Any, ...]:
    return (
        config.difficulty,
        config.sample_count,
        config.train_percentage,
        config.validation_percentage,
        config.test_percentage,
        config.input_feature_count,
        config.random_seed,
    )


def _get_train_dataset(config: DatasetTrainConfig) -> Any:
    return _get_dataset_train_data(*_train_data_cache_key(config))


def _train_dataset_splits(data: Any) -> list[tuple[str, np.ndarray, np.ndarray]]:
    feature_means = data.training_features_raw.mean(axis=0)
    feature_stds = data.training_features_raw.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
    return [
        ("training", data.training_features_raw, data.training_labels),
        (
            "validation",
            data.validation_features * feature_stds + feature_means,
            data.validation_labels,
        ),
        ("testing", data.testing_features * feature_stds + feature_means, data.testing_labels),
    ]


def _class_counts(labels: np.ndarray) -> list[int]:
    return [int(np.count_nonzero(labels == class_label)) for class_label in range(CLASS_COUNT)]


def _train_scatter_points(
    data: Any,
    feature_x: int,
    feature_y: int,
    points_per_class: int = 120,
) -> list[dict[str, Any]]:
    split_rows = _train_dataset_splits(data)
    features = np.vstack([rows[1] for rows in split_rows])
    labels = np.concatenate([rows[2] for rows in split_rows])
    split_labels = np.concatenate(
        [np.full(len(rows[1]), split_name, dtype=object) for split_name, *rows in split_rows]
    )
    points: list[dict[str, Any]] = []
    for class_label in range(CLASS_COUNT):
        class_indices = np.flatnonzero(labels == class_label)
        if len(class_indices) > points_per_class:
            class_indices = class_indices[
                np.linspace(0, len(class_indices) - 1, points_per_class, dtype=int)
            ]
        points.extend(
            {
                "index": int(index),
                "label": int(labels[index]),
                "split": str(split_labels[index]),
                "x": float(features[index, feature_x]),
                "y": float(features[index, feature_y]),
            }
            for index in class_indices
        )
    return points


@app.get("/predict/meta")
async def predict_meta() -> dict[str, Any]:
    artifact = _get_digits_artifact()
    return {
        "ready": True,
        "dataset": artifact["dataset"],
        "preprocessing": {"name": artifact["preprocessing"]["name"]},
        "test_indices": get_test_indices(artifact),
        "default_sample_index": int(artifact.get("demo_sample_index", get_test_indices(artifact)[0])),
        "models": public_model_metadata(artifact),
    }


@app.get("/dataset/predict/meta")
async def dataset_predict_meta() -> dict[str, Any]:
    artifact = _get_digits_artifact()
    labels = np.asarray(artifact["test_labels"], dtype=int)
    representatives = []
    for class_label in range(CLASS_COUNT):
        class_indices = np.flatnonzero(labels == class_label)
        if len(class_indices):
            representatives.append(serialize_sample(artifact, int(class_indices[0])))
    return {
        "dataset": artifact["dataset"],
        "preprocessing": {"name": artifact["preprocessing"]["name"]},
        "class_counts": _class_counts(labels),
        "representatives": representatives,
    }


@app.get("/dataset/predict/samples")
async def dataset_predict_samples(
    label: int | None = Query(default=None, ge=0, le=CLASS_COUNT - 1),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=40),
) -> dict[str, Any]:
    artifact = _get_digits_artifact()
    labels = np.asarray(artifact["test_labels"], dtype=int)
    indices = np.flatnonzero(labels == label) if label is not None else np.arange(len(labels))
    page_indices = indices[offset:offset + limit]
    return {
        "items": [serialize_sample(artifact, int(index)) for index in page_indices],
        "offset": offset,
        "limit": limit,
        "total": int(len(indices)),
        "has_more": offset + len(page_indices) < len(indices),
    }


@app.post("/dataset/train/preview")
async def dataset_train_preview(request: DatasetTrainPreviewRequest) -> dict[str, Any]:
    data = _get_train_dataset(request)
    split_counts = {
        split_name: int(len(features))
        for split_name, features, _labels in _train_dataset_splits(data)
    }
    labels = np.concatenate([rows[2] for rows in _train_dataset_splits(data)])
    return {
        "sample_count": int(len(labels)),
        "class_count": CLASS_COUNT,
        "class_names": list(data.class_names),
        "feature_names": list(data.feature_names),
        "class_counts": _class_counts(labels),
        "split_counts": split_counts,
        "feature_x": request.feature_x,
        "feature_y": request.feature_y,
        "points": _train_scatter_points(data, request.feature_x, request.feature_y),
    }


@app.post("/dataset/train/samples")
async def dataset_train_samples(request: DatasetTrainSamplesRequest) -> dict[str, Any]:
    data = _get_train_dataset(request)
    rows = _train_dataset_splits(data)
    features = np.vstack([row[1] for row in rows])
    labels = np.concatenate([row[2] for row in rows])
    split_labels = np.concatenate(
        [np.full(len(row[1]), split_name, dtype=object) for split_name, *row in rows]
    )
    indices = np.arange(len(labels))
    if request.class_label is not None:
        indices = indices[labels == request.class_label]
    if request.split != "all":
        indices = indices[split_labels[indices] == request.split]
    page_indices = indices[request.offset:request.offset + request.limit]
    return {
        "items": [
            {
                "index": int(index),
                "label": int(labels[index]),
                "split": str(split_labels[index]),
                "features": [float(value) for value in features[index]],
            }
            for index in page_indices
        ],
        "offset": request.offset,
        "limit": request.limit,
        "total": int(len(indices)),
        "has_more": request.offset + len(page_indices) < len(indices),
    }


@app.get("/predict/sample")
async def predict_sample(index: int | None = Query(default=None, ge=0)) -> dict[str, Any]:
    artifact = _get_digits_artifact()
    test_indices = get_test_indices(artifact)
    selected_index = index if index is not None else test_indices[0]
    if selected_index not in set(test_indices):
        raise HTTPException(status_code=400, detail="Sample index must belong to the test set.")
    try:
        return serialize_sample(artifact, selected_index)
    except IndexError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict")
async def predict(request: PredictRequest) -> dict[str, Any]:
    artifact = _get_digits_artifact()
    if request.sample_index is not None and request.sample_index not in set(
        get_test_indices(artifact)
    ):
        raise HTTPException(status_code=400, detail="Sample index must belong to the test set.")
    try:
        if request.drawing is not None:
            return predict_drawing(artifact, np.asarray(request.drawing, dtype=float))
        return predict_digits(artifact, np.asarray(request.pixels, dtype=float), request.sample_index)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@dataclass
class InspectSession:
    model: SimpleANN
    feature_means: np.ndarray
    feature_stds: np.ndarray
    test_features_raw: np.ndarray
    test_labels: np.ndarray
    activation: str
    hidden_neuron_count: int
    hidden_layers: list[tuple[int, str]]
    epochs_trained: int
    final_training_loss: float
    validation_accuracy: float


INSPECT_SESSIONS: dict[str, InspectSession] = {}


class InspectLayerConfig(BaseModel):
    neurons: int = Field(default=8, ge=1, le=32)
    activation: str = "tanh"

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, value: str) -> str:
        if value not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {value}")
        return value


class InspectBuildConfig(BaseModel):
    activation: str = "tanh"
    hidden_neuron_count: int = Field(default=8, ge=1, le=32)
    hidden_layers: list[InspectLayerConfig] | None = Field(default=None, max_length=4)
    epochs: int = Field(default=200, ge=1, le=5000)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    sample_count: int = Field(default=300, ge=30, le=10000)
    batch_size: int = Field(default=32, ge=1, le=10000)
    random_seed: int = 42

    @field_validator("activation")
    @classmethod
    def validate_activation(cls, value: str) -> str:
        if value not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {value}")
        return value


class InspectForwardRequest(BaseModel):
    model_id: str
    features: list[float] = Field(min_length=8, max_length=8)


def _get_inspect_session(model_id: str) -> InspectSession:
    session = INSPECT_SESSIONS.get(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Inspect model not found. Build one first.")
    return session


def _serialize_forward(session: InspectSession, features_raw: np.ndarray) -> dict[str, Any]:
    features_raw = np.asarray(features_raw, dtype=float).reshape(1, 8)
    features_normalized = (features_raw - session.feature_means) / session.feature_stds
    trace = session.model.trace(features_normalized)
    hidden_pre_activations = trace["hidden_pre_activations"]
    hidden_outputs = trace["hidden_outputs"]
    logits = trace["output_pre_activation"][0]
    probabilities = trace["output_probabilities"][0]

    serialized_layers = []
    for layer_index, ((neuron_count, activation), layer_z, layer_h, weights, biases) in enumerate(
        zip(
            session.hidden_layers,
            hidden_pre_activations,
            hidden_outputs,
            session.model.hidden_layer_weights,
            session.model.hidden_layer_biases_list,
        )
    ):
        neurons = [
            {
                "index": neuron_index,
                "z": float(layer_z[0, neuron_index]),
                "h": float(layer_h[0, neuron_index]),
                "weights": [float(value) for value in weights[:, neuron_index]],
                "bias": float(biases[0, neuron_index]),
            }
            for neuron_index in range(neuron_count)
        ]
        serialized_layers.append(
            {
                "index": layer_index,
                "activation": activation,
                "neuron_count": neuron_count,
                "neurons": neurons,
            }
        )

    output_weights = session.model.output_layer_weights
    output_biases = session.model.output_layer_biases[0]
    output_neurons = [
        {
            "index": index,
            "logit": float(logits[index]),
            "probability": float(probabilities[index]),
            "weights": [float(value) for value in output_weights[:, index]],
            "bias": float(output_biases[index]),
        }
        for index in range(CLASS_COUNT)
    ]
    return {
        "features_raw": [float(value) for value in features_raw[0]],
        "features_normalized": [float(value) for value in features_normalized[0]],
        "activation": session.activation,
        "hidden_neuron_count": session.hidden_neuron_count,
        "hidden_layers": [
            {"neurons": neuron_count, "activation": activation}
            for neuron_count, activation in session.hidden_layers
        ],
        "layers": serialized_layers,
        "hidden": serialized_layers[0]["neurons"],
        "output": output_neurons,
        "predicted_class": int(probabilities.argmax()),
    }


@app.post("/inspect/build")
async def inspect_build(config: InspectBuildConfig) -> dict[str, Any]:
    data = prepare_classification_data(
        random_seed=config.random_seed,
        difficulty=config.difficulty,
        sample_count=config.sample_count,
        input_feature_count=8,
    )
    feature_means = data.training_features_raw.mean(axis=0)
    feature_stds = data.training_features_raw.std(axis=0)
    test_features_raw = (
        data.testing_features * feature_stds + feature_means
    )

    configured_layers = config.hidden_layers or [
        InspectLayerConfig(neurons=config.hidden_neuron_count, activation=config.activation)
    ]
    hidden_layers = [(layer.neurons, layer.activation) for layer in configured_layers]
    first_activation = hidden_layers[0][1]
    model = SimpleANN(
        input_feature_count=8,
        output_class_count=CLASS_COUNT,
        learning_rate=config.learning_rate,
        random_seed=config.random_seed,
        hidden_layers=hidden_layers,  # type: ignore[arg-type]
    )
    for _ in range(config.epochs):
        model.train_epoch(
            data.training_features,
            data.training_one_hot_labels,
            batch_size=config.batch_size,
        )
    validation_loss, validation_accuracy = model.evaluate(
        data.validation_features,
        data.validation_one_hot_labels,
    )

    model_id = str(uuid.uuid4())
    INSPECT_SESSIONS[model_id] = InspectSession(
        model=model,
        feature_means=feature_means,
        feature_stds=feature_stds,
        test_features_raw=test_features_raw,
        test_labels=data.testing_labels,
        activation=first_activation,
        hidden_neuron_count=hidden_layers[0][0],
        hidden_layers=hidden_layers,
        epochs_trained=config.epochs,
        final_training_loss=float(model.loss_history[-1]),
        validation_accuracy=validation_accuracy,
    )
    return {
        "model_id": model_id,
        "activation": first_activation,
        "hidden_neuron_count": hidden_layers[0][0],
        "hidden_layers": [
            {"neurons": neuron_count, "activation": activation}
            for neuron_count, activation in hidden_layers
        ],
        "epochs_trained": config.epochs,
        "final_training_loss": float(model.loss_history[-1]),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "weight_shapes": {
            "hidden": [list(weights.shape) for weights in model.hidden_layer_weights],
            "hidden_to_output": list(model.output_layer_weights.shape),
        },
    }


@app.post("/inspect/forward")
async def inspect_forward(request: InspectForwardRequest) -> dict[str, Any]:
    session = _get_inspect_session(request.model_id)
    return _serialize_forward(session, np.asarray(request.features, dtype=float))


@app.get("/inspect/sample")
async def inspect_sample(model_id: str) -> dict[str, Any]:
    session = _get_inspect_session(model_id)
    sample_index = int(
        session.model.random_generator.integers(0, len(session.test_labels))
    )
    features = session.test_features_raw[sample_index]
    return {
        "features": [float(value) for value in features],
        "label": int(session.test_labels[sample_index]),
        "sample_index": sample_index,
    }


@app.websocket("/ws/train")
async def training_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    training_task: asyncio.Task[None] | None = None
    cancel_event: asyncio.Event | None = None
    receive_task: asyncio.Task[Any] | None = asyncio.create_task(
        websocket.receive_json()
    )

    try:
        while True:
            wait_tasks = [receive_task]
            if training_task is not None:
                wait_tasks.append(training_task)
            done, _ = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if receive_task in done:
                try:
                    message = receive_task.result()
                except _WS_CLOSED:
                    break
                receive_task = asyncio.create_task(websocket.receive_json())
                message_type = message.get("type")

                if message_type == "start":
                    if training_task is not None and not training_task.done():
                        await _ws_send_json(
                            websocket,
                            {
                                "type": "error",
                                "message": "A training run is already active.",
                            },
                        )
                        continue
                    try:
                        config = TrainConfig(**message.get("config", {}))
                    except Exception as error:
                        await _ws_send_json(
                            websocket,
                            {
                                "type": "error",
                                "message": validation_error_message(error),
                            },
                        )
                        continue
                    cancel_event = asyncio.Event()
                    training_task = asyncio.create_task(
                        stream_training(websocket, config, cancel_event)
                    )
                elif message_type == "cancel" and cancel_event is not None:
                    cancel_event.set()
                elif message_type == "reset":
                    if cancel_event is not None:
                        cancel_event.set()
                    if training_task is not None and not training_task.done():
                        training_task.cancel()

            if training_task is not None and training_task in done:
                try:
                    await training_task
                except (asyncio.CancelledError, *_WS_CLOSED):
                    pass
                except Exception as error:
                    await _ws_send_json(
                        websocket, {"type": "error", "message": str(error)}
                    )
                training_task = None
                cancel_event = None
    except _WS_CLOSED:
        pass
    finally:
        if cancel_event is not None:
            cancel_event.set()
        if training_task is not None:
            training_task.cancel()
        if receive_task is not None:
            receive_task.cancel()
