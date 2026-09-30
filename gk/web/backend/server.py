"""Realtime FastAPI server for the ANN training lesson."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from .ann_core import (
    CLASS_COUNT,
    DEFAULT_SAMPLE_COUNT,
    SAMPLES_PER_CLASS,
    SUPPORTED_ACTIVATIONS,
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
    delay_seconds: float = Field(default=0.001, ge=0.0, le=2.0)
    learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    sample_count: int = Field(default=DEFAULT_SAMPLE_COUNT, ge=30, le=10000)
    batch_size: int = Field(default=32, ge=1, le=10000)
    train_percentage: float = Field(default=60.0, gt=0.0, lt=100.0)
    validation_percentage: float = Field(default=15.0, gt=0.0, lt=100.0)
    test_percentage: float = Field(default=15.0, gt=0.0, lt=100.0)
    holdout_percentage: float = Field(default=10.0, gt=0.0, lt=100.0)
    hidden_neuron_count: int = Field(default=8, ge=1, le=32)
    random_seed: int = 42
    early_stopping: bool = False
    early_stopping_patience: int = Field(default=40, ge=1, le=500)
    early_stopping_min_delta: float = Field(default=0.001, ge=0.0, le=1.0)
    activations: list[str] = Field(default_factory=lambda: list(SUPPORTED_ACTIVATIONS))

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
        split_total = (
            self.train_percentage
            + self.validation_percentage
            + self.test_percentage
            + self.holdout_percentage
        )
        if abs(split_total - 100.0) > 1e-6:
            raise ValueError("all dataset percentages must sum to 100")
        return self


def validation_error_message(error: Exception) -> str:
    return f"Invalid training configuration: {error}"


async def train_activation(
    activation_name: str,
    config: TrainConfig,
    data: Any,
    event_queue: asyncio.Queue[dict[str, Any]],
    cancel_event: asyncio.Event,
) -> dict[str, Any]:
    model = SimpleANN(
        input_feature_count=4,
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
        if should_stop_early:
            stopped_early = True
            break
        await asyncio.sleep(config.delay_seconds)

    test_accuracy = float(
        (model.predict(data.testing_features) == data.testing_labels).mean()
    )
    holdout_accuracy = float(
        (model.predict(data.holdout_features) == data.holdout_labels).mean()
    )
    return {
        "activation": activation_name,
        "epochs_completed": completed_epochs,
        "test_accuracy": test_accuracy,
        "holdout_accuracy": holdout_accuracy,
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
        holdout_percentage=config.holdout_percentage,
    )
    event_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    await websocket.send_json(
        {
            "type": "run_started",
            "run_id": run_id,
            "activations": config.activations,
            "total_epochs": config.epochs,
            "delay_seconds": config.delay_seconds,
            "difficulty": config.difficulty,
            "sample_count": config.sample_count,
            "class_count": CLASS_COUNT,
            "samples_per_class": SAMPLES_PER_CLASS,
            "batch_size": config.batch_size,
            "train_percentage": config.train_percentage,
            "validation_percentage": config.validation_percentage,
            "test_percentage": config.test_percentage,
            "holdout_percentage": config.holdout_percentage,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": config.early_stopping_patience,
            "early_stopping_min_delta": config.early_stopping_min_delta,
            "dataset": {
                "training": list(data.training_features.shape),
                "validation": list(data.validation_features.shape),
                "testing": list(data.testing_features.shape),
                "holdout": list(data.holdout_features.shape),
            },
        }
    )

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
            await websocket.send_json({"run_id": run_id, **event})

    await asyncio.gather(*completion_tasks)
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    status = "cancelled" if cancel_event.is_set() else "completed"
    await websocket.send_json(
        {
            "type": "run_cancelled" if cancel_event.is_set() else "run_completed",
            "run_id": run_id,
            "status": status,
            "duration_ms": duration_ms,
            "results": sorted(results, key=lambda result: result["activation"]),
        }
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
    holdout_features_raw: np.ndarray
    holdout_labels: np.ndarray
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
    features: list[float] = Field(min_length=4, max_length=4)


def _get_inspect_session(model_id: str) -> InspectSession:
    session = INSPECT_SESSIONS.get(model_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Inspect model not found. Build one first.")
    return session


def _serialize_forward(session: InspectSession, features_raw: np.ndarray) -> dict[str, Any]:
    features_raw = np.asarray(features_raw, dtype=float).reshape(1, 4)
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
    )
    feature_means = data.training_features_raw.mean(axis=0)
    feature_stds = data.training_features_raw.std(axis=0)
    holdout_features_raw = (
        data.holdout_features * feature_stds + feature_means
    )

    configured_layers = config.hidden_layers or [
        InspectLayerConfig(neurons=config.hidden_neuron_count, activation=config.activation)
    ]
    hidden_layers = [(layer.neurons, layer.activation) for layer in configured_layers]
    first_activation = hidden_layers[0][1]
    model = SimpleANN(
        input_feature_count=4,
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
        holdout_features_raw=holdout_features_raw,
        holdout_labels=data.holdout_labels,
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
        session.model.random_generator.integers(0, len(session.holdout_labels))
    )
    features = session.holdout_features_raw[sample_index]
    return {
        "features": [float(value) for value in features],
        "label": int(session.holdout_labels[sample_index]),
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
                except WebSocketDisconnect:
                    break
                receive_task = asyncio.create_task(websocket.receive_json())
                message_type = message.get("type")

                if message_type == "start":
                    if training_task is not None and not training_task.done():
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "A training run is already active.",
                            }
                        )
                        continue
                    try:
                        config = TrainConfig(**message.get("config", {}))
                    except Exception as error:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": validation_error_message(error),
                            }
                        )
                        continue
                    cancel_event = asyncio.Event()
                    training_task = asyncio.create_task(
                        stream_training(websocket, config, cancel_event)
                    )
                elif message_type == "cancel" and cancel_event is not None:
                    cancel_event.set()
                elif message_type == "reset" and cancel_event is not None:
                    cancel_event.set()

            if training_task is not None and training_task in done:
                try:
                    await training_task
                except Exception as error:
                    await websocket.send_json(
                        {"type": "error", "message": str(error)}
                    )
                training_task = None
                cancel_event = None
    except WebSocketDisconnect:
        pass
    finally:
        if cancel_event is not None:
            cancel_event.set()
        if training_task is not None:
            training_task.cancel()
        if receive_task is not None:
            receive_task.cancel()
