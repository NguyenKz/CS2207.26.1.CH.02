"""Realtime FastAPI server for the ANN training lesson."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
