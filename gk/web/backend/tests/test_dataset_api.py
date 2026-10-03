import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from gk.web.backend.ann_core import SUPPORTED_ACTIVATIONS, prepare_classification_data
from gk.web.backend.server import TrainConfig, app


client = TestClient(app)


def train_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "difficulty": 0.7,
        "sample_count": 300,
        "train_percentage": 70,
        "validation_percentage": 15,
        "test_percentage": 15,
        "input_feature_count": 8,
        "random_seed": 42,
    }
    payload.update(overrides)
    return payload


def test_predict_dataset_metadata_and_class_page() -> None:
    metadata = client.get("/dataset/predict/meta")
    assert metadata.status_code == 200
    body = metadata.json()
    assert body["dataset"]["sample_count"] == 70000
    assert body["dataset"]["input_shape"] == [28, 28]
    assert body["dataset"]["feature_count"] == 784
    assert sum(body["class_counts"]) == body["dataset"]["sample_count"]
    assert body["split_counts"] == {"training": 48000, "validation": 12000, "testing": 10000}
    assert len(body["representatives"]) == 10

    predict_meta = client.get("/predict/meta")
    assert predict_meta.status_code == 200
    assert predict_meta.json()["training"] == {
        "fit_samples": 48000,
        "validation_samples": 12000,
        "test_samples": 10000,
        "total_samples": 70000,
        "augmented_fit_samples": 240000,
        "augmentation_factor": 5,
        "epochs": 700,
        "batch_size": 256,
    }

    page = client.get("/dataset/predict/samples", params={"label": 7, "limit": 40})
    assert page.status_code == 200
    page_body = page.json()
    assert len(page_body["items"]) == 40
    assert all(item["label"] == 7 for item in page_body["items"])
    assert all(item["image_url"].endswith(".webp") for item in page_body["items"])
    image = client.get(page_body["items"][0]["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/webp"


def test_train_preview_is_deterministic_and_filters_samples() -> None:
    payload = train_payload(feature_x=0, feature_y=1)
    first = client.post("/dataset/train/preview", json=payload)
    second = client.post("/dataset/train/preview", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert sum(first.json()["split_counts"].values()) == 300
    assert set(first.json()["split_counts"]) == {"training", "validation", "testing"}

    page = client.post(
        "/dataset/train/samples",
        json=train_payload(class_label=3, split="training", limit=40),
    )
    assert page.status_code == 200
    items = page.json()["items"]
    assert items
    assert all(item["label"] == 3 and item["split"] == "training" for item in items)


def test_train_dataset_rejects_invalid_filter_and_axes() -> None:
    invalid_split = client.post(
        "/dataset/train/samples",
        json=train_payload(split="not-a-split"),
    )
    assert invalid_split.status_code == 422

    invalid_holdout = client.post(
        "/dataset/train/samples",
        json=train_payload(split="holdout"),
    )
    assert invalid_holdout.status_code == 422

    invalid_axes = client.post(
        "/dataset/train/preview",
        json=train_payload(feature_x=2, feature_y=2),
    )
    assert invalid_axes.status_code == 422


def test_train_runtime_has_three_splits_and_no_softplus_delay() -> None:
    config = TrainConfig()
    assert "delay_seconds" not in config.model_dump()
    assert "softplus" not in SUPPORTED_ACTIVATIONS

    data = prepare_classification_data(sample_count=300, input_feature_count=8)
    assert len(data.training_labels) == 210
    assert len(data.validation_labels) == 45
    assert len(data.testing_labels) == 45
    assert not hasattr(data, "holdout_features")

    try:
        TrainConfig(activations=["softplus"])
    except ValueError:
        pass
    else:
        raise AssertionError("Softplus should not be accepted by TrainConfig")


def main() -> None:
    test_predict_dataset_metadata_and_class_page()
    test_train_preview_is_deterministic_and_filters_samples()
    test_train_dataset_rejects_invalid_filter_and_axes()
    test_train_runtime_has_three_splits_and_no_softplus_delay()
    print("PASS: dataset API regression checks")


if __name__ == "__main__":
    main()
