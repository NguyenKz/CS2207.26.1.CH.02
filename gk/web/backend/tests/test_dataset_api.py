import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from gk.web.backend.server import app


client = TestClient(app)


def train_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "difficulty": 0.7,
        "sample_count": 300,
        "train_percentage": 60,
        "validation_percentage": 15,
        "test_percentage": 15,
        "holdout_percentage": 10,
        "input_feature_count": 8,
        "random_seed": 42,
    }
    payload.update(overrides)
    return payload


def test_predict_dataset_metadata_and_class_page() -> None:
    metadata = client.get("/dataset/predict/meta")
    assert metadata.status_code == 200
    body = metadata.json()
    assert sum(body["class_counts"]) == body["dataset"]["test_sample_count"]
    assert len(body["representatives"]) == 10

    page = client.get("/dataset/predict/samples", params={"label": 7, "limit": 40})
    assert page.status_code == 200
    page_body = page.json()
    assert len(page_body["items"]) == 40
    assert all(item["label"] == 7 for item in page_body["items"])


def test_train_preview_is_deterministic_and_filters_samples() -> None:
    payload = train_payload(feature_x=0, feature_y=1)
    first = client.post("/dataset/train/preview", json=payload)
    second = client.post("/dataset/train/preview", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert sum(first.json()["split_counts"].values()) == 300

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

    invalid_axes = client.post(
        "/dataset/train/preview",
        json=train_payload(feature_x=2, feature_y=2),
    )
    assert invalid_axes.status_code == 422


def main() -> None:
    test_predict_dataset_metadata_and_class_page()
    test_train_preview_is_deterministic_and_filters_samples()
    test_train_dataset_rejects_invalid_filter_and_axes()
    print("PASS: dataset API regression checks")


if __name__ == "__main__":
    main()
