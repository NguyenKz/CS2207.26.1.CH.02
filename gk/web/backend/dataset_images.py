"""Export and index normalized MNIST images for the Dataset browser."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .digits_config import DATASET_DIR, DATASET_PATH, PIXEL_SIZE, SPLIT_PATH


IMAGE_DIR = DATASET_DIR / "images"
MANIFEST_PATH = IMAGE_DIR / "manifest.json"
SPLIT_NAMES = ("training", "validation", "testing")


class DatasetImagesError(RuntimeError):
    """Raised when the local image export is unavailable or invalid."""


def _load_split_indices() -> tuple[set[int], set[int]]:
    try:
        split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        return set(map(int, split["fit_indices"])), set(map(int, split["validation_indices"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetImagesError(f"Could not read MNIST split metadata: {error}") from error


def _save_image(values: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(
        destination,
        format="WEBP",
        lossless=True,
        method=6,
    )


def export_dataset_images() -> dict[str, Any]:
    """Export all 70,000 normalized samples and write the browser manifest."""
    if not DATASET_PATH.exists():
        raise DatasetImagesError(f"Normalized dataset is missing: {DATASET_PATH}")

    fit_indices, validation_indices = _load_split_indices()
    with np.load(DATASET_PATH) as data:
        train_pixels = np.asarray(data["X_train_pixels"], dtype=np.float32)
        train_labels = np.asarray(data["y_train"], dtype=int)
        test_pixels = np.asarray(data["X_test_pixels"], dtype=np.float32)
        test_labels = np.asarray(data["y_test"], dtype=int)

    if train_pixels.shape[1:] != (PIXEL_SIZE, PIXEL_SIZE):
        raise DatasetImagesError(f"Unexpected train image shape: {train_pixels.shape}")
    if test_pixels.shape[1:] != (PIXEL_SIZE, PIXEL_SIZE):
        raise DatasetImagesError(f"Unexpected test image shape: {test_pixels.shape}")
    if len(fit_indices) + len(validation_indices) != len(train_pixels):
        raise DatasetImagesError("Train split indices do not cover all training images.")

    items: list[dict[str, Any]] = []
    class_counts = [0] * 10
    split_counts = {name: 0 for name in SPLIT_NAMES}

    for index, (pixels, label) in enumerate(zip(train_pixels, train_labels)):
        split_name = "training" if index in fit_indices else "validation"
        label_value = int(label)
        image_path = f"train/{label_value}/{index}.webp"
        _save_image(pixels, IMAGE_DIR / image_path)
        items.append({
            "id": f"train:{index}",
            "source": "train",
            "index": index,
            "split": split_name,
            "label": label_value,
            "image": image_path,
        })
        class_counts[label_value] += 1
        split_counts[split_name] += 1

    for index, (pixels, label) in enumerate(zip(test_pixels, test_labels)):
        label_value = int(label)
        image_path = f"test/{label_value}/{index}.webp"
        _save_image(pixels, IMAGE_DIR / image_path)
        items.append({
            "id": f"test:{index}",
            "source": "test",
            "index": index,
            "split": "testing",
            "label": label_value,
            "image": image_path,
        })
        class_counts[label_value] += 1
        split_counts["testing"] += 1

    manifest = {
        "version": 1,
        "pixel_size": PIXEL_SIZE,
        "sample_count": len(items),
        "class_counts": class_counts,
        "split_counts": split_counts,
        "items": items,
    }
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    load_dataset_catalog.cache_clear()
    return manifest


@lru_cache(maxsize=1)
def load_dataset_catalog() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise DatasetImagesError(
            f"Dataset image export is missing: {MANIFEST_PATH}. "
            "Run `python -m gk.web.backend.dataset_images`."
        )
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        items = tuple(manifest["items"])
        class_items = {
            label: tuple(item for item in items if int(item["label"]) == label)
            for label in range(10)
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetImagesError(f"Could not read dataset image manifest: {error}") from error
    return {"manifest": manifest, "items": items, "class_items": class_items}


def dataset_image_url(item: dict[str, Any]) -> str:
    return f"/dataset/images/{item['image']}"


def public_dataset_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "source": item["source"],
        "index": int(item["index"]),
        "split": item["split"],
        "label": int(item["label"]),
        "image_url": dataset_image_url(item),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = export_dataset_images()
    print(
        f"Exported {manifest['sample_count']:,} images to {IMAGE_DIR} "
        f"({manifest['pixel_size']}x{manifest['pixel_size']}, WebP)."
    )


if __name__ == "__main__":
    main()
