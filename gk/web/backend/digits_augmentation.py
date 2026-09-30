"""Small NumPy-only training augmentations for handwritten digits."""

from __future__ import annotations

import numpy as np

from .digits_config import PIXEL_COUNT
from .digits_predict import _resize_area_average


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an image without wrapping pixels around the opposite edge."""
    height, width = image.shape
    result = np.zeros_like(image)
    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy) if dy >= 0 else height
    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx) if dx >= 0 else width
    target_y_start = max(0, dy)
    target_y_end = target_y_start + max(0, source_y_end - source_y_start)
    target_x_start = max(0, dx)
    target_x_end = target_x_start + max(0, source_x_end - source_x_start)
    if target_y_end > target_y_start and target_x_end > target_x_start:
        result[target_y_start:target_y_end, target_x_start:target_x_end] = image[
            source_y_start:source_y_end, source_x_start:source_x_end
        ]
    return result


def _scale_about_center(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize around the image center with nearest-neighbor sampling."""
    size = image.shape[0]
    center = (size - 1) / 2.0
    coordinates = np.rint((np.arange(size) - center) / scale + center).astype(int)
    result = np.zeros_like(image)
    valid = (coordinates >= 0) & (coordinates < size)
    result[np.ix_(valid, valid)] = image[np.ix_(coordinates[valid], coordinates[valid])]
    return result


def _change_stroke(image: np.ndarray, mode: str) -> np.ndarray:
    """Make the binary ink mildly thicker or thinner without changing labels."""
    mask = image >= max(8.0, float(image.max()) * 0.25)
    if mode == "thick":
        result = image.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                result = np.maximum(result, _shift(image, dx, dy))
        return result
    if mode == "thin":
        neighbors = np.zeros(mask.shape, dtype=np.int16)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbors += (_shift(mask.astype(np.uint8), dx, dy) > 0).astype(np.int16)
        return np.where(mask & (neighbors >= 3), image, 0.0)
    raise ValueError(f"Unknown stroke variant: {mode}")


def _normalize_variant(image: np.ndarray, pixel_size: int) -> np.ndarray:
    """Fast equivalent of the UI normalization for generated MNIST variants.

    Augmentations do not add isolated noise, so the foreground bounding box is
    the same as keeping the largest connected component. The original UI
    function remains the source of truth for user drawings and raw dataset
    creation.
    """
    maximum = float(image.max())
    threshold = max(1.0, maximum * 0.5)
    binary = image >= threshold
    rows, columns = np.where(binary)
    if rows.size == 0:
        return np.zeros(pixel_size * pixel_size, dtype=np.float32)
    cropped = np.where(
        binary[rows.min():rows.max() + 1, columns.min():columns.max() + 1],
        maximum,
        0.0,
    )
    height, width = cropped.shape
    square_size = max(height, width)
    square = np.zeros((square_size, square_size), dtype=float)
    top = (square_size - height) // 2
    left = (square_size - width) // 2
    square[top:top + height, left:left + width] = cropped
    resized = _resize_area_average(square, pixel_size, pixel_size)
    resized_maximum = float(resized.max())
    return np.rint(np.clip(resized * (16.0 / resized_maximum), 0.0, 16.0)).astype(np.float32).reshape(-1)


def _variant(
    image: np.ndarray,
    index: int,
    seed: int,
    shift_pixels: int,
    scale_range: tuple[float, float],
    stroke_variants: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed + index * 7919)
    dx, dy = rng.integers(-shift_pixels, shift_pixels + 1, size=2)
    scale = float(rng.uniform(*scale_range))
    transformed = _scale_about_center(_shift(image, int(dx), int(dy)), scale)
    if stroke_variants:
        transformed = _change_stroke(transformed, "thick" if index % 2 else "thin")
    return transformed


def build_augmented_variants(
    raw_images: np.ndarray,
    *,
    factor: int = 3,
    shift_pixels: int = 2,
    stroke_variants: bool = True,
    scale_range: tuple[float, float] = (0.90, 1.10),
    seed: int = 42,
) -> np.ndarray:
    """Return normalized variants only; callers keep original train samples separately."""
    images = np.asarray(raw_images, dtype=float)
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("MNIST augmentation expects images with shape (n, 28, 28).")
    if factor < 1:
        raise ValueError("Augmentation factor must be at least 1.")
    if shift_pixels < 0 or scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
        raise ValueError("Invalid augmentation shift or scale range.")
    if factor == 1 or len(images) == 0:
        return np.empty((0, PIXEL_COUNT), dtype=np.float32)

    variant_count = len(images) * (factor - 1)
    normalized = np.empty((variant_count, PIXEL_COUNT), dtype=np.float32)
    output_index = 0
    for image_index, image in enumerate(images):
        for variant_index in range(1, factor):
            augmented = _variant(
                image,
                variant_index,
                seed + image_index,
                shift_pixels,
                scale_range,
                stroke_variants,
            )
            normalized[output_index] = _normalize_variant(augmented, int(np.sqrt(PIXEL_COUNT)))
            output_index += 1
    return normalized


def repeat_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    """Match labels to original samples followed by their generated variants."""
    values = np.asarray(labels, dtype=int)
    if factor < 1:
        raise ValueError("Augmentation factor must be at least 1.")
    return np.concatenate((values, np.repeat(values, factor - 1)))
