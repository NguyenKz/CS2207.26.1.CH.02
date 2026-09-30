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


def _rotate_about_center(image: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate around center with nearest-neighbor sampling (no SciPy)."""
    if abs(degrees) < 1e-6:
        return image
    size = image.shape[0]
    center = (size - 1) / 2.0
    theta = np.deg2rad(degrees)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    yy, xx = np.indices(image.shape)
    x = xx - center
    y = yy - center
    src_x = np.rint(cos_t * x + sin_t * y + center).astype(int)
    src_y = np.rint(-sin_t * x + cos_t * y + center).astype(int)
    result = np.zeros_like(image)
    valid = (src_x >= 0) & (src_x < size) & (src_y >= 0) & (src_y < size)
    result[valid] = image[src_y[valid], src_x[valid]]
    return result


def _box_blur3(image: np.ndarray) -> np.ndarray:
    """3×3 mean filter via padded sums."""
    size = image.shape[0]
    padded = np.pad(image.astype(float), 1, mode="edge")
    blurred = np.zeros_like(image, dtype=float)
    for dy in range(3):
        for dx in range(3):
            blurred += padded[dy:dy + size, dx:dx + size]
    return blurred / 9.0


def _sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask: emphasize ink edges after soft blur."""
    maximum = float(image.max())
    if maximum <= 0 or amount <= 0:
        return image
    sharpened = image.astype(float) + amount * (image.astype(float) - _box_blur3(image))
    return np.clip(sharpened, 0.0, maximum)


def _change_stroke(image: np.ndarray, mode: str) -> np.ndarray:
    """Make the binary ink thicker or thinner without changing labels."""
    mask = image >= max(8.0, float(image.max()) * 0.25)
    if mode == "thick":
        result = image.copy()
        # Two dilate passes so augmented strokes closer match freehand canvas mass.
        for _ in range(2):
            dilated = result.copy()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    dilated = np.maximum(dilated, _shift(result, dx, dy))
            result = dilated
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
    rotate_degrees: float,
    sharpen: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed + index * 7919)
    dx, dy = rng.integers(-shift_pixels, shift_pixels + 1, size=2)
    scale = float(rng.uniform(*scale_range))
    transformed = _scale_about_center(_shift(image, int(dx), int(dy)), scale)
    if rotate_degrees > 0:
        angle = float(rng.uniform(-rotate_degrees, rotate_degrees))
        transformed = _rotate_about_center(transformed, angle)
    if stroke_variants:
        # Cycle thick / thin / keep so canvas-like fat strokes appear often.
        stroke_mode = ("thick", "thin", "thick")[index % 3]
        transformed = _change_stroke(transformed, stroke_mode)
    if sharpen and rng.random() < 0.5:
        transformed = _sharpen(transformed, amount=float(rng.uniform(0.6, 1.4)))
    return transformed


def build_augmented_variants(
    raw_images: np.ndarray,
    *,
    factor: int = 5,
    shift_pixels: int = 3,
    stroke_variants: bool = True,
    scale_range: tuple[float, float] = (0.85, 1.15),
    rotate_degrees: float = 15.0,
    sharpen: bool = True,
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
    if rotate_degrees < 0:
        raise ValueError("rotate_degrees must be non-negative.")
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
                rotate_degrees,
                sharpen,
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


if __name__ == "__main__":
    digit = np.zeros((28, 28), dtype=float)
    digit[6:22, 10:18] = 200
    rotated = _rotate_about_center(digit, 12)
    sharpened = _sharpen(digit, 1.0)
    thick = _change_stroke(digit, "thick")
    assert rotated.sum() > 0 and not np.allclose(rotated, digit)
    assert sharpened.max() >= digit.max() * 0.9
    assert (thick > 0).sum() > (digit > 0).sum()
    batch = np.stack([digit, digit])
    variants = build_augmented_variants(batch, factor=3, seed=0)
    assert variants.shape == (4, PIXEL_COUNT)
    assert np.all((variants >= 0) & (variants <= 16))
    print("ok", variants.shape, float(variants.mean()))
