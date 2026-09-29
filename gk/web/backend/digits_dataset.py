"""MNIST download and the preprocessing shared by training and the UI."""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from .digits_predict import preprocess_dataset_pixels


MNIST_CACHE = Path('/tmp/ann-mnist')
MNIST_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist'


def _download_mnist_file(filename: str) -> Path:
    destination = MNIST_CACHE / filename
    if not destination.exists():
        MNIST_CACHE.mkdir(parents=True, exist_ok=True)
        print(f'Downloading {filename}...')
        urllib.request.urlretrieve(f'{MNIST_URL}/{filename}', destination)
    return destination


def _read_mnist_images(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as stream:
        magic, count, height, width = struct.unpack('>IIII', stream.read(16))
        if magic != 2051 or (height, width) != (28, 28):
            raise ValueError(f'Invalid MNIST image file: {path}')
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    return values.reshape(count, height, width).astype(float)


def _read_mnist_labels(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as stream:
        magic, count = struct.unpack('>II', stream.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid MNIST label file: {path}')
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f'MNIST label count does not match: {path}')
    return values.astype(int)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = _read_mnist_images(_download_mnist_file('train-images-idx3-ubyte.gz'))
    train_labels = _read_mnist_labels(_download_mnist_file('train-labels-idx1-ubyte.gz'))
    test_images = _read_mnist_images(_download_mnist_file('t10k-images-idx3-ubyte.gz'))
    test_labels = _read_mnist_labels(_download_mnist_file('t10k-labels-idx1-ubyte.gz'))
    if len(train_images) != len(train_labels) or len(test_images) != len(test_labels):
        raise ValueError('MNIST image and label counts do not match.')
    return train_images, train_labels, test_images, test_labels


def normalize_mnist_images(
    train_images: np.ndarray,
    test_images: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same crop-square-resize transform used by the freehand UI."""
    print('Normalizing MNIST images with the same crop-square-resize pipeline as the UI...')
    train_features = np.stack([preprocess_dataset_pixels(image) for image in train_images])
    test_features = np.stack([preprocess_dataset_pixels(image) for image in test_images])
    return train_features, test_features
