"""Standard MNIST, restricted to the single four-class task 0, 1, 2, 3."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torchvision.datasets as tvds

from .image_data import ArrayImageDataset, balanced_loaders


def get_dataset_config(name: str = "mnist") -> dict:
    """Keep digit order identical to the four model-output labels."""
    if name != "mnist":
        raise ValueError(f"Unknown MNIST dataset: {name}")
    return {
        "original_labels": [0, 1, 2, 3],
        "class_names": ["0", "1", "2", "3"],
        "source": "https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html",
        "native_image_size": 28,
        "n_channels": 1,
        "normalization": "pixel / 255",
        "split_protocol": "official train/test; disjoint balanced train/validation",
    }


def _read_split(data_dir: Path, *, train: bool):
    """Reuse local IDX files; request a torchvision download if data are missing."""
    try:
        source = tvds.MNIST(root=str(data_dir), train=train, download=False)
    except RuntimeError:
        source = tvds.MNIST(root=str(data_dir), train=train, download=True)
    # The source images are uint8, 28x28 and grayscale. Fixed scaling never fits
    # statistics on validation/test data. Other digits remain outside the sampler.
    return np.asarray(source.data, dtype=np.float32) / 255.0, np.asarray(source.targets)


def _dataset(data_dir, image_size, *, train):
    labels = get_dataset_config()["original_labels"]
    return ArrayImageDataset(
        *_read_split(Path(data_dir), train=train), image_size,
        {label: target for target, label in enumerate(labels)},
    )


def get_test_dataset(*, dataset_name, data_dir, image_size):
    """Expose official test images to analysis with identical preprocessing."""
    get_dataset_config(dataset_name)
    return _dataset(data_dir, image_size, train=False)


def load_data(
    *, dataset_name: str, data_dir: Path, image_size: int, seed: int,
    batch_size: int, train_per_class: int, val_per_class: int, test_per_class: int,
):
    """Build balanced splits; validation never overlaps the training selection."""
    get_dataset_config(dataset_name)
    train = _dataset(data_dir, image_size, train=True)
    test = _dataset(data_dir, image_size, train=False)
    # Sharing this object tells balanced_loaders to start validation after the
    # training indices in the same seeded per-class permutations.
    return balanced_loaders(
        train, train, test, seed=seed, batch_size=batch_size,
        train_per_class=train_per_class, val_per_class=val_per_class,
        test_per_class=test_per_class,
    )
