"""OrganAMNIST: four organs from the official 28x28 grayscale axial CT archive."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .image_data import ArrayImageDataset, balanced_loaders, download_file


# MedMNIST's official label dictionary: keep the requested organ order when
# remapping to QCNN outputs 0, 1, 2, 3. Labels 4 and 7 identify the LEFT
# kidney and LEFT lung respectively; display names are kept in English.
ORGAN_CLASSES = ((4, "Left kidney"), (7, "Left lung"), (9, "Pancreas"), (6, "Liver"))
FILENAME = "organamnist.npz"
DOWNLOAD_URL = "https://zenodo.org/records/10519652/files/organamnist.npz?download=1"


def get_dataset_config(name: str = "organamnist") -> dict:
    """Record source labels and anatomical names in model-output order."""
    if name != "organamnist":
        raise ValueError(f"Unknown OrganAMNIST dataset: {name}")
    return {
        "original_labels": [label for label, _ in ORGAN_CLASSES],
        "class_names": [name for _, name in ORGAN_CLASSES],
        "source": "https://medmnist.com/",
        "native_image_size": 28,
        "n_channels": 1,
        "normalization": "pixel / 255",
        "split_protocol": "official train/val/test; balanced sampling within each split",
    }


def _read_split(data_dir: Path, *, split: str):
    """Read one official pool without mixing CT scans across its boundaries.

    Use the small native archive directly, so no extra dataset library is needed.
    In particular, the 64/128/224 variants are never downloaded or resized to 28.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown OrganAMNIST split: {split}")
    path = download_file(DOWNLOAD_URL, Path(data_dir) / "OrganAMNIST" / FILENAME)
    with np.load(path, allow_pickle=False) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"]
    # Validate the real archive layout before normalizing or flattening. This
    # rejects a larger-resolution archive accidentally copied to the same path.
    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
        raise ValueError(f"Invalid OrganAMNIST {split} images: expected uint8 shape (N, 28, 28).")
    if (labels.shape not in {(len(images),), (len(images), 1)}
            or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(labels, range(11)).all()):
        raise ValueError(f"Invalid OrganAMNIST {split} labels: expected one label in 0..10 per image.")
    return images.astype(np.float32) / 255.0, labels.reshape(-1)


def _dataset(data_dir, image_size, *, split):
    labels = get_dataset_config()["original_labels"]
    return ArrayImageDataset(
        *_read_split(Path(data_dir), split=split), image_size,
        {label: target for target, label in enumerate(labels)},
    )


def get_test_dataset(*, dataset_name, data_dir, image_size):
    """Expose only the official test pool to analysis notebooks."""
    get_dataset_config(dataset_name)
    return _dataset(data_dir, image_size, split="test")


def load_data(
    *, dataset_name: str, data_dir: Path, image_size: int, seed: int,
    batch_size: int, train_per_class: int, val_per_class: int, test_per_class: int,
):
    """Sample each organ equally, preserving all three official source pools."""
    get_dataset_config(dataset_name)
    # Unlike MNIST, OrganAMNIST already supplies a validation pool. Keeping it
    # separate preserves the official CT-scan partition and prevents leakage.
    train, validation, test = (
        _dataset(data_dir, image_size, split=split) for split in ("train", "val", "test")
    )
    return balanced_loaders(
        train, validation, test, seed=seed, batch_size=batch_size,
        train_per_class=train_per_class, val_per_class=val_per_class,
        test_per_class=test_per_class,
    )
