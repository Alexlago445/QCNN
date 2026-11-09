"""Shared adapters for native 28x28 grayscale images and QCNN input grids."""
from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, Subset

from .fashion_mnist import balanced_indices


# -----------------------------------------------------------------------------
# Local download cache for dataset archives and numeric image files.

def download_file(url: str, destination: Path) -> Path:
    """Cache downloads atomically; an interrupted download is never reused."""
    destination = Path(destination)
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish the final filename only after the entire response has been written.
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}. Download it manually to {destination}."
        ) from exc
    return destination


# -----------------------------------------------------------------------------
# Adapt square image arrays to the flattened tensor interface used by every QCNN.
# Dataset-specific loaders handle value normalization before creating this view.

class ArrayImageDataset(Dataset):
    """Float images in [0,1], resized and flattened for the existing encoders.

    Original labels remain in targets for sampling. The balanced samplers select
    only labels in class_map before __getitem__ remaps them to model outputs.
    """

    def __init__(self, images, targets, image_size: int, class_map: dict[int, int], *, native_size=28):
        if image_size not in {8, 16, 28}:
            raise ValueError(f"Unsupported image size: {image_size}; expected 8, 16 or 28.")
        if native_size != 28:
            raise ValueError(f"Unsupported native image size: {native_size}; expected 28.")
        images = np.asarray(images, dtype=np.float32).reshape(-1, native_size, native_size)
        self.targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        if (len(images) != len(self.targets) or not np.isfinite(images).all()
                or (images < 0).any() or (images > 1).any()):
            raise ValueError("Images and labels must match; pixels must be finite values in [0,1].")
        # Interpolation expects [N, channels, height, width]; each image has one channel.
        self.images = torch.from_numpy(images.copy()).unsqueeze(1)
        if image_size != native_size:
            # Match the encoder's input grid. Antialiasing limits high-frequency
            # artifacts when reducing native 28x28 images; resizing adds no samples.
            self.images = functional.interpolate(
                self.images, size=(image_size, image_size), mode="bilinear",
                align_corners=False, antialias=image_size < native_size,
            )
        # Reupload receives 64 features; E1/E3 receive 256. The native 28x28
        # analysis view retains all 784 pixels and is not a QCNN input grid.
        self.images = self.images.flatten(1)
        self.class_map = class_map

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.class_map[int(self.targets[index])]


# -----------------------------------------------------------------------------
# Select equal class counts using seeded permutations of each source class.

def balanced_loaders(
    train, validation, test, *, seed, batch_size,
    train_per_class, val_per_class, test_per_class,
):
    """Use the same seeded, balanced sampler as Fashion-MNIST."""
    if batch_size <= 0 or min(train_per_class, val_per_class, test_per_class) <= 0:
        raise ValueError("Batch size and samples per class must be positive.")
    loaders = []
    # A shared train/validation object means both selections use one source pool.
    # Reuse the same seeded class permutations and offset validation past training
    # to guarantee disjoint selections. Separate pools require no such offset.
    for dataset, count, offset, shuffle in (
        (train, train_per_class, 0, True),
        (validation, val_per_class, train_per_class if validation is train else 0, False),
        (test, test_per_class, 0, False),
    ):
        indices = balanced_indices(dataset, dataset.class_map, count, offset=offset, seed=seed)
        loaders.append(DataLoader(
            Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle,
        ))
    return tuple(loaders)
