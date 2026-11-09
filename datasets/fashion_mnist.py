from __future__ import annotations

from pathlib import Path

import numpy as np
import torchvision.datasets as tvds
import torchvision.transforms as tvtf
from torch.utils.data import DataLoader, Dataset, Subset

# Classes are listed in output-label order (0, 1, 2, 3) for each Fashion-MNIST subset.
FASHION4_GROUPS = {
    "fashion4_1": ((0, "T-shirt"), (1, "Trouser"), (7, "Sneaker"), (8, "Bag")),
    "fashion4_2": ((2, "Pullover"), (5, "Sandal"), (1, "Trouser"), (4, "Coat")),
    "fashion4_3": ((3, "Dress"), (9, "Ankle boot"), (6, "Shirt"), (8, "Bag")),
}

#--------------------- UTILITY FUNCTIONS ------------------------------#


# Get the configuration for a specific four-class Fashion-MNIST subset.
def get_dataset_config(name: str) -> dict:
    """Class metadata in output-label order (0, 1, 2, 3)."""
    try:
        group = FASHION4_GROUPS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {', '.join(sorted(FASHION4_GROUPS))}"
        ) from exc
    return {
        "original_labels": [label for label, _ in group],
        "class_names": [class_name for _, class_name in group],
    }


# Reshape the images to a flat vector and remap the labels to 0, 1, 2, 3 for the four-class Fashion-MNIST subsets.
class Fashion4Dataset(Dataset):
    """Four-class Fashion-MNIST view used by all QCNN models."""

    def __init__(self, subset: Dataset, class_map: dict[int, int]):
        self.subset = subset
        self.class_map = class_map

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        return x.reshape(-1), self.class_map[int(y)]

# Shared balanced sampling for Fashion-MNIST, MNIST and OrganAMNIST.
def balanced_indices(
    ds,
    class_map: dict[int, int],
    n: int,  # Number of samples per selected class.
    offset: int = 0,  # Skip training samples when validation shares the same pool.
    seed: int = 42,
):
    """Class-balanced sampling shared by every model and every image resize."""
    targets = np.asarray(ds.targets)  # Original source labels, before remapping.
    rng = np.random.default_rng(seed)  # Local generator; no global RNG state changes.
    selected = []

    # Repeating the same seed and class order reproduces each class permutation.
    # Offsets then give disjoint slices when training and validation share a pool.
    for original_class in class_map:
        arr = np.flatnonzero(targets == original_class).copy()
        rng.shuffle(arr)
        if offset + n > len(arr):
            raise ValueError(
                f"Not enough samples for class {original_class}: "
                f"requested offset={offset}, n={n}, available={len(arr)}"
            )
        selected.extend(arr[offset : offset + n].tolist())

    # Shuffle the selected indices to mix the classes and return them.
    rng.shuffle(selected)
    return selected

# Load the Fashion-MNIST dataset, using a local copy if available, or downloading it if not.
def _fashion_split(data_dir: Path, *, train: bool, transform):
    """Use the local copy first; download only if torchvision cannot find it."""
    try:
        return tvds.FashionMNIST(
            root=str(data_dir),
            train=train,
            download=False,
            transform=transform,
        )
    except RuntimeError:
        split = "train" if train else "test"
        print(f"Fashion-MNIST {split} split not found in {data_dir}; downloading...")
        return tvds.FashionMNIST(
            root=str(data_dir),
            train=train,
            download=True,
            transform=transform,
        )

#------------------------------------- DATASET ENTRY POINT ------------------------------#
def get_test_dataset(*, dataset_name, data_dir, image_size):
    """Provide analysis with the same Resize/ToTensor used during training."""
    config = get_dataset_config(dataset_name)
    if image_size not in {8, 16, 28}:
        raise ValueError(f"Unsupported image size: {image_size}; expected 8, 16 or 28.")
    source = _fashion_split(
        Path(data_dir), train=False,
        transform=tvtf.Compose([tvtf.Resize((image_size, image_size)), tvtf.ToTensor()]),
    )
    dataset = Fashion4Dataset(
        source, {label: target for target, label in enumerate(config["original_labels"])},
    )
    # Keep original labels available to the common sampler; __getitem__ remaps
    # only the selected classes, just as in the regular training loaders.
    dataset.targets = np.asarray(source.targets)
    return dataset


# Build balanced loaders for a named four-class Fashion-MNIST subset, 
# main entry point for the dataset module.
def load_data(
    *,
    dataset_name: str,
    data_dir: Path,
    image_size: int,
    seed: int,
    batch_size: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
):
    """Build balanced loaders for a named four-class Fashion-MNIST subset."""
    config = get_dataset_config(dataset_name)
    class_map = {
        original: target for target, original in enumerate(config["original_labels"])
    }
    if image_size not in {8, 16}:
        raise ValueError(f"Unsupported Fashion-MNIST resize: {image_size}x{image_size}")

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Define the transformation pipeline for the images: resize to the specified image size and convert to tensor.
    transform = tvtf.Compose([
        tvtf.Resize((image_size, image_size)),
        tvtf.ToTensor(),
    ])

    trn = _fashion_split(data_dir, train=True, transform=transform)
    tst = _fashion_split(data_dir, train=False, transform=transform)

    # Fashion-MNIST has no separate validation pool. Use disjoint slices of the
    # official training pool; testing uses only the official test pool.
    train_idx = balanced_indices(
        trn, class_map, train_per_class, offset=0, seed=seed
    )
    val_idx = balanced_indices(
        trn, class_map, val_per_class, offset=train_per_class, seed=seed
    )
    test_idx = balanced_indices(
        tst, class_map, test_per_class, offset=0, seed=seed
    )

    common = dict(
        batch_size=batch_size,
    )

    # Create the data loaders for the training, validation, and testing sets.
    train_loader = DataLoader(
        Fashion4Dataset(Subset(trn, train_idx), class_map),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        Fashion4Dataset(Subset(trn, val_idx), class_map),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        Fashion4Dataset(Subset(tst, test_idx), class_map),
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader
