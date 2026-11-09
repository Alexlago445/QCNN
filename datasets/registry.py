from __future__ import annotations

from functools import partial

from torch.utils.data import DataLoader, Subset

from . import fashion_mnist, mnist, organamnist
from .fashion_mnist import balanced_indices


# Map every public dataset name to its loader and metadata module.
# Training and analysis select loaders and metadata through the package interface.
_MODULES = {
    **{name: fashion_mnist for name in fashion_mnist.FASHION4_GROUPS},
    "mnist": mnist,
    "organamnist": organamnist,
}
# Bind the chosen subset name while retaining one common loader signature.
_DATASETS = {
    name: partial(module.load_data, dataset_name=name) for name, module in _MODULES.items()
}


# Return class and preprocessing metadata for the selected dataset family.
def get_dataset_config(name: str) -> dict:
    get_dataset_loader(name)  # Keep the same useful unknown-dataset error.
    return _MODULES[name].get_dataset_config(name)


def get_dataset_loader(name: str):
    try:
        return _DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}") from exc


def list_datasets() -> list[str]:
    return sorted(_DATASETS)


def get_test_loader(name: str, *, data_dir, image_size, seed, batch_size, test_per_class):
    """Sample only the official test pool, with the training preprocessing.

    Analysis notebooks use this entry point so switching datasets never silently
    loads images from another source or requires the training/validation pools.
    A 28x28 view is also available for inspecting the native grayscale images.
    """
    get_dataset_loader(name)
    if batch_size <= 0 or test_per_class <= 0:
        raise ValueError("Batch size and samples per class must be positive.")
    dataset = _MODULES[name].get_test_dataset(
        dataset_name=name, data_dir=data_dir, image_size=image_size,
    )
    indices = balanced_indices(dataset, dataset.class_map, test_per_class, seed=seed)
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
