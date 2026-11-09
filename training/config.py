from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_SEEDS = [11, 42, 59]

TRAIN_PER_CLASS = 4000
VAL_PER_CLASS = 700
TEST_PER_CLASS = 700

# -----------------------------------------------------------------------------
# Dataset-specific sample counts, in (training, validation, test) order per class.
# Fashion-MNIST uses the constants above. CLI counts can override each value.
# MNIST and OrganAMNIST use 2400/500/500 images PER CLASS: with four classes
# each run contains 9600 training, 2000 validation and 2000 test images.
# MNIST takes disjoint training/validation selections from its training pool;
# OrganAMNIST retains its separate official training, validation and test pools.
DATASET_SPLIT_DEFAULTS = {
    "mnist": (2400, 500, 500),
    "organamnist": (2400, 500, 500),
}


# Resolve dataset defaults while keeping the shared Fashion-MNIST fallback.
def get_dataset_defaults(name: str) -> dict[str, int]:
    counts = DATASET_SPLIT_DEFAULTS.get(
        name, (TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS),
    )
    return dict(zip(("train_per_class", "val_per_class", "test_per_class"), counts))


BATCH_SIZE = 24
EPOCHS_DEFAULT = 20

LR_QKERNEL = 1e-3
LR_EMBED = 1e-3
WD_QKERNEL = 0.0
WD_EMBED = 0.0

# SPSA gains, applied once per training mini-batch (no AdamW or cosine scheduler).
SPSA_A = 0.1
SPSA_C = 0.1
SPSA_ALPHA = 0.602
SPSA_GAMMA = 0.101

LABEL_SMOOTHING = 0.1
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 10
# Accuracy is a fraction: 0.005 means 0.5 percentage points. The engine requires
# a gain strictly greater than this threshold to replace the best checkpoint.
EARLY_STOP_MIN_ACC_DELTA = 0.005
