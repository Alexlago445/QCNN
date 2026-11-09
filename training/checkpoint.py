from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch

# Save to a temporary file, then replace the checkpoint so an interrupted write preserves the previous version.
def atomic_torch_save(obj, path: Path) -> None:
    """Write a checkpoint atomically to reduce corruption risk."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# Random generator states saved and restored by training/engine.py during resume.
# Preserve the position in each random sequence, including PyTorch's batch shuffling.

# Capture the current Python, NumPy and PyTorch generator states for the checkpoint.
def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    return state


# Restore the saved generator states before the next epoch, so random sequences
# continue from the checkpoint instead of restarting from the initial seed.
def restore_rng_state(state: dict | None) -> None:
    if not state:
        return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


# Rebuild metrics.csv from the checkpoint history to keep epoch metrics consistent when resuming training.
def write_metrics_from_history(metrics_path: Path, history: list[dict]) -> None:
    """Keep metrics.csv synchronized with the checkpointed history."""
    metrics_path = Path(metrics_path)
    tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")

    with open(tmp, "w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"]
        )

        for rec in history:
            writer.writerow(
                [
                    rec["epoch"],
                    f'{rec["train_loss"]:.6f}',
                    f'{rec["train_acc"]:.6f}',
                    f'{rec["val_loss"]:.6f}',
                    f'{rec["val_acc"]:.6f}',
                    f'{rec["time_sec"]:.2f}',
                ]
            )

        mf.flush()
        os.fsync(mf.fileno())

    os.replace(tmp, metrics_path)
