from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from datasets import get_dataset_config
from models.utilities.cost_function import probability_cross_entropy
from .checkpoint import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    write_metrics_from_history,
)
from .config import (
    EARLY_STOP_MIN_ACC_DELTA,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    LABEL_SMOOTHING,
    LR_EMBED,
    LR_QKERNEL,
    SPSA_A,
    SPSA_ALPHA,
    SPSA_C,
    SPSA_GAMMA,
    WD_EMBED,
    WD_QKERNEL,
)
from .spsa import SPSA


# -----------------------------------------------------------------------------
# Training utilities used to initialize randomness, measure accuracy and build
# the optimizer shared by all QCNN models.

# Seed Python, NumPy and PyTorch before model and dataset initialization.
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Return the fraction of samples whose most probable class matches the target.
def accuracy(probs: torch.Tensor, y: torch.Tensor) -> float:
    return (probs.argmax(dim=1) == y).float().mean().item()


# Select SPSA or AdamW with separate quantum-kernel and encoding parameter groups.
def build_optimizer(model, opt="back"):
    if opt == "spsa":
        return SPSA(model.parameters(), lr=SPSA_A, c=SPSA_C,
                    alpha=SPSA_ALPHA, gamma=SPSA_GAMMA)
    if opt != "back":
        raise ValueError(f"Unknown optimization method: {opt!r}.")
    groups = model.optimizer_parameter_groups()

    param_groups = [
        {
            "params": groups["qkernel"],
            "lr": LR_QKERNEL,
            "weight_decay": WD_QKERNEL,
        }
    ]

    # E3 has no trainable encoding parameters, so its embedding group is empty.
    if groups["embedding"]:
        param_groups.append(
            {
                "params": groups["embedding"],
                "lr": LR_EMBED,
                "weight_decay": WD_EMBED,
            }
        )

    return torch.optim.AdamW(param_groups)


# -----------------------------------------------------------------------------
# Shared batch loop for training, validation and testing, called by train_one_run.
# Passing an optimizer enables parameter updates; otherwise only evaluate the model.
# Return loss and accuracy averaged over all samples in the loader.

def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    # SPSA evaluates perturbed losses without building an autograd graph.
    use_spsa = isinstance(optimizer, SPSA)
    context = torch.enable_grad() if training and not use_spsa else torch.no_grad()
    with context:
        for x_flat, y in loader:
            x_flat = x_flat.to(device)
            y = y.to(device)

            if training:
                optimizer.zero_grad()

            # Run the model's forward method and compute loss from class probabilities.
            probs = model(x_flat)
            loss = probability_cross_entropy(
                probs,
                y,
                label_smoothing=LABEL_SMOOTHING if training else 0.0,
            )

            if training:
                if use_spsa:
                    # Both perturbations use this same mini-batch. The forward pass
                    # above measures training metrics at the unperturbed parameters.
                    def closure():
                        return probability_cross_entropy(
                            model(x_flat), y, label_smoothing=LABEL_SMOOTHING,
                        )
                    optimizer.step(closure)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                    optimizer.step()

            # Weight batch metrics by sample count
            bs = x_flat.shape[0]
            total_loss += loss.item() * bs
            total_acc += accuracy(probs.detach(), y) * bs
            total_n += bs

    return total_loss / total_n, total_acc / total_n


# -----------------------------------------------------------------------------
# Confusion matrix helper, used in train_one_run to evaluate the best model on the test set after training is complete.

def confusion_matrix(model, loader, device, n_classes: int):
    model.eval()
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    with torch.no_grad():
        for x_flat, y in loader:
            probs = model(x_flat.to(device))
            pred = probs.argmax(dim=1).cpu()

            for target, p in zip(y, pred):
                cm[int(target), int(p)] += 1

    return cm

# Complete experiment for one model, dataset and seed, called by training/train.py.
# Set up or resume training, save epoch checkpoints and evaluate the best model.

def train_one_run(
    *,
    model_factory,
    model_name: str,
    dataset_name: str,
    dataset_loader,
    data_dir: Path,
    run_dir: Path,
    seed: int,
    epochs: int,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    batch_size: int,
    auto_resume: bool = True,
    opt: str = "back",
):
    # Initialize the model and balanced data loaders, then optimizer and scheduler.
    # Set the seed first so parameter initialization and sample selection can repeat.
    if opt not in {"back", "spsa"}:
        raise ValueError(f"Unknown optimization method: {opt!r}.")
    set_seed(seed)

    device = torch.device("cpu")
    model = model_factory().to(device)

    # Check that registered trainable parameters match the architecture's expected count.
    n_params = model.n_trainable_params()
    expected = getattr(model, "expected_n_params", None)
    if expected is not None and n_params != expected:
        raise RuntimeError(
            f"Parameter-count mismatch: model has {n_params}, expected {expected}."
        )

    train_loader, val_loader, test_loader = dataset_loader(
        data_dir=data_dir,
        image_size=model.image_size,
        seed=seed,
        batch_size=batch_size,
        train_per_class=train_per_class,
        val_per_class=val_per_class,
        test_per_class=test_per_class,
    )

    # Backpropagation retains its AdamW schedule; SPSA uses per-step power gains.
    optimizer = build_optimizer(model, opt)
    scheduler = None
    if opt == "back":
        t0_sched = max(1, epochs // 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0_sched,
            T_mult=1,
            eta_min=1e-4,
        )

    optimizer_config = (
        {"lr_qkernel": LR_QKERNEL, "lr_embed": LR_EMBED,
         "weight_decay_qkernel": WD_QKERNEL, "weight_decay_embed": WD_EMBED}
        if opt == "back" else
        {"spsa_a": SPSA_A, "spsa_c": SPSA_C,
         "spsa_alpha": SPSA_ALPHA, "spsa_gamma": SPSA_GAMMA}
    )

    # -------------------------------------------------------------------------
    # Experiment configuration and output paths, used for saving and resume checks.
    # Combine dataset details, training settings and the model's identifying names.

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": model_name,
        "dataset": dataset_name,
        **get_dataset_config(dataset_name),
        "seed": seed,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "test_per_class": test_per_class,
        "batch_size": batch_size,
        "epochs": epochs,
        "opt": opt,
        **optimizer_config,
        "label_smoothing": LABEL_SMOOTHING,
        "gradient_clip_norm": GRAD_CLIP_NORM if opt == "back" else None,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "early_stop_min_acc_delta": EARLY_STOP_MIN_ACC_DELTA,
        **model.model_config(),
    }
    # MNIST/OrganAMNIST loaders wrap their dataset in a Subset; Fashion-MNIST uses
    # its own view. Optional metadata is read from the appropriate underlying view.
    training_dataset = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    preprocessing = getattr(training_dataset, "preprocessing", None)
    if preprocessing is not None:
        config["preprocessing"] = preprocessing

    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.csv"
    best_path = run_dir / "best_state.pt"
    latest_path = run_dir / "checkpoint_latest.pt"

    # -------------------------------------------------------------------------
    # Run state: track the best model, early-stopping patience and completed epochs.
    # Start with fresh values, then replace them if resuming an existing checkpoint.

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state = None
    patience = 0
    last_epoch = 0
    train_acc_final = 0.0
    val_acc_final = 0.0
    history = []
    start_epoch = 1
    training_done = False
    resumed = False

    # Restore a checkpoint only when its experiment configuration matches this run.
    if auto_resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)

        saved_config = checkpoint.get("config")
        # Checkpoints created before --opt was added used backpropagation.
        if isinstance(saved_config, dict):
            saved_config = {"opt": "back", **saved_config}
        if saved_config != config:
            raise RuntimeError(
                "A resumable checkpoint exists, but its experiment configuration "
                "does not match the current command.\n"
                f"Checkpoint: {latest_path}\n"
                "Use the same settings or run "
                "with --no-resume to restart this seed."
            )

        # Recover parameters, optimizer history and the learning-rate schedule.
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        best_val_acc = float(checkpoint["best_val_acc"])
        best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_model_state = checkpoint.get("best_model_state")
        patience = int(checkpoint["patience"])
        last_epoch = int(checkpoint["last_epoch"])
        train_acc_final = float(checkpoint["train_acc_final"])
        val_acc_final = float(checkpoint["val_acc_final"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["next_epoch"])
        training_done = bool(checkpoint.get("training_done", False))

        # Continue random sequences and rebuild the CSV from checkpointed metrics.
        restore_rng_state(checkpoint.get("rng_state"))
        write_metrics_from_history(metrics_path, history)

        # Recreate the best-model file in case saving it was interrupted previously.
        if best_model_state is not None:
            atomic_torch_save(
                {
                    "model_state": best_model_state,
                    "epoch": best_epoch,
                    "val_acc": best_val_acc,
                    "val_loss": best_val_loss,
                    "seed": seed,
                    "n_params": n_params,
                },
                best_path,
            )

        resumed = True
    else:
        if not auto_resume:
            # Invalidate old run artifacts before starting a replacement run.
            # An interrupted restart must not look completed or resume old weights.
            for stale_path in (run_dir / "test.json", latest_path, best_path):
                stale_path.unlink(missing_ok=True)
        write_metrics_from_history(metrics_path, history)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Report the experiment settings and the starting point of this execution.
    print("=" * 72)
    print(f"{model_name.upper()} | dataset={dataset_name}")
    print(f"seed={seed} | opt={opt} | device={device} | PennyLane={qml.__version__}")
    print("classes (output labels 0-3):", config["class_names"])
    print(f"trainable parameters={n_params}")
    if resumed:
        print(
            f"AUTO-RESUME: completed epoch {last_epoch}; "
            f"next epoch {start_epoch}; patience={patience}."
        )
    else:
        print("AUTO-RESUME: starting a new run.")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Epoch loop: train, validate, select the best state and save progress.
    # A completed checkpoint skips this loop and proceeds to final evaluation.

    if not training_done:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
            )
            val_loss, val_acc = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
            )
            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - epoch_start

            # Select a new best state only after a sufficient validation accuracy gain.
            improved = val_acc > best_val_acc + EARLY_STOP_MIN_ACC_DELTA
            if improved:
                best_val_acc = float(val_acc)
                best_val_loss = float(val_loss)
                best_epoch = int(epoch)
                patience = 0
                # Clone tensors so later optimizer steps cannot change the saved best state.
                best_model_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                patience += 1

            train_acc_final = float(train_acc)
            val_acc_final = float(val_acc)
            last_epoch = int(epoch)

            # Keep one metrics record per completed epoch for CSV export and resume.
            epoch_record = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "time_sec": float(elapsed),
            }
            history.append(epoch_record)

            training_done = (
                patience >= EARLY_STOP_PATIENCE
                or epoch >= epochs
            )

            # Save all state needed to continue from the next epoch, including randomness.
            latest_checkpoint = {
                "version": 1,
                "config": config,
                "model_state": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_model_state": best_model_state,
                "best_val_acc": float(best_val_acc),
                "best_val_loss": float(best_val_loss),
                "best_epoch": int(best_epoch),
                "patience": int(patience),
                "last_epoch": int(last_epoch),
                "next_epoch": int(epoch + 1),
                "train_acc_final": float(train_acc_final),
                "val_acc_final": float(val_acc_final),
                "training_done": bool(training_done),
                "history": list(history),
                "rng_state": capture_rng_state(),
            }
            atomic_torch_save(latest_checkpoint, latest_path)

            # Store the best validation model separately from the latest training state.
            if improved and best_model_state is not None:
                atomic_torch_save(
                    {
                        "model_state": best_model_state,
                        "epoch": best_epoch,
                        "val_acc": best_val_acc,
                        "val_loss": best_val_loss,
                        "seed": seed,
                        "n_params": n_params,
                    },
                    best_path,
                )

            write_metrics_from_history(metrics_path, history)

            print(
                f"epoch {epoch:02d}/{epochs} | "
                f"train {train_loss:.4f}/{train_acc:.4f} | "
                f"val {val_loss:.4f}/{val_acc:.4f} | "
                f"best {best_val_acc:.4f} @ {best_epoch} | "
                f"{elapsed:.1f}s"
            )
            print(f"  checkpoint: {latest_path}")

            # Stop after the allowed number of epochs without a sufficient improvement.
            if patience >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

    # -------------------------------------------------------------------------
    # Final test evaluation using the best validation state, without parameter updates.
    # Compute test loss, accuracy and confusion counts after model selection is complete.

    if best_model_state is None:
        raise RuntimeError("No best model state is available.")

    model.load_state_dict(best_model_state)

    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
    )
    cm = confusion_matrix(
        model,
        test_loader,
        device,
        n_classes=model.n_classes,
    )

    # -------------------------------------------------------------------------
    # Save test results and run metadata for later summaries and analysis notebooks.

    result = {
        "model": model_name,
        "dataset": dataset_name,
        **get_dataset_config(dataset_name),
        "architecture": model.architecture_name,
        "opt": opt,
        "seed": seed,
        "n_params": n_params,
        "n_epochs_trained": last_epoch,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "val_acc_best": best_val_acc,
        "val_acc_final": val_acc_final,
        "train_acc_final": train_acc_final,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_size": len(test_loader.dataset),
        "confusion_matrix": cm.tolist(),
        "logical_resources": model.logical_resource_summary(),
        "auto_resume": auto_resume,
    }
    # Store preprocessing settings with the test metrics for reproducible evaluation.
    if preprocessing is not None:
        result["preprocessing"] = preprocessing

    with open(run_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("-" * 72)
    print(f"BEST VAL : {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"TEST     : loss={test_loss:.4f}, acc={test_acc:.4f}")
    print("CONFUSION MATRIX")
    print(cm.numpy())
    print("results:", run_dir)
    print("-" * 72)

    return result
