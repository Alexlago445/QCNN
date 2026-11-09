from __future__ import annotations

import argparse
import json
import time

import numpy as np

from datasets import get_dataset_config, get_dataset_loader, list_datasets
from models import get_model, list_models
from .config import (
    BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_SEEDS,
    EPOCHS_DEFAULT,
    get_dataset_defaults,
)
from .engine import train_one_run


# -----------------------------------------------------------------------------
# Command-line interface for training

MAX_MODELS_PER_COMMAND = 5


# Parse model, dataset, seed and training options, rejecting incompatible selections.
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"""Train modular QCNN models sequentially.

Usage examples (run from the project root):
  One model, one dataset and one seed:
    python -m training.train --model model5 --dataset fashion4_1 --seed 42
  Up to {MAX_MODELS_PER_COMMAND} distinct models, in the specified order:
    python -m training.train --models model5 model6 model7 --dataset fashion4_1 --all-seeds
  All registered models with all default seeds:
    python -m training.train --all-models --dataset fashion4_1 --all-seeds
  Select a registered four-class dataset:
    python -m training.train --model model5 --dataset fashion4_2 --all-seeds
  Train on MNIST digits 0..3 or OrganAMNIST (2400/500/500 samples per class):
    python -m training.train --model model5 --dataset mnist --all-seeds
    python -m training.train --model model5 --dataset organamnist --all-seeds
  A custom seed list:
    python -m training.train --model model5 --dataset fashion4_1 --seeds 11 59
  Customize epochs, samples per class and batch size:
    python -m training.train --model model5 --dataset fashion4_1 --seed 42 --epochs 2 --train-per-class 50 --val-per-class 20 --test-per-class 20 --batch-size 10
  Train without backpropagation using SPSA:
    python -m training.train --model model17 --dataset mnist --seed 42 --opt spsa
  Restart selected runs from scratch, replacing previous results:
    python -m training.train --all-models --dataset fashion4_1 --all-seeds --no-resume
  Show every available option:
    python -m training.train --help

Every training command requires exactly one of --model / --models / --all-models,
an explicit --dataset, and exactly one of --seed / --seeds / --all-seeds.
--all-seeds selects {DEFAULT_SEEDS}.
The dataset selects the loader; input files are read from {DEFAULT_DATA_DIR}.
Without --no-resume, completed runs are skipped and interrupted runs resume
from the last completed epoch when the checkpoint configuration matches.
Results are always saved under {DEFAULT_RESULTS_DIR}.
SPSA uses the separate {DEFAULT_RESULTS_DIR / "spsa"} subtree.
""",
    )

    # Choose a single model, an explicit list or all models; these options cannot combine.
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list_models(),
        help="Train one model, e.g. --model model5.",
    )
    model_group.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=list_models(),
        help="Train 1 to 5 models sequentially, e.g. --models model5 model6 model7.",
    )
    model_group.add_argument(
        "--all-models",
        dest="all_models",
        action="store_true",
        help=(
            "Train all registered models sequentially. Seeds with an existing "
            "test.json are skipped unless --no-resume is specified."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list_datasets(),
        help="Required registered four-class dataset; download and input path are automatic.",
    )

    # Explicitly choose one seed, a custom list or all standard seeds.
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Run one seed, e.g. --seed 42.",
    )
    seed_group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run a custom seed list, e.g. --seeds 11 42 59.",
    )
    seed_group.add_argument(
        "--all-seeds",
        action="store_true",
        help=f"Run all standard seeds: {DEFAULT_SEEDS}.",
    )

    # Override training sizes; input and result paths are fixed by the project.
    parser.add_argument(
        "--opt", choices=("back", "spsa"), default="back",
        help="Optimization method: backpropagation with AdamW or SPSA (default: %(default)s).",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS_DEFAULT,
        help="Maximum training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--train-per-class", type=int, default=None,
        help="Training samples per class (Fashion: 4000; MNIST/OrganAMNIST: 2400).",
    )
    parser.add_argument(
        "--val-per-class", type=int, default=None,
        help="Validation samples per class (Fashion: 700; MNIST/OrganAMNIST: 500).",
    )
    parser.add_argument(
        "--test-per-class", type=int, default=None,
        help="Test samples per class (Fashion: 700; MNIST/OrganAMNIST: 500).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Samples per batch (default: %(default)s).",
    )
    parser.set_defaults(results_dir=DEFAULT_RESULTS_DIR)
    # Force selected runs to restart, including seeds with completed test results.
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Restart every selected seed from epoch 1, including completed seeds, in results/.",
    )

    args = parser.parse_args(argv)
    # Keep the existing backpropagation layout and isolate SPSA experiments.
    if args.opt == "spsa":
        args.results_dir = args.results_dir / "spsa"
    # Fill only unspecified counts so explicit CLI values always take precedence
    # over the configured sampling protocol for the selected dataset.
    for key, value in get_dataset_defaults(args.dataset).items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    # Reject empty splits before dataset loading or creating any training artifacts.
    for key in ("epochs", "batch_size", "train_per_class", "val_per_class", "test_per_class"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive.")

    if args.models is not None and len(args.models) > MAX_MODELS_PER_COMMAND:
        parser.error(
            f"--models accepts at most {MAX_MODELS_PER_COMMAND} models per command."
        )

    return args


# -----------------------------------------------------------------------------
# Selection helpers used by main() to turn command-line options into ordered lists.

# Resolve the required model selection.
def _selected_models(args) -> list[str]:
    if args.all_models:
        return list_models()
    if args.models is not None:
        # Preserve command-line order, but reject duplicates to avoid accidental reruns.
        if len(set(args.models)) != len(args.models):
            raise ValueError("The same model cannot appear twice in --models.")
        return list(args.models)
    if args.model is not None:
        return [args.model]
    raise ValueError("Select --model, --models or --all-models.")


# Resolve the required seed selection; only --all-seeds uses DEFAULT_SEEDS.
def _selected_seeds(args) -> list[int]:
    if args.all_seeds:
        return list(DEFAULT_SEEDS)
    if args.seeds is not None:
        return list(args.seeds)
    if args.seed is not None:
        return [args.seed]
    raise ValueError("Select --seed, --seeds or --all-seeds.")


# -----------------------------------------------------------------------------
# Run one model across the selected seeds, delegating training and evaluation
# to engine.train_one_run and collecting results under results/<model>/<dataset>/.

def _run_model(args, model_name: str, seeds: list[int], dataset_loader):
    experiment_dir = args.results_dir / model_name / args.dataset
    experiment_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    model_start = time.time()

    print(model_name)

    # Process seeds sequentially, keeping checkpoints and metrics in separate folders.
    for idx, seed in enumerate(seeds, start=1):
        run_dir = experiment_dir / f"seed{seed}"
        completed = run_dir / "test.json"

        # Reuse existing test results unless a fresh run was explicitly requested.
        if completed.exists() and not args.no_resume:
            with open(completed, "r", encoding="utf-8") as f:
                previous_result = json.load(f)
            if previous_result.get("opt", "back") != args.opt:
                raise RuntimeError(
                    f"Completed run uses a different optimization method: {completed}. "
                    "Use --no-resume to explicitly replace it."
                )
            print(f"[{idx}/{len(seeds)}] SKIP completed seed: {seed}")
            all_results.append(previous_result)
            continue

        # Report whether the engine will start from scratch or resume a checkpoint.
        latest = run_dir / "checkpoint_latest.pt"
        if latest.exists() and not args.no_resume:
            print(f"[{idx}/{len(seeds)}] RESUME seed {seed}")
        else:
            print(f"[{idx}/{len(seeds)}] START seed {seed}")

        # Defer model creation through a factory so the engine sets the seed first.
        result = train_one_run(
            model_factory=lambda name=model_name: get_model(name),
            model_name=model_name,
            dataset_name=args.dataset,
            dataset_loader=dataset_loader,
            data_dir=DEFAULT_DATA_DIR,
            run_dir=run_dir,
            seed=seed,
            epochs=args.epochs,
            train_per_class=args.train_per_class,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            batch_size=args.batch_size,
            auto_resume=not args.no_resume,
            opt=args.opt,
        )
        all_results.append(result)

    # -------------------------------------------------------------------------
    # Per-model summary: combine reused and newly computed seed results.
    # Save individual accuracies and their mean.

    accs = np.asarray([r["test_acc"] for r in all_results], dtype=float)
    summary = {
        "model": model_name,
        "dataset": args.dataset,
        "opt": args.opt,
        **get_dataset_config(args.dataset),
        "seeds": seeds,
        "test_accuracies": accs.tolist(),
        "mean_test_accuracy": float(accs.mean()) if len(accs) else None,
        "elapsed_seconds": time.time() - model_start,
    }

    with open(experiment_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print(f"MODEL SUMMARY: {model_name}")
    print("test acc:", accs.tolist())
    if len(accs):
        print(f"mean: {accs.mean():.6f}")
    print("results:", experiment_dir)
    print("-" * 72)

    return summary


# -----------------------------------------------------------------------------
# Command entry point: resolve selections and run each model sequentially on one dataset.
# The selected seeds are handled within _run_model for every model.

def main():
    args = parse_args()
    models = _selected_models(args)
    seeds = _selected_seeds(args)
    dataset_loader = get_dataset_loader(args.dataset)

    suite_start = time.time()
    print("=" * 72)
    print("MODELS TO RUN:", models)
    print("DATASET      :", args.dataset)
    print("OPTIMIZATION :", args.opt)
    print("SEEDS        :", seeds)
    print("=" * 72)

    # Collect one summary per model after all of its selected seeds are processed.
    suite_summaries = []
    for pos, model_name in enumerate(models, start=1):
        print(f"\n### MODEL {pos}/{len(models)}: {model_name} ###")
        suite_summaries.append(
            _run_model(args, model_name, seeds, dataset_loader)
        )

    # Save a combined record of this command's models, seeds, dataset and elapsed time.
    suite_result = {
        "models": models,
        "dataset": args.dataset,
        "opt": args.opt,
        **get_dataset_config(args.dataset),
        "seeds": seeds,
        "elapsed_seconds": time.time() - suite_start,
        "summaries": suite_summaries,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "last_suite.json", "w", encoding="utf-8") as f:
        json.dump(suite_result, f, indent=2)

    print("=" * 72)
    print("ALL REQUESTED MODELS COMPLETED")
    print("models:", models)
    print(f"elapsed: {(time.time() - suite_start) / 60:.1f} min")
    print("=" * 72)


# -----------------------------------------------------------------------------
# Launch the command when executed as python -m training.train.

if __name__ == "__main__":
    main()
