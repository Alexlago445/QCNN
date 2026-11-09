# QCNN – Image Classification

This repository contains 24 Quantum Convolutional Neural Network (QCNN) configurations (`model5`–`model28`) for four-class image classification. It includes dataset loaders, model definitions, training scripts, saved results, and analysis notebooks. Training and analysis run on CPU.

## Project structure

```text
qcnn/
├── analyses/
│   ├── data_visualization.ipynb       # Quantum-state and image analysis
│   ├── qcnn_transpilation.ipynb        # Hardware-oriented circuit analysis
│   └── results_visualization.ipynb    # Training and test results
├── datasets/
│   ├── data/                          # Downloaded/cached datasets
│   ├── fashion_mnist.py
│   ├── mnist.py
│   ├── organamnist.py
│   ├── image_data.py
│   └── registry.py
├── models/
│   ├── model5.py ... model28.py       # Individual QCNN configurations
│   ├── registry.py
│   └── utilities/                    # Encoding, ansatz, pooling, QCNN, loss
├── training/
│   ├── config.py                     # Training defaults
│   ├── checkpoint.py
│   ├── engine.py
│   └── train.py                      # Training entry point
├── results/                          # Checkpoints, metrics, and summaries
├── requirements.txt
└── README.md
```

## Setup

Run commands from the **project root** (the directory containing `training/` and `requirements.txt`):

```bash
python -m pip install -r requirements.txt
```

Datasets are loaded automatically and cached under `datasets/data/`.

## Datasets

All tasks use four classes. The dataset names accepted by `--dataset` are:

| Name | Classes (output-label order) |
|---|---|
| `fashion4_1` | T-shirt, Trouser, Sneaker, Bag |
| `fashion4_2` | Pullover, Sandal, Trouser, Coat |
| `fashion4_3` | Dress, Ankle boot, Shirt, Bag |
| `mnist` | Digits 0, 1, 2, 3 |
| `organamnist` | Left kidney, Left lung, Pancreas, Liver |

Images are resized automatically to **8×8** for `reupload` models and **16×16** for `E1`/`E3` models.

Default train / validation / test samples **per class**:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| Fashion-MNIST subsets | 4000 | 700 | 700 |
| MNIST | 2400 | 500 | 500 |
| OrganAMNIST | 2400 | 500 | 500 |

## Model configurations

The 24 models combine two connectivity patterns (`spatial_diag`, `benchmark`), four convolutional blocks, and three encodings. Model IDs in each cell are listed in **reupload / E1 / E3** order:

| Convolution block | `spatial_diag` | `benchmark` |
|---|---|---|
| Hur6 | `model5` / `model6` / `model7` | `model14` / `model15` / `model16` |
| Hur8 | `model8` / `model9` / `model10` | `model17` / `model18` / `model19` |
| Hur9 | `model11` / `model12` / `model13` | `model20` / `model21` / `model22` |
| Custom | `model23` / `model24` / `model25` | `model26` / `model27` / `model28` |

## Training commands

**One model, one dataset, one seed:**

```bash
python -m training.train --model model17 --dataset fashion4_1 --seed 42
```

**One model with all standard seeds** (`11`, `42`, `59`):

```bash
python -m training.train --model model17 --dataset mnist --all-seeds
```

**Several models** (up to five with `--models`):

```bash
python -m training.train --models model17 model18 model19 --dataset fashion4_2 --all-seeds
```

**All 24 models on a dataset:**

```bash
python -m training.train --all-models --dataset fashion4_1 --all-seeds
```

**Use SPSA instead of the default backpropagation optimizer:**

```bash
python -m training.train --model model17 --dataset mnist --seed 42 --opt spsa
```

**Quick test with fewer images and epochs:**

```bash
python -m training.train --model model17 --dataset mnist --seed 42 --epochs 2 --train-per-class 50 --val-per-class 20 --test-per-class 20
```

**Restart a run from scratch** (instead of resuming/skipping existing results):

```bash
python -m training.train --model model17 --dataset mnist --seed 42 --no-resume
```

**View all command-line options:**

```bash
python -m training.train --help
```

Training uses backpropagation with AdamW by default (`--opt back`). To select a different seed, use `--seed N` or `--seeds N ...`. Sample counts and training hyperparameters can be changed through command-line flags or `training/config.py`.

Completed runs are skipped automatically, and interrupted runs resume from their latest checkpoint. **Note:** a quick-test run uses the same output location as a full run; use `--no-resume` to replace it.

## Output files

Default training results are stored under:

```text
results/
└── model17/
    └── fashion4_1/
        ├── seed42/
        │   ├── config.json
        │   ├── metrics.csv
        │   ├── best_state.pt
        │   ├── checkpoint_latest.pt
        │   └── test.json
        └── summary.json
```

Other seeds and models follow the same structure. SPSA runs are stored separately under `results/spsa/<model>/<dataset>/seed<seed>/`.

## Analysis notebooks

Open the notebooks in `analyses/` after training the relevant models:

| Notebook | Purpose | Main settings to change |
|---|---|---|
| `results_visualization.ipynb` | Accuracy, loss, confusion matrices, model comparisons | `MODELS`, `DATASETS`, `SEEDS` |
| `data_visualization.ipynb` | Image and quantum-state representation analysis | `MODEL_NAME`, `DATASET_NAME`, `SEED` |
| `qcnn_transpilation.ipynb` | Gate counts, depth, and transpilation on simulated hardware backends | `MODELS`, `DATASET`, `TRAINING_SEED` |

The notebooks display analyses inline and do not automatically export plots or tables. Some analyses require extra packages (e.g. `pandas`, `matplotlib`, `scikit-learn`, or Qiskit) beyond the training dependencies.
