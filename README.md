# QCNN modular training project

This folder contains 24 QCNN configurations (`model5` ... `model28`) trained with one common protocol.

All project code runs exclusively on CPU, including training, evaluation and analysis.

## Datasets

All registered tasks have four classes, matching the two-qubit output. Classes map to
output labels 0, 1, 2 and 3 in the order shown below.

| Dataset | Classes in output-label order | Source labels |
|---|---|---|
| `fashion4_1` | T-shirt, Trouser, Sneaker, Bag | 0, 1, 7, 8 |
| `fashion4_2` | Pullover, Sandal, Trouser, Coat | 2, 5, 1, 4 |
| `fashion4_3` | Dress, Ankle boot, Shirt, Bag | 3, 9, 6, 8 |
| `mnist` | Digit 0, Digit 1, Digit 2, Digit 3 | 0, 1, 2, 3 |
| `organamnist` | Left kidney, Left lung, Pancreas, Liver | 4, 7, 9, 6 |

MNIST has one fixed task containing only digits 0 through 3. OrganAMNIST has one
fixed task containing exactly the four organs listed above.

Dataset names select loaders and metadata through `datasets/registry.py`. All
source images are **28x28 grayscale**. The model's encoding selects an 8x8 input
for `reupload`, or 16x16 for `e1` and `e3`. All loaders return flattened float32
images in [0,1] and class labels in output order. For MNIST and OrganAMNIST,
pixels are divided by 255 and downsampled with antialiased bilinear interpolation.
Seeds 11, 42 and 59 select reproducible class-balanced samples; the source images
selected for one seed are identical across the different model input sizes.

| Dataset | Training per class | Validation per class | Test per class | Total train / val / test |
|---|---:|---:|---:|---|
| Fashion-MNIST | 4000 | 700 | 700 | 16000 / 2800 / 2800 |
| `mnist` | **2400** | **500** | **500** | **9600 / 2000 / 2000** |
| `organamnist` | **2400** | **500** | **500** | **9600 / 2000 / 2000** |

The requested 2400/500/500 split is **per class**, not the overall dataset size.
These defaults are set in `training/config.py`. Explicit `--train-per-class`,
`--val-per-class` and `--test-per-class` values override them. Sampling is without
replacement; a request exceeding a class's available images raises an error.

### MNIST

The loader uses the standard [Torchvision MNIST dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html),
cached in `datasets/data/MNIST/raw/`. Missing files are downloaded automatically.
Only digits 0, 1, 2 and 3 are selected. Training and validation use disjoint seeded
selections from the official training pool (2400 + 500 images per digit). The
500 test images per digit come exclusively from the official test pool.

### OrganAMNIST

[MedMNIST OrganAMNIST](https://medmnist.com/) contains axial CT organ crops. The
loader downloads the official **28x28, single-channel** archive from
[Zenodo](https://zenodo.org/records/10519652/files/organamnist.npz?download=1)
to `datasets/data/OrganAMNIST/organamnist.npz`. It reads NumPy arrays directly;
no additional dataset package is required. The
[official label dictionary](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py)
identifies left kidney = 4, left lung = 7, pancreas = 9 and liver = 6.

The official training, validation and test pools remain separate. In particular,
validation is sampled from `val_images` and never from `train_images`, preserving
the source CT-scan partition. Each pool is sampled independently and balanced.
The native archive has enough images for the requested counts:

| Organ | Available training | Available validation | Available test |
|---|---:|---:|---:|
| Left kidney | 3963 | 568 | 2064 |
| Left lung | 3919 | 1033 | 1747 |
| Pancreas | 3031 | 529 | 1622 |
| Liver | 6164 | 1033 | 3285 |

### Installation and execution

From the project root, install the project dependencies if needed:

```powershell
python -m pip install -r requirements.txt
```

Train one model with all three standard seeds, using the requested split:

```powershell
python -m training.train --model model22 --dataset organamnist --all-seeds
python -m training.train --model model22 --dataset mnist --all-seeds
```

Replace `--model model22` with `--all-models` to execute all 24 models. To run both
datasets sequentially and stop if either command fails:

```powershell
foreach ($dataset in 'organamnist', 'mnist') {
    python -m training.train --all-models --dataset $dataset --all-seeds
    if ($LASTEXITCODE -ne 0) { break }
}
```

For one seed, replace `--all-seeds` with `--seed 42`. The split defaults need no
extra flags. Downloads and model-specific resizing are automatic.

Runs use `results/<model>/<dataset>/seed<seed>/` with `config.json`, `metrics.csv`,
`best_state.pt`, `checkpoint_latest.pt` and `test.json`. Each model/dataset folder
contains `summary.json`. Interrupted runs resume from the latest completed epoch;
completed runs are skipped. Add `--no-resume` to explicitly restart selected runs.

All analysis notebooks support the registered dataset names. The image-analysis
notebooks use `get_test_loader` to select images only from the chosen dataset's
official test pool, with exactly the same preprocessing as training. The results
notebook reads class labels and test sizes from each saved run's metadata.

## Common training protocol

All models share one training implementation. The default backpropagation protocol uses:

- PennyLane + Torch backpropagation with AdamW by default (`--opt back`);
- learning rate 1e-3 for quantum-kernel and trainable encoding parameters;
- weight decay 0;
- batch size 24;
- max 20 epochs;
- label smoothing 0.1 during training only;
- gradient clipping with norm 1.0;
- CosineAnnealingWarmRestarts with `T_0 = epochs // 3`, `T_mult = 1`, `eta_min = 1e-4`;
- early stopping on validation accuracy, patience 10, minimum absolute improvement 0.005 (0.5 percentage points);
- seeds 11, 42 and 59;
- epoch-level auto-resume through `checkpoint_latest.pt`.

All model definitions (`model5` ... `model28`) inherit from `ConfigurableQCNN` and use the same architecture/resource metadata schema. Each model fixes its topology, ansatz and encoding; the dataset selects the images and class mapping.

### Choosing the optimization method

`--opt back` preserves the default training protocol above. To use SPSA:

```powershell
python -m training.train --model model17 --dataset mnist --seed 42 --opt spsa
```

SPSA perturbs all trainable parameters simultaneously, including the encoding.
Each mini-batch uses two loss evaluations at opposite random perturbations for
the update, plus one unperturbed forward pass for training metrics. No backward
pass, AdamW, gradient clipping or cosine scheduler is used in this mode.
The two-sided update follows the [SPSA gain sequences](https://docs.pennylane.ai/en/stable/code/api/pennylane.SPSAOptimizer.html):
`a_k = a / k**alpha`, `c_k = c / k**gamma`, with `k` starting at 1.
Defaults in `training/config.py` are `a=0.1`, `c=0.1`, `alpha=0.602`,
`gamma=0.101` (stability constant `A=0`); these can be tuned for an experiment.
Validation, best-model selection, early stopping and testing are shared with
backpropagation. Checkpoints preserve the SPSA iteration and random state.

Backpropagation keeps `results/<model>/<dataset>/seed<seed>/`. SPSA saves to
`results/spsa/<model>/<dataset>/seed<seed>/`, with its own summaries and
`results/spsa/last_suite.json`. Completed runs are skipped and `--no-resume`
restarts runs within the selected optimization method only. Existing checkpoints
without an `opt` field are treated as backpropagation checkpoints.

## Shared ansatz / pooling rules

- HUR6: 6 convolution parameters per layer + HUR pooling (2 parameters per pooling layer).
- HUR8: 10 convolution parameters per layer + HUR pooling (2 parameters per pooling layer).
- HUR9: 15 convolution parameters per layer + **trace-out pooling** as in the qcnn_benchmark 9b variant. No pooling gates and no trainable pooling parameters are introduced; after each reduction only the surviving wires are used by the following layer/readout.
- custom: 11-parameter Cartan-inspired two-qubit convolution block + custom transfer pooling (3 parameters per pooling layer).

All model variants terminate after the second pooling stage and directly read the output probabilities on wires `[0,4]`. No additional two-qubit classifier is applied after the final pooling stage.

## Model mapping

### Spatial-diagonal family

Fine layer: shared horizontal + vertical interactions.  
Coarse 2x2 layer: shared horizontal + vertical + both diagonal interactions.

| Model | Topology | Ansatz | Pooling | Encoding | Resize |
|---|---|---|---|---|---|
| model5 | spatial_diag | HUR6 | HUR | reupload | 8x8 |
| model6 | spatial_diag | HUR6 | HUR | E1 | 16x16 |
| model7 | spatial_diag | HUR6 | HUR | E3 | 16x16 |
| model8 | spatial_diag | HUR8 | HUR | reupload | 8x8 |
| model9 | spatial_diag | HUR8 | HUR | E1 | 16x16 |
| model10 | spatial_diag | HUR8 | HUR | E3 | 16x16 |
| model11 | spatial_diag | HUR9 | trace-out | reupload | 8x8 |
| model12 | spatial_diag | HUR9 | trace-out | E1 | 16x16 |
| model13 | spatial_diag | HUR9 | trace-out | E3 | 16x16 |
| model23 | spatial_diag | custom | custom transfer | reupload | 8x8 |
| model24 | spatial_diag | custom | custom transfer | E1 | 16x16 |
| model25 | spatial_diag | custom | custom transfer | E3 | 16x16 |

### qcnn_benchmark family

Convolution topology: even pairs followed by shifted pairs in each active-wire layer.

| Model | Topology | Ansatz | Pooling | Encoding | Resize |
|---|---|---|---|---|---|
| model14 | benchmark | HUR6 | HUR | reupload | 8x8 |
| model15 | benchmark | HUR6 | HUR | E1 | 16x16 |
| model16 | benchmark | HUR6 | HUR | E3 | 16x16 |
| model17 | benchmark | HUR8 | HUR | reupload | 8x8 |
| model18 | benchmark | HUR8 | HUR | E1 | 16x16 |
| model19 | benchmark | HUR8 | HUR | E3 | 16x16 |
| model20 | benchmark | HUR9 | trace-out | reupload | 8x8 |
| model21 | benchmark | HUR9 | trace-out | E1 | 16x16 |
| model22 | benchmark | HUR9 | trace-out | E3 | 16x16 |
| model26 | benchmark | custom | custom transfer | reupload | 8x8 |
| model27 | benchmark | custom | custom transfer | E1 | 16x16 |
| model28 | benchmark | custom | custom transfer | E3 | 16x16 |

## Custom and HUR9 model parameter counts

The counts below include all independent trainable parameters, including the encoding.

| Model | Parameters |
|---|---:|
| model23 | 508 |
| model24 | 44 |
| model25 | 28 |
| model26 | 508 |
| model27 | 44 |
| model28 | 28 |

HUR9 models use trace-out pooling for both topologies, so their full parameter counts are:

- model11 / model20: 510 (reupload)
- model12 / model21: 46 (E1)
- model13 / model22: 30 (E3)

## Project layout

```text
qcnn/
├── analyses/
│   ├── data_visualization.ipynb
│   ├── qcnn_transpilation.ipynb
│   └── results_visualization.ipynb
├── datasets/
│   ├── data/
│   ├── fashion_mnist.py
│   ├── mnist.py
│   ├── organamnist.py
│   ├── image_data.py
│   └── registry.py
├── models/
│   ├── model5.py
│   ├── ...
│   ├── model28.py
│   ├── registry.py
│   └── utilities/
│       ├── encoding.py
│       ├── ansatz.py
│       ├── pooling.py
│       ├── cost_function.py
│       └── qcnn_model.py
├── training/
│   ├── config.py
│   ├── checkpoint.py
│   ├── engine.py
│   └── train.py
├── results/
├── README.md
└── requirements.txt
```

## Commands

Every training command must specify:

- exactly one of `--model NAME`, `--models NAME ...` or `--all-models`;
- `--dataset NAME`;
- exactly one of `--seed N`, `--seeds N ...` or `--all-seeds`.

`--all-seeds` selects seeds 11, 42 and 59. The dataset name selects its loader;
all dataset files are cached under `datasets/data/`.
The input path is automatic and cannot be overridden from the command line.
`python -m training.train --help` (or `-h`) works without the required selections.

Run model17 on one dataset with one seed:

```bash
python -m training.train --model model17 --dataset organamnist --seed 42
python -m training.train --model model17 --dataset mnist --seed 42
```

Run one model with all default seeds:

```bash
python -m training.train --model model23 --dataset fashion4_1 --all-seeds
```

Run up to five models sequentially, each over the same requested seeds:

```bash
python -m training.train --models model23 model24 model25 model26 model27 --dataset fashion4_1 --all-seeds
```

More than five names in `--models` is rejected intentionally.

Run the complete model suite (all registered models):

```bash
python -m training.train --all-models --dataset fashion4_1 --all-seeds
```

Run the complete model suite on either of the other subsets with the same default seeds:

```bash
python -m training.train --all-models --dataset fashion4_2 --all-seeds
python -m training.train --all-models --dataset fashion4_3 --all-seeds
```

To train all three subsets sequentially in PowerShell with one command:

```powershell
foreach ($dataset in 'fashion4_1', 'fashion4_2', 'fashion4_3') { python -m training.train --all-models --dataset $dataset --all-seeds; if ($LASTEXITCODE -ne 0) { break } }
```

For every model, seeds that already contain a `test.json` in the results directory are skipped automatically unless `--no-resume` is specified.

All results are always saved under the project's `results/` directory. To retrain the complete suite from scratch in that same directory, replacing previous results:

```bash
python -m training.train --all-models --dataset fashion4_1 --all-seeds --no-resume
```

`--no-resume` restarts every selected seed, including completed ones. Run the command without this flag to resume interrupted training.

Run a small trial with reduced image counts and two epochs:

```bash
python -m training.train --model model17 --dataset mnist --seed 42 --epochs 2 --train-per-class 50 --val-per-class 20 --test-per-class 20
```

Trial runs use the same result folder as a full run of the same model/dataset/seed.
To replace a completed trial with the full experiment, use the default counts and
explicitly add `--no-resume`.

Run a custom seed list:

```bash
python -m training.train --model model23 --dataset fashion4_1 --seeds 11 59
```

## Google Colab

Use a CPU runtime, mount Drive, then change to the complete updated project folder
(for example `%cd /content/drive/MyDrive/qcnn`). Install the dependencies with
`%pip install -r requirements.txt`, then run either dataset:

```python
!python -m training.train --model model22 --dataset organamnist --all-seeds
!python -m training.train --model model22 --dataset mnist --all-seeds
```

Replace `--model model22` with `--all-models` to run the full suite. Results and
resumable checkpoints are saved in the Drive project's `results/` directory.

## Results

Each model and dataset is stored independently under `results/<model>/<dataset>/`.
`config.json`, `test.json` and `summary.json` record `original_labels` and
`class_names` in output-label order, including the row/column order of the
confusion matrix. `checkpoint_latest.pt` contains the full configuration and
training state for resume; `best_state.pt` contains the best model state and its
epoch/validation metadata.

```text
results/
└── modelX/
    └── fashion4_1/
        ├── seed11/
        │   ├── config.json
        │   ├── metrics.csv
        │   ├── best_state.pt
        │   ├── checkpoint_latest.pt
        │   └── test.json
        ├── seed42/
        ├── seed59/
        └── summary.json
```

`results/last_suite.json` records the last training command summary, including
commands that select a single model.

## Analysis notebooks

The notebooks read the existing project and training results. Select dataset names
with the configuration variables in each notebook; all five registered tasks are
supported. Their defaults select Fashion-MNIST runs already present in the project.
To analyse the new tasks, set the dataset variable to `mnist` or `organamnist` and
select models/seeds that have completed training.

Notebook imports require additional packages beyond the training requirements:
`pandas`, `matplotlib` and `IPython`; Hilbert-space analysis also uses
`scikit-learn` and optionally `tqdm`; transpilation uses `qiskit` and
`qiskit-ibm-runtime`.

### Hilbert-space and data visualization

Open `analyses/data_visualization.ipynb`. Set `MODEL_NAME`, `DATASET_NAME` and
`SEED` (defaults: `model17`, `fashion4_2`, `42`). The notebook requires that run's
`config.json`, `test.json` and `best_state.pt`. It samples 500 official test images
per class by default, using `SAMPLE_SEED = 123`; this selection can differ from
the test subset saved during training.

It compares quantum-state fidelities after encoding, after the first pooling and
before measurement, then displays MDS maps for the three stages. Tables and plots
are displayed inline; the notebook does not automatically export analysis files.

### Post-training transpilation analysis

Open `analyses/qcnn_transpilation.ipynb` after the selected models have finished
training on the dataset and training seed chosen for analysis. `MODELS` defaults
to all 24 registered models and can be reduced to the completed selection. All
analysis functions are embedded in the notebook. Set `DATASET` and
`TRAINING_SEED` in the configuration cell (defaults: `fashion4_1` and `42`).

The notebook loads `results/<model>/<DATASET>/seed<TRAINING_SEED>/best_state.pt`
for each selected model. Each checkpoint contains that run's best validation
state. Other datasets or training seeds are not required. Trainable parameters
come from the checkpoints; image-dependent encoding features are computed from
the selected test image. Missing runs cause an explicit error.

A single reproducible random image from the selected dataset's official test split is shared by every model, with the same dataset-specific preprocessing used for training. The selected image and its identifying metadata stay in memory.

Circuits are exported from the current PennyLane QNodes to Qiskit, including E3
state preparation and measurements on the two output wires. Defaults use local
`FakeBerlin`, `FakeSherbrooke` and `FakeTorino` snapshots, optimization level 3 and
transpiler seed 42, for 72 compilations of the 24 model circuits; no hardware jobs
or account are required. Each selected model is compiled once per backend; the
resource tables show the resulting counts and depths directly. A final cell
reports the RZ share of one-qubit gates on FakeBerlin for each selected model and
its arithmetic mean across models. The transpiler seed controls compilation and
is separate from the training seed used to load weights.

Analysis results, converted circuits, and the shared input stay in memory; tables are displayed inline. The notebook ends with the hardware transpilation results and contains no plots or matched topology comparisons. It does not automatically save analysis files.

### Training results analysis

Open `analyses/results_visualization.ipynb`. All loading, statistics and plotting functions are embedded; it reads only `config.json`, `test.json` and `metrics.csv`, without loading models or checkpoints.

The configuration selects models, datasets and seeds (defaults: all 24 models,
all three Fashion-MNIST subsets, seeds 11/42/59). `TRAINING_DATASET` and
`TRAINING_SEED` choose the training curves, defaulting to `fashion4_2` and `42`.
The curve dataset and seed must also occur in `DATASETS` and `SEEDS`. For example,
to analyse both new datasets, set `DATASETS = ["mnist", "organamnist"]` and
`TRAINING_DATASET = "mnist"` after completing the selected runs.

The notebook checks that config and test files exist, run identities match their
folders, class names and labels match between each run's config and test files,
and each model's encoding, ansatz, topology and parameter count remain consistent
across runs. It requires metrics CSVs only for the dataset/seed selected for the
training curves. A separate comparison cell selects `model5`, `model6` and
`model25`; adjust that selection too if analysing fewer models.

Figures show training/validation accuracy and loss with best-epoch markers, mean
confusion matrices, per-dataset and pooled test-accuracy boxplots, a comparative
accuracy table and pooled mean/standard-deviation bar charts. Training plots,
per-dataset confusions and boxplots use groups of six models by default
(`MODELS_PER_FIGURE`). Confusion percentages are normalized by true-class row
before averaging the selected seeds. `CONFUSION_VIEW="counts"` switches to mean
counts. Per-class recall statistics are computed in `class_by_seed` and
`class_summary`; the notebook does not plot separate per-class accuracy bars.

Boxplots show individual observations, medians, quartiles and a mean diamond.
Pooled summaries weight each selected dataset/seed combination equally; the
observations do not imply independent repetitions across datasets. Standard
deviations are sample standard deviations; a single observation has no estimated
sample standard deviation. The number of figures depends on the selections.

Tables and figures are displayed inline and retained in memory. The notebook does
not automatically write CSV files or PNG/SVG figures to `results/`. Existing
training results remain unchanged.

