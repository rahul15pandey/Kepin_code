# KePIN: Koopman-Enhanced Physics-Informed Network for Predictive Modelling

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/tensorflow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Paper:** *KePIN: Koopman-Enhanced Physics-Informed Network for Remaining Useful Life Prediction and Multi-Domain Forecasting*
>
> A hybrid deep-learning architecture that embeds a **Koopman operator** (SVD-parameterised) into an encoder–decoder backbone with a seven-component physics-informed loss.  
> KePIN captures latent linear dynamics of nonlinear systems and enforces physical constraints—monotonicity, spectral stability, and multi-step consistency—through learnable Kendall uncertainty weighting.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Configuration & Reproducibility](#configuration--reproducibility)
6. [Quick Start](#quick-start)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Ablation Study](#ablation-study)
10. [Reproducing Paper Figures](#reproducing-paper-figures)
11. [Key Results](#key-results)
12. [Citation](#citation)
13. [License](#license)

---

## Architecture

```
Input (B, T, d)
  │
  ├─ ResConv1D + Squeeze-and-Excitation  ×N blocks
  │
  ├─ Bidirectional LSTM
  │
  ├─ Multi-Head Attention
  │
  ├─ SVD-Parameterised Koopman Operator
  │      K = U · diag(σ(s)) · Vᵀ     (latent_dim × latent_dim)
  │      ├─ 1-step prediction: ẑ_{t+1} = K ẑ_t
  │      ├─ Multi-step rollout: ẑ_{t+k} = K^k ẑ_t
  │      └─ Spectral features:  |λ|, ∠λ, Re(λ), Im(λ)
  │
  ├─ Concatenation  [encoder_out ‖ spectral_features]
  │
  └─ Deep Prediction Head (256 → 128 → 64, skip connections)
       └─ ŷ  (RUL or forecast)
```

**Auto-configuration tiers:**

| Tier   | Features (d) | Latent dim | Blocks | Heads | Approx. params |
|--------|:------------:|:----------:|:------:|:-----:|:--------------:|
| Small  | ≤ 10         | 64         | 3      | 4     | ~504 K         |
| Medium | ≤ 16         | 128        | 4      | 4     | ~1.4 M         |
| Large  | > 16         | 128        | 4      | 8     | ~1.7 M         |

---

## Repository Structure

```
KPDD/
├── kepin/                          # Refactored Python package
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kepin_model.py          # KePINModel, auto_configure(), build_kepin_model()
│   │   ├── koopman.py              # KoopmanOperator layer
│   │   └── baselines.py           # 7 baseline architectures
│   ├── losses/
│   │   ├── __init__.py
│   │   └── composite.py           # 7-component loss + Kendall uncertainty
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # KePINTrainer, EnhancedKePINTrainer
│   │   └── augmentation.py        # mixup, noise injection, label smoothing
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py              # Dataset loading (re-exports legacy loaders)
│   └── utils/
│       ├── __init__.py
│       ├── gpu.py                 # GPU setup, A100 optimisation
│       ├── metrics.py             # RMSE, MAE, R², NASA score, eigenvalue recovery
│       └── preprocessing.py       # 4D→3D, EMA smoothing
├── scripts/
│   ├── train.py                   # CLI: training entry point
│   ├── evaluate.py                # CLI: evaluation entry point
│   └── ablation.py                # CLI: ablation study entry point
├── configs/
│   ├── datasets_kepin_config.json      # C-MAPSS datasets
│   ├── datasets_all_config.json        # All datasets (incl. synthetic)
│   ├── datasets_cmapss_config.json     # C-MAPSS only
│   ├── datasets_multidomain_config.json # Cross-domain datasets
│   └── run_kepin.json                   # Run config (seed, data_root, training)
├── code/                          # Original research scripts (preserved)
│   ├── kepin_model.py
│   ├── kepin_losses.py
│   ├── kepin_training.py
│   ├── kepin_cmapss_optimized.py
│   ├── GenericTimeSeriesDataset.py
│   ├── CMAPSSDataset.py
│   └── ...
├── experiments_result/             # Saved models, predictions, figures
├── paper/                          # LaTeX source and figures
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

### Prerequisites

- **Python** ≥ 3.10
- **CUDA** ≥ 12.0 (for GPU training; tested on NVIDIA A100-PCIE-40GB)
- **conda** (recommended)

For bit-for-bit replication, record the exact CUDA driver and cuDNN versions
used when generating paper results and archive them alongside the release.

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/KPDD.git
cd KPDD

# Create and activate Conda environment
conda create -n kepin python=3.10 -y
conda activate kepin

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Optional: PyTorch baselines (install the CUDA-matched wheel from pytorch.org)
# pip install torch==2.3.1
```

### Key dependencies

| Package       | Version |
|---------------|---------|
| tensorflow    | ≥ 2.16 |
| keras         | ≥ 3.0  |
| numpy         | ≥ 1.24 |
| pandas        | ≥ 2.0  |
| scikit-learn  | ≥ 1.3  |
| matplotlib    | ≥ 3.8  |
| scipy         | ≥ 1.11 |
| torch (optional) | ≥ 2.0 |

---

## Data Preparation

### C-MAPSS (Turbofan Engine Degradation)

The C-MAPSS dataset files (`train_FD001.txt`, …, `RUL_FD004.txt`) should be placed in:

```
code/C-MAPSS-Data/
```

> The data is included in this repository. If missing, download from the [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq).

### Cross-Domain Datasets

For Jena Climate, Cylinder Wake, and Building Energy datasets:

```bash
python code/download_datasets.py
```

This downloads datasets into `code/datasets/`.

### Synthetic Datasets

Synthetic ODE, weather, and finance datasets are generated on-the-fly by `GenericTimeSeriesDataset` — no download required.

---

## Configuration & Reproducibility

### Run configuration

Use a single JSON run config to control seeds, dataset configs, and training defaults:

```bash
python scripts/train.py --run_config configs/run_kepin.json
```

You can still override any field with CLI flags (e.g., `--epochs`, `--seed`, `--output_dir`).

### Data root handling

All dataset paths are resolved relative to `KEPIN_DATA_ROOT` (or `KPDD_DATA_ROOT`) if set. This
lets you keep large datasets outside the repo while reusing the same configs.

```bash
export KEPIN_DATA_ROOT=/path/to/data
python scripts/train.py --config configs/datasets_kepin_config.json
```

### Deterministic runs

For deterministic kernels, pass `--seed` and `--deterministic`. The CLI disables XLA and mixed
precision automatically in deterministic mode to minimize nondeterministic kernels.

```bash
python scripts/train.py --config configs/datasets_kepin_config.json \
    --seed 42 --deterministic
```

Exact hardware, driver, and CUDA/cuDNN versions still matter for bit-for-bit replication.

---

## Quick Start

```python
from kepin.models import KePINModel, auto_configure, build_kepin_model
from kepin.losses import make_kepin_loss
from kepin.training import KePINTrainer
from kepin.utils.gpu import setup_gpu
import keras

# Setup GPU
setup_gpu()

# Build model (auto-configures architecture based on data shape)
model = build_kepin_model(seq_len=30, n_features=15, n_train=16000)

# Create loss and optimizer
loss_fn = make_kepin_loss(
    loss_weights_layer=model.loss_weight_layer,
    use_auto_weights=True,
    domain_mode="degradation",
)
optimizer = keras.optimizers.Adam(learning_rate=8e-4)

# Train
trainer = KePINTrainer(model, loss_fn, optimizer)
history = trainer.fit(X_train, Y_train, X_val, Y_val,
                      epochs=200, batch_size=512, patience=40)

# Predict
import tensorflow as tf
y_pred = model.predict_rul(tf.constant(X_test)).numpy()
```

---

## Training

### Train from a run config

```bash
python scripts/train.py --run_config configs/run_kepin.json
```

### Train on all C-MAPSS datasets

```bash
python scripts/train.py --config configs/datasets_kepin_config.json \
    --epochs 200 --patience 40 --n_runs 3
```

### Train a single dataset (by index)

```bash
python scripts/train.py --config configs/datasets_kepin_config.json \
    --dataset_idx 0 --epochs 200
```

### Enhanced training (SWA + mixup + warmup)

```bash
python scripts/train.py --config configs/datasets_kepin_config.json \
    --enhanced --epochs 250 --n_runs 3 --patience 50
```

### Quick synthetic test

```bash
python scripts/train.py --mode synthetic --epochs 50
```

### Key CLI arguments

| Argument          | Default | Description                              |
|-------------------|---------|------------------------------------------|
| `--config`        | —       | Path to JSON dataset config              |
| `--dataset_idx`   | all     | Train only this dataset index            |
| `--epochs`        | 200     | Maximum training epochs                  |
| `--batch_size`    | auto    | Override batch size (auto for A100)      |
| `--lr`            | auto    | Override learning rate                   |
| `--patience`      | 40      | Early stopping patience                  |
| `--n_runs`        | 1       | Independent runs per dataset             |
| `--enhanced`      | false   | Use SWA + mixup + warmup                |
| `--no_auto_weights` | false | Disable Kendall uncertainty weighting    |

---

## Evaluation

### Evaluate saved predictions

```bash
# Single file
python scripts/evaluate.py --npz experiments_result/kepin/CMAPSS_FD001/predictions_CMAPSS_FD001_run0.npz

# All predictions in a directory
python scripts/evaluate.py --dir experiments_result/kepin_20260224_071535 --nasa_score

# Save results to CSV
python scripts/evaluate.py --dir experiments_result/ --output_csv results_table.csv
```

---

## Ablation Study

Remove individual loss components to measure their contribution:

```bash
python scripts/ablation.py --config configs/datasets_kepin_config.json \
    --dataset_idx 0 --epochs 200 --n_runs 1

# Run specific variants only
python scripts/ablation.py --config configs/datasets_kepin_config.json \
    --dataset_idx 0 --variants full no_koopman_loss no_spectral no_monotonicity
```

### Ablation variants

| Variant              | Description                              |
|----------------------|------------------------------------------|
| `full`               | All 7 loss components (baseline)         |
| `no_koopman_loss`    | Remove Koopman one-step prediction loss  |
| `no_spectral`        | Remove spectral regularisation           |
| `no_monotonicity`    | Remove monotonicity penalty              |
| `no_multi_step`      | Remove multi-step rollout loss           |
| `no_auto_weights`    | Replace Kendall weighting with fixed     |
| `no_koopman_module`  | Remove entire Koopman operator           |

---

## Reproducing Paper Figures

The paper figures are generated from saved prediction and history artifacts
(`predictions_*.npz`, `history_*.csv`, `eigenvalues_*.npz`). A minimal
reproduction flow is:

```bash
# 1) Train all datasets into a single results directory
python scripts/train.py --config configs/datasets_all_config.json \
    --output_dir experiments_result/kepin_repro

# 2) Generate the paper figures from that directory
python code/generate_paper_figures.py \
    --results_dir experiments_result/kepin_repro \
    --cmapss_dir experiments_result/kepin_repro \
    --kepin_final_dir experiments_result/kepin_repro \
    --improved_dir experiments_result/kepin_repro \
    --fig_dir paper/figures
```

If your results are split across multiple runs, pass each directory explicitly
to `--cmapss_dir`, `--kepin_final_dir`, and `--improved_dir`.

---

## Key Results

### C-MAPSS Benchmark (3-run ensemble, test RMSE)

| Model              | FD001  | FD002  | FD003  | FD004  |
|--------------------|:------:|:------:|:------:|:------:|
| MLP                | 14.21  | 16.42  | 13.58  | 19.27  |
| LSTM               | 13.47  | 15.89  | 11.86  | 18.41  |
| BiLSTM             | 13.02  | 15.61  | 12.04  | 17.98  |
| CNN-LSTM           | 13.15  | 16.03  | 12.31  | 18.12  |
| Transformer        | 13.28  | 14.68  | 12.19  | 17.02  |
| **KePIN (ours)**   | **12.92** | **15.08** | **11.42** | **17.04** |

### Cross-Domain Generalisation

| Dataset          | Metric | Best Baseline | KePIN  |
|------------------|:------:|:-------------:|:------:|
| Jena Climate     | RMSE   | 5.777         | 5.98   |
| Cylinder Wake    | R²     | 0.8994        | 0.900  |
| Building Energy  | R²     | 0.9642        | 0.963  |

---

## Citation

If you use this code in your research, please cite:

Paper DOI: TBD

Zenodo DOI: TBD


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
