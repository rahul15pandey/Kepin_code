# -*- coding: utf-8 -*-
"""
KePIN Ablation Study — Systematic evaluation of each novelty component.

Five ablation configurations tested across all 6 domains:

  A) Baseline FCN (no Koopman, no physics loss — vanilla Conv1D + SE)
  B) KePIN w/o spectral loss  (Koopman operator but no eigenvalue constraint)
  C) KePIN w/o multi-step     (one-step Koopman only, no rollout fidelity)
  D) KePIN w/o auto-weights   (replace Kendall uncertainty with fixed weights)
  E) KePIN Full               (all components — proposed method)

For full baseline comparison (MLP, LSTM, BiLSTM, CNN-LSTM, Vanilla FCN,
PI-DP-FCN, Transformer) combined with ablation, use:
  python kepin_comparison_study.py --config datasets_kepin_config.json

Outputs:
  - Grouped bar charts (per metric)
  - Radar / spider chart (multi-metric overview)
  - Cross-domain heatmap
  - LaTeX-ready results table
  - Summary CSV

Usage:
  # Full ablation on all 6 datasets
  python kepin_ablation.py --config datasets_kepin_config.json

  # Single dataset (by index)
  python kepin_ablation.py --config datasets_kepin_config.json --dataset_idx 3

  # Custom epochs & runs
  python kepin_ablation.py --config datasets_kepin_config.json --epochs 100 --n_runs 3

  # From pre-saved results (skip training)
  python kepin_ablation.py --results_dir experiments_result/kepin_ablation_20240101
"""

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# Local imports
import GenericTimeSeriesDataset as GDS
from kepin_model import KePINModel, build_kepin_model, convert_4d_to_3d, auto_configure
from kepin_losses import (
    KePINLossWeights, make_kepin_loss,
    rul_mse_loss, koopman_one_step_loss, spectral_stability_loss,
    monotonicity_loss, multi_step_loss, asymmetric_loss, slope_matching_loss,
)
from kepin_training import (
    KePINTrainer, apply_ema_smoothing,
    rmse_np, mae_np, physics_metrics_np, eigenvalue_recovery_error,
)
from gpu_config import setup_gpu, get_batch_size, get_learning_rate

# ---------- GPU setup (A100 40 GB) ----------
setup_gpu(mixed_precision=False, xla=False, verbose=True)

# ---------- Reproducibility ----------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- Paths ----------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))

# =========================================================================
# Plot style
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colour palette (colour-blind friendly)
COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]
HATCHES = ["", "//", "\\\\", "xx", "..", "oo"]


# =========================================================================
# Ablation configuration definitions
# =========================================================================

def get_ablation_configs():
    """Return the 5 ablation configurations.

    Each config defines which loss terms to include and whether to use
    auto-weighting. The configs progressively add components to isolate
    the contribution of each novelty.

    Returns:
        list of dicts, each with:
            name:       short human-readable label
            tag:        filesystem-safe tag
            use_koopman: whether to use Koopman module
            loss_config: dict for make_kepin_loss or custom weights
            description: one-line description
    """
    return [
        {
            "name": "A: Baseline FCN",
            "tag": "A_baseline_fcn",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.0, "spectral": 0.0,
                "mono": 0.0, "multi_step": 0.0, "asym": 0.0, "slope": 0.0,
            },
            "description": "Vanilla Conv1D + SE + dual pooling, MSE loss only",
        },
        {
            "name": "B: w/o Spectral",
            "tag": "B_no_spectral",
            "use_koopman": True,
            "use_auto_weights": True,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.1, "spectral": 0.0,
                "mono": 0.001, "multi_step": 0.05, "asym": 0.05, "slope": 0.0003,
            },
            "description": "KePIN with spectral stability loss disabled",
        },
        {
            "name": "C: w/o Multi-step",
            "tag": "C_no_multistep",
            "use_koopman": True,
            "use_auto_weights": True,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.1, "spectral": 0.01,
                "mono": 0.001, "multi_step": 0.0, "asym": 0.05, "slope": 0.0003,
            },
            "description": "KePIN without multi-step rollout loss",
        },
        {
            "name": "D: w/o Auto-Wt",
            "tag": "D_no_autoweight",
            "use_koopman": True,
            "use_auto_weights": False,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.1, "spectral": 0.01,
                "mono": 0.001, "multi_step": 0.05, "asym": 0.05, "slope": 0.0003,
            },
            "description": "KePIN with fixed weights (no Kendall uncertainty)",
        },
        {
            "name": "E: KePIN Full",
            "tag": "E_kepin_full",
            "use_koopman": True,
            "use_auto_weights": True,
            "fixed_weights": None,  # auto-weighted, but defaults used as init
            "description": "Full KePIN with all components (proposed method)",
        },
    ]


# =========================================================================
# Additional metrics
# =========================================================================

def r_squared_np(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true.flatten() - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_true.flatten() - np.mean(y_true.flatten())) ** 2) + 1e-10
    return float(1.0 - ss_res / ss_tot)


def nasa_score_np(y_true, y_pred):
    """NASA asymmetric scoring metric (lower is better)."""
    s = 0.0
    yt = y_true.flatten()
    yp = y_pred.flatten()
    for i in range(len(yp)):
        if yp[i] > yt[i]:
            s += math.exp((yp[i] - yt[i]) / 10.0) - 1.0
        else:
            s += math.exp((yt[i] - yp[i]) / 13.0) - 1.0
    return float(s)


def compute_all_metrics(y_true, y_pred) -> dict:
    """Compute full metric suite for one prediction set."""
    mono_viol, slope_err = physics_metrics_np(y_true, y_pred)
    return {
        "RMSE": rmse_np(y_true, y_pred),
        "MAE": mae_np(y_true, y_pred),
        "R2": r_squared_np(y_true, y_pred),
        "MonoViol": mono_viol,
        "SlopeRMSE": slope_err,
    }


# =========================================================================
# Build model for a specific ablation config
# =========================================================================

class BaselineFCN(keras.Model):
    """Minimal Conv1D + SE + dual pooling baseline (no Koopman).

    This isolates the encoder architecture contribution from the Koopman
    operator and physics-informed losses.
    """

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, **kwargs):
        super().__init__(**kwargs)

        seq_len, n_features = input_shape_tuple
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len, n_train)

        self.arch_config = arch_config
        self.seq_len = seq_len
        self.n_features = n_features

        # Encoder blocks + SE layers (all created in __init__ to avoid tf.function issues)
        self.encoder_blocks = []
        self.se_blocks = []
        for i in range(arch_config["n_blocks"]):
            f = arch_config["filters"][i]
            self.encoder_blocks.append({
                "conv": keras.layers.Conv1D(
                    f,
                    arch_config["kernels"][i],
                    padding="same",
                    kernel_initializer="he_normal",
                    name=f"enc_conv_{i}",
                ),
                "bn": keras.layers.BatchNormalization(name=f"enc_bn_{i}"),
                "relu": keras.layers.Activation("relu", name=f"enc_relu_{i}"),
            })
            se_dim = max(f // 8, 4)
            self.se_blocks.append({
                "gap": keras.layers.GlobalAveragePooling1D(name=f"se_gap_{i}"),
                "fc1": keras.layers.Dense(se_dim, activation="relu", name=f"se_fc1_{i}"),
                "fc2": keras.layers.Dense(f, activation="sigmoid", name=f"se_fc2_{i}"),
                "reshape": keras.layers.Reshape((1, f), name=f"se_reshape_{i}"),
                "mul": keras.layers.Multiply(name=f"se_mul_{i}"),
            })

        # Dual pooling
        self.gap = keras.layers.GlobalAveragePooling1D(name="gap")
        self.gmp = keras.layers.GlobalMaxPooling1D(name="gmp")
        self.concat = keras.layers.Concatenate(name="dual_concat")

        # RUL head
        last_filters = arch_config["filters"][-1]
        self.head_dense = keras.layers.Dense(
            64, activation="relu", kernel_initializer="he_normal", name="head_dense",
        )
        self.head_dropout = keras.layers.Dropout(
            arch_config["dropout"], name="head_dropout",
        )
        self.head_output = keras.layers.Dense(
            1, activation="relu", dtype="float32", name="rul_output",
        )

        # Dummy loss weights (unused but keeps interface compatible)
        self.loss_weight_layer = KePINLossWeights(n_losses=7, name="loss_weights")

        # Build
        dummy = tf.zeros((1, seq_len, n_features))
        _ = self(dummy, training=False)

    def call(self, inputs, training=None):
        x = inputs
        for i, block in enumerate(self.encoder_blocks):
            x = block["conv"](x)
            x = block["bn"](x, training=training)
            x = block["relu"](x)
            # SE block (all layers pre-created in __init__)
            se = self.se_blocks[i]
            s = se["gap"](x)
            s = se["fc1"](s)
            s = se["fc2"](s)
            s = se["reshape"](s)
            x = se["mul"]([x, s])

        pool_avg = self.gap(x)
        pool_max = self.gmp(x)
        pooled = self.concat([pool_avg, pool_max])

        h = self.head_dense(pooled)
        h = self.head_dropout(h, training=training)
        rul_pred = self.head_output(h)

        # Return dummy koopman outputs for interface compatibility
        d = self.arch_config.get("latent_dim", 32)
        batch_size = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        dummy_koopman = {
            "one_step_pred": tf.zeros((batch_size, T - 1, d)),
            "one_step_target": tf.zeros((batch_size, T - 1, d)),
            "multi_step_pred": tf.zeros((batch_size, 1, 1, d)),
            "multi_step_target": tf.zeros((batch_size, 1, 1, d)),
            "eigenvalues": tf.zeros((d,), dtype=tf.complex64),
            "final_state": tf.zeros((batch_size, d)),
        }

        return rul_pred, dummy_koopman

    def predict_rul(self, inputs):
        rul_pred, _ = self(inputs, training=False)
        return rul_pred

    def get_eigenvalues(self):
        """Return zeros (no Koopman operator)."""
        d = self.arch_config.get("latent_dim", 32)
        return np.zeros(d, dtype=np.complex128)

    def get_koopman_matrix(self):
        d = self.arch_config.get("latent_dim", 32)
        return np.zeros((d, d))


def build_ablation_model(ab_config: dict, seq_len: int, n_features: int,
                         n_train: int = None):
    """Build a model for a specific ablation configuration.

    Args:
        ab_config:  one item from get_ablation_configs()
        seq_len:    sequence length
        n_features: number of input features
        n_train:    number of training samples

    Returns:
        model:   KePINModel or BaselineFCN
        loss_fn: composite loss function
    """
    arch_config = auto_configure(n_features, seq_len, n_train)

    if not ab_config["use_koopman"]:
        # Baseline: no Koopman operator
        model = BaselineFCN(
            input_shape_tuple=(seq_len, n_features),
            arch_config=arch_config,
            n_train=n_train,
        )
    else:
        # KePIN variant — domain_mode determines active losses
        domain_mode = ab_config.get("domain_mode", "degradation")
        n_active_losses = 4 if domain_mode == "forecasting" else 7
        model = build_kepin_model(
            seq_len, n_features,
            n_train=n_train,
            arch_config=arch_config,
            n_active_losses=n_active_losses,
        )

    # Build loss function
    if ab_config["use_auto_weights"] and ab_config["use_koopman"]:
        loss_fn = make_kepin_loss(
            loss_weights_layer=model.loss_weight_layer,
            use_auto_weights=True,
        )
    else:
        fixed_w = ab_config.get("fixed_weights") or {
            "rul": 1.0, "koopman": 0.0, "spectral": 0.0,
            "mono": 0.0, "multi_step": 0.0, "asym": 0.0, "slope": 0.0,
        }
        loss_fn = make_kepin_loss(
            loss_weights_layer=None,
            use_auto_weights=False,
            fixed_weights=fixed_w,
        )

    return model, loss_fn


# =========================================================================
# Single ablation run
# =========================================================================

def run_single_ablation(ab_config: dict, ds_config: dict,
                        output_dir: str,
                        epochs: int = 200, batch_size: int = None,
                        lr: float = None, patience: int = 40,
                        run_id: int = 0, verbose: int = 1) -> dict:
    """Train one ablation configuration on one dataset.

    Returns:
        result dict with metrics, predictions, and config metadata
    """
    ab_name = ab_config["name"]
    ds_name = ds_config.get("name", "unknown")
    tag = f"{ab_config['tag']}_{ds_name}_run{run_id}"

    print(f"\n  --- {ab_name} on {ds_name} (run {run_id}) ---")

    # Load dataset
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()

    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)

    # EMA smoothing
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)

    seq_len, n_feat, n_train = X_train.shape[1], X_train.shape[2], X_train.shape[0]

    # Auto batch size / LR (A100-optimised)
    if batch_size is None:
        model_cat = "kepin" if ab_config["use_koopman"] else "kepin"
        batch_size = get_batch_size(n_train, seq_len, n_feat, model_type=model_cat)
    if lr is None:
        lr = get_learning_rate(batch_size, base_lr=0.001, base_batch=256)

    # Build model and loss
    model, loss_fn = build_ablation_model(
        ab_config, seq_len, n_feat, n_train,
    )

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    # Train
    trainer = KePINTrainer(model, loss_fn, optimizer)
    history = trainer.fit(
        X_train, Y_train, X_test, Y_test,
        epochs=epochs, batch_size=batch_size,
        patience=patience, initial_lr=lr,
        verbose=verbose,
    )

    # Evaluate
    Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
    metrics = compute_all_metrics(Y_test, Y_pred)

    # Additional metrics
    try:
        metrics["NASAScore"] = nasa_score_np(Y_test, Y_pred)
    except OverflowError:
        metrics["NASAScore"] = float("inf")

    # Eigenvalue info
    eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(eigs))[::-1]

    # Eigenvalue recovery (synthetic ODE only)
    eig_recovery = None
    if hasattr(ds, "ode_true_K_eigenvalues"):
        eig_recovery = eigenvalue_recovery_error(eigs, ds.ode_true_K_eigenvalues)

    # Print results
    print(f"    RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
          f"R2={metrics['R2']:.4f}  Mono={metrics['MonoViol']:.6f}")

    # Save predictions
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        os.path.join(output_dir, f"pred_{tag}.npz"),
        y_true=Y_test, y_pred=Y_pred,
    )

    # Save history
    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
    })
    hist_df.to_csv(os.path.join(output_dir, f"history_{tag}.csv"), index=False)

    result = {
        "ablation": ab_name,
        "ablation_tag": ab_config["tag"],
        "dataset": ds_name,
        "run_id": run_id,
        **metrics,
        "epochs_trained": len(history["epoch"]),
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "top_eig_mags": eig_mags[:5].tolist(),
        "eig_recovery": eig_recovery,
        "description": ab_config["description"],
    }
    return result


# =========================================================================
# Full ablation study
# =========================================================================

def run_full_ablation(config_path: str, output_base: str = None,
                      epochs: int = 200, n_runs: int = 1,
                      dataset_idx: int = None,
                      batch_size: int = None, lr: float = None,
                      patience: int = 40, verbose: int = 1) -> pd.DataFrame:
    """Run the complete ablation study: 5 configs x N datasets x M runs.

    Args:
        config_path:  path to datasets JSON config
        output_base:  output directory
        epochs:       max epochs per run
        n_runs:       independent runs per config-dataset pair
        dataset_idx:  if set, only run on this dataset index
        batch_size:   override auto batch size
        lr:           override auto learning rate
        patience:     early stopping patience
        verbose:      verbosity

    Returns:
        results_df: DataFrame with all results
    """
    with open(config_path, "r") as f:
        ds_configs = json.load(f)

    if dataset_idx is not None:
        ds_configs = [ds_configs[dataset_idx]]

    ab_configs = get_ablation_configs()

    if output_base is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"kepin_ablation_{timestamp}")

    os.makedirs(output_base, exist_ok=True)

    print(f"{'='*70}")
    print(f"  KePIN ABLATION STUDY")
    print(f"  {len(ab_configs)} configs x {len(ds_configs)} datasets x {n_runs} runs")
    print(f"  Output: {output_base}")
    print(f"{'='*70}")

    all_results = []

    for ds_config in ds_configs:
        ds_name = ds_config.get("name", "unknown")
        ds_dir = os.path.join(output_base, ds_name)

        for ab_config in ab_configs:
            for run in range(n_runs):
                try:
                    result = run_single_ablation(
                        ab_config, ds_config, ds_dir,
                        epochs=epochs, batch_size=batch_size,
                        lr=lr, patience=patience,
                        run_id=run, verbose=verbose,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"    FAILED: {ab_config['name']} on {ds_name} run {run}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "ablation": ab_config["name"],
                        "ablation_tag": ab_config["tag"],
                        "dataset": ds_name,
                        "run_id": run,
                        "error": str(e),
                    })

    # Build results dataframe
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(output_base, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved raw results to {results_path}")

    # JSON backup
    json_path = os.path.join(output_base, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate plots and tables
    valid_df = results_df[~results_df.get("error", pd.Series(dtype=str)).notna() |
                          (results_df.get("RMSE", pd.Series(dtype=float)).notna())]
    if "RMSE" in valid_df.columns and len(valid_df) > 0:
        try:
            generate_ablation_plots(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: Plot generation failed: {e}")
        try:
            generate_latex_table(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: LaTeX table generation failed: {e}")
        try:
            generate_heatmap(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: Heatmap generation failed: {e}")

    return results_df


# =========================================================================
# Visualization: Grouped Bar Charts
# =========================================================================

def generate_ablation_plots(df: pd.DataFrame, output_dir: str):
    """Publication-quality grouped bar charts comparing ablation configs."""

    metrics_to_plot = ["RMSE", "MAE", "R2", "MonoViol", "SlopeRMSE"]
    lower_is_better = {"RMSE": True, "MAE": True, "R2": False,
                       "MonoViol": True, "SlopeRMSE": True}

    # Aggregate: mean ± std across runs
    agg = df.groupby(["ablation", "dataset"]).agg(
        {m: ["mean", "std"] for m in metrics_to_plot if m in df.columns}
    ).reset_index()

    datasets = df["dataset"].unique()
    ablations = df["ablation"].unique()
    n_datasets = len(datasets)
    n_ablations = len(ablations)

    for metric in metrics_to_plot:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(8, n_datasets * 1.5), 5))

        x = np.arange(n_datasets)
        width = 0.8 / n_ablations

        for i, ab in enumerate(ablations):
            means = []
            stds = []
            for ds in datasets:
                mask = (df["ablation"] == ab) & (df["dataset"] == ds)
                vals = df.loc[mask, metric].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)

            bars = ax.bar(
                x + i * width - (n_ablations - 1) * width / 2,
                means, width,
                yerr=stds if any(s > 0 for s in stds) else None,
                label=ab,
                color=COLORS[i % len(COLORS)],
                hatch=HATCHES[i % len(HATCHES)],
                edgecolor="black", linewidth=0.5,
                capsize=3,
            )

        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.set_title(f"Ablation: {metric} across domains")
        ax.set_xticks(x)
        ax.set_xticklabels([_shorten_name(d) for d in datasets],
                           rotation=30, ha="right")
        ax.legend(fontsize=7, ncol=2, loc="best")

        # Add "lower is better" / "higher is better" annotation
        direction = "lower" if lower_is_better.get(metric, True) else "higher"
        ax.annotate(f"({direction} is better)", xy=(0.99, 0.01),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=8, fontstyle="italic", alpha=0.6)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"ablation_{metric}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"ablation_{metric}.pdf"))
        plt.close(fig)

    print(f"  Saved grouped bar charts to {output_dir}")


# =========================================================================
# Visualization: Radar / Spider Chart
# =========================================================================

def generate_radar_chart(df: pd.DataFrame, output_dir: str):
    """Multi-metric radar chart comparing ablation configs (per dataset)."""

    metrics = ["RMSE", "MAE", "MonoViol", "SlopeRMSE", "R2"]
    available_metrics = [m for m in metrics if m in df.columns]

    if len(available_metrics) < 3:
        print("  Skipping radar chart: not enough metrics")
        return

    datasets = df["dataset"].unique()

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        ablations = ds_df["ablation"].unique()

        if len(ablations) < 2:
            continue

        # Normalise metrics to [0, 1] for radar
        normalised = {}
        for ab in ablations:
            ab_df = ds_df[ds_df["ablation"] == ab]
            vals = []
            for m in available_metrics:
                v = ab_df[m].mean()
                vals.append(v)
            normalised[ab] = vals

        # Normalise per metric (min-max over ablation configs)
        norm_data = {}
        for ab in ablations:
            norm_data[ab] = []

        for mi, m in enumerate(available_metrics):
            all_vals = [normalised[ab][mi] for ab in ablations]
            vmin, vmax = min(all_vals), max(all_vals)
            rng = vmax - vmin if vmax > vmin else 1.0
            for ab in ablations:
                # For R2, higher is better → invert
                if m == "R2":
                    norm_data[ab].append((normalised[ab][mi] - vmin) / rng)
                else:
                    # Lower is better → invert so that outer = better
                    norm_data[ab].append(1.0 - (normalised[ab][mi] - vmin) / rng)

        # Plot
        n_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        for i, ab in enumerate(ablations):
            values = norm_data[ab] + [norm_data[ab][0]]
            ax.plot(angles, values, "o-", label=ab,
                    color=COLORS[i % len(COLORS)], linewidth=1.5, markersize=4)
            ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

        ax.set_thetagrids(
            np.degrees(angles[:-1]),
            available_metrics,
        )
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Ablation Radar — {_shorten_name(ds)}", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)

        plt.tight_layout()
        safe_ds = ds.replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(output_dir, f"radar_{safe_ds}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"radar_{safe_ds}.pdf"))
        plt.close(fig)

    print(f"  Saved radar charts to {output_dir}")


# =========================================================================
# Visualization: Cross-domain heatmap
# =========================================================================

def generate_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap of RMSE (or key metric) across ablation configs x datasets.

    Rows = ablation configs, Columns = datasets.
    """
    metric = "RMSE"
    if metric not in df.columns:
        return

    pivot = df.groupby(["ablation", "dataset"])[metric].mean().unstack(fill_value=np.nan)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                    max(4, len(pivot.index) * 0.8)))

    data = pivot.values
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmedian(data) else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=9)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([_shorten_name(c) for c in pivot.columns],
                       rotation=35, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Ablation Config")
    ax.set_title(f"Cross-Domain Heatmap — {metric}")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric)

    # Highlight best (lowest RMSE) per dataset column
    for j in range(data.shape[1]):
        col = data[:, j]
        if not np.all(np.isnan(col)):
            best_i = np.nanargmin(col)
            ax.add_patch(plt.Rectangle((j - 0.5, best_i - 0.5), 1, 1,
                                       fill=False, edgecolor="green",
                                       linewidth=2.5))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"heatmap_{metric}.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, f"heatmap_{metric}.pdf"))
    plt.close(fig)

    print(f"  Saved heatmap to {output_dir}")


# =========================================================================
# LaTeX Table
# =========================================================================

def generate_latex_table(df: pd.DataFrame, output_dir: str):
    """Generate publication-ready LaTeX table of ablation results.

    Format: rows = datasets, columns = metrics for each ablation config.
    Best values are bolded.
    """
    metrics = ["RMSE", "MAE", "R2"]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return

    datasets = sorted(df["dataset"].unique())
    ablations = sorted(df["ablation"].unique())

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study results across 6 domains. "
                 "Best values per dataset are \\textbf{bolded}.}")
    lines.append("\\label{tab:ablation}")

    # Column spec
    n_cols = 1 + len(available_metrics) * len(ablations)
    col_spec = "l" + "|".join(["c" * len(available_metrics)] * len(ablations))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header row 1: ablation config names
    header1 = "Dataset"
    for ab in ablations:
        short_ab = ab.split(":")[0].strip() if ":" in ab else ab
        header1 += f" & \\multicolumn{{{len(available_metrics)}}}{{c}}{{{short_ab}}}"
    header1 += " \\\\"
    lines.append(header1)

    # Header row 2: metric names
    header2 = ""
    for ab in ablations:
        for m in available_metrics:
            header2 += f" & {m}"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    # Find best per (dataset, metric)
    best_vals = {}
    for ds in datasets:
        for m in available_metrics:
            vals = {}
            for ab in ablations:
                mask = (df["ablation"] == ab) & (df["dataset"] == ds)
                v = df.loc[mask, m].dropna()
                if len(v) > 0:
                    vals[ab] = v.mean()
            if vals:
                if m == "R2":
                    best_vals[(ds, m)] = max(vals.values())
                else:
                    best_vals[(ds, m)] = min(vals.values())

    # Data rows
    for ds in datasets:
        short_ds = _shorten_name(ds)
        row = short_ds
        for ab in ablations:
            for m in available_metrics:
                mask = (df["ablation"] == ab) & (df["dataset"] == ds)
                v = df.loc[mask, m].dropna()
                if len(v) > 0:
                    mean_val = v.mean()
                    std_val = v.std() if len(v) > 1 else 0
                    is_best = abs(mean_val - best_vals.get((ds, m), float("inf"))) < 1e-6
                    val_str = f"{mean_val:.2f}"
                    if std_val > 0:
                        val_str += f"$\\pm${std_val:.2f}"
                    if is_best:
                        val_str = f"\\textbf{{{val_str}}}"
                    row += f" & {val_str}"
                else:
                    row += " & --"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    latex_str = "\n".join(lines)
    latex_path = os.path.join(output_dir, "ablation_table.tex")
    with open(latex_path, "w") as f:
        f.write(latex_str)

    print(f"  Saved LaTeX table to {latex_path}")


# =========================================================================
# Component improvement chart
# =========================================================================

def generate_improvement_chart(df: pd.DataFrame, output_dir: str):
    """Bar chart showing % improvement of each component over baseline.

    For each dataset, compute % RMSE improvement from A→E and from each
    intermediate config, to visualise marginal contribution of each piece.
    """
    metric = "RMSE"
    if metric not in df.columns:
        return

    datasets = df["dataset"].unique()
    ablations = sorted(df["ablation"].unique())

    if "A: Baseline FCN" not in ablations:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 1.5), 5))

    x = np.arange(len(datasets))

    # Compute % improvements over baseline
    improvements = {}
    for ab in ablations:
        if ab == "A: Baseline FCN":
            continue
        imps = []
        for ds in datasets:
            base_mask = (df["ablation"] == "A: Baseline FCN") & (df["dataset"] == ds)
            ab_mask = (df["ablation"] == ab) & (df["dataset"] == ds)
            base_rmse = df.loc[base_mask, metric].mean()
            ab_rmse = df.loc[ab_mask, metric].mean()
            if base_rmse > 0 and not np.isnan(ab_rmse):
                imp = (base_rmse - ab_rmse) / base_rmse * 100
            else:
                imp = 0.0
            imps.append(imp)
        improvements[ab] = imps

    n_bars = len(improvements)
    width = 0.8 / n_bars

    for i, (ab, imps) in enumerate(improvements.items()):
        ax.bar(
            x + i * width - (n_bars - 1) * width / 2,
            imps, width,
            label=ab,
            color=COLORS[(i + 1) % len(COLORS)],
            hatch=HATCHES[(i + 1) % len(HATCHES)],
            edgecolor="black", linewidth=0.5,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("RMSE Improvement over Baseline (%)")
    ax.set_title("Component Contribution Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels([_shorten_name(d) for d in datasets],
                       rotation=30, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "improvement_chart.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "improvement_chart.pdf"))
    plt.close(fig)

    print(f"  Saved improvement chart to {output_dir}")


# =========================================================================
# Load pre-saved results for plot-only mode
# =========================================================================

def load_results(results_dir: str) -> pd.DataFrame:
    """Load previously saved ablation results."""
    csv_path = os.path.join(results_dir, "ablation_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    json_path = os.path.join(results_dir, "ablation_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return pd.DataFrame(data)
    raise FileNotFoundError(f"No results found in {results_dir}")


# =========================================================================
# Helper
# =========================================================================

def _shorten_name(name: str) -> str:
    """Shorten dataset/config names for plot labels."""
    replacements = {
        "CMAPSS_FD001": "FD001",
        "PHM2012_PRONOSTIA_Bearings": "PHM2012",
        "NASA_Battery_Capacity": "Battery",
        "Weather_Dynamics": "Weather",
        "Finance_Market_Dynamics": "Finance",
        "Synthetic_ODE_Validation": "Syn.ODE",
    }
    for full, short in replacements.items():
        if full in name:
            return short
    # Generic shortening
    if len(name) > 15:
        return name[:12] + "..."
    return name


# =========================================================================
# Print summary table
# =========================================================================

def print_summary(df: pd.DataFrame):
    """Print a nicely formatted summary of ablation results."""
    metrics = ["RMSE", "MAE", "R2", "MonoViol", "SlopeRMSE"]
    available = [m for m in metrics if m in df.columns]

    print(f"\n{'='*80}")
    print("  ABLATION STUDY SUMMARY")
    print(f"{'='*80}")

    # Per-dataset summary
    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        print(f"\n  {ds}:")
        print(f"  {'Config':<25}", end="")
        for m in available:
            print(f"  {m:>10}", end="")
        print()
        print(f"  {'-'*25}", end="")
        for m in available:
            print(f"  {'----------':>10}", end="")
        print()

        for ab in sorted(ds_df["ablation"].unique()):
            ab_df = ds_df[ds_df["ablation"] == ab]
            print(f"  {ab:<25}", end="")
            for m in available:
                if m in ab_df.columns:
                    mean = ab_df[m].mean()
                    print(f"  {mean:10.4f}", end="")
                else:
                    print(f"  {'--':>10}", end="")
            print()

    # Cross-dataset average
    print(f"\n  CROSS-DATASET AVERAGE:")
    print(f"  {'Config':<25}", end="")
    for m in available:
        print(f"  {m:>10}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for m in available:
        print(f"  {'----------':>10}", end="")
    print()

    for ab in sorted(df["ablation"].unique()):
        ab_df = df[df["ablation"] == ab]
        print(f"  {ab:<25}", end="")
        for m in available:
            if m in ab_df.columns:
                mean = ab_df[m].mean()
                print(f"  {mean:10.4f}", end="")
            else:
                print(f"  {'--':>10}", end="")
        print()

    print(f"\n{'='*80}")


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="KePIN Ablation Study — systematic component evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default="datasets_kepin_config.json",
                        help="Path to datasets JSON config")
    parser.add_argument("--dataset_idx", type=int, default=None,
                        help="Run ablation on specific dataset only")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training epochs per run")
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Independent runs per config-dataset pair")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)

    # Plot-only mode
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Load pre-saved results and regenerate plots only")
    parser.add_argument("--plots_only", action="store_true",
                        help="Only generate plots from results in output_dir")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.results_dir or args.plots_only:
        # Plot-only mode
        results_dir = args.results_dir or args.output_dir
        if not results_dir:
            print("Error: provide --results_dir or --output_dir for plot-only mode")
            sys.exit(1)

        print(f"Loading results from {results_dir}...")
        df = load_results(results_dir)
        print(f"  Loaded {len(df)} result rows")

        print_summary(df)
        generate_ablation_plots(df, results_dir)
        generate_radar_chart(df, results_dir)
        generate_heatmap(df, results_dir)
        generate_improvement_chart(df, results_dir)
        generate_latex_table(df, results_dir)

    else:
        # Full training + evaluation
        df = run_full_ablation(
            config_path=args.config,
            output_base=args.output_dir,
            epochs=args.epochs,
            n_runs=args.n_runs,
            dataset_idx=args.dataset_idx,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            verbose=args.verbose,
        )

        print_summary(df)

        # Generate all plots
        output_dir = args.output_dir or os.path.dirname(
            os.path.join(_project_dir, "experiments_result",
                         "kepin_ablation_latest", ""))
        if "RMSE" in df.columns:
            generate_radar_chart(df, os.path.dirname(
                df.to_csv.__self__  # placeholder — use output_base
            ) if hasattr(df, "_output_dir") else output_dir)
            generate_improvement_chart(df, output_dir)


if __name__ == "__main__":
    main()
