# -*- coding: utf-8 -*-
"""
KePIN Ablation & Baseline Comparison Study.

Comprehensive evaluation combining:
  A) Ablation study — 5 KePIN variants (isolating each contribution)
  B) Baseline comparison — 7 established methods vs KePIN

Total: 12 configurations × 6 domains × N runs

Configurations:
  --- Baselines ---
  B1. MLP              — no temporal modelling
  B2. LSTM             — recurrent baseline
  B3. BiLSTM           — bidirectional recurrent
  B4. CNN-LSTM         — hybrid CNN + recurrent
  B5. Vanilla FCN      — Conv1D + GAP only (no SE, no dual-pool)
  B6. PI-DP-FCN        — original method (Conv1D + SE + dual-pool + physics loss)
  B7. Transformer      — self-attention encoder baseline

  --- KePIN Ablation ---
  A1. KePIN w/o Koopman  — Conv1D + SE + dual-pool, MSE only (= BaselineFCN)
  A2. KePIN w/o Spectral — Koopman but no eigenvalue constraint
  A3. KePIN w/o Multi-step — no rollout fidelity loss
  A4. KePIN w/o Auto-Wt  — fixed weights (no Kendall uncertainty)
  A5. KePIN Full         — all components (proposed method)

Outputs:
  - Comprehensive grouped bar charts (all 12 configs × 6 datasets)
  - Radar/spider charts per domain
  - Cross-domain heatmap
  - Pairwise statistical tests (Wilcoxon signed-rank)
  - Component improvement analysis
  - LaTeX-ready results tables
  - Summary CSV

Usage:
  # Full study (all 12 configs × 6 datasets × 1 run)
  python kepin_comparison_study.py --config datasets_kepin_config.json

  # Multiple runs for statistical tests
  python kepin_comparison_study.py --config datasets_kepin_config.json --n_runs 5

  # Single dataset
  python kepin_comparison_study.py --config datasets_kepin_config.json --dataset_idx 0

  # Plot-only from saved results
  python kepin_comparison_study.py --results_dir experiments_result/kepin_study_20260101
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
    rmse_np, mae_np, physics_metrics_np,
)
from kepin_baseline_models import (
    BASELINE_REGISTRY, build_baseline_model,
    MLPBaseline, LSTMBaseline, BiLSTMBaseline,
    CNNLSTMBaseline, VanillaFCN, PIDPFCNBaseline, TransformerBaseline,
)
from kepin_ablation import BaselineFCN
from gpu_config import setup_gpu, build_tf_dataset, get_batch_size, get_learning_rate, is_mixed_precision_enabled

# ---------- GPU setup (A100 40 GB) ----------
setup_gpu(mixed_precision=True, xla=True, verbose=True)

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
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Extended colour palette (12 configs)
COLORS_12 = [
    "#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161",
    "#56B4E9", "#F0E442", "#009E73", "#E69F00", "#AA4499", "#332288",
]
HATCHES = ["", "//", "\\\\", "xx", "..", "oo", "||", "--", "++", "OO", "**", "##"]


# =========================================================================
# Configuration registry
# =========================================================================

def get_all_configs():
    """Return all 12 configurations: 7 baselines + 5 KePIN ablation variants.

    Returns:
        list of dicts, each with:
            name:        display name
            tag:         filesystem-safe tag
            category:    'baseline' or 'ablation'
            model_type:  model class identifier
            use_koopman: whether to use Koopman module
            use_auto_weights: Kendall uncertainty weighting
            fixed_weights: manual loss weights (if not auto)
            loss_type:   'mse' or 'physics' or 'kepin'
            description: one-line description
    """
    configs = [
        # --- Baselines ---
        {
            "name": "MLP",
            "tag": "B1_mlp",
            "category": "baseline",
            "model_type": "mlp",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Flatten → Dense (no temporal modelling)",
        },
        {
            "name": "LSTM",
            "tag": "B2_lstm",
            "category": "baseline",
            "model_type": "lstm",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Stacked LSTM → Dense",
        },
        {
            "name": "BiLSTM",
            "tag": "B3_bilstm",
            "category": "baseline",
            "model_type": "bilstm",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Bidirectional LSTM → Dense",
        },
        {
            "name": "CNN-LSTM",
            "tag": "B4_cnn_lstm",
            "category": "baseline",
            "model_type": "cnn_lstm",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Conv1D → LSTM → Dense",
        },
        {
            "name": "Vanilla FCN",
            "tag": "B5_vanilla_fcn",
            "category": "baseline",
            "model_type": "vanilla_fcn",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Conv1D + BN + GAP (no SE, no dual-pool)",
        },
        {
            "name": "PI-DP-FCN",
            "tag": "B6_pi_dp_fcn",
            "category": "baseline",
            "model_type": "pi_dp_fcn",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _physics_loss_weights(),
            "loss_type": "physics",
            "description": "Original Conv1D + SE + Dual-Pool + physics loss",
        },
        {
            "name": "Transformer",
            "tag": "B7_transformer",
            "category": "baseline",
            "model_type": "transformer",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Transformer encoder with self-attention",
        },
        # --- KePIN ablation variants ---
        {
            "name": "KePIN-noK",
            "tag": "A1_kepin_nok",
            "category": "ablation",
            "model_type": "baseline_fcn",
            "use_koopman": False,
            "use_auto_weights": False,
            "fixed_weights": _mse_only_weights(),
            "loss_type": "mse",
            "description": "Conv1D + SE + Dual-Pool, MSE only (no Koopman)",
        },
        {
            "name": "KePIN-noSpec",
            "tag": "A2_kepin_nospec",
            "category": "ablation",
            "model_type": "kepin",
            "use_koopman": True,
            "use_auto_weights": True,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.1, "spectral": 0.0,
                "mono": 0.001, "multi_step": 0.05, "asym": 0.05, "slope": 0.0003,
            },
            "loss_type": "kepin",
            "description": "KePIN without spectral stability loss",
        },
        {
            "name": "KePIN-noMS",
            "tag": "A3_kepin_noms",
            "category": "ablation",
            "model_type": "kepin",
            "use_koopman": True,
            "use_auto_weights": True,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.1, "spectral": 0.01,
                "mono": 0.001, "multi_step": 0.0, "asym": 0.05, "slope": 0.0003,
            },
            "loss_type": "kepin",
            "description": "KePIN without multi-step rollout loss",
        },
        {
            "name": "KePIN-fixW",
            "tag": "A4_kepin_fixw",
            "category": "ablation",
            "model_type": "kepin",
            "use_koopman": True,
            "use_auto_weights": False,
            "fixed_weights": {
                "rul": 1.0, "koopman": 0.1, "spectral": 0.01,
                "mono": 0.001, "multi_step": 0.05, "asym": 0.05, "slope": 0.0003,
            },
            "loss_type": "kepin",
            "description": "KePIN with fixed weights (no Kendall uncertainty)",
        },
        {
            "name": "KePIN (Ours)",
            "tag": "A5_kepin_full",
            "category": "ablation",
            "model_type": "kepin",
            "use_koopman": True,
            "use_auto_weights": True,
            "fixed_weights": None,
            "loss_type": "kepin",
            "description": "Full KePIN with all components (proposed)",
        },
    ]
    return configs


def _mse_only_weights():
    return {
        "rul": 1.0, "koopman": 0.0, "spectral": 0.0,
        "mono": 0.0, "multi_step": 0.0, "asym": 0.0, "slope": 0.0,
    }


def _physics_loss_weights():
    return {
        "rul": 1.0, "koopman": 0.0, "spectral": 0.0,
        "mono": 0.001, "multi_step": 0.0, "asym": 0.05, "slope": 0.0003,
    }


# =========================================================================
# Metrics
# =========================================================================

def r_squared_np(y_true, y_pred):
    ss_res = np.sum((y_true.flatten() - y_pred.flatten()) ** 2)
    ss_tot = np.sum((y_true.flatten() - np.mean(y_true.flatten())) ** 2) + 1e-10
    return float(1.0 - ss_res / ss_tot)


def nasa_score_np(y_true, y_pred):
    s = 0.0
    yt = y_true.flatten()
    yp = y_pred.flatten()
    for i in range(len(yp)):
        diff = yp[i] - yt[i]
        if diff > 50:  # overflow guard
            s += 1e6
        elif diff > 0:
            s += math.exp(diff / 10.0) - 1.0
        else:
            s += math.exp(-diff / 13.0) - 1.0
    return float(s)


def mape_np(y_true, y_pred):
    """Mean Absolute Percentage Error (clipped denominator)."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    mask = np.abs(yt) > 1.0  # avoid near-zero denominators
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / np.abs(yt[mask])) * 100)


def compute_all_metrics(y_true, y_pred) -> dict:
    mono_viol, slope_err = physics_metrics_np(y_true, y_pred)
    metrics = {
        "RMSE": rmse_np(y_true, y_pred),
        "MAE": mae_np(y_true, y_pred),
        "R2": r_squared_np(y_true, y_pred),
        "MAPE": mape_np(y_true, y_pred),
        "MonoViol": mono_viol,
        "SlopeRMSE": slope_err,
    }
    try:
        metrics["Score"] = nasa_score_np(y_true, y_pred)
    except (OverflowError, ValueError):
        metrics["Score"] = float("inf")
    return metrics


# =========================================================================
# Model builder dispatcher
# =========================================================================

def build_model_for_config(cfg: dict, seq_len: int, n_features: int,
                           n_train: int = None):
    """Build model & loss function for a given configuration.

    Returns:
        model:   model instance
        loss_fn: composite loss function compatible with KePINTrainer
    """
    model_type = cfg["model_type"]
    arch_config = auto_configure(n_features, seq_len, n_train)

    # --- Build model ---
    if model_type == "kepin":
        model = build_kepin_model(seq_len, n_features,
                                  n_train=n_train, arch_config=arch_config)
    elif model_type == "baseline_fcn":
        model = BaselineFCN(
            input_shape_tuple=(seq_len, n_features),
            arch_config=arch_config, n_train=n_train,
        )
    elif model_type in BASELINE_REGISTRY:
        model = build_baseline_model(model_type, seq_len, n_features, n_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Build loss function ---
    if cfg["use_auto_weights"] and cfg["use_koopman"]:
        loss_fn = make_kepin_loss(
            loss_weights_layer=model.loss_weight_layer,
            use_auto_weights=True,
        )
    else:
        fw = cfg.get("fixed_weights") or _mse_only_weights()
        loss_fn = make_kepin_loss(
            loss_weights_layer=None,
            use_auto_weights=False,
            fixed_weights=fw,
        )

    return model, loss_fn


# =========================================================================
# Single run
# =========================================================================

def run_single(cfg: dict, ds_config: dict, output_dir: str,
               epochs: int = 200, batch_size: int = None,
               lr: float = None, patience: int = 40,
               run_id: int = 0, verbose: int = 1) -> dict:
    """Train one config on one dataset.

    Returns:
        result dict with metrics, config metadata, predictions path
    """
    cfg_name = cfg["name"]
    ds_name = ds_config.get("name", "unknown")
    tag = f"{cfg['tag']}_{ds_name}_run{run_id}"

    if verbose:
        print(f"\n  --- {cfg_name} on {ds_name} (run {run_id}) ---")
        print(f"      {cfg['description']}")

    # Load data
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()

    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)

    # EMA smoothing
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]

    # Auto batch/LR (A100-optimised)
    if batch_size is None:
        model_cat = cfg.get("model_type", "kepin")
        batch_size = get_batch_size(n_train, seq_len, n_feat, model_type=model_cat)
    if lr is None:
        lr = get_learning_rate(batch_size, base_lr=0.001, base_batch=256)

    # Build model & loss
    tf.random.set_seed(SEED + run_id)
    np.random.seed(SEED + run_id)

    model, loss_fn = build_model_for_config(cfg, seq_len, n_feat, n_train)

    # Train
    optimizer = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)
    trainer = KePINTrainer(model, loss_fn, optimizer)

    history = trainer.fit(
        X_train, Y_train, X_test, Y_test,
        epochs=epochs, batch_size=batch_size,
        patience=patience, initial_lr=lr,
        verbose=max(0, verbose - 1),
    )

    # Evaluate
    Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
    metrics = compute_all_metrics(Y_test, Y_pred)

    if verbose:
        print(f"      RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
              f"R2={metrics['R2']:.4f}  MAPE={metrics['MAPE']:.1f}%")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, f"pred_{tag}.npz"),
             y_true=Y_test, y_pred=Y_pred)

    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
    })
    hist_df.to_csv(os.path.join(output_dir, f"history_{tag}.csv"), index=False)

    result = {
        "config": cfg_name,
        "config_tag": cfg["tag"],
        "category": cfg["category"],
        "model_type": cfg["model_type"],
        "dataset": ds_name,
        "run_id": run_id,
        **metrics,
        "epochs_trained": len(history["epoch"]),
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else float("inf"),
        "n_params": model.count_params(),
        "description": cfg["description"],
    }
    return result


# =========================================================================
# Full study orchestrator
# =========================================================================

def run_full_study(config_path: str, output_base: str = None,
                   epochs: int = 200, n_runs: int = 1,
                   dataset_idx: int = None,
                   config_filter: str = None,
                   batch_size: int = None, lr: float = None,
                   patience: int = 40, verbose: int = 1) -> pd.DataFrame:
    """Run the complete comparison study.

    Args:
        config_path:    path to datasets JSON config
        output_base:    output directory
        epochs:         max epochs per run
        n_runs:         independent runs per config-dataset pair
        dataset_idx:    restrict to one dataset
        config_filter:  comma-separated config tags to run (e.g., "B1_mlp,A5_kepin_full")
        batch_size:     override
        lr:             override
        patience:       early stopping patience
        verbose:        verbosity

    Returns:
        results_df: DataFrame with all results
    """
    with open(config_path, "r") as f:
        ds_configs = json.load(f)

    if dataset_idx is not None:
        ds_configs = [ds_configs[dataset_idx]]

    all_configs = get_all_configs()

    if config_filter:
        tags = [t.strip() for t in config_filter.split(",")]
        all_configs = [c for c in all_configs if c["tag"] in tags]

    if output_base is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"kepin_study_{timestamp}")

    os.makedirs(output_base, exist_ok=True)

    n_total = len(all_configs) * len(ds_configs) * n_runs
    print(f"\n{'='*70}")
    print(f"  KePIN ABLATION & BASELINE COMPARISON STUDY")
    print(f"  {len(all_configs)} configs x {len(ds_configs)} datasets x {n_runs} runs = {n_total} experiments")
    print(f"  Output: {output_base}")
    print(f"{'='*70}")

    # Print config table
    print(f"\n  {'#':<4} {'Tag':<22} {'Cat':<10} {'Description'}")
    print(f"  {'-'*4} {'-'*22} {'-'*10} {'-'*40}")
    for i, cfg in enumerate(all_configs):
        print(f"  {i+1:<4} {cfg['tag']:<22} {cfg['category']:<10} {cfg['description']}")

    all_results = []
    completed = 0

    for ds_config in ds_configs:
        ds_name = ds_config.get("name", "unknown")
        ds_dir = os.path.join(output_base, ds_name)

        for cfg in all_configs:
            for run in range(n_runs):
                completed += 1
                progress = f"[{completed}/{n_total}]"

                try:
                    if verbose:
                        print(f"\n{progress} {cfg['name']} on {ds_name} (run {run})")

                    result = run_single(
                        cfg, ds_config, ds_dir,
                        epochs=epochs, batch_size=batch_size,
                        lr=lr, patience=patience,
                        run_id=run, verbose=verbose,
                    )
                    all_results.append(result)

                    # Incremental save
                    _save_results(all_results, output_base)

                except Exception as e:
                    print(f"    FAILED: {cfg['name']} on {ds_name} run {run}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "config": cfg["name"],
                        "config_tag": cfg["tag"],
                        "category": cfg["category"],
                        "model_type": cfg["model_type"],
                        "dataset": ds_name,
                        "run_id": run,
                        "error": str(e),
                    })

    # Final save
    _save_results(all_results, output_base)

    results_df = pd.DataFrame(all_results)

    # Generate all outputs
    valid_df = results_df[results_df.get("RMSE", pd.Series(dtype=float)).notna()].copy()

    if len(valid_df) > 0:
        print_summary_table(valid_df)

        try:
            plot_grouped_bars(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: bar plots failed: {e}")

        try:
            plot_radar_per_domain(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: radar plots failed: {e}")

        try:
            plot_heatmap(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: heatmap failed: {e}")

        try:
            plot_improvement_over_baselines(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: improvement chart failed: {e}")

        try:
            plot_param_efficiency(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: param efficiency plot failed: {e}")

        try:
            generate_latex_tables(valid_df, output_base)
        except Exception as e:
            print(f"  Warning: LaTeX tables failed: {e}")

        if n_runs >= 3:
            try:
                run_statistical_tests(valid_df, output_base)
            except Exception as e:
                print(f"  Warning: statistical tests failed: {e}")

    return results_df


def _save_results(all_results, output_base):
    """Incremental save to CSV and JSON."""
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_base, "study_results.csv"), index=False)
    with open(os.path.join(output_base, "study_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)


# =========================================================================
# Summary table
# =========================================================================

def print_summary_table(df: pd.DataFrame):
    """Print formatted summary table."""
    metrics = ["RMSE", "MAE", "R2", "MAPE"]
    available = [m for m in metrics if m in df.columns]

    print(f"\n{'='*90}")
    print("  COMPARISON STUDY RESULTS")
    print(f"{'='*90}")

    # --- Per-dataset table ---
    for ds in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == ds]
        print(f"\n  Dataset: {ds}")
        print(f"  {'Config':<20} {'Cat':<10}", end="")
        for m in available:
            print(f"  {m:>8}", end="")
        if "n_params" in ds_df.columns:
            print(f"  {'Params':>10}", end="")
        print()
        print(f"  {'-'*20} {'-'*10}", end="")
        for m in available:
            print(f"  {'--------':>8}", end="")
        if "n_params" in ds_df.columns:
            print(f"  {'----------':>10}", end="")
        print()

        # Sort: baselines first, then ablation
        configs_sorted = sorted(ds_df["config"].unique(),
                                key=lambda x: (0 if "KePIN" not in x else 1, x))

        for cfg_name in configs_sorted:
            cfg_df = ds_df[ds_df["config"] == cfg_name]
            cat = cfg_df["category"].iloc[0] if "category" in cfg_df.columns else ""
            print(f"  {cfg_name:<20} {cat:<10}", end="")
            for m in available:
                if m in cfg_df.columns:
                    val = cfg_df[m].mean()
                    std = cfg_df[m].std() if len(cfg_df) > 1 else 0
                    if std > 0:
                        print(f"  {val:>8.3f}", end="")
                    else:
                        print(f"  {val:>8.3f}", end="")
                else:
                    print(f"  {'--':>8}", end="")
            if "n_params" in cfg_df.columns:
                params = int(cfg_df["n_params"].iloc[0])
                if params >= 1_000_000:
                    print(f"  {params/1e6:>8.1f}M", end="")
                elif params >= 1_000:
                    print(f"  {params/1e3:>8.1f}K", end="")
                else:
                    print(f"  {params:>10d}", end="")
            print()

    # --- Cross-dataset average ---
    print(f"\n  CROSS-DATASET AVERAGE:")
    print(f"  {'Config':<20} {'Cat':<10}", end="")
    for m in available:
        print(f"  {m:>8}", end="")
    print()
    print(f"  {'-'*20} {'-'*10}", end="")
    for m in available:
        print(f"  {'--------':>8}", end="")
    print()

    for cfg_name in sorted(df["config"].unique(),
                           key=lambda x: (0 if "KePIN" not in x else 1, x)):
        cfg_df = df[df["config"] == cfg_name]
        cat = cfg_df["category"].iloc[0] if "category" in cfg_df.columns else ""
        print(f"  {cfg_name:<20} {cat:<10}", end="")
        for m in available:
            if m in cfg_df.columns:
                val = cfg_df[m].mean()
                print(f"  {val:>8.3f}", end="")
            else:
                print(f"  {'--':>8}", end="")
        print()

    print(f"\n{'='*90}")


# =========================================================================
# Plot: Grouped bar charts
# =========================================================================

def plot_grouped_bars(df: pd.DataFrame, output_dir: str):
    """Grouped bar charts — all configs side by side per dataset."""

    metrics_to_plot = [
        ("RMSE", True), ("MAE", True), ("R2", False), ("MAPE", True),
    ]

    datasets = sorted(df["dataset"].unique())
    configs = _sort_configs(df["config"].unique())
    n_ds = len(datasets)
    n_cfg = len(configs)

    for metric, lower_better in metrics_to_plot:
        if metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(max(10, n_ds * 2), 6))

        x = np.arange(n_ds)
        width = 0.85 / n_cfg

        for i, cfg in enumerate(configs):
            means = []
            stds = []
            for ds in datasets:
                mask = (df["config"] == cfg) & (df["dataset"] == ds)
                vals = df.loc[mask, metric].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)

            offset = i * width - (n_cfg - 1) * width / 2
            bars = ax.bar(
                x + offset, means, width,
                yerr=stds if any(s > 0 for s in stds) else None,
                label=cfg,
                color=COLORS_12[i % len(COLORS_12)],
                hatch=HATCHES[i % len(HATCHES)],
                edgecolor="black", linewidth=0.4,
                capsize=2,
            )

        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} — All Methods Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([_shorten(d) for d in datasets], rotation=25, ha="right")

        # Compact legend
        ncol = min(4, max(2, n_cfg // 3))
        ax.legend(fontsize=6.5, ncol=ncol, loc="best",
                  framealpha=0.9, borderaxespad=0.5)

        direction = "lower" if lower_better else "higher"
        ax.annotate(f"({direction} is better)", xy=(0.99, 0.01),
                    xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=7, fontstyle="italic", alpha=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"comparison_{metric}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"comparison_{metric}.pdf"))
        plt.close(fig)

    # --- Also generate baselines-only and ablation-only views ---
    for subset, label in [("baseline", "baselines"), ("ablation", "ablation")]:
        if "category" not in df.columns:
            continue
        sub_df = df[df["category"] == subset]
        # Also always include KePIN Full for reference
        if subset == "baseline":
            kepin_full = df[df["config"] == "KePIN (Ours)"]
            sub_df = pd.concat([sub_df, kepin_full])

        if len(sub_df) < 2:
            continue

        for metric, lower_better in metrics_to_plot:
            if metric not in sub_df.columns:
                continue

            configs_sub = _sort_configs(sub_df["config"].unique())
            n_cfg_sub = len(configs_sub)

            fig, ax = plt.subplots(figsize=(max(10, n_ds * 1.5), 5))
            x = np.arange(n_ds)
            width = 0.85 / n_cfg_sub

            for i, cfg in enumerate(configs_sub):
                means = []
                for ds in datasets:
                    mask = (sub_df["config"] == cfg) & (sub_df["dataset"] == ds)
                    vals = sub_df.loc[mask, metric].dropna()
                    means.append(vals.mean() if len(vals) > 0 else 0)

                is_ours = "Ours" in cfg
                offset = i * width - (n_cfg_sub - 1) * width / 2
                ax.bar(
                    x + offset, means, width,
                    label=cfg,
                    color=COLORS_12[i % len(COLORS_12)],
                    hatch=HATCHES[i % len(HATCHES)],
                    edgecolor="red" if is_ours else "black",
                    linewidth=1.5 if is_ours else 0.4,
                )

            ax.set_xlabel("Dataset")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} — {label.title()} Comparison")
            ax.set_xticks(x)
            ax.set_xticklabels([_shorten(d) for d in datasets],
                               rotation=25, ha="right")
            ax.legend(fontsize=7, ncol=min(4, n_cfg_sub))

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     f"{label}_{metric}.png"), dpi=300)
            fig.savefig(os.path.join(output_dir,
                                     f"{label}_{metric}.pdf"))
            plt.close(fig)

    print(f"  Saved grouped bar charts to {output_dir}")


# =========================================================================
# Plot: Radar / Spider charts per domain
# =========================================================================

def plot_radar_per_domain(df: pd.DataFrame, output_dir: str):
    """Radar chart for each dataset comparing all methods."""
    metrics = ["RMSE", "MAE", "R2", "MonoViol", "SlopeRMSE"]
    available = [m for m in metrics if m in df.columns]
    if len(available) < 3:
        return

    datasets = sorted(df["dataset"].unique())

    for ds in datasets:
        ds_df = df[df["dataset"] == ds]
        configs = _sort_configs(ds_df["config"].unique())

        if len(configs) < 2:
            continue

        # Get metric values
        raw_data = {}
        for cfg in configs:
            cfg_df = ds_df[ds_df["config"] == cfg]
            raw_data[cfg] = [cfg_df[m].mean() if m in cfg_df else 0 for m in available]

        # Normalise to [0, 1] — outer = better
        norm_data = {c: [] for c in configs}
        for mi, m in enumerate(available):
            vals = [raw_data[c][mi] for c in configs]
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin if vmax != vmin else 1.0
            for c in configs:
                val_norm = (raw_data[c][mi] - vmin) / rng
                # For R2: higher is better (outer = better)
                if m == "R2":
                    norm_data[c].append(val_norm)
                else:
                    # Lower is better → invert
                    norm_data[c].append(1.0 - val_norm)

        # Plot
        n_m = len(available)
        angles = np.linspace(0, 2 * np.pi, n_m, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        for i, cfg in enumerate(configs):
            values = norm_data[cfg] + [norm_data[cfg][0]]
            is_ours = "Ours" in cfg
            ax.plot(angles, values, "o-", label=cfg,
                    color=COLORS_12[i % len(COLORS_12)],
                    linewidth=2.5 if is_ours else 1.2,
                    markersize=5 if is_ours else 3,
                    alpha=1.0 if is_ours else 0.7)
            if is_ours:
                ax.fill(angles, values, alpha=0.15,
                        color=COLORS_12[i % len(COLORS_12)])

        ax.set_thetagrids(np.degrees(angles[:-1]), available)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Radar — {_shorten(ds)}", pad=20, fontsize=13)
        ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15),
                  fontsize=6.5, ncol=1)

        plt.tight_layout()
        safe_ds = ds.replace(" ", "_").replace("/", "_")
        fig.savefig(os.path.join(output_dir, f"radar_{safe_ds}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"radar_{safe_ds}.pdf"))
        plt.close(fig)

    print(f"  Saved radar charts to {output_dir}")


# =========================================================================
# Plot: Heatmap
# =========================================================================

def plot_heatmap(df: pd.DataFrame, output_dir: str):
    """Cross-domain heatmap: configs (rows) x datasets (cols) for RMSE."""
    for metric in ["RMSE", "MAE"]:
        if metric not in df.columns:
            continue

        pivot = df.groupby(["config", "dataset"])[metric].mean().unstack(
            fill_value=np.nan)

        # Sort rows nicely
        row_order = _sort_configs(pivot.index.tolist())
        pivot = pivot.reindex(row_order)

        fig, ax = plt.subplots(figsize=(max(9, len(pivot.columns) * 1.5),
                                        max(5, len(pivot.index) * 0.5)))

        data = pivot.values
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > np.nanmedian(data) else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=text_color, fontsize=8)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([_shorten(c) for c in pivot.columns],
                           rotation=30, ha="right")
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Method")
        ax.set_title(f"Cross-Domain Heatmap — {metric}")

        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label(metric)

        # Highlight best per column
        for j in range(data.shape[1]):
            col = data[:, j]
            if not np.all(np.isnan(col)):
                best_i = np.nanargmin(col)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, best_i - 0.5), 1, 1,
                    fill=False, edgecolor="lime", linewidth=2.5,
                ))

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"heatmap_{metric}.png"), dpi=300)
        fig.savefig(os.path.join(output_dir, f"heatmap_{metric}.pdf"))
        plt.close(fig)

    print(f"  Saved heatmaps to {output_dir}")


# =========================================================================
# Plot: Improvement over baselines
# =========================================================================

def plot_improvement_over_baselines(df: pd.DataFrame, output_dir: str):
    """% RMSE improvement of KePIN (Ours) vs each baseline, per dataset."""
    metric = "RMSE"
    if metric not in df.columns:
        return

    our_name = "KePIN (Ours)"
    if our_name not in df["config"].values:
        return

    datasets = sorted(df["dataset"].unique())
    other_configs = [c for c in _sort_configs(df["config"].unique()) if c != our_name]

    n_ds = len(datasets)
    n_others = len(other_configs)

    fig, ax = plt.subplots(figsize=(max(10, n_ds * 1.8), 6))
    x = np.arange(n_ds)
    width = 0.85 / n_others

    for i, other in enumerate(other_configs):
        improvements = []
        for ds in datasets:
            our_mask = (df["config"] == our_name) & (df["dataset"] == ds)
            oth_mask = (df["config"] == other) & (df["dataset"] == ds)
            our_rmse = df.loc[our_mask, metric].mean()
            oth_rmse = df.loc[oth_mask, metric].mean()
            if oth_rmse > 0 and not np.isnan(our_rmse):
                imp = (oth_rmse - our_rmse) / oth_rmse * 100
            else:
                imp = 0.0
            improvements.append(imp)

        offset = i * width - (n_others - 1) * width / 2
        bars = ax.bar(
            x + offset, improvements, width,
            label=other,
            color=COLORS_12[i % len(COLORS_12)],
            edgecolor="black", linewidth=0.4,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("RMSE Improvement over baseline (%)")
    ax.set_title("KePIN (Ours) Improvement vs. Each Method")
    ax.set_xticks(x)
    ax.set_xticklabels([_shorten(d) for d in datasets], rotation=25, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.legend(fontsize=6, ncol=min(4, n_others), loc="best")

    # Annotate average improvement
    avg_imp = {}
    for other in other_configs:
        imps = []
        for ds in datasets:
            our_mask = (df["config"] == our_name) & (df["dataset"] == ds)
            oth_mask = (df["config"] == other) & (df["dataset"] == ds)
            our_rmse = df.loc[our_mask, metric].mean()
            oth_rmse = df.loc[oth_mask, metric].mean()
            if oth_rmse > 0:
                imps.append((oth_rmse - our_rmse) / oth_rmse * 100)
        avg_imp[other] = np.mean(imps) if imps else 0

    if avg_imp:
        best_baseline = max(avg_imp, key=lambda k: avg_imp[k] if "KePIN" not in k else -999)
        baseline_bests = {k: v for k, v in avg_imp.items() if "KePIN" not in k}
        if baseline_bests:
            overall_avg = np.mean(list(baseline_bests.values()))
            ax.annotate(f"Avg improvement over baselines: {overall_avg:.1f}%",
                        xy=(0.02, 0.98), xycoords="axes fraction",
                        ha="left", va="top", fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                  edgecolor="orange", alpha=0.9))

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "improvement_vs_baselines.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "improvement_vs_baselines.pdf"))
    plt.close(fig)

    print(f"  Saved improvement chart to {output_dir}")


# =========================================================================
# Plot: Parameter efficiency
# =========================================================================

def plot_param_efficiency(df: pd.DataFrame, output_dir: str):
    """Scatter plot: RMSE vs. number of parameters (per dataset)."""
    if "n_params" not in df.columns or "RMSE" not in df.columns:
        return

    datasets = sorted(df["dataset"].unique())
    n_ds = len(datasets)
    ncols = min(3, n_ds)
    nrows = math.ceil(n_ds / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    configs = _sort_configs(df["config"].unique())

    for idx, ds in enumerate(datasets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        ds_df = df[df["dataset"] == ds]

        for i, cfg in enumerate(configs):
            cfg_df = ds_df[ds_df["config"] == cfg]
            if len(cfg_df) == 0:
                continue

            params = cfg_df["n_params"].iloc[0]
            rmse = cfg_df["RMSE"].mean()
            is_ours = "Ours" in cfg

            ax.scatter(
                params, rmse,
                s=120 if is_ours else 60,
                marker="*" if is_ours else "o",
                color=COLORS_12[i % len(COLORS_12)],
                edgecolors="red" if is_ours else "black",
                linewidth=2 if is_ours else 0.5,
                label=cfg if idx == 0 else None,
                zorder=10 if is_ours else 5,
            )

        ax.set_xlabel("Parameters")
        ax.set_ylabel("RMSE")
        ax.set_title(_shorten(ds))
        ax.set_xscale("log")

    # Remove extra axes
    for idx in range(n_ds, len(axes)):
        axes[idx].set_visible(False)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(6, len(configs)), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Parameter Efficiency: RMSE vs. Model Size", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "param_efficiency.png"), dpi=300,
                bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "param_efficiency.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved parameter efficiency plot to {output_dir}")


# =========================================================================
# LaTeX tables
# =========================================================================

def generate_latex_tables(df: pd.DataFrame, output_dir: str):
    """Generate two LaTeX tables: baseline comparison + ablation study."""

    # --- Table 1: Baselines vs KePIN (Ours) ---
    _generate_one_latex_table(
        df,
        config_filter=lambda c: c.get("category") == "baseline" or "Ours" in c.get("name", ""),
        metrics=["RMSE", "MAE", "R2"],
        caption="Comparison of KePIN with baseline methods across 6 domains. "
                "Best values per dataset are \\textbf{bolded}.",
        label="tab:baseline_comparison",
        filename="table_baseline_comparison.tex",
        output_dir=output_dir,
    )

    # --- Table 2: Ablation study ---
    _generate_one_latex_table(
        df,
        config_filter=lambda c: c.get("category") == "ablation",
        metrics=["RMSE", "MAE", "R2", "MonoViol"],
        caption="Ablation study: contribution of each KePIN component. "
                "Best values per dataset are \\textbf{bolded}.",
        label="tab:ablation_study",
        filename="table_ablation_study.tex",
        output_dir=output_dir,
    )

    print(f"  Saved LaTeX tables to {output_dir}")


def _generate_one_latex_table(df, config_filter, metrics, caption, label,
                              filename, output_dir):
    """Generate a single LaTeX table."""
    all_cfgs = get_all_configs()
    selected_cfgs = [c for c in all_cfgs if config_filter(c)]
    selected_names = [c["name"] for c in selected_cfgs]

    # Filter df
    sub_df = df[df["config"].isin(selected_names)]
    if len(sub_df) == 0:
        return

    available = [m for m in metrics if m in sub_df.columns]
    if not available:
        return

    datasets = sorted(sub_df["dataset"].unique())
    configs = [c for c in _sort_configs(sub_df["config"].unique())
               if c in selected_names]

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    # Column spec
    m_cols = "c" * len(available)
    col_groups = "|".join([m_cols] * len(configs))
    lines.append(f"\\begin{{tabular}}{{l|{col_groups}}}")
    lines.append("\\toprule")

    # Header row 1: config names
    header1 = "\\multirow{2}{*}{Dataset}"
    for cfg in configs:
        short = cfg.replace("KePIN ", "K-").replace(" (Ours)", "$^\\star$")
        header1 += f" & \\multicolumn{{{len(available)}}}{{c}}{{{short}}}"
    header1 += " \\\\"
    lines.append(header1)

    # Header row 2: metric names
    header2 = ""
    for cfg in configs:
        for m in available:
            header2 += f" & {m}"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    # Find best per (dataset, metric)
    best = {}
    for ds in datasets:
        for m in available:
            vals = {}
            for cfg in configs:
                mask = (sub_df["config"] == cfg) & (sub_df["dataset"] == ds)
                v = sub_df.loc[mask, m].dropna()
                if len(v) > 0:
                    vals[cfg] = v.mean()
            if vals:
                best[(ds, m)] = max(vals.values()) if m == "R2" else min(vals.values())

    # Data rows
    for ds in datasets:
        row = _shorten(ds)
        for cfg in configs:
            for m in available:
                mask = (sub_df["config"] == cfg) & (sub_df["dataset"] == ds)
                v = sub_df.loc[mask, m].dropna()
                if len(v) > 0:
                    mean_val = v.mean()
                    std_val = v.std() if len(v) > 1 else 0

                    # Format
                    if m in ("MonoViol",):
                        fmt = f"{mean_val:.4f}"
                    elif m == "R2":
                        fmt = f"{mean_val:.3f}"
                    else:
                        fmt = f"{mean_val:.2f}"

                    if std_val > 0:
                        fmt += f"$\\pm${std_val:.2f}"

                    is_best = abs(mean_val - best.get((ds, m), float("inf"))) < 1e-6
                    if is_best:
                        fmt = f"\\textbf{{{fmt}}}"
                    row += f" & {fmt}"
                else:
                    row += " & --"
        row += " \\\\"
        lines.append(row)

    # Average row
    lines.append("\\midrule")
    row = "\\textit{Average}"
    for cfg in configs:
        for m in available:
            mask = sub_df["config"] == cfg
            v = sub_df.loc[mask, m].dropna()
            if len(v) > 0:
                if m in ("MonoViol",):
                    row += f" & {v.mean():.4f}"
                elif m == "R2":
                    row += f" & {v.mean():.3f}"
                else:
                    row += f" & {v.mean():.2f}"
            else:
                row += " & --"
    row += " \\\\"
    lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(lines))


# =========================================================================
# Statistical tests
# =========================================================================

def run_statistical_tests(df: pd.DataFrame, output_dir: str):
    """Wilcoxon signed-rank tests: KePIN vs. each baseline.

    Requires n_runs >= 3 for meaningful p-values.
    """
    from scipy import stats

    our_name = "KePIN (Ours)"
    if our_name not in df["config"].values:
        print("  Skipping statistical tests: KePIN (Ours) not found")
        return

    other_configs = [c for c in _sort_configs(df["config"].unique())
                     if c != our_name]
    datasets = sorted(df["dataset"].unique())

    results = []

    for other in other_configs:
        for ds in datasets:
            our_mask = (df["config"] == our_name) & (df["dataset"] == ds)
            oth_mask = (df["config"] == other) & (df["dataset"] == ds)
            our_rmses = df.loc[our_mask, "RMSE"].dropna().values
            oth_rmses = df.loc[oth_mask, "RMSE"].dropna().values

            n = min(len(our_rmses), len(oth_rmses))
            if n < 3:
                continue

            our_rmses = our_rmses[:n]
            oth_rmses = oth_rmses[:n]

            # Wilcoxon signed-rank test (two-sided)
            try:
                stat, p_val = stats.wilcoxon(oth_rmses, our_rmses,
                                             alternative="two-sided")
                significant = p_val < 0.05

                # Effect size (rank-biserial correlation)
                # r = 1 - (2W / (n(n+1))
                # where W is the test statistic
                n_pairs = n
                r = 1 - (2 * stat) / (n_pairs * (n_pairs + 1))

                results.append({
                    "Comparison": f"KePIN vs {other}",
                    "Dataset": ds,
                    "Our_RMSE_mean": np.mean(our_rmses),
                    "Other_RMSE_mean": np.mean(oth_rmses),
                    "Difference": np.mean(oth_rmses) - np.mean(our_rmses),
                    "W_statistic": stat,
                    "p_value": p_val,
                    "Significant": "Yes" if significant else "No",
                    "Effect_size_r": r,
                    "n_pairs": n,
                })
            except Exception:
                pass

    if results:
        stat_df = pd.DataFrame(results)
        stat_path = os.path.join(output_dir, "statistical_tests.csv")
        stat_df.to_csv(stat_path, index=False)

        print(f"\n  STATISTICAL TESTS (Wilcoxon signed-rank, α=0.05)")
        print(f"  {'Comparison':<30} {'Dataset':<15} {'Δ RMSE':>10} {'p-value':>10} {'Sig?':>5}")
        for _, row in stat_df.iterrows():
            sig_mark = "*" if row["Significant"] == "Yes" else ""
            print(f"  {row['Comparison']:<30} {_shorten(row['Dataset']):<15} "
                  f"{row['Difference']:>10.4f} {row['p_value']:>10.4f} {sig_mark:>5}")

        # Count wins
        if len(stat_df) > 0:
            n_sig = stat_df["Significant"].value_counts().get("Yes", 0)
            n_total = len(stat_df)
            n_kepin_better = (stat_df["Difference"] > 0).sum()
            print(f"\n  KePIN better in {n_kepin_better}/{n_total} comparisons "
                  f"({n_sig} statistically significant)")

        print(f"  Saved statistical tests to {stat_path}")


# =========================================================================
# Load pre-saved results
# =========================================================================

def load_results(results_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(results_dir, "study_results.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    json_path = os.path.join(results_dir, "study_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            return pd.DataFrame(json.load(f))
    raise FileNotFoundError(f"No results in {results_dir}")


# =========================================================================
# Helpers
# =========================================================================

def _shorten(name: str) -> str:
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
    return name[:15] if len(name) > 15 else name


def _sort_configs(config_names):
    """Sort configs: baselines first (alphabetical), then ablation, KePIN last."""
    order = {
        "MLP": 0, "LSTM": 1, "BiLSTM": 2, "CNN-LSTM": 3,
        "Vanilla FCN": 4, "PI-DP-FCN": 5, "Transformer": 6,
        "KePIN-noK": 7, "KePIN-noSpec": 8, "KePIN-noMS": 9,
        "KePIN-fixW": 10, "KePIN (Ours)": 11,
    }
    return sorted(config_names, key=lambda x: order.get(x, 50))


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="KePIN Ablation & Baseline Comparison Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full study
  python kepin_comparison_study.py --config datasets_kepin_config.json

  # Multiple runs with statistical tests
  python kepin_comparison_study.py --config datasets_kepin_config.json --n_runs 5

  # Baselines only (quick)
  python kepin_comparison_study.py --config datasets_kepin_config.json \\
      --configs B1_mlp,B2_lstm,B6_pi_dp_fcn,A5_kepin_full

  # Regenerate plots from saved results
  python kepin_comparison_study.py --results_dir experiments_result/kepin_study_20260101
        """,
    )

    parser.add_argument("--config", type=str, default="datasets_kepin_config.json",
                        help="Path to datasets JSON config")
    parser.add_argument("--dataset_idx", type=int, default=None,
                        help="Run on specific dataset only")
    parser.add_argument("--configs", type=str, default=None,
                        help="Comma-separated config tags to run "
                             "(e.g., B1_mlp,A5_kepin_full)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (auto for A100 if omitted)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override LR (auto-scaled with batch size if omitted)")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable float16 mixed precision")
    parser.add_argument("--no_xla", action="store_true",
                        help="Disable XLA JIT compilation")

    # Plot-only
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Load pre-saved results and generate plots only")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        print(f"Loading results from {args.results_dir}...")
        df = load_results(args.results_dir)
        print(f"  Loaded {len(df)} result rows")

        valid_df = df[df.get("RMSE", pd.Series(dtype=float)).notna()].copy()

        if len(valid_df) > 0:
            print_summary_table(valid_df)
            plot_grouped_bars(valid_df, args.results_dir)
            plot_radar_per_domain(valid_df, args.results_dir)
            plot_heatmap(valid_df, args.results_dir)
            plot_improvement_over_baselines(valid_df, args.results_dir)
            plot_param_efficiency(valid_df, args.results_dir)
            generate_latex_tables(valid_df, args.results_dir)

            if valid_df["run_id"].nunique() >= 3:
                run_statistical_tests(valid_df, args.results_dir)
        else:
            print("  No valid results found.")

    else:
        run_full_study(
            config_path=args.config,
            output_base=args.output_dir,
            epochs=args.epochs,
            n_runs=args.n_runs,
            dataset_idx=args.dataset_idx,
            config_filter=args.configs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
