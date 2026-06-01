#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run ALL experiments for the KePIN paper:
  1. Ablation study on FD002, FD004, Weather, Finance
  2. Optimized training on Weather, Finance, Synthetic ODE (improved hyperparams)
  3. Generate all publication figures
  4. Output updated LaTeX tables

Does NOT retrain CMAPSS FD001-FD004 main results (uses existing).

Usage:
  python run_all_experiments.py
  python run_all_experiments.py --skip_ablation   # skip ablation, only optimize
  python run_all_experiments.py --skip_optimize    # skip optimize, only ablation
"""

import argparse
import datetime
import json
import math
import os
import sys
import copy

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
from kepin_losses import KePINLossWeights, make_kepin_loss
from kepin_training import (
    KePINTrainer, apply_ema_smoothing, rmse_np, mae_np,
    physics_metrics_np, eigenvalue_recovery_error, SEED,
    train_on_dataset,
)
from kepin_ablation import (
    get_ablation_configs, BaselineFCN, build_ablation_model,
    compute_all_metrics, r_squared_np,
)
from kepin_optimize_multidomain import (
    get_domain_arch_config, get_domain_training_params,
)
from gpu_config import setup_gpu, build_tf_dataset, get_batch_size

# GPU setup
setup_gpu(mixed_precision=False, xla=False, verbose=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))

# Plot style
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

COLORS = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]


# =========================================================================
# Improved domain configs (better hyperparameters)
# =========================================================================

def get_improved_weather_config():
    """Improved config for Jena Climate — lower RMSE target."""
    return {
        "type": "weather",
        "name": "Jena_Climate",
        "sequence_length": 72,   # 3 days context (was 48)
        "rul_cap": None,
        "csv_path": "datasets/jena_climate_2009_2016.csv",
        "datetime_col": "Date Time",
        "target_col": "T (degC)",
        "feature_cols": ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)",
                         "rh (%)", "VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)",
                         "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)",
                         "wv (m/s)", "max. wv (m/s)", "wd (deg)"],
        "resample_minutes": 60,
        "prediction_horizon": 24,
        "train_ratio": 0.8,
        "test_last_only": False,
    }


def get_improved_finance_config():
    """Improved config for SPY Stock — target RMSE reduction."""
    return {
        "type": "finance",
        "name": "SPY_Stock",
        "sequence_length": 60,      # 3 months context (was 30)
        "rul_cap": None,
        "csv_path": "datasets/spy_stock.csv",
        "datetime_col": "Date",
        "close_col": "Close",
        "prediction_horizon": 5,    # 1 week (was 5, keep same but more context)
        "drawdown_threshold": 0.03, # 3% threshold (was 5% → more training signal)
        "target_mode": "drawdown",
        "train_ratio": 0.8,
        "test_last_only": False,
    }


def get_improved_synthetic_config():
    """Improved config for Synthetic ODE — better eigenvalue recovery."""
    return {
        "type": "synthetic_ode",
        "name": "Synthetic_ODE",
        "sequence_length": 40,    # More temporal context (was 30)
        "rul_cap": 200,
        "n_units_train": 150,     # More training data (was 100)
        "n_units_test": 30,
        "max_life": 400,          # Longer trajectories (was 300)
        "dt": 0.1,
        "noise_std": 0.03,        # Less noise (was 0.05)
        "failure_threshold": 2.0,
        "test_last_only": False,
    }


def get_improved_arch_config(domain, n_features, seq_len, n_train):
    """Improved architecture configs per domain."""
    base = auto_configure(n_features, seq_len, n_train)

    if domain == "weather":
        base["latent_dim"] = 128
        base["lstm_units"] = 128
        base["dropout"] = 0.2        # Even less dropout — data is very abundant
        base["rollout"] = 5
        base["n_heads"] = 8
        base["head_key_dim"] = 32
    elif domain == "finance":
        base["latent_dim"] = 64
        base["lstm_units"] = 64
        base["dropout"] = 0.4        # Moderate dropout (was 0.45)
        base["rollout"] = 3
        base["n_heads"] = 4
        base["head_key_dim"] = 16
    elif domain == "synthetic_ode":
        base["latent_dim"] = 32
        base["lstm_units"] = 64
        base["dropout"] = 0.15       # Very low — clean synthetic data
        base["rollout"] = 5
        base["n_heads"] = 4
        base["head_key_dim"] = 16

    return base


def get_improved_training_params(domain):
    """Improved training hyperparameters per domain."""
    params = {
        "weather": {
            "epochs": 350,
            "patience": 60,
            "lr": 0.0004,
            "batch_size": 256,
            "clip_norm": 1.0,
        },
        "finance": {
            "epochs": 500,
            "patience": 80,
            "lr": 0.0002,
            "batch_size": 32,
            "clip_norm": 0.5,
        },
        "synthetic_ode": {
            "epochs": 400,
            "patience": 60,
            "lr": 0.0003,
            "batch_size": 32,
            "clip_norm": 1.0,
        },
    }
    return params.get(domain, {"epochs": 200, "patience": 40, "lr": 0.0008,
                                "batch_size": 128, "clip_norm": 2.0})


# =========================================================================
# FD002 config
# =========================================================================

def get_fd002_config():
    with open(os.path.join(_script_dir, "datasets_cmapss_config.json")) as f:
        configs = json.load(f)
    return configs[1]  # FD002

def get_fd004_config():
    with open(os.path.join(_script_dir, "datasets_cmapss_config.json")) as f:
        configs = json.load(f)
    return configs[3]  # FD004


# =========================================================================
# Optimized training function (improved)
# =========================================================================

def train_optimized_improved(ds_config, domain, output_dir, run_id=0, verbose=1):
    """Train with improved domain-specific hyperparameters."""

    ds_name = ds_config.get("name", "unknown")
    print(f"\n{'='*70}")
    print(f"  IMPROVED TRAINING: {ds_name} (domain={domain}, run {run_id})")
    print(f"{'='*70}")

    # Load dataset
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    print(ds.summary())

    # Convert
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)

    # EMA smoothing
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    print(f"  EMA alpha = {ema_alpha:.4f}")

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]

    # Domain-specific arch
    arch_config = get_improved_arch_config(domain, n_feat, seq_len, n_train)
    print(f"  Architecture: tier={arch_config['tier']} (improved)")
    print(f"    Latent={arch_config['latent_dim']}, LSTM={arch_config['lstm_units']}, "
          f"Heads={arch_config['n_heads']}, Dropout={arch_config['dropout']}")

    # Domain mode
    if domain in ("weather", "finance"):
        domain_mode = "forecasting"
        n_active_losses = 4
    else:
        domain_mode = "degradation"
        n_active_losses = 7

    # Build model
    model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                              arch_config=arch_config,
                              n_active_losses=n_active_losses)
    print(f"  Domain mode: {domain_mode} ({n_active_losses} losses)")
    print(model.summary_config())

    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Parameters: {n_params:,}")

    # Loss
    loss_fn = make_kepin_loss(
        loss_weights_layer=model.loss_weight_layer,
        use_auto_weights=True,
        domain_mode=domain_mode,
    )

    # Training params
    tp = get_improved_training_params(domain)
    print(f"  Training: epochs={tp['epochs']}, patience={tp['patience']}, "
          f"lr={tp['lr']}, batch={tp['batch_size']}")

    # Optimizer
    optimizer = keras.optimizers.Adam(learning_rate=tp["lr"], clipnorm=tp["clip_norm"])

    # Train
    trainer = KePINTrainer(model, loss_fn, optimizer, clip_norm=tp["clip_norm"])
    history = trainer.fit(
        X_train, Y_train, X_test, Y_test,
        epochs=tp["epochs"], batch_size=tp["batch_size"],
        patience=tp["patience"], initial_lr=tp["lr"],
        verbose=verbose,
    )

    # Evaluate
    Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
    test_rmse = rmse_np(Y_test, Y_pred)
    test_mae = mae_np(Y_test, Y_pred)
    mono_viol, slope_err = physics_metrics_np(Y_test, Y_pred)
    r2 = r_squared_np(Y_test, Y_pred)

    print(f"\n  RESULTS for {ds_name} (improved):")
    print(f"    RMSE:           {test_rmse:.4f}")
    print(f"    MAE:            {test_mae:.4f}")
    print(f"    R2:             {r2:.4f}")
    print(f"    Mono violation: {mono_viol:.6f}")
    print(f"    Slope RMSE:     {slope_err:.4f}")

    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1]
    print(f"    Top |lambda|:   {eig_mags[:5]}")

    # Eigenvalue recovery for ODE
    eig_recovery = None
    if hasattr(ds, "ode_true_K_eigenvalues"):
        true_K_eigs = ds.ode_true_K_eigenvalues
        eig_recovery = eigenvalue_recovery_error(final_eigs, true_K_eigs)
        print(f"    Eigenvalue recovery:")
        print(f"      Mean mag error:   {eig_recovery['mean_mag_error']:.6f}")
        print(f"      Max mag error:    {eig_recovery['max_mag_error']:.6f}")
        print(f"      Mean phase error: {eig_recovery['mean_phase_error']:.6f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    run_tag = f"{ds_name}_run{run_id}"

    model.save_weights(os.path.join(output_dir, f"kepin_{run_tag}.weights.h5"))
    np.savez(os.path.join(output_dir, f"predictions_{run_tag}.npz"),
             y_true=Y_test, y_pred=Y_pred)
    np.savez(os.path.join(output_dir, f"eigenvalues_{run_tag}.npz"),
             eigenvalue_history=np.array(history["eigenvalues"]),
             final_eigenvalues=final_eigs,
             koopman_matrix=model.get_koopman_matrix())

    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
    })
    hist_df.to_csv(os.path.join(output_dir, f"history_{run_tag}.csv"), index=False)

    # Training plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train", color="#0173B2")
    axes[0].plot(history["val_loss"], label="Val", color="#DE8F05")
    axes[0].set_title(f"Total Loss - {ds_name}")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_rmse"], label="Train", color="#0173B2")
    axes[1].plot(history["val_rmse"], label="Val", color="#DE8F05")
    axes[1].set_title("RMSE"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    eig_hist = np.array(history["eigenvalues"])
    for mode_idx in range(min(4, eig_hist.shape[1])):
        axes[2].plot(np.abs(eig_hist[:, mode_idx]),
                     label=f"Mode {mode_idx+1}", alpha=0.8)
    axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_title("|lambda| Convergence"); axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"training_{run_tag}.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, f"training_{run_tag}.pdf"))
    plt.close(fig)

    result = {
        "dataset": ds_name,
        "domain": domain,
        "run_id": run_id,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R2": r2,
        "MonoViol": mono_viol,
        "SlopeRMSE": slope_err,
        "Tier": arch_config["tier"],
        "Epochs": len(history["epoch"]),
        "latent_dim": arch_config["latent_dim"],
        "top_eig_mags": eig_mags[:5].tolist(),
    }
    if eig_recovery:
        result["eig_recovery"] = eig_recovery

    return result


# =========================================================================
# Ablation on a single domain
# =========================================================================

def run_ablation_on_domain(domain, ds_config, output_dir, verbose=1):
    """Run 5 ablation configs on one domain dataset."""

    ds_name = ds_config.get("name", "unknown")
    ds_type = ds_config.get("type", "csv")

    ab_configs = get_ablation_configs()

    # Add domain_mode to each ablation config
    if ds_type in ("weather", "finance"):
        for c in ab_configs:
            c["domain_mode"] = "forecasting"
    else:
        for c in ab_configs:
            c["domain_mode"] = "degradation"

    # Domain training params
    train_params = {
        "fd002": {"epochs": 200, "patience": 40, "batch_size": 128, "lr": 0.0008},
        "fd004": {"epochs": 200, "patience": 40, "batch_size": 128, "lr": 0.0008},
        "weather": {"epochs": 200, "patience": 50, "batch_size": 256, "lr": 0.0005},
        "finance": {"epochs": 200, "patience": 50, "batch_size": 64, "lr": 0.0003},
    }
    tp = train_params.get(domain, {"epochs": 200, "patience": 40, "batch_size": 128, "lr": 0.0008})

    print(f"\n{'='*60}")
    print(f"  ABLATION on {ds_name} ({domain})")
    print(f"  5 configs, epochs={tp['epochs']}, patience={tp['patience']}")
    print(f"{'='*60}")

    results = []

    for ab_config in ab_configs:
        ab_name = ab_config["name"]
        print(f"\n  --- {ab_name} on {ds_name} ---")

        try:
            # Load data
            ds = GDS.load_dataset_from_config(ds_config)
            X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()

            X_train = convert_4d_to_3d(X_train_4d)
            X_test = convert_4d_to_3d(X_test_4d)
            X_train, ema_alpha = apply_ema_smoothing(X_train)
            X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)

            seq_len = X_train.shape[1]
            n_feat = X_train.shape[2]
            n_train = X_train.shape[0]

            # Get domain arch config
            if domain in ("weather", "finance"):
                arch_config = get_improved_arch_config(domain, n_feat, seq_len, n_train)
            else:
                arch_config = auto_configure(n_feat, seq_len, n_train)

            # Domain mode
            domain_mode = ab_config.get("domain_mode", "degradation")
            n_active_losses = 4 if domain_mode == "forecasting" else 7

            # Build model
            if not ab_config["use_koopman"]:
                model = BaselineFCN(
                    input_shape_tuple=(seq_len, n_feat),
                    arch_config=arch_config,
                    n_train=n_train,
                )
            else:
                model = build_kepin_model(
                    seq_len, n_feat, n_train=n_train,
                    arch_config=arch_config,
                    n_active_losses=n_active_losses,
                )

            # Build loss
            if ab_config["use_auto_weights"] and ab_config["use_koopman"]:
                loss_fn = make_kepin_loss(
                    loss_weights_layer=model.loss_weight_layer,
                    use_auto_weights=True,
                    domain_mode=domain_mode,
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
                    domain_mode=domain_mode,
                )

            # Train
            optimizer = keras.optimizers.Adam(learning_rate=tp["lr"], clipnorm=1.0)
            trainer = KePINTrainer(model, loss_fn, optimizer, clip_norm=2.0)

            history = trainer.fit(
                X_train, Y_train, X_test, Y_test,
                epochs=tp["epochs"], batch_size=tp["batch_size"],
                patience=tp["patience"], initial_lr=tp["lr"],
                verbose=verbose,
            )

            # Evaluate
            Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
            metrics = compute_all_metrics(Y_test, Y_pred)

            print(f"    RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
                  f"R2={metrics['R2']:.4f}  Mono={metrics['MonoViol']:.6f}")

            # Save predictions
            os.makedirs(output_dir, exist_ok=True)
            run_tag = f"{ds_name}_{ab_config['tag']}_run0"
            np.savez(os.path.join(output_dir, f"predictions_{run_tag}.npz"),
                     y_true=Y_test, y_pred=Y_pred)

            result = {
                "ablation": ab_name,
                "ablation_tag": ab_config["tag"],
                "dataset": ds_name,
                "domain": domain,
                **metrics,
                "epochs_trained": len(history["epoch"]),
            }
            results.append(result)

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "ablation": ab_name,
                "ablation_tag": ab_config["tag"],
                "dataset": ds_name,
                "domain": domain,
                "error": str(e),
            })

    return results


# =========================================================================
# Generate all publication figures
# =========================================================================

def generate_all_figures(cmapss_exp_dir, multidomain_results, ablation_results,
                         output_dir):
    """Generate all figures for the paper."""

    os.makedirs(output_dir, exist_ok=True)
    DATASETS_CM = ["CMAPSS_FD001", "CMAPSS_FD002", "CMAPSS_FD003", "CMAPSS_FD004"]
    LABELS_CM = ["FD001", "FD002", "FD003", "FD004"]
    COLORS_CM = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    # 1. Training convergence (CMAPSS)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, (ds, label, color) in enumerate(zip(DATASETS_CM, LABELS_CM, COLORS_CM)):
            hist_path = os.path.join(cmapss_exp_dir, ds, f"history_{ds}_run0.csv")
            if not os.path.exists(hist_path):
                continue
            df = pd.read_csv(hist_path)
            ax = axes[i]
            ax.plot(df['epoch'], df['train_rmse'], color=color, alpha=0.8,
                    label='Train RMSE', linewidth=1.5)
            ax.plot(df['epoch'], df['val_rmse'], color=color, linestyle='--',
                    alpha=0.8, label='Val RMSE', linewidth=1.5)
            ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE')
            ax.set_title(f'{label}'); ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
        plt.suptitle('Training Convergence - KePIN', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_convergence.pdf"))
        plt.savefig(os.path.join(output_dir, "training_convergence.png"))
        plt.close()
        print("[OK] training_convergence")
    except Exception as e:
        print(f"[WARN] training_convergence failed: {e}")

    # 2. Loss components (CMAPSS)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, (ds, label, color) in enumerate(zip(DATASETS_CM, LABELS_CM, COLORS_CM)):
            hist_path = os.path.join(cmapss_exp_dir, ds, f"history_{ds}_run0.csv")
            if not os.path.exists(hist_path):
                continue
            df = pd.read_csv(hist_path)
            ax = axes[i]
            ax.plot(df['epoch'], df['train_loss'], label='Total', linewidth=1.5, color='black')
            if 'train_rul_mse' in df.columns:
                ax.plot(df['epoch'], df['train_rul_mse'], label='RUL', linewidth=1.2, alpha=0.8)
            if 'train_spectral' in df.columns:
                ax.plot(df['epoch'], df['train_spectral'], label='Spectral', linewidth=1.2, alpha=0.8)
            if 'train_multi_step' in df.columns:
                ax.plot(df['epoch'], df['train_multi_step'], label='Multi-step', linewidth=1.2, alpha=0.8)
            ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
            ax.set_title(f'{label}'); ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3); ax.set_yscale('log')
        plt.suptitle('Loss Component Evolution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_components.pdf"))
        plt.savefig(os.path.join(output_dir, "loss_components.png"))
        plt.close()
        print("[OK] loss_components")
    except Exception as e:
        print(f"[WARN] loss_components failed: {e}")

    # 3. Eigenvalue spectrum (CMAPSS)
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        for i, (ds, label, color) in enumerate(zip(DATASETS_CM, LABELS_CM, COLORS_CM)):
            eig_path = os.path.join(cmapss_exp_dir, ds, f"eigenvalues_{ds}_run0.npz")
            if not os.path.exists(eig_path):
                continue
            data = np.load(eig_path)
            eigs = data['final_eigenvalues']
            ax = axes[i]
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8, alpha=0.4)
            ax.scatter(eigs.real, eigs.imag, c=color, s=40, zorder=5,
                       edgecolors='black', linewidths=0.5, alpha=0.8)
            ax.set_xlabel('Real'); ax.set_ylabel('Imaginary')
            ax.set_title(f'{label}'); ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axvline(0, color='grey', linewidth=0.5)
            lim = max(1.2, np.max(np.abs(eigs))*1.3)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        plt.suptitle('Koopman Eigenvalue Spectrum', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eigenvalue_spectrum.pdf"))
        plt.savefig(os.path.join(output_dir, "eigenvalue_spectrum.png"))
        plt.close()
        print("[OK] eigenvalue_spectrum")
    except Exception as e:
        print(f"[WARN] eigenvalue_spectrum failed: {e}")

    # 4. Eigenvalue convergence
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, (ds, label, color) in enumerate(zip(DATASETS_CM, LABELS_CM, COLORS_CM)):
            eig_path = os.path.join(cmapss_exp_dir, ds, f"eigenvalues_{ds}_run0.npz")
            if not os.path.exists(eig_path):
                continue
            data = np.load(eig_path)
            eig_hist = data['eigenvalue_history']
            mags = np.abs(eig_hist)
            ax = axes[i]
            epochs_arr = np.arange(mags.shape[0])
            final_mags = mags[-1]
            top_idx = np.argsort(final_mags)[::-1][:5]
            cmap = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
            for j, idx in enumerate(top_idx):
                ax.plot(epochs_arr, mags[:, idx], color=cmap[j], linewidth=1.2,
                        label=f'$\\lambda_{{{j+1}}}$', alpha=0.85)
            ax.axhline(1.0, color='red', linewidth=0.8, linestyle=':', alpha=0.6)
            ax.set_xlabel('Epoch'); ax.set_ylabel('$|\\lambda|$')
            ax.set_title(f'{label}'); ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.suptitle('Eigenvalue Magnitude Convergence', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eigenvalue_convergence.pdf"))
        plt.savefig(os.path.join(output_dir, "eigenvalue_convergence.png"))
        plt.close()
        print("[OK] eigenvalue_convergence")
    except Exception as e:
        print(f"[WARN] eigenvalue_convergence failed: {e}")

    # 5. Prediction scatter (CMAPSS)
    try:
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for i, (ds, label, color) in enumerate(zip(DATASETS_CM, LABELS_CM, COLORS_CM)):
            pred_path = os.path.join(cmapss_exp_dir, ds, f"predictions_{ds}_run0.npz")
            if not os.path.exists(pred_path):
                continue
            data = np.load(pred_path)
            y_true = data['y_true'].flatten()
            y_pred = data['y_pred'].flatten()
            error = y_pred - y_true

            ax = axes[0, i]
            ax.scatter(y_true, y_pred, c=color, s=15, alpha=0.6, edgecolors='none')
            lim = max(y_true.max(), y_pred.max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5)
            ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
            ax.set_title(f'{label}'); ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            ax2 = axes[1, i]
            ax2.hist(error, bins=25, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.axvline(0, color='black', linewidth=1, linestyle='--')
            ax2.set_xlabel('Error'); ax2.set_ylabel('Count')
            ax2.set_title(f'{label} Error Dist.')
            ax2.grid(True, alpha=0.3)

        plt.suptitle('Predictions vs Ground Truth', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predictions_scatter.pdf"))
        plt.savefig(os.path.join(output_dir, "predictions_scatter.png"))
        plt.close()
        print("[OK] predictions_scatter")
    except Exception as e:
        print(f"[WARN] predictions_scatter failed: {e}")

    # 6. Results bar (CMAPSS)
    try:
        results_json = os.path.join(cmapss_exp_dir, "kepin_results.json")
        if os.path.exists(results_json):
            with open(results_json) as f:
                cm_results = json.load(f)
            rmse_vals = [r['rmse'] for r in cm_results if 'rmse' in r]
            mae_vals = [r['mae'] for r in cm_results if 'mae' in r]

            x = np.arange(len(LABELS_CM))
            width = 0.35
            fig, ax = plt.subplots(figsize=(8, 5))
            bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE',
                           color='#2196F3', edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE',
                           color='#FF9800', edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Dataset'); ax.set_ylabel('Error')
            ax.set_title('KePIN Performance on C-MAPSS', fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(LABELS_CM)
            ax.legend(); ax.grid(True, alpha=0.3, axis='y')
            for bar in bars1:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
            for bar in bars2:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "results_bar.pdf"))
            plt.savefig(os.path.join(output_dir, "results_bar.png"))
            plt.close()
            print("[OK] results_bar")
    except Exception as e:
        print(f"[WARN] results_bar failed: {e}")

    # 7. Eigenvalue histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (ds, label, color) in enumerate(zip(DATASETS_CM, LABELS_CM, COLORS_CM)):
            eig_path = os.path.join(cmapss_exp_dir, ds, f"eigenvalues_{ds}_run0.npz")
            if not os.path.exists(eig_path):
                continue
            data = np.load(eig_path)
            mags = np.abs(data['final_eigenvalues'])
            ax.hist(mags, bins=20, alpha=0.5, color=color, label=label,
                    edgecolor='black', linewidth=0.5)
        ax.axvline(1.0, color='red', linewidth=1.5, linestyle='--', label='Unit circle')
        ax.set_xlabel('$|\\lambda|$'); ax.set_ylabel('Count')
        ax.set_title('Koopman Eigenvalue Magnitudes', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "eigenvalue_histogram.pdf"))
        plt.savefig(os.path.join(output_dir, "eigenvalue_histogram.png"))
        plt.close()
        print("[OK] eigenvalue_histogram")
    except Exception as e:
        print(f"[WARN] eigenvalue_histogram failed: {e}")

    # 8. Architecture diagram
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
        blocks = [
            (1.0, 3.0, 2.0, 1.5, 'Input\n$\\mathbf{X} \\in \\mathbb{R}^{T \\times d}$', '#E3F2FD'),
            (3.5, 3.0, 2.0, 1.5, 'Conv1D + SE\nEncoder\n$f_\\theta(\\cdot)$', '#BBDEFB'),
            (6.0, 3.0, 2.2, 1.5, 'Koopman\nOperator\n$\\mathbf{K} = \\mathbf{U}\\Sigma\\mathbf{V}^\\top$', '#90CAF9'),
            (8.7, 3.0, 2.0, 1.5, 'Spectral\nFeatures\n$\\phi(\\lambda_i)$', '#64B5F6'),
            (11.2, 3.0, 2.0, 1.5, 'Prediction\nHead\n$\\hat{y} = g_\\psi(\\cdot)$', '#42A5F5'),
        ]
        for bx, by, bw, bh, text, color in blocks:
            rect = plt.Rectangle((bx, by-bh/2), bw, bh, linewidth=1.5, edgecolor='black',
                                 facecolor=color, zorder=2, clip_on=False)
            ax.add_patch(rect)
            ax.text(bx + bw/2, by, text, ha='center', va='center', fontsize=9,
                    fontweight='bold', zorder=3)
        arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
        for x_s, x_e in [(3.0, 3.5), (5.5, 6.0), (8.2, 8.7), (10.7, 11.2)]:
            ax.annotate('', xy=(x_e, 3.0), xytext=(x_s, 3.0), arrowprops=arrow_props)
        loss_y = 0.8
        losses = [
            (2.0, 'Prediction\n$\\mathcal{L}_{\\text{pred}}$'),
            (4.0, 'Monotonicity\n$\\mathcal{L}_{\\text{mono}}$'),
            (6.0, 'Spectral\n$\\mathcal{L}_{\\text{spec}}$'),
            (8.0, 'Multi-step\n$\\mathcal{L}_{\\text{multi}}$'),
            (10.0, 'Koopman\n$\\mathcal{L}_{\\text{koop}}$'),
            (12.0, 'Asymmetric\n$\\mathcal{L}_{\\text{asym}}$'),
        ]
        for lx, ltext in losses:
            ax.text(lx, loss_y, ltext, ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                              edgecolor='black', linewidth=0.8))
        ax.text(7.0, 1.85, 'Physics-Informed Composite Loss (Kendall Weighting)',
                ha='center', va='bottom', fontsize=10, style='italic', color='#333')
        ax.set_title('KePIN Architecture Overview', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "architecture.pdf"))
        plt.savefig(os.path.join(output_dir, "architecture.png"))
        plt.close()
        print("[OK] architecture")
    except Exception as e:
        print(f"[WARN] architecture failed: {e}")

    # 9. Multi-domain results bar chart
    if multidomain_results:
        try:
            valid_md = [r for r in multidomain_results if "error" not in r]
            if valid_md:
                ds_names = [r["dataset"] for r in valid_md]
                rmse_vals = [r["RMSE"] for r in valid_md]
                mae_vals = [r["MAE"] for r in valid_md]

                short_names = []
                for n in ds_names:
                    if "Jena" in n: short_names.append("Jena Climate")
                    elif "SPY" in n: short_names.append("SPY Stock")
                    elif "Synthetic" in n: short_names.append("Synth. ODE")
                    else: short_names.append(n[:12])

                x = np.arange(len(short_names))
                width = 0.35
                fig, ax = plt.subplots(figsize=(8, 5))
                bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE',
                               color='#2196F3', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE',
                               color='#FF9800', edgecolor='black', linewidth=0.5)
                ax.set_xlabel('Dataset'); ax.set_ylabel('Error')
                ax.set_title('KePIN Multi-Domain Performance (Improved)', fontweight='bold')
                ax.set_xticks(x); ax.set_xticklabels(short_names)
                ax.legend(); ax.grid(True, alpha=0.3, axis='y')
                for bar in bars1:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
                for bar in bars2:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "multidomain_results_bar.pdf"))
                plt.savefig(os.path.join(output_dir, "multidomain_results_bar.png"))
                plt.close()
                print("[OK] multidomain_results_bar")
        except Exception as e:
            print(f"[WARN] multidomain_results_bar failed: {e}")

    # 10. Ablation heatmap (cross-domain)
    if ablation_results:
        try:
            valid_ab = [r for r in ablation_results if "error" not in r and "RMSE" in r]
            if valid_ab:
                ab_df = pd.DataFrame(valid_ab)
                pivot = ab_df.groupby(["ablation", "dataset"])["RMSE"].mean().unstack(fill_value=np.nan)

                fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                                max(4, len(pivot.index) * 0.8)))
                data = pivot.values
                im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
                for ii in range(data.shape[0]):
                    for jj in range(data.shape[1]):
                        val = data[ii, jj]
                        if not np.isnan(val):
                            text_color = "white" if val > np.nanmedian(data) else "black"
                            ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                                    color=text_color, fontsize=9)

                short_col_names = []
                for c in pivot.columns:
                    if "FD002" in c: short_col_names.append("FD002")
                    elif "FD004" in c: short_col_names.append("FD004")
                    elif "Jena" in c: short_col_names.append("Jena")
                    elif "SPY" in c: short_col_names.append("SPY")
                    else: short_col_names.append(c[:10])

                ax.set_xticks(np.arange(len(pivot.columns)))
                ax.set_yticks(np.arange(len(pivot.index)))
                ax.set_xticklabels(short_col_names, rotation=35, ha="right")
                ax.set_yticklabels(pivot.index)
                ax.set_xlabel("Dataset"); ax.set_ylabel("Ablation Config")
                ax.set_title("Cross-Domain Ablation Heatmap (RMSE)")
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label("RMSE")

                # Highlight best
                for jj in range(data.shape[1]):
                    col = data[:, jj]
                    if not np.all(np.isnan(col)):
                        best_i = np.nanargmin(col)
                        ax.add_patch(plt.Rectangle((jj - 0.5, best_i - 0.5), 1, 1,
                                                   fill=False, edgecolor="green", linewidth=2.5))

                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, "ablation_heatmap.pdf"))
                fig.savefig(os.path.join(output_dir, "ablation_heatmap.png"))
                plt.close()
                print("[OK] ablation_heatmap")
        except Exception as e:
            print(f"[WARN] ablation_heatmap failed: {e}")

    # 11. Ablation grouped bar chart
    if ablation_results:
        try:
            valid_ab = [r for r in ablation_results if "error" not in r and "RMSE" in r]
            if valid_ab:
                ab_df = pd.DataFrame(valid_ab)
                datasets = ab_df["dataset"].unique()
                ablations = ab_df["ablation"].unique()
                n_ds = len(datasets)
                n_ab = len(ablations)

                fig, ax = plt.subplots(figsize=(max(10, n_ds * 2), 6))
                x = np.arange(n_ds)
                width = 0.8 / n_ab

                for i, ab in enumerate(ablations):
                    means = []
                    for ds in datasets:
                        mask = (ab_df["ablation"] == ab) & (ab_df["dataset"] == ds)
                        vals = ab_df.loc[mask, "RMSE"].dropna()
                        means.append(vals.mean() if len(vals) > 0 else 0)
                    ax.bar(x + i * width - (n_ab-1)*width/2, means, width,
                           label=ab, color=COLORS[i % len(COLORS)],
                           edgecolor="black", linewidth=0.5)

                short_ds = []
                for d in datasets:
                    if "FD002" in d: short_ds.append("FD002")
                    elif "FD004" in d: short_ds.append("FD004")
                    elif "Jena" in d: short_ds.append("Jena")
                    elif "SPY" in d: short_ds.append("SPY")
                    else: short_ds.append(d[:10])

                ax.set_xlabel("Dataset"); ax.set_ylabel("RMSE")
                ax.set_title("Ablation Study: RMSE Across Domains")
                ax.set_xticks(x); ax.set_xticklabels(short_ds, rotation=30, ha="right")
                ax.legend(fontsize=7, ncol=2, loc="best")
                ax.annotate("(lower is better)", xy=(0.99, 0.01),
                            xycoords="axes fraction", ha="right", va="bottom",
                            fontsize=8, fontstyle="italic", alpha=0.6)
                plt.tight_layout()
                fig.savefig(os.path.join(output_dir, "ablation_rmse_bar.pdf"))
                fig.savefig(os.path.join(output_dir, "ablation_rmse_bar.png"))
                plt.close()
                print("[OK] ablation_rmse_bar")
        except Exception as e:
            print(f"[WARN] ablation_rmse_bar failed: {e}")

    # 12. Cross-domain bar chart (all 7 datasets)
    try:
        all_results_list = []
        # Add CMAPSS from existing results
        results_json = os.path.join(cmapss_exp_dir, "kepin_results.json")
        if os.path.exists(results_json):
            with open(results_json) as f:
                cm = json.load(f)
            for r in cm:
                if 'rmse' in r:
                    all_results_list.append({
                        "dataset": r["dataset"],
                        "RMSE": r["rmse"],
                        "MAE": r["mae"],
                    })
        # Add multi-domain
        if multidomain_results:
            for r in multidomain_results:
                if "error" not in r:
                    all_results_list.append({
                        "dataset": r["dataset"],
                        "RMSE": r["RMSE"],
                        "MAE": r["MAE"],
                    })

        if all_results_list:
            ds_names = [r["dataset"] for r in all_results_list]
            rmse_vals = [r["RMSE"] for r in all_results_list]

            short_names = []
            for n in ds_names:
                if "FD001" in n: short_names.append("FD001")
                elif "FD002" in n: short_names.append("FD002")
                elif "FD003" in n: short_names.append("FD003")
                elif "FD004" in n: short_names.append("FD004")
                elif "Jena" in n: short_names.append("Jena")
                elif "SPY" in n: short_names.append("SPY")
                elif "Synth" in n: short_names.append("Synth")
                else: short_names.append(n[:8])

            domain_colors = []
            for n in ds_names:
                if "FD" in n: domain_colors.append("#2196F3")
                elif "Jena" in n: domain_colors.append("#4CAF50")
                elif "SPY" in n: domain_colors.append("#FF9800")
                else: domain_colors.append("#9C27B0")

            x = np.arange(len(short_names))
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(x, rmse_vals, color=domain_colors, edgecolor='black', linewidth=0.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
            ax.set_xlabel('Dataset'); ax.set_ylabel('RMSE')
            ax.set_title('KePIN Cross-Domain RMSE', fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(short_names, rotation=30, ha="right")
            ax.grid(True, alpha=0.3, axis='y')

            # Legend for domains
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2196F3', edgecolor='black', label='Maintenance'),
                Patch(facecolor='#4CAF50', edgecolor='black', label='Weather'),
                Patch(facecolor='#FF9800', edgecolor='black', label='Finance'),
                Patch(facecolor='#9C27B0', edgecolor='black', label='Synthetic'),
            ]
            ax.legend(handles=legend_elements, loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "cross_domain_rmse.pdf"))
            plt.savefig(os.path.join(output_dir, "cross_domain_rmse.png"))
            plt.close()
            print("[OK] cross_domain_rmse")
    except Exception as e:
        print(f"[WARN] cross_domain_rmse failed: {e}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Run all KePIN experiments")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_optimize", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"kepin_full_{timestamp}")
    else:
        output_base = args.output_dir

    os.makedirs(output_base, exist_ok=True)

    # Best existing CMAPSS experiment dir
    cmapss_exp_dir = os.path.join(_project_dir, "experiments_result",
                                   "kepin_20260224_002102")

    # ---- Step 1: Optimized training on Weather, Finance, Synthetic ODE ----
    multidomain_results = []
    if not args.skip_optimize:
        print("\n" + "="*80)
        print("  STEP 1: OPTIMIZED MULTI-DOMAIN TRAINING")
        print("="*80)

        domain_configs = [
            ("weather", get_improved_weather_config()),
            ("finance", get_improved_finance_config()),
            ("synthetic_ode", get_improved_synthetic_config()),
        ]

        for domain, ds_config in domain_configs:
            ds_name = ds_config["name"]
            ds_dir = os.path.join(output_base, "optimized", ds_name)
            try:
                result = train_optimized_improved(
                    ds_config, domain, ds_dir,
                    run_id=0, verbose=args.verbose
                )
                multidomain_results.append(result)
                print(f"\n  >>> {ds_name}: RMSE={result['RMSE']:.4f}")
            except Exception as e:
                print(f"\n  FAILED: {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                multidomain_results.append({"dataset": ds_name, "domain": domain, "error": str(e)})

        # Save
        md_path = os.path.join(output_base, "multidomain_results.json")
        with open(md_path, "w") as f:
            json.dump(multidomain_results, f, indent=2, default=str)
        print(f"\n  Saved multidomain results to {md_path}")

    # ---- Step 2: Ablation on FD002, FD004, Weather, Finance ----
    ablation_results = []
    if not args.skip_ablation:
        print("\n" + "="*80)
        print("  STEP 2: ABLATION STUDIES (FD002, FD004, Weather, Finance)")
        print("="*80)

        ablation_domains = [
            ("fd002", get_fd002_config()),
            ("fd004", get_fd004_config()),
            ("weather", get_improved_weather_config()),
            ("finance", get_improved_finance_config()),
        ]

        for domain, ds_config in ablation_domains:
            ds_name = ds_config.get("name", domain)
            ab_dir = os.path.join(output_base, "ablation", ds_name)
            results = run_ablation_on_domain(
                domain, ds_config, ab_dir, verbose=args.verbose
            )
            ablation_results.extend(results)

            # Save intermediate
            ab_path = os.path.join(output_base, "ablation_results.json")
            with open(ab_path, "w") as f:
                json.dump(ablation_results, f, indent=2, default=str)

        # Final save
        ab_df = pd.DataFrame(ablation_results)
        ab_df.to_csv(os.path.join(output_base, "ablation_results.csv"), index=False)
        print(f"\n  Saved ablation results to {output_base}")

        # Print summary
        valid_ab = [r for r in ablation_results if "error" not in r and "RMSE" in r]
        if valid_ab:
            print(f"\n{'='*80}")
            print("  ABLATION SUMMARY")
            print(f"{'='*80}")
            ab_df_valid = pd.DataFrame(valid_ab)
            for ds in sorted(ab_df_valid["dataset"].unique()):
                ds_df = ab_df_valid[ab_df_valid["dataset"] == ds]
                print(f"\n  {ds}:")
                print(f"  {'Config':<25}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}  {'Mono':>8}")
                print(f"  {'-'*25}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
                for _, row in ds_df.iterrows():
                    print(f"  {row['ablation']:<25}  {row['RMSE']:8.4f}  {row['MAE']:8.4f}  "
                          f"{row.get('R2', 0):8.4f}  {row.get('MonoViol', 0):8.4f}")

    # ---- Step 3: Generate all figures ----
    if not args.skip_plots:
        print("\n" + "="*80)
        print("  STEP 3: GENERATING PUBLICATION FIGURES")
        print("="*80)

        figures_dir = os.path.join(_project_dir, "paper", "figures")
        generate_all_figures(cmapss_exp_dir, multidomain_results, ablation_results,
                             figures_dir)

    # ---- Final Summary ----
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)

    if multidomain_results:
        print("\n  Multi-Domain (Improved):")
        for r in multidomain_results:
            if "error" not in r:
                print(f"    {r['dataset']:20s}: RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  "
                      f"Tier={r['Tier']}  Epochs={r['Epochs']}")
            else:
                print(f"    {r['dataset']:20s}: FAILED")

    if ablation_results:
        valid_ab = [r for r in ablation_results if "error" not in r]
        if valid_ab:
            print(f"\n  Ablation: {len(valid_ab)} successful runs across "
                  f"{len(set(r['dataset'] for r in valid_ab))} datasets")

    print(f"\n  All results saved to: {output_base}")
    print(f"  Figures saved to: {os.path.join(_project_dir, 'paper', 'figures')}")
    print("="*80)


if __name__ == "__main__":
    main()
