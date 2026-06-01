#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Multi-Domain Training for KePIN.

Domain-specific optimizations:
  1. Weather (Jena Climate):
     - Longer sequence (72h → captures 3 diurnal cycles)
     - Lower LR (5e-4) for smoother convergence
     - More epochs (300) + patience (60) since dataset is large
     - Target normalization for stable Huber loss

  2. Finance (SPY Stock):
     - Use "return" mode instead of "drawdown" (cleaner regression target)
     - Sequence length 60 (3 months of trading days for context)
     - Higher dropout (0.45) to handle noise
     - Longer prediction horizon (10 days)

  3. Synthetic ODE:
     - More training units (200 train, 50 test) for richer dynamics
     - Lower noise (0.02) for cleaner eigenvalue recovery
     - Focus on eigenvalue recovery metrics

Does NOT modify CMAPSS results in any way.
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
from koopman_module import extract_spectral_features
from gpu_config import setup_gpu, build_tf_dataset, get_batch_size, is_mixed_precision_enabled
from kepin_training import (
    rmse_np, mae_np, physics_metrics_np, eigenvalue_recovery_error,
    apply_ema_smoothing, KePINTrainer, SEED,
)

# ---------- GPU setup ----------
setup_gpu(mixed_precision=False, xla=False, verbose=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))


# =========================================================================
# Domain-specific configs
# =========================================================================

DOMAIN_CONFIGS = {
    "Jena_Climate": {
        "type": "weather",
        "name": "Jena_Climate",
        "sequence_length": 72,          # 72 hours = 3 full diurnal cycles
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
        # Training hyperparameters
        "epochs": 300,
        "patience": 60,
        "lr": 5e-4,
        "batch_size": 256,
        "arch_override": {
            "n_blocks": 4,
            "filters": [64, 128, 128, 256],
            "kernels": [7, 5, 5, 3],
            "latent_dim": 96,            # Slightly smaller latent for efficiency
            "lstm_units": 96,
            "n_heads": 4,
            "head_key_dim": 24,
            "dropout": 0.25,             # Lower dropout for large dataset
            "rollout": 5,                # More rollout for temporal patterns
            "spectral_k": 5,
            "tier": "medium",
        },
    },
    "SPY_Stock": {
        "type": "finance",
        "name": "SPY_Stock",
        "sequence_length": 60,           # 60 trading days (~3 months context)
        "rul_cap": None,
        "csv_path": "datasets/spy_stock.csv",
        "datetime_col": "Date",
        "close_col": "Close",
        "prediction_horizon": 10,        # 10-day forward return
        "drawdown_threshold": 0.05,
        "target_mode": "return",         # Use return instead of drawdown
        "train_ratio": 0.8,
        "test_last_only": False,
        # Training hyperparameters
        "epochs": 400,
        "patience": 80,
        "lr": 3e-4,
        "batch_size": 64,
        "arch_override": {
            "n_blocks": 3,
            "filters": [32, 64, 128],
            "kernels": [7, 5, 3],
            "latent_dim": 48,            # Smaller model to avoid overfitting
            "lstm_units": 48,
            "n_heads": 4,
            "head_key_dim": 12,
            "dropout": 0.45,             # High dropout for noisy data
            "rollout": 3,
            "spectral_k": 4,
            "tier": "small",
        },
    },
    "Synthetic_ODE": {
        "type": "synthetic_ode",
        "name": "Synthetic_ODE",
        "sequence_length": 50,            # Longer sequences for dynamics
        "rul_cap": 200,
        "n_units_train": 200,             # More units for richer training
        "n_units_test": 50,
        "max_life": 400,                  # Longer trajectories
        "dt": 0.1,
        "noise_std": 0.02,               # Less noise for cleaner eigvals
        "failure_threshold": 2.0,
        "test_last_only": False,
        # Training hyperparameters
        "epochs": 300,
        "patience": 60,
        "lr": 5e-4,
        "batch_size": 64,
        "arch_override": {
            "n_blocks": 3,
            "filters": [64, 128, 128],
            "kernels": [7, 5, 3],
            "latent_dim": 32,            # Small latent (true system is 2D)
            "lstm_units": 48,
            "n_heads": 4,
            "head_key_dim": 8,
            "dropout": 0.2,
            "rollout": 5,                # More rollout steps for dynamics
            "spectral_k": 4,
            "tier": "small",
        },
    },
}


def train_domain_optimized(domain_name: str, output_dir: str,
                           run_id: int = 0, verbose: int = 1):
    """Train KePIN with domain-optimized hyperparameters.

    Returns:
        results: dict with metrics
    """
    cfg = DOMAIN_CONFIGS[domain_name]
    ds_config = {k: v for k, v in cfg.items()
                 if k not in ("epochs", "patience", "lr", "batch_size", "arch_override")}

    epochs = cfg["epochs"]
    patience = cfg["patience"]
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    arch_override = cfg.get("arch_override", None)

    print(f"\n{'='*70}")
    print(f"  OPTIMIZED KePIN Training: {domain_name} (run {run_id})")
    print(f"  epochs={epochs}, patience={patience}, lr={lr}, batch={batch_size}")
    print(f"{'='*70}")

    # --- Load dataset ---
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    print(ds.summary())

    # --- Convert 4D → 3D ---
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)

    # --- EMA smoothing ---
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    print(f"  EMA α = {ema_alpha:.4f}")

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]

    # --- Architecture config ---
    if arch_override:
        arch_config = arch_override.copy()
        # Adjust kernels to be valid for actual seq_len
        arch_config["kernels"] = [min(k, seq_len) for k in arch_config["kernels"]]
        arch_config["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch_config["kernels"]]
    else:
        arch_config = auto_configure(n_feat, seq_len, n_train)

    print(f"  Architecture: {arch_config['tier']}, latent={arch_config['latent_dim']}")
    print(f"  Filters: {arch_config['filters']}, Dropout: {arch_config['dropout']}")

    # --- Domain mode ---
    ds_type = ds_config.get("type", "csv")
    if ds_type in ("weather", "finance"):
        domain_mode = "forecasting"
        n_active_losses = 4
    else:
        domain_mode = "degradation"
        n_active_losses = 7

    # --- Build model ---
    model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                              arch_config=arch_config,
                              n_active_losses=n_active_losses)
    print(f"  Domain mode: {domain_mode} ({n_active_losses} active losses)")
    print(model.summary_config())

    n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Total params: {n_params:,}")

    # --- Loss function ---
    loss_fn = make_kepin_loss(
        loss_weights_layer=model.loss_weight_layer,
        use_auto_weights=True,
        domain_mode=domain_mode,
    )

    # --- Optimizer ---
    optimizer = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    # --- Build trainer ---
    trainer = KePINTrainer(model, loss_fn, optimizer)

    # --- Train ---
    history = trainer.fit(
        X_train, Y_train, X_test, Y_test,
        epochs=epochs, batch_size=batch_size,
        patience=patience, initial_lr=lr,
        verbose=verbose,
    )

    # --- Evaluate ---
    Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
    test_rmse = rmse_np(Y_test, Y_pred)
    test_mae = mae_np(Y_test, Y_pred)
    mono_viol, slope_err = physics_metrics_np(Y_test, Y_pred)

    # R² score
    ss_res = np.sum((Y_test.flatten() - Y_pred.flatten()) ** 2)
    ss_tot = np.sum((Y_test.flatten() - np.mean(Y_test.flatten())) ** 2) + 1e-10
    r2 = 1.0 - ss_res / ss_tot

    print(f"\n  Results for {domain_name}:")
    print(f"    RMSE:    {test_rmse:.4f}")
    print(f"    MAE:     {test_mae:.4f}")
    print(f"    R²:      {r2:.4f}")
    print(f"    Mono:    {mono_viol:.6f}")
    print(f"    Slope:   {slope_err:.4f}")

    # --- Eigenvalue analysis ---
    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1]
    print(f"    Top |λ|: {eig_mags[:5]}")

    # --- Eigenvalue recovery (synthetic ODE) ---
    eig_recovery = None
    if hasattr(ds, "ode_true_K_eigenvalues"):
        true_K_eigs = ds.ode_true_K_eigenvalues
        eig_recovery = eigenvalue_recovery_error(final_eigs, true_K_eigs)
        print(f"    Eigenvalue recovery:")
        print(f"      Mean mag error:   {eig_recovery['mean_mag_error']:.6f}")
        print(f"      Max mag error:    {eig_recovery['max_mag_error']:.6f}")
        print(f"      Mean phase error: {eig_recovery['mean_phase_error']:.6f}")

    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)
    run_tag = f"{domain_name}_run{run_id}"

    # Save model weights
    model_path = os.path.join(output_dir, f"kepin_{run_tag}.weights.h5")
    model.save_weights(model_path)

    # Save predictions
    pred_path = os.path.join(output_dir, f"predictions_{run_tag}.npz")
    np.savez(pred_path, y_true=Y_test.flatten(), y_pred=Y_pred.flatten())

    # Save eigenvalues
    eig_path = os.path.join(output_dir, f"eigenvalues_{run_tag}.npz")
    np.savez(eig_path,
             eigenvalue_history=np.array(history["eigenvalues"]),
             final_eigenvalues=final_eigs,
             koopman_matrix=model.get_koopman_matrix())

    # Save training history
    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
    })
    hist_path = os.path.join(output_dir, f"history_{run_tag}.csv")
    hist_df.to_csv(hist_path, index=False)

    # --- Generate training convergence plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#0173B2")
    axes[0].plot(history["val_loss"], label="Val", color="#DE8F05")
    axes[0].set_title(f"Total Loss — {domain_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_rmse"], label="Train", color="#0173B2")
    axes[1].plot(history["val_rmse"], label="Val", color="#DE8F05")
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    eig_hist = np.array(history["eigenvalues"])
    for mode_idx in range(min(4, eig_hist.shape[1])):
        axes[2].plot(np.abs(eig_hist[:, mode_idx]),
                     label=f"Mode {mode_idx+1}", alpha=0.8)
    axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="|λ|=1")
    axes[2].set_title("Koopman |λ| Evolution")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"training_{run_tag}.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, f"training_{run_tag}.pdf"))
    plt.close()

    results = {
        "dataset": domain_name,
        "run_id": run_id,
        "rmse": test_rmse,
        "mae": test_mae,
        "r2": r2,
        "mono_violation": mono_viol,
        "slope_rmse": slope_err,
        "best_val_loss": float(min(history["val_loss"])),
        "epochs_trained": len(history["epoch"]),
        "eigenvalue_mags": eig_mags.tolist(),
        "eigenvalue_recovery": eig_recovery,
        "ema_alpha": ema_alpha,
        "arch_tier": arch_config["tier"],
        "arch_config": {k: v for k, v in arch_config.items()
                        if k not in ("kernels",)},
        "batch_size": batch_size,
        "lr": lr,
        "n_params": int(n_params),
        "y_test_mean": float(np.mean(Y_test)),
        "y_test_std": float(np.std(Y_test)),
        "y_test_range": [float(np.min(Y_test)), float(np.max(Y_test))],
    }

    return results


# =========================================================================
# Publication figure generation
# =========================================================================

def generate_multidomain_figures(results_dir: str, output_fig_dir: str):
    """Generate publication-quality figures for multi-domain results.

    Produces:
      1. Multi-domain convergence comparison
      2. Prediction scatter plots for each domain
      3. Eigenvalue spectrum comparison
      4. Multi-domain results bar chart
    """
    os.makedirs(output_fig_dir, exist_ok=True)

    domains = ["Jena_Climate", "SPY_Stock", "Synthetic_ODE"]
    domain_labels = {"Jena_Climate": "Jena Climate (Weather)",
                     "SPY_Stock": "SPY Stock (Finance)",
                     "Synthetic_ODE": "Synthetic ODE"}
    colors = {"Jena_Climate": "#0173B2", "SPY_Stock": "#DE8F05",
              "Synthetic_ODE": "#029E73"}

    # ---- 1. Training convergence comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, domain in enumerate(domains):
        hist_path = os.path.join(results_dir, domain,
                                 f"history_{domain}_run0.csv")
        if not os.path.exists(hist_path):
            continue
        hist = pd.read_csv(hist_path)
        axes[i].plot(hist["train_rmse"], label="Train", color="#0173B2",
                     alpha=0.8, linewidth=1.2)
        axes[i].plot(hist["val_rmse"], label="Val", color="#DE8F05",
                     alpha=0.8, linewidth=1.2)
        axes[i].set_title(domain_labels.get(domain, domain), fontsize=11)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("RMSE")
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)
    plt.suptitle("Training Convergence Across Domains", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "multidomain_convergence.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_fig_dir, "multidomain_convergence.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 2. Prediction scatter plots ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    units = {
        "Jena_Climate": "Temperature (°C)",
        "SPY_Stock": "Return (%)",
        "Synthetic_ODE": "RUL (steps)",
    }
    for i, domain in enumerate(domains):
        pred_path = os.path.join(results_dir, domain,
                                 f"predictions_{domain}_run0.npz")
        if not os.path.exists(pred_path):
            continue
        data = np.load(pred_path)
        y_true = data["y_true"].flatten()
        y_pred = data["y_pred"].flatten()

        # Subsample if too many points
        if len(y_true) > 3000:
            idx = np.random.choice(len(y_true), 3000, replace=False)
            y_true_plot, y_pred_plot = y_true[idx], y_pred[idx]
        else:
            y_true_plot, y_pred_plot = y_true, y_pred

        axes[i].scatter(y_true_plot, y_pred_plot, alpha=0.15, s=8,
                        color=colors[domain], edgecolors="none")
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()),
                max(y_true.max(), y_pred.max())]
        axes[i].plot(lims, lims, "r--", alpha=0.6, linewidth=1.5,
                     label="Perfect")
        axes[i].set_xlabel(f"True {units[domain]}")
        axes[i].set_ylabel(f"Predicted {units[domain]}")
        axes[i].set_title(domain_labels.get(domain, domain), fontsize=11)

        # Add R² annotation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-10
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        axes[i].text(0.05, 0.92, f"R² = {r2:.3f}\nRMSE = {rmse:.2f}",
                     transform=axes[i].transAxes, fontsize=9,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                               alpha=0.7))
        axes[i].legend(fontsize=9)
        axes[i].grid(alpha=0.3)

    plt.suptitle("Predicted vs. True Values Across Domains",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "multidomain_predictions.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_fig_dir, "multidomain_predictions.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 3. Eigenvalue spectrum comparison ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, domain in enumerate(domains):
        eig_path = os.path.join(results_dir, domain,
                                f"eigenvalues_{domain}_run0.npz")
        if not os.path.exists(eig_path):
            continue
        data = np.load(eig_path)
        eigs = data["final_eigenvalues"]

        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        axes[i].plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3,
                     linewidth=1)
        axes[i].scatter(np.real(eigs), np.imag(eigs), c=colors[domain],
                        s=40, alpha=0.7, edgecolors="black", linewidths=0.5,
                        zorder=5)

        # For ODE, also plot true eigenvalues
        if domain == "Synthetic_ODE":
            from scipy.linalg import expm
            A = np.array([[-0.05, 0.1], [-0.1, -0.03]])
            K_true = expm(A * 0.1)
            true_eigs = np.linalg.eigvals(K_true)
            axes[i].scatter(np.real(true_eigs), np.imag(true_eigs),
                            c="red", s=100, marker="x", linewidths=2,
                            zorder=10, label="True eigenvalues")

        axes[i].set_xlabel("Re(λ)")
        axes[i].set_ylabel("Im(λ)")
        axes[i].set_title(domain_labels.get(domain, domain), fontsize=11)
        axes[i].set_aspect("equal")
        axes[i].grid(alpha=0.3)
        if domain == "Synthetic_ODE":
            axes[i].legend(fontsize=9)

    plt.suptitle("Koopman Eigenvalue Spectra Across Domains",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "multidomain_eigenvalues.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_fig_dir, "multidomain_eigenvalues.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ---- 4. Multi-domain error distribution ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, domain in enumerate(domains):
        pred_path = os.path.join(results_dir, domain,
                                 f"predictions_{domain}_run0.npz")
        if not os.path.exists(pred_path):
            continue
        data = np.load(pred_path)
        errors = data["y_pred"].flatten() - data["y_true"].flatten()

        axes[i].hist(errors, bins=60, color=colors[domain], alpha=0.7,
                     edgecolor="black", linewidth=0.3, density=True)
        axes[i].axvline(x=0, color="red", linestyle="--", alpha=0.6)
        axes[i].set_xlabel("Prediction Error")
        axes[i].set_ylabel("Density")
        axes[i].set_title(domain_labels.get(domain, domain), fontsize=11)
        axes[i].text(0.05, 0.92,
                     f"μ = {np.mean(errors):.2f}\nσ = {np.std(errors):.2f}",
                     transform=axes[i].transAxes, fontsize=9,
                     verticalalignment="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                               alpha=0.7))
        axes[i].grid(alpha=0.3)

    plt.suptitle("Prediction Error Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_fig_dir, "multidomain_errors.pdf"),
                bbox_inches="tight")
    fig.savefig(os.path.join(output_fig_dir, "multidomain_errors.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n  Saved publication figures to {output_fig_dir}")


# =========================================================================
# Multi-domain ablation study
# =========================================================================

def run_ablation_single(domain_name: str, output_dir: str,
                        run_id: int = 0, verbose: int = 1):
    """Run ablation on a single domain with 5 configs.

    Configs:
      A: Baseline (no Koopman, MSE loss only)
      B: KePIN w/o spectral loss
      C: KePIN w/o multi-step loss
      D: KePIN w/o auto-weighting (fixed weights)
      E: KePIN Full (all components)
    """
    from kepin_ablation import (
        get_ablation_configs, run_single_ablation,
        generate_ablation_plots, generate_latex_table, generate_heatmap,
        print_summary,
    )

    cfg = DOMAIN_CONFIGS[domain_name]
    ds_config = {k: v for k, v in cfg.items()
                 if k not in ("epochs", "patience", "lr", "batch_size", "arch_override")}

    ab_configs = get_ablation_configs()
    os.makedirs(output_dir, exist_ok=True)

    epochs = cfg["epochs"]
    patience = cfg["patience"]
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]

    all_results = []
    for ab_config in ab_configs:
        print(f"\n  --- {ab_config['name']} on {domain_name} (run {run_id}) ---")
        try:
            result = run_single_ablation(
                ab_config, ds_config, output_dir,
                epochs=epochs, batch_size=batch_size,
                lr=lr, patience=patience,
                run_id=run_id, verbose=verbose,
            )
            all_results.append(result)
        except Exception as e:
            print(f"    FAILED: {ab_config['name']} on {domain_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "ablation": ab_config["name"],
                "ablation_tag": ab_config["tag"],
                "dataset": domain_name,
                "run_id": run_id,
                "error": str(e),
            })

    return all_results


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized Multi-Domain KePIN Training")
    parser.add_argument("--domain", type=str, default=None,
                        choices=list(DOMAIN_CONFIGS.keys()),
                        help="Run single domain (default: all)")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study instead of optimized training")
    parser.add_argument("--figures_only", action="store_true",
                        help="Generate figures from existing results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        output_base = args.output_dir
    else:
        prefix = "kepin_ablation_md" if args.ablation else "kepin_optimized_md"
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"{prefix}_{timestamp}")

    domains = [args.domain] if args.domain else list(DOMAIN_CONFIGS.keys())

    if args.figures_only:
        generate_multidomain_figures(output_base,
                                     os.path.join(_project_dir, "paper", "figures"))
        return

    os.makedirs(output_base, exist_ok=True)
    all_results = []

    for domain in domains:
        ds_dir = os.path.join(output_base, domain)

        if args.ablation:
            results = run_ablation_single(domain, ds_dir, verbose=args.verbose)
            all_results.extend(results)
        else:
            result = train_domain_optimized(domain, ds_dir, verbose=args.verbose)
            all_results.append(result)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    summary_rows = []
    for r in all_results:
        if "error" not in r:
            row = {
                "Dataset": r.get("dataset", "?"),
                "RMSE": r.get("rmse", "?"),
                "MAE": r.get("mae", "?"),
                "R²": r.get("r2", "?"),
                "Epochs": r.get("epochs_trained", "?"),
                "Tier": r.get("arch_tier", "?"),
            }
            if r.get("eigenvalue_recovery"):
                row["EigRecov"] = r["eigenvalue_recovery"]["mean_mag_error"]
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(output_base, "optimized_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        json_path = os.path.join(output_base, "optimized_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n  Saved results to {output_base}")

    # --- Generate publication figures ---
    if not args.ablation:
        try:
            generate_multidomain_figures(
                output_base,
                os.path.join(_project_dir, "paper", "figures"),
            )
        except Exception as e:
            print(f"  Warning: Figure generation failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
