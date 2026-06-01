# -*- coding: utf-8 -*-
"""
KePIN Analysis — Koopman-specific interpretability and visualization tools.

Provides novel analysis capabilities unique to the Koopman framework:
  1. Eigenvalue trajectory plot (dynamics discovery convergence)
  2. Koopman mode decomposition (dominant degradation modes)
  3. Multi-step rollout accuracy (linear dynamics validity horizon)
  4. Cross-domain spectral comparison (domain-specific vs universal patterns)
  5. Synthetic ground-truth eigenvalue recovery validation
  6. Latent state trajectory visualization
  7. Loss weight evolution (auto-balancing dynamics)

Usage:
  python kepin_analysis.py --results_dir experiments_result/kepin_YYYYMMDD

Or import and call individual analysis functions programmatically.
"""

import argparse
import glob
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tensorflow as tf

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
MARKERS = ["o", "s", "^", "D", "v", "P"]


# =========================================================================
# 1. Eigenvalue Trajectory Plot
# =========================================================================

def plot_eigenvalue_trajectory(eig_history: np.ndarray,
                               true_eigs: np.ndarray = None,
                               title: str = "Koopman Eigenvalue Discovery",
                               save_path: str = None):
    """Plot how learned eigenvalues evolve during training.

    Shows convergence of the Koopman operator to the true dynamics.

    Args:
        eig_history: (n_epochs, d) complex array
        true_eigs:   (d_true,) complex array — ground-truth eigenvalues
        title:       plot title
        save_path:   path to save figure
    """
    n_epochs, d = eig_history.shape

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Eigenvalue magnitudes over training ---
    ax = axes[0]
    for mode in range(min(d, 8)):
        mags = np.abs(eig_history[:, mode])
        ax.plot(mags, label=f"Mode {mode+1}", alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="|λ|=1 (stability)")
    if true_eigs is not None:
        for i, te in enumerate(true_eigs):
            ax.axhline(y=abs(te), color="green", linestyle=":",
                       alpha=0.4, label=f"True |λ_{i+1}|" if i < 3 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalue Magnitude Convergence")
    ax.legend(fontsize=7, ncol=2)

    # --- Panel 2: Complex plane trajectory ---
    ax = axes[1]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit circle")

    # Plot final eigenvalues as large markers
    final_eigs = eig_history[-1]
    ax.scatter(final_eigs.real, final_eigs.imag, s=100, c=COLORS[0],
               zorder=5, label="Learned (final)", edgecolors="black")

    # Plot trajectory for top modes
    for mode in range(min(d, 4)):
        traj = eig_history[:, mode]
        ax.plot(traj.real, traj.imag, alpha=0.3, color=COLORS[mode % len(COLORS)])
        # Starting point
        ax.scatter(traj[0].real, traj[0].imag, s=30, marker="x",
                   color=COLORS[mode % len(COLORS)])

    if true_eigs is not None:
        ax.scatter(true_eigs.real, true_eigs.imag, s=120, c="green",
                   marker="*", zorder=6, label="True eigenvalues", edgecolors="black")

    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Complex Plane — Eigenvalue Trajectory")
    ax.set_aspect("equal")
    ax.legend(fontsize=7)

    # --- Panel 3: Decay rate and frequency evolution ---
    ax = axes[2]
    top_k = min(d, 4)
    for mode in range(top_k):
        log_eigs = np.log(eig_history[:, mode].astype(np.complex128) + 1e-10)
        decay = -log_eigs.real
        freq = np.abs(log_eigs.imag)
        ax.plot(decay, label=f"σ_{mode+1} (decay)", linestyle="-",
                color=COLORS[mode % len(COLORS)])
        ax.plot(freq, label=f"ω_{mode+1} (freq)", linestyle="--",
                color=COLORS[mode % len(COLORS)], alpha=0.6)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Decay Rates & Frequencies")
    ax.legend(fontsize=7, ncol=2)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# 2. Koopman Mode Decomposition
# =========================================================================

def plot_koopman_modes(K_matrix: np.ndarray,
                       feature_names: List[str] = None,
                       title: str = "Koopman Mode Decomposition",
                       save_path: str = None):
    """Visualise the Koopman operator structure and mode importance.

    Args:
        K_matrix: (d, d) learned Koopman operator matrix
        feature_names: names for latent dimensions
        title: plot title
        save_path: path to save figure
    """
    d = K_matrix.shape[0]
    eig_vals, eig_vecs = np.linalg.eig(K_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: K matrix heatmap ---
    ax = axes[0]
    im = ax.imshow(K_matrix, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(K_matrix).max(), vmax=np.abs(K_matrix).max())
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Koopman Matrix K")
    ax.set_xlabel("Latent dim j")
    ax.set_ylabel("Latent dim i")

    # --- Panel 2: Eigenvalue spectrum ---
    ax = axes[1]
    mags = np.abs(eig_vals)
    sorted_idx = np.argsort(mags)[::-1]
    colors = [COLORS[i % len(COLORS)] for i in range(d)]

    bars = ax.bar(range(d), mags[sorted_idx], color=colors, alpha=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mode index (sorted by magnitude)")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalue Spectrum")

    # --- Panel 3: Top eigenvector contributions ---
    ax = axes[2]
    top_modes = sorted_idx[:min(4, d)]
    x = np.arange(d)
    width = 0.2

    for i, mode_idx in enumerate(top_modes):
        vec = np.abs(eig_vecs[:, mode_idx])
        ax.bar(x + i * width, vec, width, label=f"Mode {i+1} (|λ|={mags[mode_idx]:.3f})",
               color=COLORS[i % len(COLORS)], alpha=0.8)

    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("|v_i|")
    ax.set_title("Top Eigenvector Magnitudes")
    ax.legend(fontsize=7)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# 3. Multi-step Rollout Accuracy
# =========================================================================

def plot_rollout_accuracy(model, X_test: np.ndarray,
                          max_horizon: int = 10,
                          title: str = "Multi-step Rollout Accuracy",
                          save_path: str = None):
    """Plot ||K^k·z(t) - z(t+k)|| vs horizon k.

    Shows how far ahead the linear dynamics model remains valid.

    Args:
        model:       KePINModel instance
        X_test:      (N, seq_len, n_feat) test data
        max_horizon: maximum rollout horizon to evaluate
        title:       plot title
        save_path:   path to save figure
    """
    # Get latent states
    Z = model.get_latent_states(tf.constant(X_test[:100]))  # (N, T, d)
    K = model.get_koopman_matrix()

    T = Z.shape[1]
    max_h = min(max_horizon, T - 1)

    # Compute rollout errors for each horizon
    horizons = list(range(1, max_h + 1))
    mean_errors = []
    std_errors = []

    K_power = np.eye(K.shape[0])
    for h in horizons:
        K_power = K_power @ K  # K^h
        errors = []
        for t in range(T - h):
            z_pred = Z[:, t, :] @ K_power.T    # (N, d)
            z_true = Z[:, t + h, :]             # (N, d)
            err = np.sqrt(np.mean((z_pred - z_true) ** 2, axis=-1))  # (N,)
            errors.extend(err.tolist())
        mean_errors.append(np.mean(errors))
        std_errors.append(np.std(errors))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.errorbar(horizons, mean_errors, yerr=std_errors,
                fmt="o-", color=COLORS[0], capsize=3, linewidth=2,
                label="Rollout error ± σ")
    ax.fill_between(horizons,
                    np.array(mean_errors) - np.array(std_errors),
                    np.array(mean_errors) + np.array(std_errors),
                    alpha=0.2, color=COLORS[0])

    ax.set_xlabel("Rollout Horizon k")
    ax.set_ylabel("Mean ||K^k·z(t) - z(t+k)||₂")
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()

    return horizons, mean_errors


# =========================================================================
# 4. Cross-Domain Spectral Comparison
# =========================================================================

def plot_cross_domain_spectra(results_dirs: Dict[str, str],
                               title: str = "Cross-Domain Eigenvalue Spectra",
                               save_path: str = None):
    """Overlay eigenvalue spectra from multiple domains.

    Shows which spectral patterns are universal vs domain-specific.

    Args:
        results_dirs: {dataset_name: path_to_eigenvalues.npz}
        title: plot title
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel 1: Complex plane ---
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3)

    for i, (name, npz_path) in enumerate(results_dirs.items()):
        data = np.load(npz_path, allow_pickle=True)
        eigs = data.get("final_eigenvalues", data.get("eigenvalues", None))
        if eigs is None:
            continue
        ax.scatter(eigs.real, eigs.imag, s=80,
                   c=COLORS[i % len(COLORS)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=name, alpha=0.8, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Eigenvalue Spectra (Complex Plane)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    # --- Panel 2: Magnitude spectra ---
    ax = axes[1]
    all_spectra = {}
    for i, (name, npz_path) in enumerate(results_dirs.items()):
        data = np.load(npz_path, allow_pickle=True)
        eigs = data.get("final_eigenvalues", data.get("eigenvalues", None))
        if eigs is None:
            continue
        mags = np.sort(np.abs(eigs))[::-1]
        ax.plot(range(len(mags)), mags,
                marker=MARKERS[i % len(MARKERS)],
                color=COLORS[i % len(COLORS)],
                label=name, linewidth=2, markersize=6)
        all_spectra[name] = mags

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalue Magnitude Spectra")
    ax.legend(fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# 5. Synthetic Eigenvalue Recovery Validation
# =========================================================================

def plot_eigenvalue_recovery(learned_eigs: np.ndarray,
                              true_eigs: np.ndarray,
                              eig_history: np.ndarray = None,
                              title: str = "Eigenvalue Recovery Validation",
                              save_path: str = None):
    """Detailed validation plot for synthetic ODE eigenvalue recovery.

    Args:
        learned_eigs: (d,) complex — final learned eigenvalues
        true_eigs:    (d_true,) complex — ground-truth eigenvalues
        eig_history:  (n_epochs, d) complex — training trajectory
        title: plot title
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Complex plane comparison ---
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.2)

    ax.scatter(learned_eigs.real, learned_eigs.imag, s=80,
               c=COLORS[0], marker="o", label="Learned", edgecolors="black", zorder=5)
    ax.scatter(true_eigs.real, true_eigs.imag, s=150,
               c="green", marker="*", label="True", edgecolors="black", zorder=6)

    # Draw arrows from learned to nearest true
    from scipy.optimize import linear_sum_assignment
    cost = np.abs(learned_eigs[:, None] - true_eigs[None, :])
    row_ind, col_ind = linear_sum_assignment(np.abs(cost))
    for r, c in zip(row_ind, col_ind):
        ax.annotate("", xy=(true_eigs[c].real, true_eigs[c].imag),
                     xytext=(learned_eigs[r].real, learned_eigs[r].imag),
                     arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5))

    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("Learned vs True Eigenvalues")
    ax.set_aspect("equal")
    ax.legend()

    # --- Panel 2: Magnitude comparison bar chart ---
    ax = axes[1]
    true_mags = np.sort(np.abs(true_eigs))[::-1]
    learned_mags = np.sort(np.abs(learned_eigs))[::-1]

    x = np.arange(min(len(true_mags), 8))
    width = 0.35
    ax.bar(x - width / 2, true_mags[:len(x)], width, label="True |λ|",
           color="green", alpha=0.7)
    ax.bar(x + width / 2, learned_mags[:len(x)], width, label="Learned |λ|",
           color=COLORS[0], alpha=0.7)
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("|λ|")
    ax.set_title("Eigenvalue Magnitude Comparison")
    ax.legend()

    # --- Panel 3: Recovery error per mode ---
    ax = axes[2]
    errors = []
    for r, c in zip(row_ind, col_ind):
        err = abs(abs(learned_eigs[r]) - abs(true_eigs[c])) / (abs(true_eigs[c]) + 1e-10)
        errors.append(err * 100)

    ax.bar(range(len(errors)), errors, color=COLORS[2], alpha=0.8)
    ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    ax.set_xlabel("Matched Mode Index")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Per-Mode Recovery Error")
    ax.legend()

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# 6. Latent State Trajectory Visualization
# =========================================================================

def plot_latent_trajectories(model, X_test: np.ndarray,
                              Y_test: np.ndarray,
                              n_samples: int = 5,
                              title: str = "Latent State Trajectories",
                              save_path: str = None):
    """Visualise latent state evolution coloured by RUL.

    Args:
        model:     KePINModel instance
        X_test:    (N, seq_len, n_feat)
        Y_test:    (N, 1)
        n_samples: number of test samples to show
    """
    Z = model.get_latent_states(tf.constant(X_test[:n_samples]))  # (n, T, d)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PCA-like: use first 2 latent dims
    ax = axes[0]
    for i in range(n_samples):
        z = Z[i]  # (T, d)
        rul = float(Y_test[i])
        color = plt.cm.RdYlGn(rul / max(Y_test.max(), 1))
        ax.plot(z[:, 0], z[:, 1], "-", alpha=0.6, color=color, linewidth=1.5)
        ax.scatter(z[0, 0], z[0, 1], marker="o", color=color, s=40, zorder=5)
        ax.scatter(z[-1, 0], z[-1, 1], marker="x", color=color, s=60, zorder=5)

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title("Latent Space Trajectories (colour = RUL)")

    # Latent norm over time
    ax = axes[1]
    for i in range(n_samples):
        z = Z[i]
        norms = np.linalg.norm(z, axis=1)
        rul = float(Y_test[i])
        color = plt.cm.RdYlGn(rul / max(Y_test.max(), 1))
        ax.plot(norms, alpha=0.7, color=color, linewidth=1.5,
                label=f"RUL={rul:.0f}" if i < 5 else None)

    ax.set_xlabel("Time step")
    ax.set_ylabel("||z(t)||₂")
    ax.set_title("Latent State Norm Evolution")
    ax.legend(fontsize=7)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# 7. Loss Weight Evolution
# =========================================================================

def plot_loss_weight_evolution(weight_history: list,
                                loss_names: list = None,
                                title: str = "Auto-Balanced Loss Weights",
                                save_path: str = None):
    """Plot how uncertainty-weighted loss coefficients evolve during training.

    Args:
        weight_history: list of (n_losses,) arrays, one per epoch
        loss_names: names for each loss component
    """
    if not weight_history or weight_history[0] is None:
        print("  No weight history available (auto-weights may be disabled).")
        return

    weights = np.array([w for w in weight_history if w is not None])
    n_losses = weights.shape[1]

    if loss_names is None:
        loss_names = ["RUL", "Koopman", "Spectral", "Mono",
                      "Multi-step", "Asym", "Slope"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(n_losses):
        ax.plot(weights[:, i], label=loss_names[i] if i < len(loss_names) else f"L_{i}",
                color=COLORS[i % len(COLORS)], linewidth=2, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective Weight exp(-s_i)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# 8. Prediction Scatter + Error Distribution
# =========================================================================

def plot_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                              dataset_name: str = "",
                              save_path: str = None):
    """Prediction scatter plot and error distribution.

    Args:
        y_true: (N, 1) ground truth RUL
        y_pred: (N, 1) predicted RUL
        dataset_name: for title
        save_path: path to save figure
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    errors = yp - yt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Scatter plot
    ax = axes[0]
    ax.scatter(yt, yp, s=20, alpha=0.5, color=COLORS[0])
    lim = max(yt.max(), yp.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect")
    ax.fill_between([0, lim], [0 - 10, lim - 10], [0 + 10, lim + 10],
                    alpha=0.1, color="green", label="±10 band")
    ax.set_xlabel("True RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"Predictions — {dataset_name}")
    ax.legend(fontsize=8)

    # Error histogram
    ax = axes[1]
    ax.hist(errors, bins=40, color=COLORS[2], alpha=0.7, edgecolor="black")
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prediction Error (pred - true)")
    ax.set_ylabel("Count")
    ax.set_title(f"Error Distribution (RMSE={np.sqrt(np.mean(errors**2)):.2f})")

    # Error by RUL range
    ax = axes[2]
    bins = [(0, 20), (20, 50), (50, 80), (80, 200)]
    bin_labels = []
    bin_errors = []
    for lo, hi in bins:
        mask = (yt >= lo) & (yt < hi)
        if mask.any():
            bin_labels.append(f"[{lo},{hi})")
            bin_errors.append(errors[mask])

    if bin_errors:
        ax.boxplot(bin_errors, labels=bin_labels)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("True RUL Range")
    ax.set_ylabel("Error")
    ax.set_title("Error by RUL Range")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =========================================================================
# Master analysis runner
# =========================================================================

def run_full_analysis(results_dir: str, output_dir: str = None):
    """Run all analysis on a trained KePIN experiment directory.

    Expects the directory structure from kepin_training.py:
      results_dir/
        <dataset_name>/
          eigenvalues_<name>_run0.npz
          predictions_<name>_run0.npz
          history_<name>_run0.csv
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  KePIN Analysis: {results_dir}")
    print(f"{'='*60}")

    # Discover datasets
    eig_files = {}
    pred_files = {}
    hist_files = {}

    for f in glob.glob(os.path.join(results_dir, "**", "*.npz"), recursive=True):
        fname = os.path.basename(f)
        if fname.startswith("eigenvalues_"):
            ds_name = fname.replace("eigenvalues_", "").replace(".npz", "")
            ds_name = ds_name.rsplit("_run", 1)[0]
            eig_files[ds_name] = f
        elif fname.startswith("predictions_"):
            ds_name = fname.replace("predictions_", "").replace(".npz", "")
            ds_name = ds_name.rsplit("_run", 1)[0]
            pred_files[ds_name] = f

    for f in glob.glob(os.path.join(results_dir, "**", "*.csv"), recursive=True):
        fname = os.path.basename(f)
        if fname.startswith("history_"):
            ds_name = fname.replace("history_", "").replace(".csv", "")
            ds_name = ds_name.rsplit("_run", 1)[0]
            hist_files[ds_name] = f

    print(f"  Found datasets: {list(eig_files.keys())}")

    # 1. Per-dataset analyses
    for ds_name in eig_files:
        print(f"\n  --- {ds_name} ---")
        ds_out = os.path.join(output_dir, ds_name)
        os.makedirs(ds_out, exist_ok=True)

        # Eigenvalue trajectory
        eig_data = np.load(eig_files[ds_name], allow_pickle=True)
        if "eigenvalue_history" in eig_data:
            eig_hist = eig_data["eigenvalue_history"]
            plot_eigenvalue_trajectory(
                eig_hist, title=f"Eigenvalue Discovery — {ds_name}",
                save_path=os.path.join(ds_out, "eigenvalue_trajectory.png"),
            )

        # Koopman mode decomposition
        if "koopman_matrix" in eig_data:
            K = eig_data["koopman_matrix"]
            plot_koopman_modes(
                K, title=f"Koopman Modes — {ds_name}",
                save_path=os.path.join(ds_out, "koopman_modes.png"),
            )

        # Prediction analysis
        if ds_name in pred_files:
            pred_data = np.load(pred_files[ds_name])
            plot_prediction_analysis(
                pred_data["y_true"], pred_data["y_pred"],
                dataset_name=ds_name,
                save_path=os.path.join(ds_out, "predictions.png"),
            )

    # 2. Cross-domain spectral comparison
    if len(eig_files) > 1:
        print(f"\n  --- Cross-Domain Analysis ---")
        plot_cross_domain_spectra(
            eig_files,
            title="Cross-Domain Koopman Spectra",
            save_path=os.path.join(output_dir, "cross_domain_spectra.png"),
        )

    print(f"\n  ✓ Analysis complete. Figures saved to {output_dir}")


# =========================================================================
# CLI
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KePIN Analysis Tools")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing KePIN training results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for analysis figures")
    args = parser.parse_args()

    run_full_analysis(args.results_dir, args.output_dir)
