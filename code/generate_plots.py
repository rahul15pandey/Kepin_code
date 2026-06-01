#!/usr/bin/env python3
"""Generate publication-quality plots for KePIN research paper."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

# ---------- Configuration ----------
EXP_DIR = "../experiments_result/kepin_20260224_002102"
OUT_DIR = "../paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = ["CMAPSS_FD001", "CMAPSS_FD002", "CMAPSS_FD003", "CMAPSS_FD004"]
LABELS   = ["FD001", "FD002", "FD003", "FD004"]
COLORS   = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ======================================================================
# 1. Training Convergence (RMSE) — all 4 datasets in one figure
# ======================================================================
def plot_training_convergence():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.flatten()
    for i, (ds, label, color) in enumerate(zip(DATASETS, LABELS, COLORS)):
        hist_path = os.path.join(EXP_DIR, ds, f"history_{ds}_run0.csv")
        df = pd.read_csv(hist_path)
        ax = axes[i]
        ax.plot(df['epoch'], df['train_rmse'], color=color, alpha=0.8, label='Train RMSE', linewidth=1.5)
        ax.plot(df['epoch'], df['val_rmse'], color=color, linestyle='--', alpha=0.8, label='Val RMSE', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title(f'{label}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    plt.suptitle('Training Convergence — KePIN Model', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_convergence.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "training_convergence.png"))
    plt.close()
    print("[OK] training_convergence")


# ======================================================================
# 2. Loss component evolution — stacked area or multi-line
# ======================================================================
def plot_loss_components():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, (ds, label, color) in enumerate(zip(DATASETS, LABELS, COLORS)):
        hist_path = os.path.join(EXP_DIR, ds, f"history_{ds}_run0.csv")
        df = pd.read_csv(hist_path)
        ax = axes[i]
        ax.plot(df['epoch'], df['train_loss'], label='Total Loss', linewidth=1.5, color='black')
        ax.plot(df['epoch'], df['train_rul_mse'], label='RUL MSE', linewidth=1.2, alpha=0.8)
        ax.plot(df['epoch'], df['train_spectral'], label='Spectral', linewidth=1.2, alpha=0.8)
        ax.plot(df['epoch'], df['train_multi_step'], label='Multi-step', linewidth=1.2, alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{label}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.suptitle('Loss Component Evolution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_components.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "loss_components.png"))
    plt.close()
    print("[OK] loss_components")


# ======================================================================
# 3. Eigenvalue spectrum on unit circle
# ======================================================================
def plot_eigenvalue_spectrum():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i, (ds, label, color) in enumerate(zip(DATASETS, LABELS, COLORS)):
        eig_path = os.path.join(EXP_DIR, ds, f"eigenvalues_{ds}_run0.npz")
        data = np.load(eig_path)
        eigs = data['final_eigenvalues']
        ax = axes[i]
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8, alpha=0.4)
        # Eigenvalues
        ax.scatter(eigs.real, eigs.imag, c=color, s=40, zorder=5, edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'{label} — Koopman Eigenvalues')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.axvline(0, color='grey', linewidth=0.5)
        lim = max(1.2, np.max(np.abs(eigs))*1.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    plt.suptitle('Koopman Operator Eigenvalue Spectrum', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eigenvalue_spectrum.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "eigenvalue_spectrum.png"))
    plt.close()
    print("[OK] eigenvalue_spectrum")


# ======================================================================
# 4. Eigenvalue magnitude convergence over training
# ======================================================================
def plot_eigenvalue_convergence():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, (ds, label, color) in enumerate(zip(DATASETS, LABELS, COLORS)):
        eig_path = os.path.join(EXP_DIR, ds, f"eigenvalues_{ds}_run0.npz")
        data = np.load(eig_path)
        eig_hist = data['eigenvalue_history']  # (epochs, latent_dim) complex
        mags = np.abs(eig_hist)  # magnitudes
        ax = axes[i]
        epochs = np.arange(mags.shape[0])
        # Plot top-5 eigenvalue magnitudes
        # Sort by final magnitude (descending)
        final_mags = mags[-1]
        top_idx = np.argsort(final_mags)[::-1][:5]
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
        for j, idx in enumerate(top_idx):
            ax.plot(epochs, mags[:, idx], color=cmap[j], linewidth=1.2,
                    label=f'$\\lambda_{{{j+1}}}$', alpha=0.85)
        # Mean + std band for all
        mean_mag = mags.mean(axis=1)
        std_mag = mags.std(axis=1)
        ax.fill_between(epochs, mean_mag-std_mag, mean_mag+std_mag,
                        alpha=0.15, color='grey')
        ax.plot(epochs, mean_mag, 'k--', linewidth=1, alpha=0.5, label='Mean')
        ax.axhline(1.0, color='red', linewidth=0.8, linestyle=':', alpha=0.6, label='Unit circle')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('$|\\lambda|$')
        ax.set_title(f'{label}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Eigenvalue Magnitude Convergence', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eigenvalue_convergence.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "eigenvalue_convergence.png"))
    plt.close()
    print("[OK] eigenvalue_convergence")


# ======================================================================
# 5. Prediction vs True RUL scatter + error histogram
# ======================================================================
def plot_predictions():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for i, (ds, label, color) in enumerate(zip(DATASETS, LABELS, COLORS)):
        pred_path = os.path.join(EXP_DIR, ds, f"predictions_{ds}_run0.npz")
        data = np.load(pred_path)
        y_true = data['y_true'].flatten()
        y_pred = data['y_pred'].flatten()
        error = y_pred - y_true

        # Scatter
        ax = axes[0, i]
        ax.scatter(y_true, y_pred, c=color, s=15, alpha=0.6, edgecolors='none')
        lim = max(y_true.max(), y_pred.max()) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5)
        ax.set_xlabel('True RUL')
        ax.set_ylabel('Predicted RUL')
        ax.set_title(f'{label}')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Error histogram
        ax2 = axes[1, i]
        ax2.hist(error, bins=25, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axvline(0, color='black', linewidth=1, linestyle='--')
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{label} — Error Distribution')
        ax2.grid(True, alpha=0.3)

    plt.suptitle('RUL Predictions vs Ground Truth', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "predictions_scatter.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "predictions_scatter.png"))
    plt.close()
    print("[OK] predictions_scatter")


# ======================================================================
# 6. Combined bar chart of RMSE and MAE
# ======================================================================
def plot_results_bar():
    import json
    with open(os.path.join(EXP_DIR, "kepin_results.json")) as f:
        results = json.load(f)
    rmse_vals = [r['rmse'] for r in results]
    mae_vals  = [r['mae'] for r in results]

    x = np.arange(len(LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='#2196F3', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, mae_vals, width, label='MAE', color='#FF9800', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Error')
    ax.set_title('KePIN Performance on C-MAPSS Datasets', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "results_bar.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "results_bar.png"))
    plt.close()
    print("[OK] results_bar")


# ======================================================================
# 7. Architecture diagram (text-based schematic)
# ======================================================================
def plot_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define blocks
    blocks = [
        (1.0, 3.0, 2.0, 1.5, 'Input\n$\\mathbf{X} \\in \\mathbb{R}^{T \\times d}$', '#E3F2FD'),
        (3.5, 3.0, 2.0, 1.5, 'Conv1D + SE\nEncoder\n$f_\\theta(\\cdot)$', '#BBDEFB'),
        (6.0, 3.0, 2.2, 1.5, 'Koopman\nOperator\n$\\mathbf{K} = \\mathbf{U}\\Sigma\\mathbf{V}^\\top$', '#90CAF9'),
        (8.7, 3.0, 2.0, 1.5, 'Spectral\nFeatures\n$\\phi(\\lambda_i)$', '#64B5F6'),
        (11.2, 3.0, 2.0, 1.5, 'RUL Head\n$\\hat{y} = g_\\psi(\\cdot)$', '#42A5F5'),
    ]

    for x, y, w, h, text, color in blocks:
        rect = plt.Rectangle((x, y-h/2), w, h, linewidth=1.5, edgecolor='black',
                             facecolor=color, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y, text, ha='center', va='center', fontsize=9,
                fontweight='bold', zorder=3)

    # Arrows
    arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
    for x_start, x_end in [(3.0, 3.5), (5.5, 6.0), (8.2, 8.7), (10.7, 11.2)]:
        ax.annotate('', xy=(x_end, 3.0), xytext=(x_start, 3.0),
                    arrowprops=arrow_props)

    # Physics losses below
    loss_y = 0.8
    losses = [
        (2.0, 'RUL MSE\n$\\mathcal{L}_{\\text{rul}}$'),
        (4.0, 'Monotonicity\n$\\mathcal{L}_{\\text{mono}}$'),
        (6.0, 'Spectral\n$\\mathcal{L}_{\\text{spec}}$'),
        (8.0, 'Multi-step\n$\\mathcal{L}_{\\text{multi}}$'),
        (10.0, 'Koopman\n$\\mathcal{L}_{\\text{koop}}$'),
        (12.0, 'Asymmetric\n$\\mathcal{L}_{\\text{asym}}$'),
    ]
    for lx, ltext in losses:
        ax.text(lx, loss_y, ltext, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4', edgecolor='black', linewidth=0.8))

    # Bracket
    ax.annotate('', xy=(1.2, 1.7), xytext=(12.8, 1.7),
                arrowprops=dict(arrowstyle='-', lw=1, color='grey'))
    ax.text(7.0, 1.85, 'Physics-Informed Composite Loss $\\mathcal{L} = \\sum_i w_i \\mathcal{L}_i$ (Kendall Weighting)',
            ha='center', va='bottom', fontsize=10, style='italic', color='#333')

    ax.set_title('KePIN Architecture Overview', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "architecture.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "architecture.png"))
    plt.close()
    print("[OK] architecture")


# ======================================================================
# 8. Eigenvalue magnitude histogram for all datasets
# ======================================================================
def plot_eigenvalue_histogram():
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (ds, label, color) in enumerate(zip(DATASETS, LABELS, COLORS)):
        eig_path = os.path.join(EXP_DIR, ds, f"eigenvalues_{ds}_run0.npz")
        data = np.load(eig_path)
        mags = np.abs(data['final_eigenvalues'])
        ax.hist(mags, bins=20, alpha=0.5, color=color, label=label, edgecolor='black', linewidth=0.5)
    ax.axvline(1.0, color='red', linewidth=1.5, linestyle='--', label='Unit circle')
    ax.set_xlabel('Eigenvalue Magnitude $|\\lambda|$')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Koopman Eigenvalue Magnitudes', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "eigenvalue_histogram.pdf"))
    plt.savefig(os.path.join(OUT_DIR, "eigenvalue_histogram.png"))
    plt.close()
    print("[OK] eigenvalue_histogram")


# ======================================================================
# Main
# ======================================================================
if __name__ == "__main__":
    print(f"Generating plots from {EXP_DIR} → {OUT_DIR}")
    plot_training_convergence()
    plot_loss_components()
    plot_eigenvalue_spectrum()
    plot_eigenvalue_convergence()
    plot_predictions()
    plot_results_bar()
    plot_architecture()
    plot_eigenvalue_histogram()
    print("\nAll plots generated successfully!")
