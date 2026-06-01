#!/usr/bin/env python3
"""
Generate all publication-quality figures for the KePIN paper.
Reads experiment results from experiments_result/ and saves to paper/figures/.

Figures generated:
  1. architecture.pdf          - KePIN architecture diagram
  2. eigenvalue_spectrum.pdf   - Koopman eigenvalues in complex plane (6 datasets)
  3. predictions_scatter.pdf   - Predicted vs true for C-MAPSS FD001-FD004
  4. loss_components.pdf       - Training loss component evolution
  5. training_convergence.pdf  - RMSE convergence curves for all datasets
  6. predictions_new.pdf       - Predictions for Cylinder Wake & Building Energy
  7. results_bar.pdf           - Cross-domain RMSE/R² bar chart
  8. ablation_radar.pdf        - Ablation study radar chart (FD004)
  9. eigenvalue_convergence.pdf - Top-5 eigenvalue magnitude convergence
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, 'experiments_result')
FIG_DIR = os.path.join(BASE, 'paper', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Data sources — using the best available runs
CMAPSS_DIR = os.path.join(RESULTS, 'kepin_20260223_231044')  # original 3-run ensemble
KEPIN_FINAL = os.path.join(RESULTS, 'kepin_final')           # PyTorch single-run
IMPROVED = os.path.join(RESULTS, 'kepin_improved_data')       # Cylinder Wake, Building Energy

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'FD001': '#2196F3', 'FD002': '#FF5722', 'FD003': '#4CAF50', 'FD004': '#9C27B0',
    'Jena': '#FF9800', 'Cylinder': '#00BCD4', 'Building': '#E91E63',
    'pred': '#1565C0', 'koop': '#E65100', 'spec': '#2E7D32', 'multi': '#6A1B9A',
}


def load_predictions(base_dir, dataset, run_suffix=''):
    """Load predictions .npz."""
    fname = f'predictions_{dataset}{run_suffix}.npz'
    path = os.path.join(base_dir, dataset, fname)
    if not os.path.exists(path):
        path = os.path.join(base_dir, fname)
    d = np.load(path)
    return d['y_true'].flatten(), d['y_pred'].flatten()


def load_eigenvalues(base_dir, dataset, run_suffix=''):
    """Load eigenvalues .npz."""
    fname = f'eigenvalues_{dataset}{run_suffix}.npz'
    path = os.path.join(base_dir, dataset, fname)
    if not os.path.exists(path):
        path = os.path.join(base_dir, fname)
    d = np.load(path)
    hist = d['eigenvalue_history'] if 'eigenvalue_history' in d else None
    return d['final_eigenvalues'], hist


def load_history(base_dir, dataset, run_suffix=''):
    """Load training history CSV."""
    fname = f'history_{dataset}{run_suffix}.csv'
    path = os.path.join(base_dir, dataset, fname)
    if not os.path.exists(path):
        path = os.path.join(base_dir, fname)
    return pd.read_csv(path)


# ======================================================================
# Figure 1: Architecture Diagram
# ======================================================================
def fig_architecture():
    print("  [1/9] architecture.pdf")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    blocks = [
        (0.7,  1.75, 1.0, 1.8, 'Input\n($T \\times d$)',           '#E3F2FD'),
        (2.3,  1.75, 1.2, 1.8, 'ResConv1D\n+ SE\nAttention',       '#BBDEFB'),
        (4.0,  1.75, 1.0, 1.8, 'BiLSTM',                           '#90CAF9'),
        (5.5,  1.75, 1.0, 1.8, 'Multi-Head\nAttention',            '#64B5F6'),
        (7.2,  1.75, 1.2, 1.8, 'Koopman\nOperator\n$\\mathbf{K}$', '#42A5F5'),
        (9.0,  1.75, 1.0, 1.8, 'Prediction\nHead\n$\\hat{y}$',     '#1E88E5'),
    ]

    for x, y, w, h, label, color in blocks:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=color, edgecolor='#0D47A1', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=9,
                fontweight='bold', color='#0D47A1')

    arrow_kw = dict(arrowstyle='Simple,tail_width=3,head_width=10,head_length=6',
                    color='#0D47A1', mutation_scale=1)
    for i in range(len(blocks) - 1):
        x1 = blocks[i][0] + blocks[i][2] / 2
        x2 = blocks[i + 1][0] - blocks[i + 1][2] / 2
        ax.add_patch(FancyArrowPatch((x1, 1.75), (x2, 1.75), **arrow_kw))

    ax.annotate('Eigenvalue\nFeatures', xy=(7.2, 0.7), fontsize=8,
                ha='center', color='#E65100', fontstyle='italic')
    ax.annotate('', xy=(8.4, 1.2), xytext=(7.2, 0.85),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.5))

    ax.text(5.0, 0.15,
            'Domain-Aware Composite Loss: 7 components (degradation) / 4 components (forecasting)',
            ha='center', va='center', fontsize=8.5, fontstyle='italic', color='#424242',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor='#F9A825', alpha=0.9))

    ax.text(7.2, 0.35, '$\\mathbf{K} = \\mathbf{U}\\Sigma\\mathbf{V}^\\top$',
            ha='center', va='center', fontsize=9, color='#E65100',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF3E0',
                      edgecolor='#E65100', alpha=0.8))

    fig.savefig(os.path.join(FIG_DIR, 'architecture.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 2: Eigenvalue Spectrum (6 datasets, complex plane)
# ======================================================================
def fig_eigenvalue_spectrum():
    print("  [2/9] eigenvalue_spectrum.pdf")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))

    datasets = [
        ('CMAPSS_FD001', CMAPSS_DIR, '_run0', 'C-MAPSS FD001', COLORS['FD001']),
        ('CMAPSS_FD002', CMAPSS_DIR, '_run0', 'C-MAPSS FD002', COLORS['FD002']),
        ('CMAPSS_FD004', CMAPSS_DIR, '_run0', 'C-MAPSS FD004', COLORS['FD004']),
        ('Jena_Climate',  KEPIN_FINAL, '', 'Jena Climate',   COLORS['Jena']),
        ('Cylinder_Wake', IMPROVED,    '', 'Cylinder Wake',  COLORS['Cylinder']),
        ('Building_Energy', IMPROVED,  '', 'Building Energy', COLORS['Building']),
    ]

    for idx, (ds, base, suffix, title, color) in enumerate(datasets):
        ax = axes.flat[idx]
        eigs, _ = load_eigenvalues(base, ds, suffix)

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, lw=1)

        real, imag = np.real(eigs), np.imag(eigs)
        mags = np.abs(eigs)

        sc = ax.scatter(real, imag, c=mags, cmap='viridis', s=25, alpha=0.8,
                        edgecolors='white', linewidth=0.3, vmin=0, vmax=1)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axhline(0, color='grey', lw=0.5, alpha=0.3)
        ax.axvline(0, color='grey', lw=0.5, alpha=0.3)
        ax.set_title(title, fontweight='bold', color=color)
        ax.set_xlabel('Re($\\lambda$)')
        ax.set_ylabel('Im($\\lambda$)')

        n_stable = int(np.sum(mags <= 1.0))
        ax.text(0.03, 0.97,
                f'$|\\lambda|_{{max}}$={mags.max():.3f}\n{n_stable}/{len(eigs)} stable',
                transform=ax.transAxes, fontsize=7.5, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    fig.tight_layout(h_pad=2, w_pad=1.5)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='$|\\lambda|$')
    fig.subplots_adjust(right=0.9)
    fig.savefig(os.path.join(FIG_DIR, 'eigenvalue_spectrum.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 3: C-MAPSS Predictions Scatter + Error Distribution
# ======================================================================
def fig_predictions_cmapss():
    print("  [3/9] predictions_scatter.pdf")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5))

    cmapss = [
        ('CMAPSS_FD001', '_run0', 'FD001', COLORS['FD001']),
        ('CMAPSS_FD002', '_run0', 'FD002', COLORS['FD002']),
        ('CMAPSS_FD003', '_run0', 'FD003', COLORS['FD003']),
        ('CMAPSS_FD004', '_run0', 'FD004', COLORS['FD004']),
    ]

    for i, (ds, suffix, label, color) in enumerate(cmapss):
        y_true, y_pred = load_predictions(CMAPSS_DIR, ds, suffix)

        # Top: scatter
        ax = axes[0, i]
        ax.scatter(y_true, y_pred, c=color, s=20, alpha=0.6,
                   edgecolors='white', linewidth=0.3)
        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.5, lw=1)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
        ax.set_title(label, fontweight='bold', color=color)

        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.05, 0.95, f'RMSE={rmse:.2f}\n$R^2$={r2:.2f}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        # Bottom: error histogram
        ax2 = axes[1, i]
        errors = y_pred - y_true
        ax2.hist(errors, bins=25, color=color, alpha=0.7, edgecolor='white', density=True)
        ax2.axvline(0, color='k', lw=1, ls='--', alpha=0.5)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Density')
        ax2.text(0.05, 0.95, f'$\\mu$={errors.mean():.1f}\n$\\sigma$={errors.std():.1f}',
                 transform=ax2.transAxes, fontsize=8, va='top',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'predictions_scatter.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 4: Loss Component Evolution (log scale)
# ======================================================================
def fig_loss_components():
    print("  [4/9] loss_components.pdf")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    info = [
        ('CMAPSS_FD001', KEPIN_FINAL, '', 'C-MAPSS FD001'),
        ('CMAPSS_FD002', KEPIN_FINAL, '', 'C-MAPSS FD002'),
        ('CMAPSS_FD004', KEPIN_FINAL, '', 'C-MAPSS FD004'),
        ('Jena_Climate',   KEPIN_FINAL, '', 'Jena Climate'),
        ('Cylinder_Wake',  IMPROVED,    '', 'Cylinder Wake'),
        ('Building_Energy', IMPROVED,   '', 'Building Energy'),
    ]

    loss_cols = {
        'pred_loss':  (COLORS['pred'],  'Prediction'),
        'koop_loss':  (COLORS['koop'],  'Koopman'),
        'spec_loss':  (COLORS['spec'],  'Spectral'),
        'multi_loss': (COLORS['multi'], 'Multi-step'),
    }

    for idx, (ds, base, suffix, title) in enumerate(info):
        ax = axes.flat[idx]
        h = load_history(base, ds, suffix)

        for col, (clr, lbl) in loss_cols.items():
            if col in h.columns:
                v = h[col].values.copy()
                v = np.where(v > 0, v, 1e-8)
                ax.semilogy(h['epoch'], v, label=lbl, color=clr, lw=1.5, alpha=0.85)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log)')
        ax.set_title(title, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'loss_components.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 5: Training Convergence (validation RMSE)
# ======================================================================
def fig_training_convergence():
    print("  [5/9] training_convergence.pdf")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    info = [
        ('CMAPSS_FD001', KEPIN_FINAL, '', 'C-MAPSS FD001', COLORS['FD001']),
        ('CMAPSS_FD002', KEPIN_FINAL, '', 'C-MAPSS FD002', COLORS['FD002']),
        ('CMAPSS_FD004', KEPIN_FINAL, '', 'C-MAPSS FD004', COLORS['FD004']),
        ('Jena_Climate',   KEPIN_FINAL, '', 'Jena Climate',   COLORS['Jena']),
        ('Cylinder_Wake',  IMPROVED,    '', 'Cylinder Wake',  COLORS['Cylinder']),
        ('Building_Energy', IMPROVED,   '', 'Building Energy', COLORS['Building']),
    ]

    for idx, (ds, base, suffix, title, color) in enumerate(info):
        ax = axes.flat[idx]
        h = load_history(base, ds, suffix)

        ax.plot(h['epoch'], h['train_loss'], color=color, lw=1.5, alpha=0.8,
                label='Train loss')
        ax.plot(h['epoch'], h['val_loss'], color=color, lw=1.5, ls='--', alpha=0.8,
                label='Val loss')

        if 'val_rmse' in h.columns:
            ax2 = ax.twinx()
            ax2.plot(h['epoch'], h['val_rmse'], color='#757575', lw=1.2, ls=':',
                     label='Val RMSE')
            ax2.set_ylabel('Val RMSE', color='#757575', fontsize=8)
            ax2.tick_params(axis='y', labelcolor='#757575', labelsize=8)
            ax2.spines['right'].set_visible(True)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title, fontweight='bold', color=color)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'training_convergence.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 6: New-Domain Predictions (Cylinder Wake & Building Energy)
# ======================================================================
def fig_predictions_new():
    print("  [6/9] predictions_new.pdf")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for col, (ds, label, color) in enumerate([
        ('Cylinder_Wake',  'Cylinder Wake',  COLORS['Cylinder']),
        ('Building_Energy', 'Building Energy', COLORS['Building']),
    ]):
        y_t, y_p = load_predictions(IMPROVED, ds)
        rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
        ss = np.sum((y_t - y_p) ** 2)
        st = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - ss / st if st > 0 else 0

        # Top: scatter
        ax = axes[0, col]
        ax.scatter(y_t, y_p, c=color, s=5, alpha=0.3, rasterized=True)
        lo = min(y_t.min(), y_p.min())
        hi = max(y_t.max(), y_p.max())
        m = (hi - lo) * 0.05
        ax.plot([lo - m, hi + m], [lo - m, hi + m], 'k--', alpha=0.5, lw=1)
        ax.set_xlim(lo - m, hi + m); ax.set_ylim(lo - m, hi + m)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted')
        ax.set_title(f'{label} — Scatter', fontweight='bold', color=color)
        ax.text(0.05, 0.95,
                f'RMSE={rmse:.4f}\n$R^2$={r2:.2f}\nn={len(y_t):,}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        # Bottom: time-series snippet
        ax2 = axes[1, col]
        n = min(500, len(y_t))
        ax2.plot(range(n), y_t[:n], color='#424242', lw=1, alpha=0.7, label='True')
        ax2.plot(range(n), y_p[:n], color=color, lw=1, alpha=0.8, label='Predicted')
        ax2.set_xlabel('Sample Index'); ax2.set_ylabel('Value')
        ax2.set_title(f'{label} — Time Series', fontweight='bold', color=color)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'predictions_new.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 7: Cross-Domain Results Bar Chart
# ======================================================================
def fig_results_bar():
    print("  [7/9] results_bar.pdf")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: RMSE (C-MAPSS + Jena)
    ds_left  = ['FD001', 'FD002', 'FD003', 'FD004', 'Jena\nClimate']
    rmse_left = [12.92, 15.08, 11.42, 17.04, 5.98]
    c_left   = [COLORS['FD001'], COLORS['FD002'], COLORS['FD003'],
                COLORS['FD004'], COLORS['Jena']]

    bars = ax1.bar(ds_left, rmse_left, color=c_left, edgecolor='white', width=0.6)
    for b, v in zip(bars, rmse_left):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax1.set_ylabel('RMSE')
    ax1.set_title('C-MAPSS & Weather RMSE', fontweight='bold')
    ax1.set_ylim(0, max(rmse_left) * 1.15)

    # Right: R²
    ds_r2  = ['FD001', 'FD002', 'FD003', 'FD004', 'Cyl.\nWake', 'Building\nEnergy']
    r2_vals = [0.85, 0.82, 0.86, 0.85, 0.90, 0.96]
    c_r2   = [COLORS['FD001'], COLORS['FD002'], COLORS['FD003'],
              COLORS['FD004'], COLORS['Cylinder'], COLORS['Building']]

    bars2 = ax2.bar(ds_r2, r2_vals, color=c_r2, edgecolor='white', width=0.6)
    for b, v in zip(bars2, r2_vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                 f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    ax2.set_ylabel('$R^2$')
    ax2.set_title('Coefficient of Determination ($R^2$)', fontweight='bold')
    ax2.set_ylim(0.7, 1.02)
    ax2.axhline(0.9, color='grey', ls=':', alpha=0.5, lw=1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'results_bar.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 8: Ablation Study Radar (FD004)
# ======================================================================
def fig_ablation_radar():
    print("  [8/9] ablation_radar.pdf")

    configs = ['A: Base FCN', 'B: w/o Spec.', 'C: w/o Multi',
               'D: w/o Auto-Wt', 'E: Full KePIN']
    metrics = ['RMSE$\\downarrow$', 'MAE$\\downarrow$', '$R^2\\uparrow$']

    raw = {
        'RMSE': [17.22, 17.47, 17.01, 18.97, 16.83],
        'MAE':  [12.78, 13.06, 12.36, 13.65, 11.85],
        'R2':   [0.84,  0.83,  0.84,  0.80,  0.85],
    }

    def norm(vals, higher=False):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        return [((v - mn) / (mx - mn)) if higher else ((mx - v) / (mx - mn))
                for v in vals]

    data = np.array([norm(raw['RMSE']), norm(raw['MAE']),
                     norm(raw['R2'], True)]).T

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ccols = ['#9E9E9E', '#FF9800', '#4CAF50', '#F44336', '#2196F3']

    fig, ax = plt.subplots(figsize=(6, 5.5), subplot_kw=dict(polar=True))
    for i, (cfg, cc) in enumerate(zip(configs, ccols)):
        v = data[i].tolist() + [data[i][0]]
        lw = 2.5 if i == 4 else 1.5
        ax.plot(angles, v, 'o-', lw=lw, label=cfg, color=cc, markersize=4)
        ax.fill(angles, v, alpha=0.08, color=cc)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '0.5', '', '1.0'], fontsize=8)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=8.5)
    ax.set_title('Ablation — C-MAPSS FD004\n(Normalized, 1.0 = best)',
                 fontweight='bold', pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ablation_radar.pdf'))
    plt.close(fig)


# ======================================================================
# Figure 9: Eigenvalue Convergence (top-5 magnitudes over training)
# ======================================================================
def fig_eigenvalue_convergence():
    print("  [9/9] eigenvalue_convergence.pdf")
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))

    info = [
        ('CMAPSS_FD001', CMAPSS_DIR, '_run0', 'C-MAPSS FD001', COLORS['FD001']),
        ('CMAPSS_FD002', CMAPSS_DIR, '_run0', 'C-MAPSS FD002', COLORS['FD002']),
        ('CMAPSS_FD004', CMAPSS_DIR, '_run0', 'C-MAPSS FD004', COLORS['FD004']),
        ('Jena_Climate',   KEPIN_FINAL, '', 'Jena Climate',   COLORS['Jena']),
        ('Cylinder_Wake',  IMPROVED,    '', 'Cylinder Wake',  COLORS['Cylinder']),
        ('Building_Energy', IMPROVED,   '', 'Building Energy', COLORS['Building']),
    ]

    for idx, (ds, base, suffix, title, color) in enumerate(info):
        ax = axes.flat[idx]
        _, eig_hist = load_eigenvalues(base, ds, suffix)

        if eig_hist is not None and eig_hist.ndim == 2:
            mags = np.abs(eig_hist)
            top5 = np.argsort(mags[-1])[-5:][::-1]
            cm = plt.cm.get_cmap('tab10')
            for rank, ei in enumerate(top5):
                ax.plot(range(len(mags)), mags[:, ei], color=cm(rank), lw=1.5,
                        alpha=0.8, label=f'$\\lambda_{{{rank+1}}}$: {mags[-1, ei]:.3f}')
            ax.axhline(1.0, color='red', ls='--', alpha=0.3, lw=1)
            ax.legend(fontsize=6.5, loc='lower right')
        else:
            ax.text(0.5, 0.5, 'No history', transform=ax.transAxes, ha='center')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('$|\\lambda|$')
        ax.set_title(title, fontweight='bold', color=color)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'eigenvalue_convergence.pdf'))
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================
def _parse_args():
    p = argparse.ArgumentParser(description="Generate KePIN paper figures")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Base experiments_result directory")
    p.add_argument("--fig_dir", type=str, default=None,
                   help="Output directory for figures")
    p.add_argument("--cmapss_dir", type=str, default=None,
                   help="C-MAPSS results directory (predictions/eigenvalues)")
    p.add_argument("--kepin_final_dir", type=str, default=None,
                   help="Jena Climate results directory")
    p.add_argument("--improved_dir", type=str, default=None,
                   help="Cylinder Wake/Building Energy results directory")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    RESULTS = args.results_dir or RESULTS
    FIG_DIR = args.fig_dir or FIG_DIR
    CMAPSS_DIR = args.cmapss_dir or os.path.join(RESULTS, 'kepin_20260223_231044')
    KEPIN_FINAL = args.kepin_final_dir or os.path.join(RESULTS, 'kepin_final')
    IMPROVED = args.improved_dir or os.path.join(RESULTS, 'kepin_improved_data')

    os.makedirs(FIG_DIR, exist_ok=True)

    print(f"Output dir: {FIG_DIR}")
    print("=" * 60)

    fig_architecture()
    fig_eigenvalue_spectrum()
    fig_predictions_cmapss()
    fig_loss_components()
    fig_training_convergence()
    fig_predictions_new()
    fig_results_bar()
    fig_ablation_radar()
    fig_eigenvalue_convergence()

    print("=" * 60)
    for f in sorted(os.listdir(FIG_DIR)):
        sz = os.path.getsize(os.path.join(FIG_DIR, f)) / 1024
        print(f"  {f:35s} {sz:7.1f} KB")
    print(f"\nDone — {len(os.listdir(FIG_DIR))} figures saved.")
