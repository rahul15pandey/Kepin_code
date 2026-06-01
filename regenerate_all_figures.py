#!/usr/bin/env python3
"""Regenerate all 8 paper figures from the 3 experiment directories."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

FIGS = 'paper/figures'
MD1  = 'experiments_result/kepin_optimized_md_20260225_000516'
MD2  = 'experiments_result/kepin_optimized_md_20260224_151712'
CM   = 'experiments_result/kepin_optimized_cmapss_merged_20260225_090600'

def load_pred(path):
    d = np.load(path)
    return d['y_true'].flatten(), d['y_pred'].flatten()

def calc_metrics(yt, yp):
    rmse = np.sqrt(np.mean((yt - yp)**2))
    mae  = np.mean(np.abs(yt - yp))
    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - np.mean(yt))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return rmse, mae, r2

histories = [
    (f'{CM}/CMAPSS_FD001/history_CMAPSS_FD001_run0.csv', 'FD001'),
    (f'{CM}/CMAPSS_FD002/history_CMAPSS_FD002_run0.csv', 'FD002'),
    (f'{CM}/CMAPSS_FD003/history_CMAPSS_FD003_run0.csv', 'FD003'),
    (f'{CM}/CMAPSS_FD004/history_CMAPSS_FD004_run0.csv', 'FD004'),
    (f'{MD1}/Jena_Climate/history_Jena_Climate_run0.csv', 'Jena Climate'),
    (f'{MD1}/SPY_Stock/history_SPY_Stock_run0.csv', 'SPY Stock'),
    (f'{MD2}/Synthetic_ODE/history_Synthetic_ODE_run0.csv', 'Synth. ODE'),
]

eig_sources = [
    (f'{CM}/CMAPSS_FD001/eigenvalues_CMAPSS_FD001_run0.npz', 'FD001'),
    (f'{CM}/CMAPSS_FD002/eigenvalues_CMAPSS_FD002_run0.npz', 'FD002'),
    (f'{CM}/CMAPSS_FD003/eigenvalues_CMAPSS_FD003_run0.npz', 'FD003'),
    (f'{CM}/CMAPSS_FD004/eigenvalues_CMAPSS_FD004_run0.npz', 'FD004'),
    (f'{MD1}/Jena_Climate/eigenvalues_Jena_Climate_run0.npz', 'Jena Climate'),
    (f'{MD1}/SPY_Stock/eigenvalues_SPY_Stock_run0.npz', 'SPY Stock'),
    (f'{MD2}/Synthetic_ODE/eigenvalues_Synthetic_ODE_run0.npz', 'Synth. ODE'),
]

domain_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e',
                 '#9467bd', '#8c564b', '#e377c2']

# =====================================================================
# 1. MULTIDOMAIN PREDICTIONS (Jena + SPY + ODE) — 1×3 scatter
# =====================================================================
print("[1/8] multidomain_predictions.pdf")
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6))
md_domains = [
    (f'{MD1}/Jena_Climate/predictions_Jena_Climate_run0.npz',
     'Jena Climate (Weather)', 'Temperature (°C)', '#9467bd'),
    (f'{MD1}/SPY_Stock/predictions_SPY_Stock_run0.npz',
     'S&P 500 ETF (Finance)', 'Forward Return (%)', '#8c564b'),
    (f'{MD2}/Synthetic_ODE/predictions_Synthetic_ODE_run0.npz',
     'Synthetic ODE (Dynamics)', 'State Value', '#e377c2'),
]
for ax, (path, title, ylabel, color) in zip(axes, md_domains):
    yt, yp = load_pred(path)
    rmse, mae, r2 = calc_metrics(yt, yp)
    ax.scatter(yt, yp, alpha=0.15, s=4, c=color, edgecolors='none')
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    pad = (lims[1] - lims[0]) * 0.05
    ax.plot([lims[0] - pad, lims[1] + pad], [lims[0] - pad, lims[1] + pad],
            'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(lims[0] - pad, lims[1] + pad)
    ax.set_ylim(lims[0] - pad, lims[1] + pad)
    ax.set_xlabel(f'True {ylabel}')
    ax.set_ylabel(f'Predicted {ylabel}')
    ax.set_title(title, fontsize=10)
    ax.text(0.05, 0.95, f'$R^2$={r2:.3f}\nRMSE={rmse:.2f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, edgecolor='gray'))
    ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
fig.savefig(f'{FIGS}/multidomain_predictions.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/multidomain_predictions.pdf'), "bytes")

# =====================================================================
# 2. PREDICTIONS SCATTER (FD001-FD004) — 2×4 full width
# =====================================================================
print("[2/8] predictions_scatter.pdf")
fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
fd_colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']
for i, ds in enumerate(['FD001', 'FD002', 'FD003', 'FD004']):
    yt, yp = load_pred(f'{CM}/CMAPSS_{ds}/predictions_CMAPSS_{ds}_ensemble.npz')
    rmse, mae, r2 = calc_metrics(yt, yp)
    err = yp - yt
    ax = axes[0, i]
    ax.scatter(yt, yp, alpha=0.5, s=14, c=fd_colors[i], edgecolors='none')
    ax.plot([0, 130], [0, 130], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlim(-5, 135); ax.set_ylim(-5, 135)
    ax.set_title(ds, fontweight='bold')
    ax.set_xlabel('True RUL'); ax.set_ylabel('Predicted RUL')
    ax.text(0.05, 0.95, f'$R^2$={r2:.3f}\nRMSE={rmse:.2f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, edgecolor='gray'))
    ax2 = axes[1, i]
    ax2.hist(err, bins=30, color=fd_colors[i], alpha=0.75, edgecolor='white', lw=0.4)
    ax2.axvline(0, color='black', ls='--', lw=0.8)
    ax2.set_xlabel('Prediction Error'); ax2.set_ylabel('Count')
    mu, sigma = np.mean(err), np.std(err)
    ax2.text(0.05, 0.95, f'μ={mu:.1f}\nσ={sigma:.1f}',
             transform=ax2.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85, edgecolor='gray'))
plt.tight_layout()
fig.savefig(f'{FIGS}/predictions_scatter.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/predictions_scatter.pdf'), "bytes")

# =====================================================================
# 3. TRAINING CONVERGENCE — 2×4 (7 panels)
# =====================================================================
print("[3/8] training_convergence.pdf")
fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
for idx, (path, title) in enumerate(histories):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    df = pd.read_csv(path)
    c = domain_colors[idx]
    ax.plot(df['epoch'], df['train_rmse'], label='Train', lw=1.0, color=c, alpha=0.8)
    ax.plot(df['epoch'], df['val_rmse'], label='Val', lw=1.0, color=c, ls='--')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)
axes[1, 3].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS}/training_convergence.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/training_convergence.pdf'), "bytes")

# =====================================================================
# 4. LOSS COMPONENTS — 2×4 (7 panels)
# =====================================================================
print("[4/8] loss_components.pdf")
fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
lc_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
for idx, (path, title) in enumerate(histories):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    df = pd.read_csv(path)
    loss_cols = [c for c in df.columns if c.startswith('train_') and c not in ['train_loss', 'train_rmse']]
    for ci, col_name in enumerate(loss_cols):
        label = col_name.replace('train_', '').replace('_', ' ').title()
        vals = np.clip(df[col_name].values, 1e-12, None)
        ax.semilogy(df['epoch'], vals, lw=0.9, label=label,
                    color=lc_colors[ci % len(lc_colors)])
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log)')
    ax.legend(loc='upper right', fontsize=5.5, ncol=1)
    ax.grid(True, alpha=0.3, linewidth=0.5)
axes[1, 3].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS}/loss_components.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/loss_components.pdf'), "bytes")

# =====================================================================
# 5. EIGENVALUE SPECTRUM — complex plane, 2×4 (7 panels)
# =====================================================================
print("[5/8] eigenvalue_spectrum.pdf")
fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
for idx, (path, title) in enumerate(eig_sources):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    d = np.load(path)
    eigs = d['final_eigenvalues']
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=0.5, alpha=0.4)
    c = domain_colors[idx]
    ax.scatter(eigs.real, eigs.imag, s=18, alpha=0.7, c=c, edgecolors='k', lw=0.3, zorder=5)
    ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Real'); ax.set_ylabel('Imag')
    ax.set_aspect('equal')
    lim = max(1.15, np.max(np.abs(eigs)) * 1.1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
axes[1, 3].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS}/eigenvalue_spectrum.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/eigenvalue_spectrum.pdf'), "bytes")

# =====================================================================
# 6. EIGENVALUE CONVERGENCE — top-5 magnitude, 2×4 (7 panels)
# =====================================================================
print("[6/8] eigenvalue_convergence.pdf")
fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
eig_top_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
for idx, (path, title) in enumerate(eig_sources):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    d = np.load(path)
    hist = d['eigenvalue_history']
    mags = np.abs(hist)
    final_mags = mags[-1]
    top5_idx = np.argsort(final_mags)[-5:][::-1]
    for rank, ei in enumerate(top5_idx):
        ax.plot(mags[:, ei], lw=1.0, label=f'λ$_{rank+1}$', color=eig_top_colors[rank])
    ax.axhline(1.0, color='gray', ls=':', lw=0.5)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('Epoch'); ax.set_ylabel('|λ|')
    ax.legend(loc='lower right', fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3, linewidth=0.5)
axes[1, 3].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS}/eigenvalue_convergence.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/eigenvalue_convergence.pdf'), "bytes")

# =====================================================================
# 7. EIGENVALUE HISTOGRAM — magnitude distribution, 2×4 (7 panels)
# =====================================================================
print("[7/8] eigenvalue_histogram.pdf")
fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
for idx, (path, title) in enumerate(eig_sources):
    row, col = idx // 4, idx % 4
    ax = axes[row, col]
    d = np.load(path)
    eigs = d['final_eigenvalues']
    mags = np.abs(eigs)
    c = domain_colors[idx]
    ax.hist(mags, bins=20, color=c, alpha=0.75, edgecolor='white', lw=0.4)
    ax.axvline(1.0, color='red', ls='--', lw=1.2, alpha=0.8, label='|λ|=1')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xlabel('|λ|'); ax.set_ylabel('Count')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.5, axis='y')
axes[1, 3].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIGS}/eigenvalue_histogram.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/eigenvalue_histogram.pdf'), "bytes")

# =====================================================================
# 8. RESULTS BAR — all 7 datasets
# =====================================================================
print("[8/8] results_bar.pdf")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))
labels = ['FD001', 'FD002', 'FD003', 'FD004', 'Jena', 'SPY', 'ODE']
rmses  = [12.92, 15.08, 11.42, 17.04, 5.98, 2.99, 55.93]
maes   = [9.52, 11.51, 8.38, 13.10, 4.82, 2.38, 42.04]
x = np.arange(len(labels))

ax1.bar(x, rmses, color=domain_colors, alpha=0.85, edgecolor='white', lw=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30, ha='right')
ax1.set_ylabel('RMSE'); ax1.set_title('RMSE Across All Domains', fontweight='bold')
for i, v in enumerate(rmses):
    ax1.text(i, v + max(rmses) * 0.02, f'{v:.1f}', ha='center', fontsize=7, fontweight='bold')
ax1.grid(True, alpha=0.2, axis='y')

ax2.bar(x, maes, color=domain_colors, alpha=0.85, edgecolor='white', lw=0.5)
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30, ha='right')
ax2.set_ylabel('MAE'); ax2.set_title('MAE Across All Domains', fontweight='bold')
for i, v in enumerate(maes):
    ax2.text(i, v + max(maes) * 0.02, f'{v:.1f}', ha='center', fontsize=7, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
fig.savefig(f'{FIGS}/results_bar.pdf', bbox_inches='tight')
plt.close()
print("  OK", os.path.getsize(f'{FIGS}/results_bar.pdf'), "bytes")

print("\nAll 8 figures regenerated successfully!")
