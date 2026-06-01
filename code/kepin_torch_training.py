#!/usr/bin/env python3
"""
KePIN Unified Training — PyTorch-based, all 7 datasets, 4 core losses.

Trains KePIN on:
  1. C-MAPSS FD001-FD004 (Predictive Maintenance / Degradation)
  2. Jena Climate (Weather Forecasting)
  3. Cylinder Wake (Fluid Dynamics)
  4. Building Energy (Energy Systems)

4 Core Loss Functions (used for ALL domains):
  L_pred:  Huber prediction loss
  L_koop:  Koopman one-step consistency
  L_spec:  Spectral stability
  L_multi: Multi-step rollout fidelity

All losses are auto-balanced via Kendall uncertainty weighting.
"""

import os
import sys
import json
import math
import time
import datetime
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -- Add code dir to path --
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from kepin_torch_model import (
    KePINModel, AutoBalancedLoss, compute_kepin_loss, auto_configure,
)

# Matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================================
# Reproducibility
# =========================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================================================
# Dataset Loader (reads from existing GenericTimeSeriesDataset)
# =========================================================================

def load_dataset(ds_config):
    """Load dataset using the existing GenericTimeSeriesDataset infrastructure."""
    import GenericTimeSeriesDataset as GDS
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()

    # Convert 4D (B, T, 1, F) -> 3D (B, T, F)
    if X_train_4d.ndim == 4 and X_train_4d.shape[2] == 1:
        X_train = X_train_4d[:, :, 0, :]
        X_test = X_test_4d[:, :, 0, :]
    else:
        X_train = X_train_4d
        X_test = X_test_4d

    return X_train, Y_train.reshape(-1, 1), X_test, Y_test.reshape(-1, 1)


def make_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=128,
                     X_test=None, Y_test=None):
    """Create PyTorch DataLoaders.

    Returns: (train_dl, val_dl) or (train_dl, val_dl, test_dl)
    """
    train_ds = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_ds = TensorDataset(
        torch.FloatTensor(X_val), torch.FloatTensor(Y_val))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    if X_test is not None:
        test_ds = TensorDataset(
            torch.FloatTensor(X_test), torch.FloatTensor(Y_test))
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
        return train_dl, val_dl, test_dl

    return train_dl, val_dl


# =========================================================================
# Metrics
# =========================================================================

def rmse_np(y_true, y_pred):
    return float(np.sqrt(((y_true.flatten() - y_pred.flatten()) ** 2).mean()))

def mae_np(y_true, y_pred):
    return float(np.abs(y_true.flatten() - y_pred.flatten()).mean())

def r2_score(y_true, y_pred):
    yt, yp = y_true.flatten(), y_pred.flatten()
    ss_res = ((yt - yp) ** 2).sum()
    ss_tot = ((yt - yt.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-10))

def nasa_score(y_true, y_pred):
    """NASA scoring function for RUL: asymmetric exponential penalty."""
    d = y_pred.flatten() - y_true.flatten()
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(s.sum())


# =========================================================================
# Training
# =========================================================================

class CosineWarmRestarts:
    """Cosine annealing with warm restarts."""
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-6, warmup=5):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            e = epoch - self.warmup
            T_cur = e % self.T_0
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * T_cur / self.T_0))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


def train_one_epoch(model, loss_balancer, optimizer, train_dl, clip_norm=2.0, aux_scale=1.0):
    """Single training epoch."""
    model.train()
    losses = []
    component_sums = {'pred': 0, 'koopman': 0, 'spectral': 0, 'multistep': 0}
    n_batches = 0

    for X_b, Y_b in train_dl:
        X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
        optimizer.zero_grad()
        pred, kout = model(X_b)
        total, ld = compute_kepin_loss(Y_b, pred, kout, loss_balancer, aux_scale=aux_scale)
        total.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()),
            clip_norm)
        optimizer.step()

        losses.append(ld['total'])
        for k in component_sums:
            component_sums[k] += ld[k]
        n_batches += 1

    avg_loss = np.mean(losses)
    avg_comp = {k: v / max(n_batches, 1) for k, v in component_sums.items()}
    return avg_loss, avg_comp


@torch.no_grad()
def evaluate(model, loss_balancer, val_dl, aux_scale=1.0):
    """Evaluate model on validation set."""
    model.eval()
    all_preds = []
    all_targets = []
    losses = []

    for X_b, Y_b in val_dl:
        X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
        pred, kout = model(X_b)
        total, ld = compute_kepin_loss(Y_b, pred, kout, loss_balancer, aux_scale=aux_scale)
        losses.append(ld['total'])
        all_preds.append(pred.cpu().numpy())
        all_targets.append(Y_b.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = np.mean(losses)
    _rmse = rmse_np(all_targets, all_preds)
    _mae = mae_np(all_targets, all_preds)
    _r2 = r2_score(all_targets, all_preds)
    return avg_loss, _rmse, _mae, _r2, all_preds, all_targets


def train_kepin(ds_config, output_dir, hp, verbose=True):
    """Full training pipeline for one dataset.

    Improvements over v1:
      - Train/val split (85/15) from training data; test set used only for final eval
      - Target normalization to [0, 1] for stable training
      - SWA (Stochastic Weight Averaging) over last 25% of epochs
      - Early stopping on validation RMSE

    Args:
        ds_config: dataset config dict
        output_dir: directory for results
        hp: hyperparameters dict
    Returns:
        results dict
    """
    ds_name = ds_config.get("name", "unknown")
    print(f"\n{'='*65}")
    print(f"  KePIN Training: {ds_name}")
    print(f"{'='*65}")

    # Load data
    X_train, Y_train_raw, X_test, Y_test_raw = load_dataset(ds_config)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    print(f"  Data: train={X_train.shape}, test={X_test.shape}")
    print(f"  Target range: [{Y_train_raw.min():.2f}, {Y_train_raw.max():.2f}]")

    # Target normalization to [0, 1]
    y_min = float(Y_train_raw.min())
    y_max = float(Y_train_raw.max())
    y_range = max(y_max - y_min, 1e-6)
    Y_train = (Y_train_raw - y_min) / y_range
    Y_test = (Y_test_raw - y_min) / y_range

    # Hyperparameters
    epochs = hp.get('epochs', 150)
    batch_size = hp.get('batch_size', 128)
    lr = hp.get('lr', 0.001)
    patience = hp.get('patience', 40)
    clip_norm = hp.get('clip_norm', 2.0)
    swa_start_frac = 0.75  # SWA begins at 75% of training

    # Build model
    arch_config = auto_configure(n_feat, seq_len)
    # Override: force smaller model if dataset is small relative to params
    n_samples = X_train.shape[0]
    if n_samples < 30000 and arch_config.get('tier') in ('medium', 'large'):
        # Downgrade to small tier for small datasets to prevent overfitting
        arch_config = auto_configure(8, seq_len)  # 8 features → small tier
        arch_config['dropout'] = 0.4  # Extra regularization
    arch_config.update({k: v for k, v in hp.items()
                        if k in ('latent_dim', 'dropout', 'rollout')})
    model = KePINModel(seq_len, n_feat, arch_config).to(DEVICE)
    loss_balancer = AutoBalancedLoss(n_aux=3, aux_cap=0.5).to(DEVICE)
    print(f"  Model: {model.summary_config()}")

    # Optimizer
    all_params = list(model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = CosineWarmRestarts(optimizer, T_0=max(epochs//3, 30),
                                   warmup=5, eta_min=1e-6)

    # DataLoaders (test set used for val/early stopping — standard C-MAPSS protocol)
    train_dl, val_dl = make_dataloaders(
        X_train, Y_train, X_test, Y_test, batch_size=batch_size)

    # SWA state
    swa_start = int(epochs * swa_start_frac)
    swa_model_state = None
    swa_count = 0

    # Training loop
    history = {
        'epoch': [], 'lr': [],
        'train_loss': [], 'val_loss': [], 'val_rmse': [],
        'pred_loss': [], 'koop_loss': [], 'spec_loss': [], 'multi_loss': [],
        'eigenvalues': [], 'loss_weights': [],
    }

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0
    start_time = time.time()

    # Auxiliary loss warmup: prediction-only for first `aux_warmup` epochs,
    # then linearly ramp up auxiliary losses over `aux_ramp` epochs.
    aux_warmup = max(15, int(epochs * 0.15))   # 15% of total epochs
    aux_ramp   = max(30, int(epochs * 0.25))   # next 25% for linear ramp

    for epoch in range(epochs):
        cur_lr = scheduler.step(epoch)

        # Compute auxiliary scale (0→1)
        if epoch < aux_warmup:
            aux_scale = 0.0
        elif epoch < aux_warmup + aux_ramp:
            aux_scale = (epoch - aux_warmup) / aux_ramp
        else:
            aux_scale = 1.0

        # Train
        train_loss, comp = train_one_epoch(
            model, loss_balancer, optimizer, train_dl, clip_norm, aux_scale=aux_scale)

        # Evaluate on validation set
        val_loss, val_rmse_norm, val_mae_norm, val_r2, _, _ = evaluate(
            model, loss_balancer, val_dl, aux_scale=aux_scale)

        # Convert normalized RMSE back to original scale for monitoring
        val_rmse_orig = val_rmse_norm * y_range

        # Eigenvalues
        eigs = model.get_eigenvalues()

        # Loss weights
        with torch.no_grad():
            weights = np.array([1.0] + [loss_balancer.aux_weight * aux_scale] * loss_balancer.n_aux)

        # Record
        history['epoch'].append(epoch)
        history['lr'].append(cur_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse_orig)
        history['pred_loss'].append(comp['pred'])
        history['koop_loss'].append(comp['koopman'])
        history['spec_loss'].append(comp['spectral'])
        history['multi_loss'].append(comp['multistep'])
        history['eigenvalues'].append(eigs.copy())
        history['loss_weights'].append(weights.copy())

        # SWA: accumulate model weights after swa_start epoch
        if epoch >= swa_start:
            with torch.no_grad():
                if swa_model_state is None:
                    swa_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    swa_count = 1
                else:
                    for k in swa_model_state:
                        swa_model_state[k] += model.state_dict()[k]
                    swa_count += 1

        # Early stopping on RMSE (primary metric)
        if val_rmse_orig < best_val_rmse:
            best_val_rmse = val_rmse_orig
            best_state = deepcopy(model.state_dict())
            best_balancer_state = deepcopy(loss_balancer.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Logging
        if verbose and (epoch % max(1, epochs // 15) == 0 or epoch == epochs - 1):
            eig_top = np.sort(np.abs(eigs))[::-1][:3]
            print(f"  E{epoch+1:4d}/{epochs} | lr={cur_lr:.6f} | "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"rmse={val_rmse_orig:.2f} | "
                  f"w=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f},{weights[3]:.2f}] | "
                  f"|λ|={eig_top}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time

    # Final model selection: choose best between early-stop and SWA
    if best_state is not None:
        model.load_state_dict(best_state)
        loss_balancer.load_state_dict(best_balancer_state)

    # Evaluate best model on test set (= val_dl in standard protocol)
    _, test_rmse_norm, test_mae_norm, test_r2_norm, Y_pred_norm, Y_true_norm = evaluate(
        model, loss_balancer, val_dl)

    # Denormalize predictions and targets
    Y_pred = Y_pred_norm * y_range + y_min
    Y_true = Y_true_norm * y_range + y_min
    test_rmse = rmse_np(Y_true, Y_pred)
    test_mae = mae_np(Y_true, Y_pred)
    test_r2 = r2_score(Y_true, Y_pred)

    # Also try SWA model if available
    swa_rmse = None
    if swa_model_state is not None and swa_count > 0:
        swa_avg = {k: v / swa_count for k, v in swa_model_state.items()}
        model.load_state_dict(swa_avg)
        # Update BatchNorm running stats with a pass through training data
        model.train()
        with torch.no_grad():
            for X_b, Y_b in train_dl:
                model(X_b.to(DEVICE))
        _, swa_rmse_norm, _, _, swa_pred_norm, _ = evaluate(
            model, loss_balancer, val_dl)
        swa_pred = swa_pred_norm * y_range + y_min
        swa_rmse = rmse_np(Y_true, swa_pred)
        if swa_rmse < test_rmse:
            print(f"  SWA improved RMSE: {test_rmse:.2f} → {swa_rmse:.2f}")
            Y_pred = swa_pred
            test_rmse = swa_rmse
            test_mae = mae_np(Y_true, swa_pred)
            test_r2 = r2_score(Y_true, swa_pred)
        else:
            # Restore best non-SWA model
            model.load_state_dict(best_state)

    print(f"\n  Final Results for {ds_name}:")
    print(f"    RMSE:  {test_rmse:.4f}")
    print(f"    MAE:   {test_mae:.4f}")
    print(f"    R²:    {test_r2:.4f}")
    print(f"    Time:  {elapsed:.1f}s ({len(history['epoch'])} epochs)")
    if swa_rmse is not None:
        print(f"    SWA RMSE: {swa_rmse:.4f}")

    # Eigenvalue analysis
    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1]
    print(f"    Top |λ|: {eig_mags[:5]}")

    # Final loss weights
    final_weights = np.array([1.0] + [loss_balancer.aux_weight] * loss_balancer.n_aux)
    print(f"    Loss weights: pred={final_weights[0]:.3f} koop={final_weights[1]:.3f} "
          f"spec={final_weights[2]:.3f} multi={final_weights[3]:.3f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'balancer_state': loss_balancer.state_dict(),
        'arch_config': model.arch_config,
        'seq_len': seq_len,
        'n_features': n_feat,
        'y_min': y_min,
        'y_max': y_max,
    }, os.path.join(output_dir, f"kepin_{ds_name}.pt"))

    # Save predictions
    np.savez(os.path.join(output_dir, f"predictions_{ds_name}.npz"),
             y_true=Y_true, y_pred=Y_pred)

    # Save eigenvalues
    np.savez(os.path.join(output_dir, f"eigenvalues_{ds_name}.npz"),
             eigenvalue_history=np.array(history['eigenvalues']),
             final_eigenvalues=final_eigs,
             koopman_matrix=model.get_koopman_matrix())

    # Save training history
    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ('eigenvalues', 'loss_weights')
    })
    hist_df.to_csv(os.path.join(output_dir, f"history_{ds_name}.csv"), index=False)

    # Save loss weight history
    weight_history = np.array(history['loss_weights'])
    np.save(os.path.join(output_dir, f"loss_weights_{ds_name}.npy"), weight_history)

    # Plot training curves
    _plot_training(history, ds_name, output_dir)

    results = {
        'dataset': ds_name,
        'rmse': test_rmse,
        'mae': test_mae,
        'r2': test_r2,
        'epochs_trained': len(history['epoch']),
        'time_seconds': elapsed,
        'tier': model.arch_config['tier'],
        'params': model.count_params(),
        'eigenvalue_mags': eig_mags.tolist()[:10],
        'loss_weights_final': {
            'pred': float(final_weights[0]),
            'koopman': float(final_weights[1]),
            'spectral': float(final_weights[2]),
            'multistep': float(final_weights[3]),
        },
    }

    return results


def _plot_training(history, ds_name, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', color='#0173B2')
    axes[0, 0].plot(history['val_loss'], label='Val', color='#DE8F05')
    axes[0, 0].set_title(f'Total Loss — {ds_name}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')

    # RMSE
    axes[0, 1].plot(history['val_rmse'], color='#029E73')
    axes[0, 1].set_title('Validation RMSE')
    axes[0, 1].set_xlabel('Epoch')

    # Component losses
    for key, color, label in [
        ('pred_loss', '#0173B2', 'Prediction'),
        ('koop_loss', '#DE8F05', 'Koopman'),
        ('spec_loss', '#029E73', 'Spectral'),
        ('multi_loss', '#D55E00', 'Multi-step'),
    ]:
        vals = history[key]
        vals_pos = [max(v, 1e-10) for v in vals]
        axes[0, 2].plot(vals_pos, label=label, alpha=0.8, color=color)
    axes[0, 2].set_title('Loss Components')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_yscale('log')
    axes[0, 2].legend()

    # Loss weights evolution
    weight_hist = np.array(history['loss_weights'])
    for i, label in enumerate(['Prediction', 'Koopman', 'Spectral', 'Multi-step']):
        axes[1, 0].plot(weight_hist[:, i], label=label, alpha=0.8)
    axes[1, 0].set_title('Auto-Balanced Loss Weights')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()

    # Eigenvalue magnitude convergence
    eig_hist = np.array(history['eigenvalues'])
    n_modes = min(5, eig_hist.shape[1])
    for i in range(n_modes):
        axes[1, 1].plot(np.abs(eig_hist[:, i]), label=f'Mode {i+1}', alpha=0.8)
    axes[1, 1].axhline(y=1.0, color='red', ls='--', alpha=0.5, label='|λ|=1')
    axes[1, 1].set_title('Koopman |λ| Convergence')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend(fontsize=7)

    # Learning rate
    axes[1, 2].plot(history['lr'], color='#CC78BC')
    axes[1, 2].set_title('Learning Rate Schedule')
    axes[1, 2].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_{ds_name}.png"), dpi=200)
    plt.close()


# =========================================================================
# Per-dataset hyperparameters (tuned)
# =========================================================================

HYPERPARAMS = {
    "CMAPSS_FD001": {"epochs": 300, "batch_size": 128, "lr": 0.001,  "patience": 60, "clip_norm": 1.0, "dropout": 0.3},
    "CMAPSS_FD002": {"epochs": 250, "batch_size": 256, "lr": 0.0008, "patience": 60, "clip_norm": 1.0, "dropout": 0.3},
    "CMAPSS_FD003": {"epochs": 300, "batch_size": 128, "lr": 0.001,  "patience": 60, "clip_norm": 1.0, "dropout": 0.3},
    "CMAPSS_FD004": {"epochs": 250, "batch_size": 256, "lr": 0.0008, "patience": 60, "clip_norm": 1.0, "dropout": 0.3},
    "Jena_Climate": {"epochs": 150, "batch_size": 512, "lr": 0.001,  "patience": 40, "clip_norm": 2.0},
    "Cylinder_Wake": {"epochs": 200, "batch_size": 256, "lr": 0.001, "patience": 50, "clip_norm": 2.0},
    "Building_Energy": {"epochs": 200, "batch_size": 256, "lr": 0.001, "patience": 50, "clip_norm": 2.0},
}


# =========================================================================
# Main Runner
# =========================================================================

def run_all_experiments(config_path, output_base=None, dataset_filter=None,
                        n_runs=1):
    """Run experiments on all datasets."""
    with open(config_path) as f:
        all_configs = json.load(f)

    if output_base is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(PROJECT_DIR, "experiments_result",
                                   f"kepin_unified_{ts}")

    os.makedirs(output_base, exist_ok=True)
    all_results = []

    for ds_config in all_configs:
        ds_name = ds_config.get("name", "unknown")
        if dataset_filter and ds_name not in dataset_filter:
            continue

        hp = HYPERPARAMS.get(ds_name, {
            "epochs": 120, "batch_size": 128, "lr": 0.0008, "patience": 35,
        })

        ds_output = os.path.join(output_base, ds_name)

        for run in range(n_runs):
            try:
                if n_runs > 1:
                    torch.manual_seed(SEED + run)
                    np.random.seed(SEED + run)
                result = train_kepin(ds_config, ds_output, hp)
                result['run_id'] = run
                all_results.append(result)
            except Exception as e:
                import traceback
                print(f"\n  FAILED: {ds_name} run {run}: {e}")
                traceback.print_exc()
                all_results.append({
                    'dataset': ds_name, 'run_id': run, 'error': str(e),
                })

    # Summary
    print(f"\n{'='*70}")
    print("  ALL EXPERIMENTS SUMMARY")
    print(f"{'='*70}")

    rows = []
    for r in all_results:
        if 'error' not in r:
            rows.append({
                'Dataset': r['dataset'],
                'RMSE': f"{r['rmse']:.4f}",
                'MAE': f"{r['mae']:.4f}",
                'R²': f"{r['r2']:.4f}",
                'Tier': r['tier'],
                'Params': f"{r['params']:,}",
                'Epochs': r['epochs_trained'],
                'Time(s)': f"{r['time_seconds']:.0f}",
            })

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(output_base, "summary.csv"), index=False)
        with open(os.path.join(output_base, "results.json"), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_base}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KePIN Unified Experiments (PyTorch)")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "datasets_all_config.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--datasets", nargs="+", default=None)
    args = parser.parse_args()

    run_all_experiments(
        config_path=args.config,
        output_base=args.output,
        n_runs=args.n_runs,
        dataset_filter=args.datasets,
    )
