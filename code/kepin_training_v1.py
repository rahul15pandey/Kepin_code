# -*- coding: utf-8 -*-
"""
KePIN Training Pipeline — Domain-Independent Koopman-Enhanced Prognostics.

Unified training script that:
  - Loads ANY dataset via JSON config (no C-MAPSS-specific code paths)
  - Auto-configures architecture from data dimensions
  - Uses custom training loop for Koopman-aware loss computation
  - Auto-balances loss weights via Kendall uncertainty weighting
  - Logs Koopman eigenvalues at each epoch for dynamics convergence analysis
  - Saves model, eigenvalue history, predictions, and metrics

Usage:
  # Train on all 4 domains from config
  python kepin_training.py --config datasets_kepin_config.json

  # Train on specific dataset from config (by index)
  python kepin_training.py --config datasets_kepin_config.json --dataset_idx 0

  # Quick test on synthetic ODE
  python kepin_training.py --mode synthetic_ode

  # Train on synthetic degradation
  python kepin_training.py --mode synthetic
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
from gpu_config import setup_gpu, build_tf_dataset, get_batch_size, get_learning_rate, is_mixed_precision_enabled

# ---------- GPU setup (A100 40 GB) ----------
setup_gpu(mixed_precision=False, xla=False, verbose=True)

# ---------- Reproducibility ----------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))


# =========================================================================
# Metrics (domain-independent)
# =========================================================================

def rmse_np(y_true, y_pred):
    return float(np.sqrt(((y_true.flatten() - y_pred.flatten()) ** 2).mean()))


def mae_np(y_true, y_pred):
    return float(np.abs(y_true.flatten() - y_pred.flatten()).mean())


def physics_metrics_np(y_true, y_pred):
    """Compute monotonicity violation and slope RMSE."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    td = yt[1:] - yt[:-1]
    pd_ = yp[1:] - yp[:-1]
    mask = (td < 0).astype(np.float32)
    denom = mask.sum() + 1e-8
    mono = float((np.maximum(pd_, 0) * mask).sum() / denom)
    slope = float(math.sqrt(((pd_ - td) ** 2 * mask).sum() / denom))
    return mono, slope


def eigenvalue_recovery_error(learned_eigs, true_eigs):
    """Compute eigenvalue recovery error (for synthetic ODE validation).

    Uses the Hungarian algorithm to match learned eigenvalues to true ones,
    then reports mean relative error of magnitudes and phase errors.
    """
    from scipy.optimize import linear_sum_assignment

    n_true = len(true_eigs)
    n_learned = len(learned_eigs)

    # Build cost matrix (magnitude distance)
    cost = np.zeros((n_true, n_learned))
    for i in range(n_true):
        for j in range(n_learned):
            cost[i, j] = abs(abs(true_eigs[i]) - abs(learned_eigs[j]))

    row_ind, col_ind = linear_sum_assignment(cost)

    mag_errors = []
    phase_errors = []
    for i, j in zip(row_ind, col_ind):
        t_mag = abs(true_eigs[i])
        l_mag = abs(learned_eigs[j])
        mag_errors.append(abs(t_mag - l_mag) / (t_mag + 1e-10))

        t_phase = np.angle(true_eigs[i])
        l_phase = np.angle(learned_eigs[j])
        phase_errors.append(abs(t_phase - l_phase))

    return {
        "mean_mag_error": float(np.mean(mag_errors)),
        "max_mag_error": float(np.max(mag_errors)),
        "mean_phase_error": float(np.mean(phase_errors)),
    }


# =========================================================================
# EMA Smoothing (auto-tuned alpha)
# =========================================================================

def apply_ema_smoothing(data_3d, alpha=None):
    """Apply exponential moving average along time axis.

    If alpha is None, auto-tune: α = 2 / (1 + avg_seq_len).
    """
    from scipy.signal import lfilter, lfilter_zi

    if alpha is None:
        seq_len = data_3d.shape[1]
        alpha = 2.0 / (1.0 + seq_len)
        alpha = max(0.05, min(alpha, 0.5))

    b = np.array([alpha], dtype=np.float64)
    a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    zi = lfilter_zi(b, a)

    smoothed = np.empty_like(data_3d)
    n_samples, n_time, n_feat = data_3d.shape
    for j in range(n_feat):
        x_2d = data_3d[:, :, j].astype(np.float64)
        zi_2d = zi * x_2d[:, 0:1]
        smoothed[:, :, j], _ = lfilter(b, a, x_2d, axis=1, zi=zi_2d)

    return smoothed.astype(np.float32), alpha


# =========================================================================
# Custom Training Step
# =========================================================================

class KePINTrainer:
    """Custom training loop that handles Koopman-aware loss computation.

    Keras model.fit() only passes (y_true, y_pred) to the loss function.
    KePIN needs the full Koopman outputs (eigenvalues, one-step predictions,
    multi-step rollouts) for the composite loss. This trainer handles that.

    Optimised for A100 40 GB:
      - Mixed precision (float16 compute, float32 loss)
      - tf.data pipeline with prefetch
      - XLA JIT compilation
    """

    def __init__(self, model: KePINModel, loss_fn, optimizer,
                 clip_norm: float = 1.0):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self._mixed = is_mixed_precision_enabled()

    @tf.function(jit_compile=False)  # XLA via global flag; jit_compile here can conflict with eigvals
    def train_step(self, X_batch, Y_batch):
        """One training step with full Koopman loss."""
        with tf.GradientTape() as tape:
            rul_pred, koopman_out = self.model(X_batch, training=True)
            # Always compute loss in float32 for numerical stability
            rul_pred_f32 = tf.cast(rul_pred, tf.float32)
            Y_batch_f32 = tf.cast(Y_batch, tf.float32)
            koopman_f32 = {k: tf.cast(v, tf.float32) if v.dtype != tf.complex64 else v
                           for k, v in koopman_out.items()
                           if isinstance(v, tf.Tensor)}
            total_loss, loss_dict = self.loss_fn(Y_batch_f32, rul_pred_f32, koopman_f32)
            # Scale loss for mixed precision gradient stability
            scaled_loss = total_loss
            if self._mixed:
                scaled_loss = self.optimizer.get_scaled_loss(total_loss) if hasattr(self.optimizer, 'get_scaled_loss') else total_loss

        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        if self._mixed and hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        # Gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )
        return total_loss, loss_dict, rul_pred_f32

    @tf.function(jit_compile=False)
    def eval_step(self, X_batch, Y_batch):
        """Evaluation step (no gradient computation)."""
        rul_pred, koopman_out = self.model(X_batch, training=False)
        rul_pred_f32 = tf.cast(rul_pred, tf.float32)
        Y_batch_f32 = tf.cast(Y_batch, tf.float32)
        koopman_f32 = {k: tf.cast(v, tf.float32) if v.dtype != tf.complex64 else v
                       for k, v in koopman_out.items()
                       if isinstance(v, tf.Tensor)}
        total_loss, loss_dict = self.loss_fn(Y_batch_f32, rul_pred_f32, koopman_f32)
        return total_loss, loss_dict, rul_pred_f32

    def fit(self, X_train, Y_train, X_val, Y_val,
            epochs=200, batch_size=512, patience=40,
            lr_schedule="cosine", initial_lr=0.003, min_lr=1e-5,
            verbose=1):
        """Full training loop with early stopping and LR scheduling.

        Uses tf.data pipeline for efficient A100 GPU feeding.

        Returns:
            history: dict with per-epoch metrics and eigenvalue snapshots
        """
        n_train = len(X_train)

        # Build tf.data pipelines for efficient GPU feeding
        train_ds = build_tf_dataset(X_train, Y_train, batch_size=batch_size,
                                    shuffle=True, seed=SEED)
        val_ds = build_tf_dataset(X_val, Y_val, batch_size=batch_size,
                                  shuffle=False)

        history = {
            "epoch": [],
            "train_loss": [], "val_loss": [],
            "train_rmse": [], "val_rmse": [],
            "eigenvalues": [],           # list of (d,) complex arrays
            "loss_weights": [],          # effective weights per epoch
            "lr": [],
            # Individual loss components
            "train_rul_mse": [], "train_koopman": [], "train_spectral": [],
            "train_mono": [], "train_multi_step": [],
        }

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            # --- Learning rate schedule ---
            if lr_schedule == "cosine":
                lr = min_lr + 0.5 * (initial_lr - min_lr) * (
                    1 + math.cos(math.pi * epoch / epochs)
                )
            else:
                lr = initial_lr
            self.optimizer.learning_rate.assign(lr)
            history["lr"].append(lr)

            # --- Training (tf.data pipeline — shuffled per epoch) ---
            epoch_losses = []
            epoch_preds = []
            epoch_component_losses = {"rul_mse": [], "koopman_1step": [],
                                      "spectral": [], "monotonicity": [],
                                      "multi_step": []}

            for X_b, Y_b in train_ds:
                total_loss, loss_dict, rul_pred = self.train_step(X_b, Y_b)
                epoch_losses.append(float(total_loss))
                epoch_preds.append(rul_pred.numpy())

                for key in epoch_component_losses:
                    if key in loss_dict:
                        epoch_component_losses[key].append(float(loss_dict[key]))

            train_loss = np.mean(epoch_losses)
            train_preds = np.concatenate(epoch_preds, axis=0)[:n_train]
            # Need Y_train ordering for RMSE — use first n samples
            train_rmse = rmse_np(Y_train[:len(train_preds)], train_preds)

            # --- Validation (tf.data pipeline — no shuffle) ---
            val_preds_list = []
            val_losses = []

            for X_vb, Y_vb in val_ds:
                v_loss, v_dict, v_pred = self.eval_step(X_vb, Y_vb)
                val_losses.append(float(v_loss))
                val_preds_list.append(v_pred.numpy())

            val_loss = np.mean(val_losses)
            val_preds = np.concatenate(val_preds_list, axis=0)
            val_rmse = rmse_np(Y_val[:len(val_preds)], val_preds)

            # --- Record eigenvalues ---
            eigs = self.model.get_eigenvalues()
            history["eigenvalues"].append(eigs.copy())

            # --- Record loss weights ---
            try:
                eff_weights = tf.exp(-self.model.loss_weight_layer.log_vars).numpy()
                history["loss_weights"].append(eff_weights.copy())
            except Exception:
                history["loss_weights"].append(None)

            # --- Record history ---
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)
            for key in epoch_component_losses:
                if f"train_{key}" in history:
                    history[f"train_{key}"].append(
                        np.mean(epoch_component_losses[key]) if epoch_component_losses[key] else 0.0
                    )

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1

            # --- Logging ---
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1):
                eig_mags = np.sort(np.abs(eigs))[::-1][:3]
                print(f"  Epoch {epoch+1:4d}/{epochs} | "
                      f"LR={lr:.6f} | "
                      f"Train: loss={train_loss:.4f} rmse={train_rmse:.2f} | "
                      f"Val: loss={val_loss:.4f} rmse={val_rmse:.2f} | "
                      f"|λ|={eig_mags}")

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

        # Restore best weights
        if best_weights is not None:
            self.model.set_weights(best_weights)
            if verbose:
                print(f"  Restored best weights (val_loss={best_val_loss:.4f})")

        return history


# =========================================================================
# Main training orchestrator
# =========================================================================

def train_on_dataset(ds_config: dict, output_dir: str,
                     epochs: int = 200, batch_size: int = None,
                     lr: float = None, patience: int = 40,
                     use_auto_weights: bool = True,
                     run_id: int = 0, verbose: int = 1):
    """Train KePIN on a single dataset.

    Args:
        ds_config:        dataset config dict (for GenericTimeSeriesDataset)
        output_dir:       directory for saving results
        epochs:           max training epochs
        batch_size:       override auto batch size
        lr:               override auto learning rate
        patience:         early stopping patience
        use_auto_weights: use Kendall uncertainty weighting
        run_id:           run index for multi-run experiments
        verbose:          verbosity level

    Returns:
        results: dict with metrics, eigenvalue history, predictions
    """
    ds_name = ds_config.get("name", "unknown")
    print(f"\n{'='*60}")
    print(f"  KePIN Training: {ds_name} (run {run_id})")
    print(f"{'='*60}")

    # --- Load dataset ---
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    print(ds.summary())

    # --- Convert 4D → 3D for KePIN ---
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)

    # --- Apply EMA smoothing (auto-tuned alpha) ---
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    print(f"  EMA α = {ema_alpha:.4f} (auto-tuned)")

    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]

    # --- Auto-configure ---
    arch_config = auto_configure(n_feat, seq_len, n_train)
    print(f"  Architecture tier: {arch_config['tier']} ({arch_config['n_blocks']} blocks)")
    print(f"  Latent dim: {arch_config['latent_dim']}, Rollout: {arch_config['rollout']}")

    # --- Auto batch size and LR (A100-optimised) ---
    if batch_size is None:
        batch_size = get_batch_size(n_train, seq_len, n_feat, model_type="kepin")
    if lr is None:
        lr = get_learning_rate(batch_size, base_lr=0.001, base_batch=256)

    print(f"  Batch size: {batch_size}, LR: {lr:.6f} (A100-optimised)")

    # --- Build model ---
    model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                              arch_config=arch_config)
    print(model.summary_config())

    # --- Create loss function ---
    loss_fn = make_kepin_loss(
        loss_weights_layer=model.loss_weight_layer if use_auto_weights else None,
        use_auto_weights=use_auto_weights,
    )

    # --- Optimizer ---
    optimizer = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    # --- Train ---
    trainer = KePINTrainer(model, loss_fn, optimizer)
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

    print(f"\n  Results for {ds_name}:")
    print(f"    RMSE:            {test_rmse:.4f}")
    print(f"    MAE:             {test_mae:.4f}")
    print(f"    Mono violation:  {mono_viol:.6f}")
    print(f"    Slope RMSE:      {slope_err:.4f}")

    # --- Koopman eigenvalue analysis ---
    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1]
    print(f"    Top |λ|:         {eig_mags[:5]}")

    # --- Eigenvalue recovery (for synthetic ODE only) ---
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
    run_tag = f"{ds_name}_run{run_id}"

    # Save model
    model_path = os.path.join(output_dir, f"kepin_{run_tag}.weights.h5")
    model.save_weights(model_path)
    print(f"    Saved weights: {model_path}")

    # Save predictions
    pred_path = os.path.join(output_dir, f"predictions_{run_tag}.npz")
    np.savez(pred_path, y_true=Y_test, y_pred=Y_pred)

    # Save eigenvalue history
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

    # Save loss convergence plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#0173B2")
    axes[0].plot(history["val_loss"], label="Val", color="#DE8F05")
    axes[0].set_title(f"Total Loss — {ds_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_rmse"], label="Train", color="#0173B2")
    axes[1].plot(history["val_rmse"], label="Val", color="#DE8F05")
    axes[1].set_title("RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # Eigenvalue magnitude evolution
    eig_hist = np.array(history["eigenvalues"])  # (n_epochs, d) complex
    for mode_idx in range(min(4, eig_hist.shape[1])):
        axes[2].plot(np.abs(eig_hist[:, mode_idx]),
                     label=f"Mode {mode_idx+1}", alpha=0.8)
    axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="|λ|=1")
    axes[2].set_title("Koopman |λ| Evolution")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"training_{run_tag}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    results = {
        "dataset": ds_name,
        "run_id": run_id,
        "rmse": test_rmse,
        "mae": test_mae,
        "mono_violation": mono_viol,
        "slope_rmse": slope_err,
        "best_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else float("inf"),
        "epochs_trained": len(history["epoch"]),
        "eigenvalue_mags": eig_mags.tolist(),
        "eigenvalue_recovery": eig_recovery,
        "ema_alpha": ema_alpha,
        "arch_tier": arch_config["tier"],
        "batch_size": batch_size,
        "lr": lr,
    }

    return results


# =========================================================================
# Multi-dataset orchestrator
# =========================================================================

def train_all(config_path: str, output_base: str = None,
              epochs: int = 200, n_runs: int = 1, **kwargs):
    """Train KePIN on all datasets in a JSON config file.

    Args:
        config_path: path to JSON config (list of dataset configs)
        output_base: base directory for results
        epochs:      max epochs per dataset
        n_runs:      number of independent runs per dataset

    Returns:
        all_results: list of result dicts
    """
    with open(config_path, "r") as f:
        configs = json.load(f)

    if output_base is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"kepin_{timestamp}")

    os.makedirs(output_base, exist_ok=True)

    all_results = []
    for ds_idx, ds_config in enumerate(configs):
        ds_name = ds_config.get("name", f"dataset_{ds_idx}")
        ds_output = os.path.join(output_base, ds_name)

        for run in range(n_runs):
            try:
                result = train_on_dataset(
                    ds_config, ds_output,
                    epochs=epochs, run_id=run, **kwargs,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n  ✗ Failed on {ds_name} run {run}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "dataset": ds_name, "run_id": run,
                    "error": str(e),
                })

    # --- Cross-dataset summary ---
    print(f"\n{'='*60}")
    print("  CROSS-DATASET SUMMARY")
    print(f"{'='*60}")

    summary_rows = []
    for r in all_results:
        if "error" not in r:
            summary_rows.append({
                "Dataset": r["dataset"],
                "Run": r["run_id"],
                "RMSE": r["rmse"],
                "MAE": r["mae"],
                "MonoViol": r["mono_violation"],
                "SlopeRMSE": r["slope_rmse"],
                "Tier": r["arch_tier"],
                "Epochs": r["epochs_trained"],
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(output_base, "kepin_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary to {summary_path}")

        # Save full results as JSON
        json_path = os.path.join(output_base, "kepin_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    return all_results


# =========================================================================
# CLI
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="KePIN: Koopman-Enhanced Physics-Informed Network Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with dataset definitions")
    parser.add_argument("--dataset_idx", type=int, default=None,
                        help="Train only on this dataset index from config")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["synthetic", "synthetic_ode", "csv",
                                 "nasa_bearing", "battery", "phm2012"],
                        help="Quick single-dataset mode (no config file needed)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (auto-selects for A100 if omitted)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override LR (auto-scaled with batch size if omitted)")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--no_auto_weights", action="store_true",
                        help="Disable Kendall uncertainty weighting")
    parser.add_argument("--no_mixed_precision", action="store_true",
                        help="Disable float16 mixed precision")
    parser.add_argument("--no_xla", action="store_true",
                        help="Disable XLA JIT compilation")

    # Quick-mode dataset paths
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        # Multi-dataset training from config
        if args.dataset_idx is not None:
            with open(args.config) as f:
                configs = json.load(f)
            config = configs[args.dataset_idx]
            output_dir = args.output_dir or os.path.join(
                _project_dir, "experiments_result", "kepin",
                config.get("name", f"ds_{args.dataset_idx}"),
            )
            train_on_dataset(
                config, output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                use_auto_weights=not args.no_auto_weights,
            )
        else:
            train_all(
                args.config,
                output_base=args.output_dir,
                epochs=args.epochs,
                n_runs=args.n_runs,
                batch_size=args.batch_size,
                lr=args.lr,
                patience=args.patience,
                use_auto_weights=not args.no_auto_weights,
            )
    elif args.mode:
        # Quick single-dataset mode
        if args.mode == "synthetic":
            config = {
                "type": "synthetic", "name": "Synthetic_Quick",
                "sequence_length": 30, "rul_cap": 125,
                "n_units_train": 80, "n_units_test": 20,
            }
        elif args.mode == "synthetic_ode":
            config = {
                "type": "synthetic_ode", "name": "Synthetic_ODE",
                "sequence_length": 30, "rul_cap": 200,
                "n_units_train": 100, "n_units_test": 25,
            }
        elif args.mode in ("nasa_bearing", "phm2012"):
            if not args.data_dir:
                print(f"Error: --data_dir required for {args.mode}")
                sys.exit(1)
            config = {
                "type": args.mode, "name": args.mode,
                "sequence_length": 40, "data_dir": args.data_dir,
            }
        elif args.mode == "battery":
            if not args.csv_path:
                print("Error: --csv_path required for battery mode")
                sys.exit(1)
            config = {
                "type": "battery", "name": "Battery",
                "sequence_length": 20, "csv_path": args.csv_path,
            }
        else:
            print(f"Mode {args.mode} requires additional arguments.")
            sys.exit(1)

        output_dir = args.output_dir or os.path.join(
            _project_dir, "experiments_result", "kepin", config["name"],
        )
        train_on_dataset(
            config, output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            use_auto_weights=not args.no_auto_weights,
        )
    else:
        print("Provide --config or --mode. Use -h for help.")
        sys.exit(1)


if __name__ == "__main__":
    main()
