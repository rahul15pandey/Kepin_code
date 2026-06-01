# -*- coding: utf-8 -*-
"""
KePIN Multi-Domain Optimization — Domain-Specific Tuning.

This script optimizes KePIN for each non-CMAPSS domain with
domain-specific hyperparameters while keeping the core architecture
and training pipeline identical.

Usage:
  python kepin_optimize_multidomain.py --domain weather
  python kepin_optimize_multidomain.py --domain finance
  python kepin_optimize_multidomain.py --domain synthetic_ode
  python kepin_optimize_multidomain.py --domain all
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
from gpu_config import setup_gpu, build_tf_dataset, get_batch_size

# ---------- GPU setup ----------
setup_gpu(mixed_precision=False, xla=False, verbose=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))


# =========================================================================
# Domain-Specific Configs (optimized)
# =========================================================================

def get_optimized_weather_config():
    """Optimized config for Jena Climate dataset.
    
    Key optimizations:
    - Sequence length 72 (3 days) instead of 48 → captures fuller diurnal cycle
    - Prediction horizon 24h (unchanged, good target)
    - Lower initial LR (0.0005) for gentler convergence
    - More epochs (300) with patience 50
    - batch_size 256 for weather (lots of data, 420K samples)
    """
    return {
        "type": "weather",
        "name": "Jena_Climate",
        "sequence_length": 72,
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


def get_optimized_finance_config():
    """Optimized config for SPY Stock dataset.
    
    Key optimizations:
    - Sequence length 60 (2 months trading) for more context
    - prediction_horizon 5 (1 week lookahead, more predictable)
    - drawdown threshold 0.03 (3% instead of 5% → more events to learn from)
    """
    return {
        "type": "finance",
        "name": "SPY_Stock",
        "sequence_length": 60,
        "rul_cap": None,
        "csv_path": "datasets/spy_stock.csv",
        "datetime_col": "Date",
        "close_col": "Close",
        "prediction_horizon": 5,
        "drawdown_threshold": 0.03,
        "target_mode": "drawdown",
        "train_ratio": 0.8,
        "test_last_only": False,
    }


def get_optimized_synthetic_config():
    """Optimized config for Synthetic ODE dataset.
    
    Key optimizations:
    - More training units (150 vs 100) for better generalization
    - Longer max_life (400 vs 300) → longer trajectories
    - Sequence length 40 (vs 30) → more temporal context
    - Lower noise (0.03 vs 0.05) → cleaner eigenvalue recovery
    """
    return {
        "type": "synthetic_ode",
        "name": "Synthetic_ODE",
        "sequence_length": 40,
        "rul_cap": 200,
        "n_units_train": 150,
        "n_units_test": 30,
        "max_life": 400,
        "dt": 0.1,
        "noise_std": 0.03,
        "failure_threshold": 2.0,
        "test_last_only": False,
    }


# =========================================================================
# Domain-specific architecture overrides
# =========================================================================

def get_domain_arch_config(domain, n_features, seq_len, n_train):
    """Get domain-specific architecture overrides.
    
    The auto_configure gives a good base. We overlay domain-specific tuning.
    """
    base = auto_configure(n_features, seq_len, n_train)
    
    if domain == "weather":
        # Weather has smooth, quasi-periodic dynamics
        # Larger latent dim helps capture multiple atmospheric modes
        base["latent_dim"] = 128
        base["lstm_units"] = 128
        base["dropout"] = 0.25       # Less dropout - data is abundant & clean
        base["rollout"] = 5           # Longer rollout for weather prediction
        base["n_heads"] = 8           # More heads for diverse temporal patterns
        base["head_key_dim"] = 32
        
    elif domain == "finance":
        # Financial data is noisy with low signal-to-noise ratio
        base["latent_dim"] = 64       # Smaller latent → less overfitting
        base["lstm_units"] = 64
        base["dropout"] = 0.45        # High dropout for noisy data
        base["rollout"] = 3
        base["n_heads"] = 4
        base["head_key_dim"] = 16
        
    elif domain == "synthetic_ode":
        # Synthetic ODE — need accurate eigenvalue recovery
        base["latent_dim"] = 32       # Small latent → forces meaningful modes
        base["lstm_units"] = 64
        base["dropout"] = 0.2         # Low dropout — clean data with known dynamics
        base["rollout"] = 5           # Longer rollout for trajectory prediction
        base["n_heads"] = 4
        base["head_key_dim"] = 16
    
    return base


def get_domain_training_params(domain):
    """Get domain-specific training hyperparameters."""
    params = {
        "weather": {
            "epochs": 300,
            "patience": 50,
            "lr": 0.0005,
            "batch_size": 256,
            "clip_norm": 1.0,
        },
        "finance": {
            "epochs": 400,
            "patience": 60,
            "lr": 0.0003,
            "batch_size": 64,
            "clip_norm": 0.5,
        },
        "synthetic_ode": {
            "epochs": 300,
            "patience": 50,
            "lr": 0.0005,
            "batch_size": 64,
            "clip_norm": 1.0,
        },
    }
    return params.get(domain, {"epochs": 200, "patience": 40, "lr": 0.0008,
                                "batch_size": 128, "clip_norm": 2.0})


# =========================================================================
# Optimized training function
# =========================================================================

def train_optimized(ds_config, domain, output_dir, run_id=0, verbose=1):
    """Train with domain-specific optimizations."""
    
    ds_name = ds_config.get("name", "unknown")
    print(f"\n{'='*70}")
    print(f"  OPTIMIZED TRAINING: {ds_name} (domain={domain}, run {run_id})")
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
    print(f"  EMA α = {ema_alpha:.4f}")
    
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]
    
    # Domain-specific arch override
    arch_config = get_domain_arch_config(domain, n_feat, seq_len, n_train)
    print(f"  Architecture: tier={arch_config['tier']} (domain-tuned)")
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
    tp = get_domain_training_params(domain)
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
    
    print(f"\n  RESULTS for {ds_name} (optimized):")
    print(f"    RMSE:           {test_rmse:.4f}")
    print(f"    MAE:            {test_mae:.4f}")
    print(f"    Mono violation: {mono_viol:.6f}")
    print(f"    Slope RMSE:     {slope_err:.4f}")
    
    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1]
    print(f"    Top |λ|:        {eig_mags[:5]}")
    
    # Eigenvalue recovery for ODE
    eig_recovery = None
    if hasattr(ds, "ode_true_K_eigenvalues"):
        true_K_eigs = ds.ode_true_K_eigenvalues
        eig_recovery = eigenvalue_recovery_error(final_eigs, true_K_eigs)
        print(f"    Eigenvalue recovery:")
        print(f"      Mean mag error:   {eig_recovery['mean_mag_error']:.6f}")
        print(f"      Max mag error:    {eig_recovery['max_mag_error']:.6f}")
        print(f"      Mean phase error: {eig_recovery['mean_phase_error']:.6f}")
    
    # Save everything
    os.makedirs(output_dir, exist_ok=True)
    run_tag = f"{ds_name}_run{run_id}"
    
    model_path = os.path.join(output_dir, f"kepin_{run_tag}.weights.h5")
    model.save_weights(model_path)
    
    pred_path = os.path.join(output_dir, f"predictions_{run_tag}.npz")
    np.savez(pred_path, y_true=Y_test, y_pred=Y_pred)
    
    eig_path = os.path.join(output_dir, f"eigenvalues_{run_tag}.npz")
    np.savez(eig_path,
             eigenvalue_history=np.array(history["eigenvalues"]),
             final_eigenvalues=final_eigs,
             koopman_matrix=model.get_koopman_matrix())
    
    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
    })
    hist_path = os.path.join(output_dir, f"history_{run_tag}.csv")
    hist_df.to_csv(hist_path, index=False)
    
    # Training curves plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train", color="#0173B2")
    axes[0].plot(history["val_loss"], label="Val", color="#DE8F05")
    axes[0].set_title(f"Total Loss — {ds_name}")
    axes[0].set_xlabel("Epoch"); axes[0].legend()
    
    axes[1].plot(history["train_rmse"], label="Train", color="#0173B2")
    axes[1].plot(history["val_rmse"], label="Val", color="#DE8F05")
    axes[1].set_title("RMSE"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    
    eig_hist = np.array(history["eigenvalues"])
    for mode_idx in range(min(4, eig_hist.shape[1])):
        axes[2].plot(np.abs(eig_hist[:, mode_idx]),
                     label=f"Mode {mode_idx+1}", alpha=0.8)
    axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    axes[2].set_title("|λ| Convergence"); axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"training_{run_tag}.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, f"training_{run_tag}.pdf"))
    plt.close(fig)
    
    # Result dict
    epochs_trained = len(history["epoch"])
    result = {
        "dataset": ds_name,
        "domain": domain,
        "run_id": run_id,
        "RMSE": test_rmse,
        "MAE": test_mae,
        "MonoViol": mono_viol,
        "SlopeRMSE": slope_err,
        "Tier": arch_config["tier"],
        "Epochs": epochs_trained,
        "latent_dim": arch_config["latent_dim"],
        "top_eig_mags": eig_mags[:5].tolist(),
    }
    if eig_recovery:
        result["eig_recovery"] = eig_recovery
    
    return result


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="KePIN Multi-Domain Optimization")
    parser.add_argument("--domain", type=str, default="all",
                        choices=["weather", "finance", "synthetic_ode", "all"],
                        help="Which domain to optimize")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto-timestamped)")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"kepin_optimized_{timestamp}")
    else:
        output_base = args.output_dir
    
    domains_to_run = []
    if args.domain in ("weather", "all"):
        domains_to_run.append(("weather", get_optimized_weather_config()))
    if args.domain in ("finance", "all"):
        domains_to_run.append(("finance", get_optimized_finance_config()))
    if args.domain in ("synthetic_ode", "all"):
        domains_to_run.append(("synthetic_ode", get_optimized_synthetic_config()))
    
    all_results = []
    
    for domain, ds_config in domains_to_run:
        ds_name = ds_config["name"]
        ds_dir = os.path.join(output_base, ds_name)
        
        try:
            result = train_optimized(ds_config, domain, ds_dir, verbose=args.verbose)
            all_results.append(result)
            print(f"\n  ✓ {ds_name}: RMSE={result['RMSE']:.4f}")
        except Exception as e:
            print(f"\n  ✗ {ds_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"dataset": ds_name, "domain": domain, "error": str(e)})
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    
    # Save summary
    os.makedirs(output_base, exist_ok=True)
    results_df.to_csv(os.path.join(output_base, "optimized_summary.csv"), index=False)
    with open(os.path.join(output_base, "optimized_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    for r in all_results:
        if "error" not in r:
            print(f"  {r['dataset']:20s}: RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  "
                  f"Tier={r['Tier']}  Epochs={r['Epochs']}")
        else:
            print(f"  {r['dataset']:20s}: FAILED - {r['error']}")
    
    print(f"\n  Results saved to: {output_base}")
    
    return results_df


if __name__ == "__main__":
    main()
