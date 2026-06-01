# -*- coding: utf-8 -*-
"""
KePIN Multi-Domain Ablation Study.

Runs the ablation study across FD004, Weather, Finance, and Synthetic ODE.
Uses different configs per domain, keeping the same 5 ablation configs.

Usage:
  python kepin_ablation_multidomain.py --domain fd004
  python kepin_ablation_multidomain.py --domain weather
  python kepin_ablation_multidomain.py --domain finance
  python kepin_ablation_multidomain.py --domain synthetic_ode
  python kepin_ablation_multidomain.py --domain all
"""

import argparse
import datetime
import json
import math
import os
import sys

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
)
from kepin_ablation import (
    get_ablation_configs, BaselineFCN, build_ablation_model,
    compute_all_metrics, print_summary,
)
from kepin_optimize_multidomain import (
    get_optimized_weather_config, get_optimized_finance_config,
    get_optimized_synthetic_config, get_domain_arch_config,
    get_domain_training_params,
)
from gpu_config import setup_gpu, build_tf_dataset

# GPU setup
setup_gpu(mixed_precision=False, xla=False, verbose=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))


# =========================================================================
# Run a single ablation config on a dataset
# =========================================================================

def run_ablation_on_dataset(ab_config, ds_config, domain, output_dir,
                            epochs=200, patience=40, batch_size=128,
                            lr=0.0008, run_id=0, verbose=1):
    """Run one ablation config on one dataset with domain-specific params."""
    
    ds_name = ds_config.get("name", "unknown")
    ab_name = ab_config["name"]
    
    print(f"\n  --- {ab_name} on {ds_name} (run {run_id}) ---")
    
    # Load and prepare data
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)
    
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]
    
    # Get domain-specific architecture
    arch_config = get_domain_arch_config(domain, n_feat, seq_len, n_train)
    
    # Determine domain mode
    ds_type = ds_config.get("type", "csv")
    if ds_type in ("weather", "finance"):
        domain_mode = "forecasting"
        n_active_losses = 4
    else:
        domain_mode = "degradation"
        n_active_losses = 7
    
    # Build model based on ablation config
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
    optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    trainer = KePINTrainer(model, loss_fn, optimizer, clip_norm=2.0)
    
    history = trainer.fit(
        X_train, Y_train, X_test, Y_test,
        epochs=epochs, batch_size=batch_size,
        patience=patience, initial_lr=lr,
        verbose=verbose,
    )
    
    # Evaluate
    Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
    metrics = compute_all_metrics(Y_test, Y_pred)
    
    print(f"    RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
          f"R2={metrics['R2']:.4f}  Mono={metrics['MonoViol']:.6f}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    run_tag = f"{ds_name}_{ab_config['tag']}_run{run_id}"
    
    pred_path = os.path.join(output_dir, f"predictions_{run_tag}.npz")
    np.savez(pred_path, y_true=Y_test, y_pred=Y_pred)
    
    model_path = os.path.join(output_dir, f"kepin_{run_tag}.weights.h5")
    model.save_weights(model_path)
    
    # Eigenvalue info
    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1][:5]
    
    eig_recovery = None
    if hasattr(ds, "ode_true_K_eigenvalues"):
        true_K_eigs = ds.ode_true_K_eigenvalues
        eig_recovery = eigenvalue_recovery_error(final_eigs, true_K_eigs)
    
    result = {
        "ablation": ab_name,
        "ablation_tag": ab_config["tag"],
        "dataset": ds_name,
        "domain": domain,
        "run_id": run_id,
        **metrics,
        "epochs_trained": len(history["epoch"]),
        "best_val_loss": float(min(history["val_loss"])),
        "top_eig_mags": eig_mags.tolist(),
        "description": ab_config.get("description", ""),
    }
    if eig_recovery:
        result["eig_recovery"] = eig_recovery
    
    return result


# =========================================================================
# Domain dataset configs
# =========================================================================

def get_fd004_config():
    """FD004 config from the CMAPSS dataset."""
    with open(os.path.join(_script_dir, "datasets_cmapss_config.json")) as f:
        configs = json.load(f)
    # FD004 is index 3
    return configs[3]


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="KePIN Multi-Domain Ablation")
    parser.add_argument("--domain", type=str, default="all",
                        choices=["fd004", "weather", "finance", "synthetic_ode", "all"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        output_base = os.path.join(_project_dir, "experiments_result",
                                   f"kepin_ablation_multi_{timestamp}")
    else:
        output_base = args.output_dir
    
    # Map domain to configs
    domain_configs = {}
    if args.domain in ("fd004", "all"):
        domain_configs["fd004"] = get_fd004_config()
    if args.domain in ("weather", "all"):
        domain_configs["weather"] = get_optimized_weather_config()
    if args.domain in ("finance", "all"):
        domain_configs["finance"] = get_optimized_finance_config()
    if args.domain in ("synthetic_ode", "all"):
        domain_configs["synthetic_ode"] = get_optimized_synthetic_config()
    
    # Domain-specific training params
    domain_train_params = {
        "fd004": {"epochs": 200, "patience": 40, "batch_size": 128, "lr": 0.0008},
        "weather": {"epochs": 200, "patience": 40, "batch_size": 256, "lr": 0.0005},
        "finance": {"epochs": 200, "patience": 50, "batch_size": 64, "lr": 0.0003},
        "synthetic_ode": {"epochs": 200, "patience": 40, "batch_size": 64, "lr": 0.0005},
    }
    
    ab_configs = get_ablation_configs()
    
    print(f"{'='*70}")
    print(f"  MULTI-DOMAIN ABLATION STUDY")
    print(f"  {len(ab_configs)} configs x {len(domain_configs)} domains")
    print(f"  Domains: {list(domain_configs.keys())}")
    print(f"  Output: {output_base}")
    print(f"{'='*70}")
    
    all_results = []
    
    for domain, ds_config in domain_configs.items():
        ds_name = ds_config.get("name", domain)
        ds_dir = os.path.join(output_base, ds_name)
        tp = domain_train_params.get(domain, {})
        
        print(f"\n{'='*60}")
        print(f"  DOMAIN: {domain} ({ds_name})")
        print(f"{'='*60}")
        
        for ab_config in ab_configs:
            try:
                result = run_ablation_on_dataset(
                    ab_config, ds_config, domain, ds_dir,
                    epochs=tp.get("epochs", args.epochs),
                    patience=tp.get("patience", args.patience),
                    batch_size=tp.get("batch_size", args.batch_size),
                    lr=tp.get("lr", args.lr),
                    run_id=0, verbose=args.verbose,
                )
                all_results.append(result)
            except Exception as e:
                print(f"    FAILED: {ab_config['name']} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "ablation": ab_config["name"],
                    "ablation_tag": ab_config["tag"],
                    "dataset": ds_name,
                    "domain": domain,
                    "run_id": 0,
                    "error": str(e),
                })
        
        # Save intermediate results after each domain
        os.makedirs(output_base, exist_ok=True)
        pd.DataFrame(all_results).to_csv(
            os.path.join(output_base, "ablation_results.csv"), index=False
        )
    
    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_base, "ablation_results.csv"), index=False)
    
    with open(os.path.join(output_base, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    valid_df = results_df[results_df.get("RMSE", pd.Series(dtype=float)).notna()].copy()
    if len(valid_df) > 0:
        print(f"\n{'='*80}")
        print("  MULTI-DOMAIN ABLATION SUMMARY")
        print(f"{'='*80}")
        
        for ds in sorted(valid_df["dataset"].unique()):
            ds_df = valid_df[valid_df["dataset"] == ds]
            print(f"\n  {ds}:")
            print(f"  {'Config':<25}  {'RMSE':>8}  {'MAE':>8}  {'R2':>8}  {'Mono':>8}")
            print(f"  {'-'*25}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}")
            for _, row in ds_df.iterrows():
                print(f"  {row['ablation']:<25}  {row['RMSE']:8.4f}  {row['MAE']:8.4f}  "
                      f"{row['R2']:8.4f}  {row['MonoViol']:8.4f}")
    
    print(f"\n  Results saved to: {output_base}")
    
    return results_df


if __name__ == "__main__":
    main()
