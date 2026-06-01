#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all experiments for the revised KePIN paper:
  1. Ablation study on ALL 4 C-MAPSS datasets + Jena Climate (5 datasets total)
  2. Uses existing optimized C-MAPSS main results
  3. Generates updated figures and LaTeX tables

Usage:
  cd /workspace/rahul/KPDD/code
  python run_paper_experiments.py
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
    physics_metrics_np, SEED,
)
from kepin_ablation import (
    get_ablation_configs, BaselineFCN, build_ablation_model,
    compute_all_metrics, r_squared_np,
)
from gpu_config import setup_gpu, build_tf_dataset, get_batch_size, get_learning_rate

# GPU setup
setup_gpu(mixed_precision=False, xla=False, verbose=True)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.abspath(os.path.join(_script_dir, ".."))

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================================
# Dataset configs
# =========================================================================

def get_all_configs():
    """Return all dataset configs for ablation: 4 C-MAPSS + Jena Climate."""
    # Load C-MAPSS configs
    config_path = os.path.join(_script_dir, "datasets_kepin_config.json")
    with open(config_path) as f:
        cmapss_configs = json.load(f)

    # Jena Climate config
    jena_config = {
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

    all_configs = cmapss_configs + [jena_config]
    return all_configs


def get_domain_for_config(ds_config):
    """Determine domain from dataset config."""
    ds_type = ds_config.get("type", "csv")
    if ds_type == "weather":
        return "weather"
    elif ds_type == "finance":
        return "finance"
    elif ds_type == "synthetic_ode":
        return "synthetic"
    else:
        return "degradation"


def get_training_params(domain, ds_name):
    """Domain-specific training parameters for ablation."""
    params = {
        "degradation": {"epochs": 200, "patience": 40, "batch_size": 128, "lr": 0.0008},
        "weather": {"epochs": 250, "patience": 50, "batch_size": 256, "lr": 0.0005},
    }

    # Per-dataset overrides for C-MAPSS
    overrides = {
        "CMAPSS_FD002": {"batch_size": 256, "lr": 0.0006},
        "CMAPSS_FD004": {"batch_size": 256, "lr": 0.0006},
    }

    tp = params.get(domain, {"epochs": 200, "patience": 40, "batch_size": 128, "lr": 0.0008})
    tp.update(overrides.get(ds_name, {}))
    return tp


def get_arch_config(domain, n_features, seq_len, n_train):
    """Get architecture config with domain-specific improvements."""
    base = auto_configure(n_features, seq_len, n_train)

    if domain == "weather":
        base["latent_dim"] = 128
        base["lstm_units"] = 128
        base["dropout"] = 0.2
        base["rollout"] = 5
        base["n_heads"] = 8
        base["head_key_dim"] = 32

    return base


# =========================================================================
# Ablation runner
# =========================================================================

def run_ablation_single(ab_config, ds_config, output_dir, verbose=1):
    """Train one ablation config on one dataset."""
    ab_name = ab_config["name"]
    ds_name = ds_config.get("name", "unknown")
    domain = get_domain_for_config(ds_config)
    tp = get_training_params(domain, ds_name)

    print(f"\n  --- {ab_name} on {ds_name} ---")

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

    # Architecture
    arch_config = get_arch_config(domain, n_feat, seq_len, n_train)

    # Domain mode
    domain_mode = "forecasting" if domain in ("weather",) else "degradation"
    n_active_losses = 4 if domain_mode == "forecasting" else 7

    # Set domain mode on ablation config
    ab_config_copy = dict(ab_config)
    ab_config_copy["domain_mode"] = domain_mode

    # Build model
    if not ab_config_copy["use_koopman"]:
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
    if ab_config_copy["use_auto_weights"] and ab_config_copy["use_koopman"]:
        loss_fn = make_kepin_loss(
            loss_weights_layer=model.loss_weight_layer,
            use_auto_weights=True,
            domain_mode=domain_mode,
        )
    else:
        fixed_w = ab_config_copy.get("fixed_weights") or {
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

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{ab_config_copy['tag']}_{ds_name}_run0"
    np.savez(os.path.join(output_dir, f"pred_{tag}.npz"),
             y_true=Y_test, y_pred=Y_pred)

    result = {
        "ablation": ab_name,
        "ablation_tag": ab_config_copy["tag"],
        "dataset": ds_name,
        "domain": domain,
        **metrics,
        "epochs_trained": len(history["epoch"]),
    }
    return result


def run_full_ablation():
    """Run ablation on all 5 datasets (4 C-MAPSS + Jena Climate)."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = os.path.join(_project_dir, "experiments_result",
                               f"kepin_ablation_full_{timestamp}")
    os.makedirs(output_base, exist_ok=True)

    all_configs = get_all_configs()
    ab_configs = get_ablation_configs()

    print(f"{'='*70}")
    print(f"  KePIN FULL ABLATION STUDY")
    print(f"  {len(ab_configs)} configs x {len(all_configs)} datasets")
    print(f"  Datasets: {[c['name'] for c in all_configs]}")
    print(f"  Output: {output_base}")
    print(f"{'='*70}")

    all_results = []

    for ds_config in all_configs:
        ds_name = ds_config.get("name", "unknown")
        ds_dir = os.path.join(output_base, ds_name)

        for ab_config in ab_configs:
            try:
                result = run_ablation_single(
                    ab_config, ds_config, ds_dir, verbose=1,
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
                    "error": str(e),
                })

        # Save intermediate results
        pd.DataFrame(all_results).to_csv(
            os.path.join(output_base, "ablation_results_partial.csv"), index=False)

    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_base, "ablation_results.csv"), index=False)

    json_path = os.path.join(output_base, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*70}")
    print("  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    for ds in results_df["dataset"].unique():
        ds_df = results_df[results_df["dataset"] == ds]
        print(f"\n  {ds}:")
        print(f"  {'Config':<25} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
        print(f"  {'-'*55}")
        for _, row in ds_df.iterrows():
            if "error" in row and pd.notna(row.get("error")):
                print(f"  {row['ablation']:<25} ERROR: {row['error']}")
            else:
                print(f"  {row['ablation']:<25} {row.get('RMSE', 0):8.4f} "
                      f"{row.get('MAE', 0):8.4f} {row.get('R2', 0):8.4f}")

    print(f"\nResults saved to: {output_base}")
    return results_df, output_base


if __name__ == "__main__":
    results_df, output_dir = run_full_ablation()
