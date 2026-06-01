#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry-point for KePIN training.

Usage examples:
    # Train on all datasets in a config file
    python scripts/train.py --config configs/datasets_kepin_config.json

    # Train a single dataset by index
    python scripts/train.py --config configs/datasets_kepin_config.json --dataset_idx 0

    # Quick synthetic test
    python scripts/train.py --mode synthetic --epochs 50

    # C-MAPSS optimised (SWA + mixup)
    python scripts/train.py --config configs/datasets_kepin_config.json --enhanced \\
        --epochs 250 --n_runs 3 --patience 50
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Resolve project root (one level up from scripts/)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)

from kepin.utils.gpu import setup_gpu, get_batch_size, get_learning_rate
from kepin.utils.metrics import rmse_np, mae_np, physics_metrics_np, eigenvalue_recovery_error
from kepin.utils.preprocessing import convert_4d_to_3d, apply_ema_smoothing
from kepin.utils.reproducibility import set_seed
from kepin.models.kepin_model import auto_configure, build_kepin_model
from kepin.losses.composite import make_kepin_loss
from kepin.training.trainer import KePINTrainer, EnhancedKePINTrainer
from kepin.data.loader import load_dataset_from_config

import tensorflow as tf
import keras


# ---------------------------------------------------------------------------
# Single-dataset training
# ---------------------------------------------------------------------------
def train_on_dataset(ds_config, output_dir, *,
                     epochs=200, batch_size=None, lr=None,
                     patience=40, run_id=0, use_auto_weights=True,
                     enhanced=False, data_root=None, seed=42, verbose=1):
    """Train KePIN on a single dataset configuration."""
    ds_name = ds_config.get("name", "unknown")
    print(f"\n{'='*60}")
    print(f"  KePIN Training: {ds_name} (run {run_id})")
    print(f"{'='*60}")

    ds = load_dataset_from_config(ds_config, data_root=data_root)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    print(ds.summary())

    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)

    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    print(f"  EMA α = {ema_alpha:.4f}")

    seq_len, n_feat, n_train = X_train.shape[1], X_train.shape[2], X_train.shape[0]

    arch_config = auto_configure(n_feat, seq_len, n_train)
    print(f"  Tier: {arch_config['tier']} ({arch_config['n_blocks']} blocks)")

    if batch_size is None:
        batch_size = get_batch_size(n_train, seq_len, n_feat, model_type="kepin")
    if lr is None:
        lr = get_learning_rate(batch_size, base_lr=0.001, base_batch=256)
    print(f"  Batch: {batch_size}, LR: {lr:.6f}")

    ds_type = ds_config.get("type", "csv")
    if ds_type in ("weather", "weather_synthetic", "finance",
                    "finance_synthetic", "fluid_dynamics", "energy_systems"):
        domain_mode, n_active_losses = "forecasting", 4
    else:
        domain_mode, n_active_losses = "degradation", 7

    model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                              arch_config=arch_config,
                              n_active_losses=n_active_losses)
    print(f"  Domain: {domain_mode} ({n_active_losses} losses)")

    loss_fn = make_kepin_loss(
        loss_weights_layer=model.loss_weight_layer if use_auto_weights else None,
        use_auto_weights=use_auto_weights,
        domain_mode=domain_mode,
    )
    optimizer = keras.optimizers.Adam(learning_rate=float(lr), clipnorm=1.0)

    if enhanced:
        trainer = EnhancedKePINTrainer(model, loss_fn, optimizer)
        history, swa_weights = trainer.fit_enhanced(
            X_train, Y_train, X_test, Y_test,
            epochs=epochs, batch_size=batch_size,
            patience=patience, initial_lr=lr, seed=seed, verbose=verbose,
        )
        if swa_weights is not None:
            model.set_weights(swa_weights)
    else:
        trainer = KePINTrainer(model, loss_fn, optimizer)
        history = trainer.fit(
            X_train, Y_train, X_test, Y_test,
            epochs=epochs, batch_size=batch_size,
            patience=patience, initial_lr=lr, seed=seed, verbose=verbose,
        )

    # --- Evaluate ---
    Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
    test_rmse = rmse_np(Y_test, Y_pred)
    test_mae = mae_np(Y_test, Y_pred)
    mono_viol, slope_err = physics_metrics_np(Y_test, Y_pred)

    print(f"\n  Results for {ds_name}:")
    print(f"    RMSE:           {test_rmse:.4f}")
    print(f"    MAE:            {test_mae:.4f}")
    print(f"    Mono violation: {mono_viol:.6f}")
    print(f"    Slope RMSE:     {slope_err:.4f}")

    final_eigs = model.get_eigenvalues()
    eig_mags = np.sort(np.abs(final_eigs))[::-1]
    print(f"    Top |λ|:        {eig_mags[:5]}")

    eig_recovery = None
    if hasattr(ds, "ode_true_K_eigenvalues"):
        true_K_eigs = ds.ode_true_K_eigenvalues
        eig_recovery = eigenvalue_recovery_error(final_eigs, true_K_eigs)

    # --- Save artefacts ---
    os.makedirs(output_dir, exist_ok=True)
    run_tag = f"{ds_name}_run{run_id}"

    model.save_weights(os.path.join(output_dir, f"kepin_{run_tag}.weights.h5"))
    np.savez(os.path.join(output_dir, f"predictions_{run_tag}.npz"),
             y_true=Y_test, y_pred=Y_pred)
    np.savez(os.path.join(output_dir, f"eigenvalues_{run_tag}.npz"),
             eigenvalue_history=np.array(history["eigenvalues"]),
             final_eigenvalues=final_eigs,
             koopman_matrix=model.get_koopman_matrix())

    hist_df = pd.DataFrame({
        k: v for k, v in history.items()
        if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
    })
    hist_df.to_csv(os.path.join(output_dir, f"history_{run_tag}.csv"), index=False)

    _save_training_figure(history, ds_name, run_tag, output_dir)

    return {
        "dataset": ds_name, "run_id": run_id,
        "rmse": test_rmse, "mae": test_mae,
        "mono_violation": mono_viol, "slope_rmse": slope_err,
        "epochs_trained": len(history["epoch"]),
        "eigenvalue_mags": eig_mags.tolist(),
        "eigenvalue_recovery": eig_recovery,
        "ema_alpha": ema_alpha, "arch_tier": arch_config["tier"],
        "batch_size": batch_size, "lr": lr,
    }


# ---------------------------------------------------------------------------
# Multi-dataset orchestrator
# ---------------------------------------------------------------------------
def train_all(config_path, output_base=None, epochs=200, n_runs=1,
              data_root=None, seed=42, **kw):
    with open(config_path) as f:
        configs = json.load(f)

    if output_base is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = os.path.join(_project_dir, "experiments_result", f"kepin_{ts}")
    os.makedirs(output_base, exist_ok=True)

    all_results = []
    for idx, cfg in enumerate(configs):
        name = cfg.get("name", f"dataset_{idx}")
        for run in range(n_runs):
            try:
                r = train_on_dataset(cfg, os.path.join(output_base, name),
                                     epochs=epochs, run_id=run,
                                     data_root=data_root, seed=seed, **kw)
                all_results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                all_results.append({"dataset": name, "run_id": run, "error": str(e)})

    _print_summary(all_results, output_base)
    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_summary(results, output_base):
    rows = [r for r in results if "error" not in r]
    if not rows:
        return
    df = pd.DataFrame(rows)
    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    print(df[["dataset", "run_id", "rmse", "mae", "mono_violation"]].to_string(index=False))
    df.to_csv(os.path.join(output_base, "kepin_summary.csv"), index=False)
    with open(os.path.join(output_base, "kepin_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


def _save_training_figure(history, ds_name, run_tag, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"], label="Train", color="#0173B2")
    axes[0].plot(history["val_loss"], label="Val", color="#DE8F05")
    axes[0].set_title(f"Total Loss — {ds_name}")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(history["train_rmse"], label="Train", color="#0173B2")
    axes[1].plot(history["val_rmse"], label="Val", color="#DE8F05")
    axes[1].set_title("RMSE"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    eig_hist = np.array(history["eigenvalues"])
    for i in range(min(4, eig_hist.shape[1])):
        axes[2].plot(np.abs(eig_hist[:, i]), label=f"Mode {i+1}", alpha=0.8)
    axes[2].axhline(1.0, color="red", linestyle="--", alpha=0.5, label="|λ|=1")
    axes[2].set_title("Koopman |λ| Evolution")
    axes[2].set_xlabel("Epoch"); axes[2].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_{run_tag}.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="KePIN Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=str, help="JSON dataset config path")
    p.add_argument("--run_config", type=str, default=None,
                   help="Optional JSON run config (overrides defaults)")
    p.add_argument("--dataset_idx", type=int, default=None)
    p.add_argument("--mode", type=str, default=None,
                   choices=["synthetic", "synthetic_ode", "csv",
                            "nasa_bearing", "battery", "phm2012"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--n_runs", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--enhanced", action="store_true",
                   help="Use EnhancedKePINTrainer (SWA + mixup)")
    p.add_argument("--no_auto_weights", action="store_true")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--csv_path", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None,
                   help="Root folder for datasets (overrides KEPIN_DATA_ROOT)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable deterministic kernels (may disable XLA/FP16)")
    p.add_argument("--no_xla", action="store_true",
                   help="Disable XLA JIT even when GPU is available")
    p.add_argument("--no_mixed_precision", action="store_true",
                   help="Disable mixed precision (force float32)")
    return p.parse_args()


def main():
    args = parse_args()
    run_cfg = {}
    if args.run_config:
        with open(args.run_config) as f:
            run_cfg = json.load(f)

    seed = args.seed if args.seed is not None else run_cfg.get("seed", 42)
    deterministic = args.deterministic or run_cfg.get("deterministic", False)
    data_root = args.data_root or run_cfg.get("data_root")

    set_seed(seed, deterministic=deterministic)

    setup_gpu(
        mixed_precision=not args.no_mixed_precision and not deterministic,
        xla=not args.no_xla and not deterministic,
    )

    epochs = args.epochs if args.epochs is not None else run_cfg.get("epochs", 200)
    batch_size = args.batch_size if args.batch_size is not None else run_cfg.get("batch_size")
    lr = args.lr if args.lr is not None else run_cfg.get("lr")
    patience = args.patience if args.patience is not None else run_cfg.get("patience", 40)
    n_runs = args.n_runs if args.n_runs is not None else run_cfg.get("n_runs", 1)
    enhanced = args.enhanced or run_cfg.get("enhanced", False)

    if args.no_auto_weights:
        use_auto_weights = False
    else:
        use_auto_weights = run_cfg.get("use_auto_weights", True)

    kw = dict(epochs=epochs, batch_size=batch_size, lr=lr,
              patience=patience, use_auto_weights=use_auto_weights,
              enhanced=enhanced, data_root=data_root)

    config_path = args.config or run_cfg.get("dataset_config")
    dataset_idx = args.dataset_idx if args.dataset_idx is not None else run_cfg.get("dataset_idx")
    output_dir = args.output_dir or run_cfg.get("output_dir")

    if config_path:
        if dataset_idx is not None:
            with open(config_path) as f:
                configs = json.load(f)
            cfg = configs[dataset_idx]
            out = output_dir or os.path.join(
                _project_dir, "experiments_result", "kepin",
                cfg.get("name", f"ds_{dataset_idx}"))
            train_on_dataset(cfg, out, seed=seed, **kw)
        else:
            train_all(config_path, output_base=output_dir,
                      n_runs=n_runs, seed=seed, **kw)
    elif args.mode:
        quick_configs = {
            "synthetic": {"type": "synthetic", "name": "Synthetic_Quick",
                          "sequence_length": 30, "rul_cap": 125,
                          "n_units_train": 80, "n_units_test": 20},
            "synthetic_ode": {"type": "synthetic_ode", "name": "Synthetic_ODE",
                              "sequence_length": 30, "rul_cap": 200,
                              "n_units_train": 100, "n_units_test": 25},
        }
        if args.mode in quick_configs:
            cfg = quick_configs[args.mode]
        elif args.mode in ("nasa_bearing", "phm2012"):
            if not args.data_dir:
                sys.exit(f"--data_dir required for {args.mode}")
            cfg = {"type": args.mode, "name": args.mode,
                   "sequence_length": 40, "data_dir": args.data_dir}
        elif args.mode == "battery":
            if not args.csv_path:
                sys.exit("--csv_path required for battery mode")
            cfg = {"type": "battery", "name": "Battery",
                   "sequence_length": 20, "csv_path": args.csv_path}
        else:
            sys.exit(f"Mode '{args.mode}' requires additional arguments.")

        out = output_dir or os.path.join(
            _project_dir, "experiments_result", "kepin", cfg["name"])
        train_on_dataset(cfg, out, seed=seed, **kw)
    else:
        sys.exit("Provide --config or --mode. Use -h for help.")


if __name__ == "__main__":
    main()
