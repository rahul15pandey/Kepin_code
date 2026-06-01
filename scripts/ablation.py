#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry-point for KePIN ablation studies.

Usage:
    python scripts/ablation.py --config configs/datasets_kepin_config.json \\
        --dataset_idx 0 --epochs 200 --patience 40
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)

from scripts.train import train_on_dataset
from kepin.utils.gpu import setup_gpu
from kepin.utils.reproducibility import set_seed


ABLATION_VARIANTS = {
    "full":              dict(),                                    # baseline: all losses
    "no_koopman_loss":   dict(domain_mode_override="no_koopman"),   # drop Koopman 1-step
    "no_spectral":       dict(domain_mode_override="no_spectral"),  # drop spectral regulariser
    "no_monotonicity":   dict(domain_mode_override="no_mono"),      # drop monotonicity
    "no_multi_step":     dict(domain_mode_override="no_mstep"),     # drop multi-step rollout
    "no_auto_weights":   dict(use_auto_weights=False),              # fixed equal weights
    "no_koopman_module": dict(domain_mode_override="no_koopman_module"),  # remove Koopman layer
}


def run_ablation(ds_config, output_base, *, epochs=200, patience=40,
                 n_runs=1, variants=None, data_root=None, seed=42):
    """Run ablation study on a single dataset."""
    if variants is None:
        variants = list(ABLATION_VARIANTS.keys())

    all_results = []
    for variant in variants:
        print(f"\n{'#'*60}")
        print(f"  ABLATION: {variant}")
        print(f"{'#'*60}")
        ablation_kw = ABLATION_VARIANTS.get(variant, {})
        out = os.path.join(output_base, variant)

        for run in range(n_runs):
            try:
                r = train_on_dataset(
                    ds_config, out,
                    epochs=epochs, patience=patience, run_id=run,
                    use_auto_weights=ablation_kw.get("use_auto_weights", True),
                    data_root=data_root, seed=seed,
                )
                r["variant"] = variant
                all_results.append(r)
            except Exception as e:
                import traceback; traceback.print_exc()
                all_results.append({"variant": variant, "run_id": run, "error": str(e)})

    rows = [r for r in all_results if "error" not in r]
    if rows:
        df = pd.DataFrame(rows)
        print(f"\n{'='*60}\n  ABLATION RESULTS\n{'='*60}")
        print(df[["variant", "rmse", "mae", "mono_violation"]].to_string(index=False))
        df.to_csv(os.path.join(output_base, "ablation_results.csv"), index=False)
        with open(os.path.join(output_base, "ablation_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    return all_results


def main():
    p = argparse.ArgumentParser(description="KePIN Ablation Study CLI")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dataset_idx", type=int, default=0)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--n_runs", type=int, default=1)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--variants", nargs="+", default=None,
                   choices=list(ABLATION_VARIANTS.keys()))
    p.add_argument("--data_root", type=str, default=None,
                   help="Root folder for datasets (overrides KEPIN_DATA_ROOT)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Enable deterministic kernels (may disable XLA/FP16)")
    p.add_argument("--no_xla", action="store_true")
    p.add_argument("--no_mixed_precision", action="store_true")
    args = p.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)
    setup_gpu(
        mixed_precision=not args.no_mixed_precision and not args.deterministic,
        xla=not args.no_xla and not args.deterministic,
    )

    with open(args.config) as f:
        configs = json.load(f)
    ds_config = configs[args.dataset_idx]

    out = args.output_dir or os.path.join(
        _project_dir, "experiments_result",
        f"ablation_{ds_config.get('name', 'unknown')}")
    os.makedirs(out, exist_ok=True)

    run_ablation(ds_config, out,
                 epochs=args.epochs, patience=args.patience,
                 n_runs=args.n_runs, variants=args.variants,
                 data_root=args.data_root, seed=args.seed)


if __name__ == "__main__":
    main()
