#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry-point for KePIN evaluation.

Usage examples:
    # Evaluate a single .npz prediction file
    python scripts/evaluate.py --npz experiments_result/kepin/CMAPSS_FD001/predictions_CMAPSS_FD001_run0.npz

    # Evaluate all .npz files in a directory
    python scripts/evaluate.py --dir experiments_result/kepin_20260224_071535

    # Evaluate with NASA scoring function
    python scripts/evaluate.py --npz path/to/predictions.npz --nasa_score
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _project_dir)

from kepin.utils.metrics import rmse_np, mae_np, r2_np, physics_metrics_np, nasa_score


def evaluate_npz(npz_path, compute_nasa=False):
    """Evaluate a single predictions .npz file."""
    data = np.load(npz_path)
    y_true = data["y_true"].flatten()
    y_pred = data["y_pred"].flatten()

    rmse = rmse_np(y_true, y_pred)
    mae = mae_np(y_true, y_pred)
    r2 = r2_np(y_true, y_pred)
    mono_viol, slope_err = physics_metrics_np(y_true, y_pred)

    result = {
        "file": os.path.basename(npz_path),
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Mono_Violation": mono_viol,
        "Slope_RMSE": slope_err,
        "N_samples": len(y_true),
    }

    if compute_nasa:
        result["NASA_Score"] = nasa_score(y_true, y_pred)

    return result


def main():
    p = argparse.ArgumentParser(description="KePIN Evaluation CLI")
    p.add_argument("--npz", type=str, help="Single .npz predictions file")
    p.add_argument("--dir", type=str, help="Directory with .npz predictions")
    p.add_argument("--nasa_score", action="store_true",
                   help="Compute asymmetric NASA scoring function")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Save results table to CSV")
    args = p.parse_args()

    if not args.npz and not args.dir:
        sys.exit("Provide --npz or --dir. Use -h for help.")

    files = []
    if args.npz:
        files.append(args.npz)
    if args.dir:
        files.extend(sorted(glob.glob(os.path.join(args.dir, "**/predictions_*.npz"),
                                       recursive=True)))

    if not files:
        sys.exit("No prediction files found.")

    rows = []
    for f in files:
        try:
            rows.append(evaluate_npz(f, compute_nasa=args.nasa_score))
        except Exception as e:
            print(f"  Error evaluating {f}: {e}")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
