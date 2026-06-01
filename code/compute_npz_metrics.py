#!/usr/bin/env python3
"""Compute metrics from saved prediction artifacts (.npz).

This repo writes prediction artifacts like:
  predictions_<DATASET>_run0.npz  (y_true, y_pred)
  predictions_<DATASET>_ensemble.npz

This script recomputes RMSE/MAE and the repo's physics metrics
(monotonicity violation + slope RMSE) directly from those arrays.

Examples:
  python compute_npz_metrics.py experiments_result/.../predictions_CMAPSS_FD004_run0.npz

  # Ensemble from multiple run files (averages y_pred)
  python compute_npz_metrics.py \
    experiments_result/.../predictions_CMAPSS_FD004_run0.npz \
    experiments_result/.../predictions_CMAPSS_FD004_run1.npz \
    experiments_result/.../predictions_CMAPSS_FD004_run2.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)
    return float(np.mean(np.abs(yt - yp)))


def physics_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Match the repo's `physics_metrics_np` behavior.

    Note: This operates on the flattened arrays in their stored order.
    """
    yt = y_true.reshape(-1).astype(np.float64)
    yp = y_pred.reshape(-1).astype(np.float64)

    td = yt[1:] - yt[:-1]
    pd = yp[1:] - yp[:-1]

    mask = (td < 0).astype(np.float64)
    denom = float(mask.sum() + 1e-8)

    mono = float((np.maximum(pd, 0.0) * mask).sum() / denom)
    slope = float(math.sqrt(((pd - td) ** 2 * mask).sum() / denom))

    return mono, slope


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "y_true" not in data or "y_pred" not in data:
        raise KeyError(f"{path} missing y_true/y_pred keys")
    return data["y_true"], data["y_pred"]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mono, slope = physics_metrics_np(y_true, y_pred)
    return {
        "rmse": rmse_np(y_true, y_pred),
        "mae": mae_np(y_true, y_pred),
        "mono": mono,
        "slope": slope,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", type=Path, help="One or more predictions_*.npz files")
    args = ap.parse_args()

    paths: List[Path] = args.npz
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    y_trues: List[np.ndarray] = []
    y_preds: List[np.ndarray] = []
    for p in paths:
        yt, yp = load_npz(p)
        y_trues.append(yt)
        y_preds.append(yp)

    # Basic consistency check
    base = y_trues[0].reshape(-1)
    for i, yt in enumerate(y_trues[1:], start=1):
        if yt.reshape(-1).shape != base.shape or not np.allclose(yt.reshape(-1), base, atol=0.0, rtol=0.0):
            raise ValueError(
                f"y_true mismatch between {paths[0]} and {paths[i]} (shape/values differ)"
            )

    # Per-file metrics
    for p, yp in zip(paths, y_preds):
        m = metrics_dict(base, yp)
        print(f"{p}")
        print(
            f"  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  Mono={m['mono']:.6f}  Slope={m['slope']:.4f}"
        )

    # Ensemble if multiple
    if len(y_preds) > 1:
        y_ens = np.mean(np.stack([yp.reshape(-1).astype(np.float64) for yp in y_preds], axis=0), axis=0)
        m = metrics_dict(base, y_ens)
        print("ENSEMBLE (mean of y_pred)")
        print(
            f"  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  Mono={m['mono']:.6f}  Slope={m['slope']:.4f}"
        )


if __name__ == "__main__":
    main()
