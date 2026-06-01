# -*- coding: utf-8 -*-
"""
Evaluation metrics for KePIN.

Includes standard regression metrics and physics-informed diagnostics.
"""

import math
import numpy as np


def rmse_np(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(((y_true.flatten() - y_pred.flatten()) ** 2).mean()))


def mae_np(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.abs(y_true.flatten() - y_pred.flatten()).mean())


def r2_np(y_true, y_pred):
    """Coefficient of determination (R²)."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2) + 1e-10
    return float(1 - ss_res / ss_tot)


def physics_metrics_np(y_true, y_pred):
    """Compute monotonicity violation and slope RMSE.

    Returns:
        (mono_violation, slope_rmse)
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    td = yt[1:] - yt[:-1]
    pd_ = yp[1:] - yp[:-1]
    mask = (td < 0).astype(np.float32)
    denom = mask.sum() + 1e-8
    mono = float((np.maximum(pd_, 0) * mask).sum() / denom)
    slope = float(math.sqrt(((pd_ - td) ** 2 * mask).sum() / denom))
    return mono, slope


def nasa_score(y_true, y_pred):
    """NASA Prognostics scoring function.

    S = Σ exp(-d/13) - 1  for early predictions (d < 0)
      + Σ exp(d/10) - 1   for late predictions (d > 0)

    where d = y_pred - y_true.
    """
    d = y_pred.flatten() - y_true.flatten()
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(np.sum(scores))


def eigenvalue_recovery_error(learned_eigs, true_eigs):
    """Compute eigenvalue recovery error using Hungarian matching.

    Returns dict with mean/max magnitude error and mean phase error.
    """
    from scipy.optimize import linear_sum_assignment

    n_true = len(true_eigs)
    n_learned = len(learned_eigs)
    cost = np.zeros((n_true, n_learned))
    for i in range(n_true):
        for j in range(n_learned):
            cost[i, j] = abs(abs(true_eigs[i]) - abs(learned_eigs[j]))

    row_ind, col_ind = linear_sum_assignment(cost)
    mag_errors, phase_errors = [], []
    for i, j in zip(row_ind, col_ind):
        t_mag = abs(true_eigs[i])
        l_mag = abs(learned_eigs[j])
        mag_errors.append(abs(t_mag - l_mag) / (t_mag + 1e-10))
        phase_errors.append(abs(np.angle(true_eigs[i]) - np.angle(learned_eigs[j])))

    return {
        "mean_mag_error": float(np.mean(mag_errors)),
        "max_mag_error": float(np.max(mag_errors)),
        "mean_phase_error": float(np.mean(phase_errors)),
    }
