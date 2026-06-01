# -*- coding: utf-8 -*-
"""
Preprocessing utilities for KePIN.
"""

import numpy as np


def convert_4d_to_3d(X):
    """Convert (N, T, 1, F) arrays to (N, T, F) for the KePIN encoder."""
    if X.ndim == 4 and X.shape[2] == 1:
        return X[:, :, 0, :]
    elif X.ndim == 3:
        return X
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}")


def apply_ema_smoothing(data_3d, alpha=None):
    """Apply exponential moving average along the time axis.

    If ``alpha`` is None, auto-tune: α = 2 / (1 + seq_len), clamped to [0.05, 0.5].

    Returns:
        (smoothed_data, alpha_used)
    """
    from scipy.signal import lfilter, lfilter_zi

    if alpha is None:
        seq_len = data_3d.shape[1]
        alpha = max(0.05, min(2.0 / (1.0 + seq_len), 0.5))

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
