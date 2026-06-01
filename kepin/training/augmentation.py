# -*- coding: utf-8 -*-
"""
Data augmentation for time-series training.
"""

import numpy as np
import tensorflow as tf


def augment_time_series(X, Y, noise_std=0.01, seed=None):
    """Additive Gaussian noise augmentation.

    Returns:
        (X_augmented, Y_unchanged)
    """
    rng = np.random.RandomState(seed)
    X_aug = X + rng.randn(*X.shape).astype(np.float32) * noise_std
    return X_aug, Y


def mixup_batch(X1, Y1, X2, Y2, alpha=0.2):
    """Mixup augmentation for time series.

    Linearly interpolates between two samples using λ ~ Beta-like.
    """
    batch_size = tf.shape(X1)[0]
    lam = tf.random.uniform((batch_size, 1, 1), 0.0, 1.0)
    if alpha > 0:
        lam = tf.pow(lam, 1.0 / (alpha + 1e-6))
    lam_y = tf.reshape(lam[:, 0, 0], (-1, 1))
    X_mix = lam * X1 + (1.0 - lam) * X2
    Y_mix = lam_y * Y1 + (1.0 - lam_y) * Y2
    return X_mix, Y_mix


def smooth_rul_labels(Y, sigma=2.0):
    """Smooth RUL labels near the cap region to soften the kink.

    Applies small Gaussian perturbation near the RUL cap.
    """
    Y_flat = Y.flatten().copy()
    if sigma > 0:
        cap_mask = Y_flat >= (Y_flat.max() * 0.9)
        cap_noise = np.random.normal(0, sigma * 0.5, cap_mask.sum()).astype(np.float32)
        Y_flat[cap_mask] = np.clip(Y_flat[cap_mask] + cap_noise, 0, Y_flat.max())
    return Y_flat.reshape(Y.shape)
